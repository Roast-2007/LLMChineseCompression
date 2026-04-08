"""Binary file format (.ztxt) reader/writer.

Format v1 (legacy, read-only):
  [magic:4] [version:1] [mode:1] [model_id:2] [token_count:4]
  [original_bytes:4] [crc32:4] [model_data_len:4]
  [model_data:var] [compressed_body:var]

Format v2 (current legacy online/offline format):
  [magic:4] [version:1] [mode:1] [flags:1] [max_order:1]
  [token_count:4] [original_bytes:4] [crc32:4]
  [model_data_len:4] [phrase_table_len:4] [reserved:4]
  [model_data:var] [phrase_table:var] [compressed_body:var]

Format v3 (structured online format):
  [magic:4] [version:1] [mode:1] [flags:1] [max_order:1]
  [token_count:4] [original_bytes:4] [crc32:4]
  [metadata_len:4] [payload_len:4] [reserved:4]
  [sections:var] [payload_body:var]

Each v3 section is encoded as:
  [section_count:uint16]
  repeated [section_type:uint8] [flags:uint8] [reserved:uint16] [length:uint32] [data:var]
"""

from __future__ import annotations

import struct
import zlib
from dataclasses import dataclass

MAGIC = b"ZTXT"
VERSION_V1 = 0x01
VERSION_V2 = 0x02
VERSION_V3 = 0x03
VERSION = VERSION_V3  # current write version

# v1 header layout (24 bytes)
_V1_FMT = "<4sBBHIIII"
_V1_SIZE = struct.calcsize(_V1_FMT)  # 24

# v2/v3 header layout (32 bytes)
_V2_FMT = "<4sBBBBIIIIII"
_V2_SIZE = struct.calcsize(_V2_FMT)  # 32
_V3_FMT = _V2_FMT
_V3_SIZE = _V2_SIZE

# v3 section entry layout (8 bytes)
_V3_SECTION_FMT = "<BBHI"
_V3_SECTION_SIZE = struct.calcsize(_V3_SECTION_FMT)

# Mode constants
MODE_ONLINE = 0x00
MODE_OFFLINE = 0x01
MODE_CODEGEN = 0x02

# Flag bits
FLAG_PHRASE_ENCODING = 0x01
FLAG_HAS_PRIORS = 0x02

# Legacy model ID constants (v1 compat only)
MODEL_DEEPSEEK_CHAT = 0x0001
MODEL_DEEPSEEK_REASONER = 0x0002

# Default PPM order
DEFAULT_MAX_ORDER = 4

# v3 section constants
SECTION_ANALYSIS = 0x01
SECTION_PHRASE_TABLE = 0x02
SECTION_SEGMENTS = 0x03
SECTION_STATS = 0x04


@dataclass(frozen=True)
class Header:
    """Unified header for v1/v2/v3 files."""

    mode: int
    model_id: int
    token_count: int
    original_bytes: int
    crc32: int
    model_data_len: int
    flags: int = 0
    max_order: int = DEFAULT_MAX_ORDER
    phrase_table_len: int = 0
    version: int = VERSION_V2


def compute_crc32(data: bytes) -> int:
    return zlib.crc32(data) & 0xFFFFFFFF


# ------------------------------------------------------------------
# Write helpers
# ------------------------------------------------------------------


def write_file(
    header: Header,
    model_data: bytes,
    compressed_body: bytes,
    phrase_table: bytes = b"",
) -> bytes:
    """Serialize a complete .ztxt v2 file to bytes."""
    header_bytes = struct.pack(
        _V2_FMT,
        MAGIC,
        VERSION_V2,
        header.mode,
        header.flags,
        header.max_order,
        header.token_count,
        header.original_bytes,
        header.crc32,
        header.model_data_len,
        len(phrase_table),
        0,
    )
    return header_bytes + model_data + phrase_table + compressed_body


def write_file_v3(
    header: Header,
    sections: dict[int, bytes],
    compressed_body: bytes,
) -> bytes:
    """Serialize a complete .ztxt v3 file to bytes."""
    metadata = _pack_v3_sections(sections)
    header_bytes = struct.pack(
        _V3_FMT,
        MAGIC,
        VERSION_V3,
        header.mode,
        header.flags,
        header.max_order,
        header.token_count,
        header.original_bytes,
        header.crc32,
        len(metadata),
        len(compressed_body),
        0,
    )
    return header_bytes + metadata + compressed_body


# ------------------------------------------------------------------
# Read dispatchers
# ------------------------------------------------------------------


def read_file(data: bytes) -> tuple[Header, bytes, bytes]:
    """Deserialize a .ztxt file.

    Returns ``(header, model_data, compressed_body)``.
    For v2 files with a phrase table, the phrase table bytes are skipped in
    this view — use :func:`read_file_v2` directly.
    For v3 files, the returned *model_data* is the raw metadata/section blob —
    use :func:`read_file_v3` directly to parse structured sections.
    """
    if len(data) < 5:
        raise ValueError(f"file too short: {len(data)} bytes")

    version = data[4]
    if data[:4] != MAGIC:
        raise ValueError(f"invalid magic: {data[:4]!r}, expected {MAGIC!r}")

    if version == VERSION_V1:
        return _read_v1(data)
    if version == VERSION_V2:
        return _read_v2(data)
    if version == VERSION_V3:
        return _read_v3(data)
    raise ValueError(f"unsupported version: {version}")


def read_file_v2(data: bytes) -> tuple[Header, bytes, bytes, bytes]:
    """Read a v2 file and return ``(header, model_data, phrase_table, body)``."""
    if len(data) < _V2_SIZE:
        raise ValueError(f"file too short for v2: {len(data)} < {_V2_SIZE}")

    fields = struct.unpack(_V2_FMT, data[:_V2_SIZE])
    (magic, version, mode, flags, max_order,
     token_count, original_bytes, crc32_val,
     model_data_len, phrase_table_len, _reserved) = fields

    if magic != MAGIC:
        raise ValueError(f"invalid magic: {magic!r}")
    if version != VERSION_V2:
        raise ValueError(f"expected v2, got version {version}")

    offset = _V2_SIZE
    model_data = data[offset:offset + model_data_len]
    offset += model_data_len
    phrase_table = data[offset:offset + phrase_table_len]
    offset += phrase_table_len
    compressed_body = data[offset:]

    header = Header(
        mode=mode,
        model_id=0,
        token_count=token_count,
        original_bytes=original_bytes,
        crc32=crc32_val,
        model_data_len=model_data_len,
        flags=flags,
        max_order=max_order,
        phrase_table_len=phrase_table_len,
        version=VERSION_V2,
    )
    return header, model_data, phrase_table, compressed_body


def read_file_v3(data: bytes) -> tuple[Header, dict[int, bytes], bytes]:
    """Read a v3 file and return ``(header, sections, body)``."""
    if len(data) < _V3_SIZE:
        raise ValueError(f"file too short for v3: {len(data)} < {_V3_SIZE}")

    fields = struct.unpack(_V3_FMT, data[:_V3_SIZE])
    (magic, version, mode, flags, max_order,
     token_count, original_bytes, crc32_val,
     metadata_len, payload_len, _reserved) = fields

    if magic != MAGIC:
        raise ValueError(f"invalid magic: {magic!r}")
    if version != VERSION_V3:
        raise ValueError(f"expected v3, got version {version}")

    metadata_start = _V3_SIZE
    metadata_end = metadata_start + metadata_len
    payload_end = metadata_end + payload_len
    if payload_end > len(data):
        raise ValueError("invalid v3 file: truncated payload")
    metadata = data[metadata_start:metadata_end]
    body = data[metadata_end:payload_end]
    sections = _unpack_v3_sections(metadata)

    header = Header(
        mode=mode,
        model_id=0,
        token_count=token_count,
        original_bytes=original_bytes,
        crc32=crc32_val,
        model_data_len=metadata_len,
        flags=flags,
        max_order=max_order,
        phrase_table_len=0,
        version=VERSION_V3,
    )
    return header, sections, body


# ------------------------------------------------------------------
# Internal readers
# ------------------------------------------------------------------


def _read_v1(data: bytes) -> tuple[Header, bytes, bytes]:
    if len(data) < _V1_SIZE:
        raise ValueError(f"file too short for v1: {len(data)} < {_V1_SIZE}")

    fields = struct.unpack(_V1_FMT, data[:_V1_SIZE])
    (magic, version, mode, model_id,
     token_count, original_bytes, crc32_val, model_data_len) = fields

    offset = _V1_SIZE
    model_data = data[offset:offset + model_data_len]
    offset += model_data_len
    compressed_body = data[offset:]

    header = Header(
        mode=mode,
        model_id=model_id,
        token_count=token_count,
        original_bytes=original_bytes,
        crc32=crc32_val,
        model_data_len=model_data_len,
        flags=0,
        max_order=DEFAULT_MAX_ORDER,
        phrase_table_len=0,
        version=VERSION_V1,
    )
    return header, model_data, compressed_body


def _read_v2(data: bytes) -> tuple[Header, bytes, bytes]:
    header, model_data, _phrase_table, compressed_body = read_file_v2(data)
    return header, model_data, compressed_body


def _read_v3(data: bytes) -> tuple[Header, bytes, bytes]:
    if len(data) < _V3_SIZE:
        raise ValueError(f"file too short for v3: {len(data)} < {_V3_SIZE}")

    fields = struct.unpack(_V3_FMT, data[:_V3_SIZE])
    (magic, version, mode, flags, max_order,
     token_count, original_bytes, crc32_val,
     metadata_len, payload_len, _reserved) = fields

    if magic != MAGIC:
        raise ValueError(f"invalid magic: {magic!r}")
    if version != VERSION_V3:
        raise ValueError(f"expected v3, got version {version}")

    metadata_start = _V3_SIZE
    metadata_end = metadata_start + metadata_len
    payload_end = metadata_end + payload_len
    if payload_end > len(data):
        raise ValueError("invalid v3 file: truncated payload")
    metadata = data[metadata_start:metadata_end]
    body = data[metadata_end:payload_end]

    header = Header(
        mode=mode,
        model_id=0,
        token_count=token_count,
        original_bytes=original_bytes,
        crc32=crc32_val,
        model_data_len=metadata_len,
        flags=flags,
        max_order=max_order,
        phrase_table_len=0,
        version=VERSION_V3,
    )
    return header, metadata, body


# ------------------------------------------------------------------
# v3 section helpers
# ------------------------------------------------------------------


def _pack_v3_sections(sections: dict[int, bytes]) -> bytes:
    items = sorted((stype, payload) for stype, payload in sections.items() if payload is not None)
    parts = [struct.pack("<H", len(items))]
    for section_type, payload in items:
        parts.append(struct.pack(_V3_SECTION_FMT, section_type, 0, 0, len(payload)))
        parts.append(payload)
    return b"".join(parts)


def _unpack_v3_sections(data: bytes) -> dict[int, bytes]:
    if not data:
        return {}
    if len(data) < 2:
        raise ValueError("invalid v3 metadata: missing section count")
    section_count = struct.unpack("<H", data[:2])[0]
    offset = 2
    sections: dict[int, bytes] = {}
    for _ in range(section_count):
        if offset + _V3_SECTION_SIZE > len(data):
            raise ValueError("invalid v3 metadata: truncated section header")
        section_type, _flags, _reserved, length = struct.unpack(
            _V3_SECTION_FMT,
            data[offset:offset + _V3_SECTION_SIZE],
        )
        offset += _V3_SECTION_SIZE
        end = offset + length
        if end > len(data):
            raise ValueError("invalid v3 metadata: truncated section payload")
        sections[section_type] = data[offset:end]
        offset = end
    return sections
