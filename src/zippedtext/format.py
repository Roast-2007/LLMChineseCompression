"""Binary file format (.ztxt) reader/writer.

Format v1 (legacy, read-only):
  [magic:4] [version:1] [mode:1] [model_id:2] [token_count:4]
  [original_bytes:4] [crc32:4] [model_data_len:4]
  [model_data:var] [compressed_body:var]

Format v2 (current):
  [magic:4] [version:1] [mode:1] [flags:1] [max_order:1]
  [token_count:4] [original_bytes:4] [crc32:4]
  [model_data_len:4] [phrase_table_len:4] [reserved:4]
  [model_data:var] [phrase_table:var] [compressed_body:var]
"""

from __future__ import annotations

import struct
import zlib
from dataclasses import dataclass

MAGIC = b"ZTXT"
VERSION_V1 = 0x01
VERSION_V2 = 0x02
VERSION = VERSION_V2  # current write version

# v1 header layout (24 bytes)
_V1_FMT = "<4sBBHIIII"
_V1_SIZE = struct.calcsize(_V1_FMT)  # 24

# v2 header layout (32 bytes)
_V2_FMT = "<4sBBBBIIIIII"
_V2_SIZE = struct.calcsize(_V2_FMT)  # 32

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


@dataclass(frozen=True)
class Header:
    """Unified header for both v1 and v2 files.

    For v1 files read from disk, ``model_id`` is populated from the
    2-byte enum field; ``flags``, ``max_order``, and ``phrase_table_len``
    are set to defaults.  For v2 files, ``model_id`` is always 0 (the
    model name is stored in model_data instead).
    """
    mode: int
    model_id: int
    token_count: int
    original_bytes: int
    crc32: int
    model_data_len: int
    # v2 fields (defaults for v1 compat)
    flags: int = 0
    max_order: int = DEFAULT_MAX_ORDER
    phrase_table_len: int = 0


def compute_crc32(data: bytes) -> int:
    return zlib.crc32(data) & 0xFFFFFFFF


# ------------------------------------------------------------------
# Write (always v2)
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
        0,  # reserved
    )
    return header_bytes + model_data + phrase_table + compressed_body


# ------------------------------------------------------------------
# Read (dispatches v1 / v2)
# ------------------------------------------------------------------

def read_file(data: bytes) -> tuple[Header, bytes, bytes]:
    """Deserialize a .ztxt file.

    Returns ``(header, model_data, compressed_body)``.
    For v2 files with a phrase table, the phrase table bytes are
    accessible via ``header.phrase_table_len`` offset inside the
    returned *model_data* — or use :func:`read_file_v2` directly.
    """
    if len(data) < 5:
        raise ValueError(f"file too short: {len(data)} bytes")

    version = data[4]
    if data[:4] != MAGIC:
        raise ValueError(f"invalid magic: {data[:4]!r}, expected {MAGIC!r}")

    if version == VERSION_V1:
        return _read_v1(data)
    elif version == VERSION_V2:
        return _read_v2(data)
    else:
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
    )
    return header, model_data, phrase_table, compressed_body


# ------------------------------------------------------------------
# Internal readers
# ------------------------------------------------------------------

def _read_v1(data: bytes) -> tuple[Header, bytes, bytes]:
    """Read a legacy v1 file."""
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
    )
    return header, model_data, compressed_body


def _read_v2(data: bytes) -> tuple[Header, bytes, bytes]:
    """Read a v2 file, returning (header, model_data, compressed_body).

    The phrase table (if any) is skipped in this view — use
    :func:`read_file_v2` to access it separately.
    """
    header, model_data, _phrase_table, compressed_body = read_file_v2(data)
    return header, model_data, compressed_body
