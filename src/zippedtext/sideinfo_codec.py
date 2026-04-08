from __future__ import annotations

import struct

from .format import (
    SECTION_CODEC_RAW,
    SECTION_CODEC_ZSTD,
    SECTION_FLAG_CODEC_MASK,
    V3Section,
)

_SMALL_SECTION_THRESHOLD = 96
_COMPRESSIBLE_SECTION_THRESHOLD = 128


def make_section(data: bytes, *, prefer_compression: bool = False) -> V3Section:
    flags = choose_section_flags(data, prefer_compression=prefer_compression)
    return V3Section(data=data, flags=flags)


def choose_section_flags(data: bytes, *, prefer_compression: bool = False) -> int:
    if not data:
        return SECTION_CODEC_RAW
    if len(data) < _SMALL_SECTION_THRESHOLD and not prefer_compression:
        return SECTION_CODEC_RAW
    if len(data) >= _COMPRESSIBLE_SECTION_THRESHOLD or prefer_compression:
        compressed = _compress_zstd(data)
        if len(compressed) + _section_header_cost() < len(data):
            return SECTION_CODEC_ZSTD
    return SECTION_CODEC_RAW


def section_stored_size(section: bytes | V3Section) -> int:
    coerced = section if isinstance(section, V3Section) else V3Section(data=section)
    return len(encode_section_payload(coerced.data, coerced.flags))


def encode_section_payload(data: bytes, flags: int) -> bytes:
    codec = flags & SECTION_FLAG_CODEC_MASK
    if codec == SECTION_CODEC_RAW:
        return data
    if codec == SECTION_CODEC_ZSTD:
        return _compress_zstd(data)
    raise ValueError(f"unsupported section codec {codec}")


def pack_string(value: str) -> bytes:
    encoded = value.encode("utf-8")
    return struct.pack("<H", len(encoded)) + encoded


def unpack_string(data: bytes, offset: int) -> tuple[str, int]:
    if offset + 2 > len(data):
        raise ValueError("truncated string length")
    length = struct.unpack("<H", data[offset:offset + 2])[0]
    offset += 2
    end = offset + length
    if end > len(data):
        raise ValueError("truncated string payload")
    return data[offset:end].decode("utf-8"), end


def _compress_zstd(data: bytes) -> bytes:
    import zstandard

    return zstandard.ZstdCompressor(level=19).compress(data)


def _section_header_cost() -> int:
    return 8
