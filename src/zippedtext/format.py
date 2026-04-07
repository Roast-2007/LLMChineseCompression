"""Binary file format (.ztxt) reader/writer.

Format v1:
  [magic:4] [version:1] [mode:1] [model_id:2] [token_count:4]
  [original_bytes:4] [crc32:4] [model_data_len:4]
  [model_data:var] [compressed_body:var]
"""

from __future__ import annotations

import struct
import zlib
from dataclasses import dataclass

MAGIC = b"ZTXT"
VERSION = 0x01
HEADER_FMT = "<4sBBHIIII"  # little-endian
HEADER_SIZE = struct.calcsize(HEADER_FMT)  # 24 bytes

MODE_ONLINE = 0x00
MODE_OFFLINE = 0x01

MODEL_DEEPSEEK_CHAT = 0x0001
MODEL_DEEPSEEK_REASONER = 0x0002


@dataclass(frozen=True)
class Header:
    mode: int
    model_id: int
    token_count: int
    original_bytes: int
    crc32: int
    model_data_len: int


def compute_crc32(data: bytes) -> int:
    return zlib.crc32(data) & 0xFFFFFFFF


def write_file(
    header: Header,
    model_data: bytes,
    compressed_body: bytes,
) -> bytes:
    """Serialize a complete .ztxt file to bytes."""
    header_bytes = struct.pack(
        HEADER_FMT,
        MAGIC,
        VERSION,
        header.mode,
        header.model_id,
        header.token_count,
        header.original_bytes,
        header.crc32,
        header.model_data_len,
    )
    return header_bytes + model_data + compressed_body


def read_file(data: bytes) -> tuple[Header, bytes, bytes]:
    """Deserialize a .ztxt file. Returns (header, model_data, compressed_body)."""
    if len(data) < HEADER_SIZE:
        raise ValueError(f"file too short: {len(data)} < {HEADER_SIZE} bytes")

    fields = struct.unpack(HEADER_FMT, data[:HEADER_SIZE])
    magic, version, mode, model_id, token_count, original_bytes, crc32_val, model_data_len = fields

    if magic != MAGIC:
        raise ValueError(f"invalid magic: {magic!r}, expected {MAGIC!r}")
    if version != VERSION:
        raise ValueError(f"unsupported version: {version}, expected {VERSION}")

    offset = HEADER_SIZE
    model_data = data[offset : offset + model_data_len]
    offset += model_data_len
    compressed_body = data[offset:]

    header = Header(
        mode=mode,
        model_id=model_id,
        token_count=token_count,
        original_bytes=original_bytes,
        crc32=crc32_val,
        model_data_len=model_data_len,
    )
    return header, model_data, compressed_body
