"""Tests for .ztxt format v2 header and backward compatibility with v1."""

import struct

import pytest

from zippedtext.format import (
    DEFAULT_MAX_ORDER,
    FLAG_HAS_PRIORS,
    FLAG_PHRASE_ENCODING,
    MAGIC,
    MODE_CODEGEN,
    MODE_OFFLINE,
    MODE_ONLINE,
    VERSION_V1,
    VERSION_V2,
    Header,
    compute_crc32,
    read_file,
    read_file_v2,
    write_file,
    _V1_FMT,
    _V1_SIZE,
    _V2_SIZE,
)


def _make_v1_bytes(
    mode: int = MODE_OFFLINE,
    model_id: int = 0x0001,
    token_count: int = 10,
    original_bytes: int = 30,
    crc32: int = 0xDEADBEEF,
    model_data: bytes = b"",
    body: bytes = b"\x01\x02\x03",
) -> bytes:
    """Construct raw v1 .ztxt bytes for testing."""
    hdr = struct.pack(
        _V1_FMT,
        MAGIC, VERSION_V1, mode, model_id,
        token_count, original_bytes, crc32, len(model_data),
    )
    return hdr + model_data + body


class TestV2WriteRead:
    """Round-trip write (v2) then read."""

    def test_basic_offline(self):
        header = Header(
            mode=MODE_OFFLINE, model_id=0,
            token_count=100, original_bytes=300,
            crc32=0x12345678, model_data_len=0,
        )
        data = write_file(header, b"", b"\xAB\xCD")
        h, md, body = read_file(data)

        assert h.mode == MODE_OFFLINE
        assert h.token_count == 100
        assert h.original_bytes == 300
        assert h.crc32 == 0x12345678
        assert h.flags == 0
        assert h.max_order == DEFAULT_MAX_ORDER
        assert h.phrase_table_len == 0
        assert md == b""
        assert body == b"\xAB\xCD"

    def test_online_with_model_data(self):
        model_data = b"\x00deepseek-chat\x00"
        header = Header(
            mode=MODE_ONLINE, model_id=0,
            token_count=50, original_bytes=150,
            crc32=0xCAFEBABE,
            model_data_len=len(model_data),
        )
        data = write_file(header, model_data, b"\xFF")
        h, md, body = read_file(data)

        assert h.mode == MODE_ONLINE
        assert md == model_data
        assert body == b"\xFF"

    def test_codegen_mode(self):
        header = Header(
            mode=MODE_CODEGEN, model_id=0,
            token_count=200, original_bytes=600,
            crc32=0, model_data_len=5,
            flags=0, max_order=4,
        )
        data = write_file(header, b"hello", b"body")
        h, md, body = read_file(data)
        assert h.mode == MODE_CODEGEN
        assert md == b"hello"
        assert body == b"body"

    def test_flags_and_max_order(self):
        header = Header(
            mode=MODE_OFFLINE, model_id=0,
            token_count=10, original_bytes=30,
            crc32=0, model_data_len=0,
            flags=FLAG_PHRASE_ENCODING | FLAG_HAS_PRIORS,
            max_order=6,
        )
        data = write_file(header, b"", b"x")
        h, _, _ = read_file(data)
        assert h.flags == (FLAG_PHRASE_ENCODING | FLAG_HAS_PRIORS)
        assert h.max_order == 6

    def test_phrase_table_via_read_file_v2(self):
        phrase_data = b"phrase1\x00phrase2\x00"
        header = Header(
            mode=MODE_OFFLINE, model_id=0,
            token_count=10, original_bytes=30,
            crc32=0, model_data_len=3,
            flags=FLAG_PHRASE_ENCODING,
            phrase_table_len=len(phrase_data),
        )
        data = write_file(header, b"abc", b"body", phrase_table=phrase_data)
        h, md, pt, body = read_file_v2(data)
        assert md == b"abc"
        assert pt == phrase_data
        assert body == b"body"
        assert h.phrase_table_len == len(phrase_data)


class TestV1BackwardCompat:
    """Ensure old v1 files still read correctly."""

    def test_read_v1_offline(self):
        raw = _make_v1_bytes(mode=MODE_OFFLINE)
        h, md, body = read_file(raw)
        assert h.mode == MODE_OFFLINE
        assert h.model_id == 0x0001
        assert h.token_count == 10
        assert h.crc32 == 0xDEADBEEF
        # v2 defaults for v1 files
        assert h.flags == 0
        assert h.max_order == DEFAULT_MAX_ORDER
        assert h.phrase_table_len == 0
        assert md == b""
        assert body == b"\x01\x02\x03"

    def test_read_v1_online_with_model_data(self):
        md_bytes = b"\x00deepseek-chat\x00"
        raw = _make_v1_bytes(
            mode=MODE_ONLINE,
            model_data=md_bytes,
            body=b"\xEE",
        )
        h, md, body = read_file(raw)
        assert h.mode == MODE_ONLINE
        assert md == md_bytes
        assert body == b"\xEE"

    def test_v1_bad_magic(self):
        raw = b"NOPE" + b"\x00" * 20
        with pytest.raises(ValueError, match="invalid magic"):
            read_file(raw)

    def test_v1_too_short(self):
        with pytest.raises(ValueError, match="too short"):
            read_file(b"ZT")


class TestV2Errors:
    """Error handling for malformed v2 data."""

    def test_unknown_version(self):
        raw = MAGIC + bytes([0x99]) + b"\x00" * 30
        with pytest.raises(ValueError, match="unsupported version"):
            read_file(raw)

    def test_v2_too_short(self):
        raw = MAGIC + bytes([VERSION_V2]) + b"\x00" * 5
        with pytest.raises(ValueError, match="too short"):
            read_file(raw)


class TestCompressDecompressV2:
    """End-to-end: compress produces v2, decompress reads it."""

    def test_roundtrip_writes_v2(self):
        from zippedtext.compressor import compress, decompress

        text = "格式升级测试"
        data = compress(text)
        # Verify it's a v2 file
        assert data[:4] == MAGIC
        assert data[4] == VERSION_V2
        # Verify decompression
        assert decompress(data) == text
