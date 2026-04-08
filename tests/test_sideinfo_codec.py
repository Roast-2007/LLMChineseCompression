import pytest

from zippedtext.sideinfo_codec import (
    choose_section_flags,
    encode_section_payload,
    make_section,
    pack_string,
    section_stored_size,
    unpack_string,
)
from zippedtext.format import SECTION_CODEC_RAW, SECTION_CODEC_ZSTD


def test_pack_and_unpack_string_roundtrip():
    payload = pack_string("结构化 side info")
    restored, offset = unpack_string(payload, 0)
    assert restored == "结构化 side info"
    assert offset == len(payload)


def test_choose_section_flags_prefers_raw_for_small_payload():
    assert choose_section_flags(b"small payload") == SECTION_CODEC_RAW


def test_choose_section_flags_can_choose_zstd_for_repetitive_payload():
    data = (b"analysis:" + b"A" * 512) * 4
    flags = choose_section_flags(data, prefer_compression=True)
    assert flags in (SECTION_CODEC_RAW, SECTION_CODEC_ZSTD)
    section = make_section(data, prefer_compression=True)
    encoded = encode_section_payload(section.data, section.flags)
    assert section_stored_size(section) == len(encoded)
    if flags == SECTION_CODEC_ZSTD:
        assert len(encoded) < len(data)


def test_unpack_string_rejects_truncated_payload():
    with pytest.raises(ValueError):
        unpack_string(b"\x05\x00ab", 0)
