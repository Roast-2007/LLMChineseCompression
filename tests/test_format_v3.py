"""Tests for .ztxt format v3 structured online files."""

import pytest

from zippedtext.format import (
    MODE_ONLINE,
    SECTION_ANALYSIS,
    SECTION_PHRASE_TABLE,
    SECTION_SEGMENTS,
    SECTION_STATS,
    VERSION_V3,
    Header,
    compute_crc32,
    read_file,
    read_file_v3,
    write_file_v3,
)


def test_write_and_read_v3_sections():
    sections = {
        SECTION_ANALYSIS: b"analysis",
        SECTION_PHRASE_TABLE: b"phrases",
        SECTION_SEGMENTS: b"segments",
        SECTION_STATS: b"stats",
    }
    header = Header(
        mode=MODE_ONLINE,
        model_id=0,
        token_count=12,
        original_bytes=24,
        crc32=compute_crc32(b"hello"),
        model_data_len=0,
        flags=0,
        max_order=4,
        version=VERSION_V3,
    )
    data = write_file_v3(header, sections, b"payload")

    h, metadata, body = read_file(data)
    assert h.version == VERSION_V3
    assert body == b"payload"
    assert metadata

    h3, sections3, body3 = read_file_v3(data)
    assert h3.version == VERSION_V3
    assert body3 == b"payload"
    assert sections3[SECTION_ANALYSIS] == b"analysis"
    assert sections3[SECTION_PHRASE_TABLE] == b"phrases"
    assert sections3[SECTION_SEGMENTS] == b"segments"
    assert sections3[SECTION_STATS] == b"stats"


def test_read_v3_rejects_truncated_section_payload():
    sections = {SECTION_ANALYSIS: b"analysis"}
    header = Header(
        mode=MODE_ONLINE,
        model_id=0,
        token_count=1,
        original_bytes=1,
        crc32=0,
        model_data_len=0,
        version=VERSION_V3,
    )
    data = write_file_v3(header, sections, b"body")
    broken = data[:-2]
    with pytest.raises(ValueError):
        read_file_v3(broken)
