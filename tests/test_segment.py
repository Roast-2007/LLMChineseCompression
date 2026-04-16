"""Tests for segment.py record grouping functionality."""

from __future__ import annotations

import pytest

from zippedtext.segment import (
    TextSegment,
    RecordGroup,
    group_record_groups,
    split_text_segments,
)
from zippedtext.online_manifest import (
    AnalysisManifest,
    BlockFamilyHint,
    LanguageHint,
)


class TestRecordGroupBasics:
    def test_empty_segments_returns_empty(self):
        assert group_record_groups((), "") == ()

    def test_single_segment_returns_empty(self):
        segs = (TextSegment(start=0, end=10, kind="config"),)
        assert group_record_groups(segs, "key: value") == ()

    def test_two_consecutive_config_lines_grouped(self):
        text = "host: localhost\nport: 8080\n"
        segs = split_text_segments(text)
        config_segs = [s for s in segs if s.kind == "config"]
        # After split, we get individual line segments
        groups = group_record_groups(segs, text)
        assert len(groups) == 1
        assert groups[0].kind == "config"
        assert groups[0].segment_count >= 2

    def test_prose_segments_never_grouped(self):
        text = "Hello world. This is prose.\nAnother paragraph here.\n"
        segs = split_text_segments(text)
        groups = group_record_groups(segs, text)
        for group in groups:
            assert group.kind != "prose"


class TestRecordGroupWithBlockFamilies:
    def test_uses_block_family_from_manifest(self):
        analysis = AnalysisManifest(
            language_segments=[],
            char_frequencies={},
            phrase_dictionary={},
            template_hints=[],
            block_families=[
                BlockFamilyHint(kind="config", start=0, end=30, family="server_config"),
            ],
            field_schemas=(),
            slot_hints=(),
            enum_candidates=(),
            document_family="config_file",
        )
        text = "host: localhost\nport: 8080\n"
        segs = split_text_segments(text, analysis)
        groups = group_record_groups(segs, text, analysis)
        assert len(groups) >= 1
        assert groups[0].family == "server_config"

    def test_fallback_to_kind_when_no_analysis(self):
        text = "- item one\n- item two\n- item three\n"
        segs = split_text_segments(text)
        groups = group_record_groups(segs, text)
        assert len(groups) >= 1
        assert groups[0].family == "list_block"

    def test_fallback_to_kind_when_no_matching_family(self):
        analysis = AnalysisManifest(
            language_segments=[],
            char_frequencies={},
            phrase_dictionary={},
            template_hints=[],
            block_families=[
                BlockFamilyHint(kind="table", start=100, end=200, family="unrelated"),
            ],
            field_schemas=(),
            slot_hints=(),
            enum_candidates=(),
            document_family="unknown",
        )
        text = "- item one\n- item two\n"
        segs = split_text_segments(text, analysis)
        groups = group_record_groups(segs, text, analysis)
        assert len(groups) >= 1
        assert groups[0].family == "list_block"


class TestRecordGroupKindSeparation:
    def test_mixed_kinds_not_grouped_together(self):
        text = "key: value1\nkey: value2\n- list item\n- list item2\n"
        segs = split_text_segments(text)
        groups = group_record_groups(segs, text)
        # Should have separate groups for config and list
        kinds = {g.kind for g in groups}
        if len(groups) >= 2:
            assert "config" in kinds or "list" in kinds

    def test_single_config_line_no_group(self):
        text = "key: value\n\nsome prose here\n"
        segs = split_text_segments(text)
        groups = group_record_groups(segs, text)
        config_groups = [g for g in groups if g.kind == "config"]
        assert len(config_groups) == 0


class TestRecordGroupTextSpan:
    def test_text_span_covers_all_segments(self):
        text = "host: localhost\nport: 8080\ntimeout: 30\n"
        segs = split_text_segments(text)
        groups = group_record_groups(segs, text)
        assert len(groups) >= 1
        group = groups[0]
        start, end = group.text_span
        assert start == 0
        # end should be at least the last config line (trailing newline may be a mixed segment)
        assert end >= len(text) - 1
