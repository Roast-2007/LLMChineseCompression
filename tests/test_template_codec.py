from zippedtext.online_manifest import AnalysisManifest
from zippedtext.template_codec import (
    TemplateCatalog,
    build_template_catalog,
    decode_template_segment,
    detect_template,
    encode_template_segment,
)
from zippedtext.segment import split_text_segments


def test_detect_template_for_key_value_line():
    match = detect_template("name: zippedtext")
    assert match is not None
    assert match.template_kind == "key_value"
    assert match.skeleton == "name: "
    assert match.slot_values == ("zippedtext",)



def test_detect_template_supports_multi_digit_numbered_list():
    match = detect_template("12. 安装 zippedtext")
    assert match is not None
    assert match.template_kind == "list_prefix"
    assert match.skeleton == "12. "



def test_detect_template_supports_tsv_rows():
    match = detect_template("name\tvalue\tdescription")
    assert match is not None
    assert match.template_kind == "table_row"
    assert match.slot_values == ("name", "value", "description")



def test_detect_template_uses_template_hints_to_boost_confidence():
    text = "name: zippedtext"
    plain = detect_template(text)
    hinted = detect_template(
        text,
        AnalysisManifest.from_api_payload({"template_hints": ["key_value"]}, len(text)),
    )
    assert plain is not None
    assert hinted is not None
    assert hinted.confidence > plain.confidence



def test_detect_template_assigns_typed_slot_from_schema_hint():
    text = "version: v1.2.3"
    manifest = AnalysisManifest.from_api_payload(
        {
            "field_schemas": [{"field": "version", "slot_type": "version"}],
            "slot_hints": [{"template_kind": "key_value", "slot_index": 0, "slot_type": "version", "field": "version"}],
        },
        len(text),
    )
    match = detect_template(text, manifest)
    assert match is not None
    assert match.slot_types == ("version",)
    assert match.slot_fields == ("version",)



def test_build_template_catalog_prunes_one_off_entries_without_hints():
    text = "name: zippedtext\nversion: 0.3.3\n\n只出现一次的段落"
    analysis = AnalysisManifest()
    segments = split_text_segments(text, analysis, max_chars=200)
    catalog = build_template_catalog(segments, text, analysis)
    # Record template may be detected from the two config lines
    # But single-entry non-record templates should still be pruned
    non_record = tuple(e for e in catalog.entries if e[0] != "record")
    assert non_record == ()



def test_template_segment_roundtrip():
    text = "name: zippedtext"
    analysis = AnalysisManifest.from_api_payload({"template_hints": ["key_value"]}, len(text))
    segments = split_text_segments(text, analysis, max_chars=200)
    catalog = build_template_catalog(segments, text, analysis)
    match = detect_template(text, analysis)
    assert match is not None
    payload = encode_template_segment(
        text,
        match,
        template_index=0,
        phrase_set=frozenset(),
        priors=None,
        max_order=4,
    )
    restored = decode_template_segment(
        payload.payload,
        catalog,
        phrase_set=frozenset(),
        char_count=len(text),
        priors=None,
        max_order=4,
        analysis=analysis,
    )
    assert restored == text



def test_template_segment_roundtrip_with_typed_slots():
    text = "endpoint: https://api.deepseek.com/v1/chat/completions"
    analysis = AnalysisManifest.from_api_payload(
        {
            "template_hints": ["key_value"],
            "field_schemas": [{"field": "endpoint", "slot_type": "path_or_url"}],
            "slot_hints": [{"template_kind": "key_value", "slot_index": 0, "slot_type": "path_or_url", "field": "endpoint"}],
        },
        len(text),
    )
    segments = split_text_segments(text, analysis, max_chars=200)
    catalog = build_template_catalog((segments[0], segments[0]), text + "\n" + text, analysis)
    match = detect_template(text, analysis)
    assert match is not None
    assert match.slot_types == ("path_or_url",)
    payload = encode_template_segment(
        text,
        match,
        template_index=0,
        phrase_set=frozenset(),
        priors=None,
        max_order=4,
    )
    restored = decode_template_segment(
        payload.payload,
        catalog,
        phrase_set=frozenset(),
        char_count=len(text),
        priors=None,
        max_order=4,
        analysis=analysis,
    )
    assert restored == text



def test_template_segment_roundtrip_with_enum_slot():
    text = "mode: online"
    analysis = AnalysisManifest.from_api_payload(
        {
            "template_hints": ["key_value"],
            "field_schemas": [{"field": "mode", "slot_type": "enum", "enum_candidates": ["online", "offline"]}],
            "slot_hints": [{"template_kind": "key_value", "slot_index": 0, "slot_type": "enum", "field": "mode", "enum_candidates": ["online", "offline"]}],
        },
        len(text),
    )
    catalog = TemplateCatalog(entries=(("key_value", "mode: "),))
    match = detect_template(text, analysis)
    assert match is not None
    assert match.slot_types == ("enum",)
    payload = encode_template_segment(
        text,
        match,
        template_index=0,
        phrase_set=frozenset(),
        priors=None,
        max_order=4,
    )
    restored = decode_template_segment(
        payload.payload,
        catalog,
        phrase_set=frozenset(),
        char_count=len(text),
        priors=None,
        max_order=4,
        analysis=analysis,
    )
    assert restored == text



def test_template_catalog_roundtrip():
    catalog = TemplateCatalog(entries=(("key_value", ": "), ("list_prefix", "- ")))
    restored = TemplateCatalog.deserialize(catalog.serialize())
    assert restored == catalog
