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


def test_build_template_catalog_prunes_one_off_entries_without_hints():
    text = "name: zippedtext\nversion: 0.3.3\n\n只出现一次的段落"
    analysis = AnalysisManifest()
    segments = split_text_segments(text, analysis, max_chars=200)
    catalog = build_template_catalog(segments, text, analysis)
    assert catalog.entries == ()


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
    )
    assert restored == text


def test_template_catalog_roundtrip():
    catalog = TemplateCatalog(entries=(("key_value", ": "), ("list_prefix", "- ")))
    restored = TemplateCatalog.deserialize(catalog.serialize())
    assert restored == catalog
