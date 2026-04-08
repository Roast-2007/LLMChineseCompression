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
    assert match.slot_values == ("name", "zippedtext")


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
