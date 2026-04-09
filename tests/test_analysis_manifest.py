import json

from zippedtext.compressor import _merge_priors
from zippedtext.online_manifest import (
    AnalysisManifest,
    StructuredOnlineStats,
    deserialize_segment_records,
    serialize_segment_records,
)


def test_analysis_manifest_normalizes_payload():
    payload = {
        "char_frequencies": {"压": 0.2, "缩": 0.1, "bad": 1.0, "文": -1},
        "top_bigrams": [["压缩", 0.8], ["文档", 0.4], ["x", 0.2]],
        "phrase_dictionary": [["压缩算法", 1], "在线模式", ["a", 0]],
        "language_segments": [
            {"start": 0, "end": 5, "lang": "zh"},
            {"start": 3, "end": 8, "lang": "en"},
            {"start": 8, "end": 12, "lang": "num"},
        ],
        "document_family": "API Docs",
        "block_families": [
            {"kind": "config", "start": 0, "end": 10, "family": "endpoint_block"},
            {"kind": "??", "start": 10, "end": 12, "family": "ignored"},
        ],
        "field_schemas": [
            {"field": "version", "slot_type": "version"},
            {"field": "mode", "slot_type": "string", "enum_candidates": ["online", "offline"]},
        ],
        "slot_hints": [
            {"template_kind": "key_value", "slot_index": 0, "slot_type": "path_or_url", "field": "endpoint"},
            {"template_kind": "key_value", "slot_index": 0, "slot_type": "oops"},
        ],
        "enum_candidates": [
            {"field": "mode", "values": ["online", "offline", "online"]},
        ],
    }
    manifest = AnalysisManifest.from_api_payload(payload, text_len=20)

    assert manifest.char_frequencies == (("压", 0.2), ("缩", 0.1))
    assert manifest.top_bigrams == (("压缩", 0.8), ("文档", 0.4))
    assert manifest.phrase_dictionary == ("压缩算法", "在线模式")
    assert [(hint.start, hint.end, hint.lang) for hint in manifest.language_segments] == [
        (0, 5, "zh"),
        (8, 12, "num"),
    ]
    assert manifest.document_family == "api_docs"
    assert len(manifest.block_families) == 2
    assert manifest.field_schema_for("version") is not None
    assert manifest.field_schema_for("mode") is not None
    assert manifest.slot_hint_for("key_value", 0, "endpoint") is not None
    assert manifest.enum_candidates_for("mode") == ("online", "offline")



def test_analysis_manifest_roundtrip_and_prior_map():
    manifest = AnalysisManifest.from_api_payload(
        {
            "char_frequencies": {"你": 3, "好": 1},
            "phrase_dictionary": ["你好世界"],
            "slot_hints": [
                {"template_kind": "key_value", "slot_index": 0, "slot_type": "identifier", "field": "name"}
            ],
        },
        text_len=4,
    )
    restored = AnalysisManifest.deserialize(manifest.serialize(), text_len=4)
    assert restored == manifest
    assert manifest.to_prior_map() == {"你": 0.75, "好": 0.25}



def test_analysis_manifest_for_storage_truncates_schema_hints():
    manifest = AnalysisManifest.from_api_payload(
        {
            "block_families": [
                {"kind": "config", "start": index, "end": index + 1, "family": f"family_{index}"}
                for index in range(40)
            ],
            "field_schemas": [
                {"field": f"field_{index}", "slot_type": "identifier"}
                for index in range(40)
            ],
            "slot_hints": [
                {"template_kind": "key_value", "slot_index": index, "slot_type": "identifier"}
                for index in range(80)
            ],
            "enum_candidates": [
                {"field": f"field_{index}", "values": ["a", "b"]}
                for index in range(40)
            ],
        },
        text_len=128,
    )
    stored = manifest.for_storage(max_block_families=5, max_field_schemas=6, max_slot_hints=7, max_enum_candidates=8)
    assert len(stored.block_families) == 5
    assert len(stored.field_schemas) == 6
    assert len(stored.slot_hints) == 7
    assert len(stored.enum_candidates) == 8



def test_merge_priors_blends_base_and_manifest_priors():
    merged = _merge_priors({"你": 0.5, "好": 0.5}, {"你": 1.0}, manifest_weight=0.4)
    assert merged is not None
    assert merged["你"] > merged["好"]
    assert abs(sum(merged.values()) - 1.0) < 1e-9



def test_segment_records_roundtrip():
    payload = serialize_segment_records(())
    assert deserialize_segment_records(payload) == ()

    records = deserialize_segment_records(
        json.dumps(
            [
                {
                    "kind": "prose",
                    "route": "phrase",
                    "char_count": 10,
                    "payload_len": 8,
                    "original_bytes": 20,
                    "encoded_bytes": 8,
                }
            ]
        ).encode("utf-8")
    )
    assert records[0].route == "phrase"
    assert records[0].char_count == 10



def test_structured_online_stats_roundtrip():
    stats = StructuredOnlineStats(
        segment_count=2,
        phrase_count=3,
        analysis_bytes=10,
        dictionary_bytes=11,
        segments_bytes=12,
        payload_bytes=13,
        route_counts=(("literal", 1), ("phrase", 1)),
        typed_slot_count=4,
        typed_template_count=2,
        template_family_counts=(("key_value:endpoint: ", 2),),
    )
    restored = StructuredOnlineStats.deserialize(stats.serialize())
    assert restored == stats
