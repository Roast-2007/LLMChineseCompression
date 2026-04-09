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
    }
    manifest = AnalysisManifest.from_api_payload(payload, text_len=20)

    assert manifest.char_frequencies == (("压", 0.2), ("缩", 0.1))
    assert manifest.top_bigrams == (("压缩", 0.8), ("文档", 0.4))
    assert manifest.phrase_dictionary == ("压缩算法", "在线模式")
    assert [(hint.start, hint.end, hint.lang) for hint in manifest.language_segments] == [
        (0, 5, "zh"),
        (8, 12, "num"),
    ]


def test_analysis_manifest_roundtrip_and_prior_map():
    manifest = AnalysisManifest.from_api_payload(
        {
            "char_frequencies": {"你": 3, "好": 1},
            "phrase_dictionary": ["你好世界"],
        },
        text_len=4,
    )
    restored = AnalysisManifest.deserialize(manifest.serialize(), text_len=4)
    assert restored == manifest
    assert manifest.to_prior_map() == {"你": 0.75, "好": 0.25}


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
    )
    restored = StructuredOnlineStats.deserialize(stats.serialize())
    assert restored == stats
