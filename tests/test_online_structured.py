from zippedtext.compressor import _structured_online_compress, decompress
from zippedtext.format import (
    MODE_ONLINE,
    SECTION_ANALYSIS,
    SECTION_PHRASE_TABLE,
    SECTION_SEGMENTS,
    SECTION_STATS,
    VERSION_V3,
    compute_crc32,
    read_file,
    read_file_v3,
)
from zippedtext.online_manifest import StructuredOnlineStats


class FakeStructuredApiClient:
    def __init__(self) -> None:
        self.model = "fake-structured-model"
        self.last_model_id = "fake-structured-model"

    def analyze_text(self, text: str):
        from zippedtext.online_manifest import AnalysisManifest

        return AnalysisManifest.from_api_payload(
            {
                "char_frequencies": {"压": 0.2, "缩": 0.18, "模": 0.16, "式": 0.14},
                "phrase_dictionary": ["压缩算法", "在线模式", "structured online"],
                "language_segments": [{"start": 0, "end": len(text), "lang": "zh"}],
            },
            len(text),
        )


def test_structured_online_roundtrip_without_api_on_decompress():
    text = (
        "压缩算法在线模式压缩算法在线模式。"
        "structured online 模式通过结构化 side info 提升压缩率。"
        "压缩算法在线模式压缩算法在线模式。"
    )
    encoded = text.encode("utf-8")
    client = FakeStructuredApiClient()
    data = _structured_online_compress(
        text=text,
        text_bytes=encoded,
        crc=compute_crc32(encoded),
        api_client=client,
        model_name=client.model,
        priors={},
        flags=0,
        max_order=4,
    )

    header, _, _ = read_file(data)
    assert header.mode == MODE_ONLINE
    assert header.version == VERSION_V3

    restored = decompress(data)
    assert restored == text


def test_structured_online_v3_sections_present():
    text = "压缩算法在线模式压缩算法在线模式。" * 4
    encoded = text.encode("utf-8")
    client = FakeStructuredApiClient()
    data = _structured_online_compress(
        text=text,
        text_bytes=encoded,
        crc=compute_crc32(encoded),
        api_client=client,
        model_name=client.model,
        priors={},
        flags=0,
        max_order=4,
    )

    header, sections, body = read_file_v3(data)
    assert header.version == VERSION_V3
    assert SECTION_ANALYSIS in sections
    assert SECTION_PHRASE_TABLE in sections
    assert SECTION_SEGMENTS in sections
    assert SECTION_STATS in sections
    assert len(body) > 0

    stats = StructuredOnlineStats.deserialize(sections[SECTION_STATS])
    assert stats.segment_count >= 1
    assert stats.analysis_bytes > 0
