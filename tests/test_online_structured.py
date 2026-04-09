from zippedtext.compressor import _structured_online_compress, compress, decompress
from zippedtext.format import (
    MODE_ONLINE,
    MODE_OFFLINE,
    SECTION_ANALYSIS,
    SECTION_PHRASE_TABLE,
    SECTION_SEGMENTS,
    SECTION_STATS,
    SECTION_TEMPLATES,
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
                "top_bigrams": [["配置", 0.8], ["模式", 0.6]],
                "phrase_dictionary": ["压缩算法", "在线模式", "structured online"],
                "language_segments": [{"start": 0, "end": len(text), "lang": "zh"}],
                "template_hints": ["key_value"],
                "field_schemas": [
                    {"field": "endpoint", "slot_type": "path_or_url"},
                    {"field": "version", "slot_type": "version"},
                    {"field": "mode", "slot_type": "enum", "enum_candidates": ["online", "offline"]},
                ],
                "slot_hints": [
                    {"template_kind": "key_value", "slot_index": 0, "slot_type": "path_or_url", "field": "endpoint"},
                    {"template_kind": "key_value", "slot_index": 0, "slot_type": "version", "field": "version"},
                    {"template_kind": "key_value", "slot_index": 0, "slot_type": "enum", "field": "mode", "enum_candidates": ["online", "offline"]},
                ],
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



def test_structured_online_uses_template_section_for_repeated_config_lines():
    text = (
        "endpoint: https://api.deepseek.com/v1/chat/completions\n"
        "endpoint: https://api.deepseek.com/v1/chat/completions\n"
        "endpoint: https://api.deepseek.com/v1/chat/completions\n"
        "endpoint: https://api.deepseek.com/v1/chat/completions"
    )
    encoded = text.encode("utf-8")
    client = FakeStructuredApiClient()
    data = _structured_online_compress(
        text=text,
        text_bytes=encoded,
        crc=compute_crc32(encoded),
        api_client=client,
        model_name=client.model,
        priors={"n": 0.2},
        flags=0,
        max_order=4,
    )

    _, sections, _ = read_file_v3(data)
    assert SECTION_TEMPLATES in sections
    stats = StructuredOnlineStats.deserialize(sections[SECTION_STATS])
    assert stats.template_count >= 1
    assert stats.template_hit_count >= 1
    assert stats.typed_slot_count >= 1
    assert decompress(data) == text



def test_public_online_structured_path_keeps_api_free_decompression():
    text = (
        "endpoint: https://api.deepseek.com/v1/chat/completions\n"
        "version: v1.2.3\n"
        "mode: online\n"
        "endpoint: https://api.deepseek.com/v1/chat/completions\n"
        "version: v1.2.4\n"
        "mode: offline"
    )
    client = FakeStructuredApiClient()
    data = compress(text, mode="online", api_client=client, sub_mode="structured")
    header, _, _ = read_file(data)
    assert header.mode in {MODE_ONLINE, MODE_OFFLINE}
    assert decompress(data) == text
