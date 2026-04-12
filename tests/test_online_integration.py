"""Integration tests for online mode with real DeepSeek API.

Requires DEEPSEEK_API_KEY environment variable to be set.
Skipped automatically when the key is not available.
"""

from __future__ import annotations

import os

import pytest

DEEPSEEK_API_KEY = os.environ.get("DEEPSEEK_API_KEY", "")
skip_no_api = pytest.mark.skipif(
    not DEEPSEEK_API_KEY,
    reason="DEEPSEEK_API_KEY not set",
)


def _make_client():
    from zippedtext.api_client import DeepSeekClient
    return DeepSeekClient(api_key=DEEPSEEK_API_KEY)


# ---------------------------------------------------------------------------
# API client smoke test
# ---------------------------------------------------------------------------

@skip_no_api
def test_generate_continuation():
    """Verify API returns valid logprobs."""
    client = _make_client()
    result = client.generate_continuation("今天天气很好", max_tokens=20)
    assert len(result.generated_text) > 0
    assert len(result.tokens) > 0
    assert result.model != ""
    for tok in result.tokens:
        assert len(tok.text) > 0
        assert len(tok.top_alternatives) > 0


# ---------------------------------------------------------------------------
# Character-level online round-trip
# ---------------------------------------------------------------------------

@skip_no_api
def test_online_char_chinese():
    """Compress/decompress Chinese text with char-level online mode."""
    from zippedtext.compressor import compress, decompress

    text = "自然语言处理是人工智能的一个重要分支，它致力于让计算机理解和生成人类语言。"
    client = _make_client()

    compressed = compress(text, mode="online", api_client=client, sub_mode="char")
    assert len(compressed) > 0

    client2 = _make_client()
    restored = decompress(compressed, api_client=client2)
    assert restored == text

    ratio = len(compressed) / len(text.encode("utf-8"))
    print(f"\n  Chinese char-level: {len(text.encode('utf-8'))}B → {len(compressed)}B (ratio: {ratio:.3f})")


@skip_no_api
def test_online_char_english():
    """Compress/decompress English text with char-level online mode."""
    from zippedtext.compressor import compress, decompress

    text = "Natural language processing is an important branch of artificial intelligence."
    client = _make_client()

    compressed = compress(text, mode="online", api_client=client, sub_mode="char")
    client2 = _make_client()
    restored = decompress(compressed, api_client=client2)
    assert restored == text


@skip_no_api
def test_online_char_mixed():
    """Compress/decompress mixed Chinese/English text."""
    from zippedtext.compressor import compress, decompress

    text = "Python 3.12是一种流行的编程语言，广泛用于AI和数据科学领域。"
    client = _make_client()

    compressed = compress(text, mode="online", api_client=client, sub_mode="char")
    client2 = _make_client()
    restored = decompress(compressed, api_client=client2)
    assert restored == text


# ---------------------------------------------------------------------------
# Token-level online round-trip
# ---------------------------------------------------------------------------

@skip_no_api
def test_online_token_chinese():
    """Compress/decompress Chinese text with token-level online mode."""
    from zippedtext.compressor import compress, decompress

    text = "深度学习模型通过大量数据训练来学习语言的规律和模式。"
    client = _make_client()

    compressed = compress(text, mode="online", api_client=client, sub_mode="token")
    assert len(compressed) > 0

    client2 = _make_client()
    restored = decompress(compressed, api_client=client2)
    assert restored == text

    ratio = len(compressed) / len(text.encode("utf-8"))
    print(f"\n  Chinese token-level: {len(text.encode('utf-8'))}B → {len(compressed)}B (ratio: {ratio:.3f})")


@skip_no_api
def test_online_structured_url_smoke():
    """Compress/decompress URL-heavy structured text with structured online path."""
    from zippedtext.compressor import structured_compress, decompress
    from zippedtext.format import SECTION_STATS, compute_crc32, read_file_v3
    from zippedtext.online_manifest import StructuredOnlineStats

    text = (
        "endpoint: https://api.deepseek.com/v1/chat/completions\n"
        "endpoint: https://api.deepseek.com/v1/chat/completions\n"
        "download_path: /opt/zippedtext/releases/v1.2.3\n"
        "download_path: /opt/zippedtext/releases/v1.2.4"
    )
    encoded = text.encode("utf-8")
    client = _make_client()

    compressed = structured_compress(
        text=text,
        text_bytes=encoded,
        crc=compute_crc32(encoded),
        api_client=client,
        model_name=client.model,
        priors={},
        flags=0,
        max_order=4,
    )
    restored = decompress(compressed)
    assert restored == text

    _, sections, _ = read_file_v3(compressed)
    stats = StructuredOnlineStats.deserialize(sections[SECTION_STATS])
    assert stats.segment_count >= 1
