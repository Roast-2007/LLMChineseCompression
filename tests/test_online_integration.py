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


# ---------------------------------------------------------------------------
# Compression ratio benchmarks
# ---------------------------------------------------------------------------

@skip_no_api
def test_compression_ratio_char():
    """Verify online char-level achieves reasonable compression ratio."""
    from zippedtext.compressor import compress

    # Use the sample Chinese text
    text_path = os.path.join(os.path.dirname(__file__), "sample_cn.txt")
    if not os.path.exists(text_path):
        pytest.skip("sample_cn.txt not found")

    with open(text_path, "rb") as f:
        text = f.read().decode("utf-8")

    client = _make_client()
    compressed = compress(text, mode="online", api_client=client, sub_mode="char")

    original_size = len(text.encode("utf-8"))
    ratio = len(compressed) / original_size
    print(f"\n  Sample CN char-level: {original_size}B → {len(compressed)}B (ratio: {ratio:.3f})")

    # Online mode should be at least as good as offline (0.54)
    # Target is < 0.35 but this depends on LLM prediction quality
    assert ratio < 0.60, f"Online char ratio {ratio:.3f} worse than expected"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
