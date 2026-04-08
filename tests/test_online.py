"""Mock-based unit tests for online mode compression.

These tests use a fake API client to verify:
  - Character probability boosting logic
  - Encode/decode symmetry (lossless round-trip)
  - Chunk boundary handling
  - Token match/fallback logic
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from unittest.mock import MagicMock

import pytest

# ---------------------------------------------------------------------------
# Fake API client that returns deterministic predictions
# ---------------------------------------------------------------------------


@dataclass
class FakeTokenInfo:
    token: str
    logprob: float
    top_logprobs: list


@dataclass
class FakeTopLogprob:
    token: str
    logprob: float


@dataclass
class FakeChoice:
    logprobs: object
    message: object = None


@dataclass
class FakeLogprobs:
    content: list[FakeTokenInfo]


@dataclass
class FakeResponse:
    choices: list[FakeChoice]
    model: str = "fake-model"


class FakeApiClient:
    """Mimics DeepSeekClient for testing without real API calls.

    Returns a fixed continuation text with deterministic logprobs.
    """

    def __init__(self, continuation_text: str = "这是一段测试文本用于验证压缩功能"):
        self.model = "fake-model"
        self.last_model_id = "fake-model"
        self._continuation = continuation_text
        self._call_count = 0

    def generate_continuation(self, context, max_tokens=200, max_top_logprobs=20):
        from zippedtext.api_client import ChunkResult, GeneratedToken

        self._call_count += 1
        # Generate tokens: each Chinese char is a separate "token"
        tokens = []
        offset = 0
        for ch in self._continuation:
            alts = [
                (ch, -0.1),  # the generated char with high probability
                ("X", -3.0),  # a low-prob alternative
            ]
            tokens.append(GeneratedToken(
                text=ch,
                logprob=-0.1,
                top_alternatives=alts,
                char_offset=offset,
            ))
            offset += len(ch)

        return ChunkResult(
            generated_text=self._continuation,
            tokens=tokens,
            model="fake-model",
        )


# ---------------------------------------------------------------------------
# Test character-level probability boosting
# ---------------------------------------------------------------------------


def test_boost_distribution():
    """Verify _boost_prob correctly boosts a single probability."""
    from zippedtext.predictor.llm import _boost_prob

    probs = [0.1, 0.3, 0.4, 0.2]  # 4-symbol distribution

    # Boost symbol 2 by 10x
    boosted = _boost_prob(probs, 2, 10.0)
    assert len(boosted) == 4
    assert abs(sum(boosted) - 1.0) < 1e-9
    # Symbol 2 should be dominant after boost
    assert boosted[2] > boosted[0]
    assert boosted[2] > boosted[1]
    assert boosted[2] > boosted[3]
    # Original relative ordering of non-boosted symbols preserved
    assert boosted[1] > boosted[0]


def test_boost_distribution_escape():
    """Boosting ESCAPE (id 0) should work correctly."""
    from zippedtext.predictor.llm import _boost_prob

    probs = [0.05, 0.5, 0.45]  # ESCAPE=0.05, char1=0.5, char2=0.45
    boosted = _boost_prob(probs, 0, 20.0)
    assert abs(sum(boosted) - 1.0) < 1e-9
    assert boosted[0] > 0.3  # ESCAPE got boosted significantly


# ---------------------------------------------------------------------------
# Test LlmCharPredictor
# ---------------------------------------------------------------------------


def test_llm_char_predictor_basics():
    """Verify LlmCharPredictor produces valid distributions."""
    from zippedtext.predictor.llm import LlmCharPredictor

    fake = FakeApiClient("你好世界")
    pred = LlmCharPredictor(fake, chunk_chars=100)

    # Simulate PPM distribution for a 3-symbol vocab: [ESC, 你, 好]
    ppm_probs = [0.1, 0.45, 0.45]
    char_to_id = {"你": 1, "好": 2}

    pred.ensure_cache()

    # At position 0, prediction is "你"
    boosted = pred.boost_distribution(ppm_probs, char_to_id)
    assert abs(sum(boosted) - 1.0) < 1e-9
    # "你" (id=1) should be boosted
    assert boosted[1] > ppm_probs[1]

    # Feed "你" and advance
    pred.feed_char("你")

    pred.ensure_cache()
    # At position 1, prediction is "好"
    boosted2 = pred.boost_distribution(ppm_probs, char_to_id)
    assert abs(sum(boosted2) - 1.0) < 1e-9
    # "好" (id=2) should be boosted
    assert boosted2[2] > ppm_probs[2]


# ---------------------------------------------------------------------------
# Test char-level online round-trip
# ---------------------------------------------------------------------------


def test_online_char_roundtrip_simple():
    """Compress and decompress with char-level online mode using fake API."""
    from zippedtext.compressor import compress, decompress

    text = "这是一段测试文本"
    fake = FakeApiClient(text)  # perfect prediction

    compressed = compress(text, mode="online", api_client=fake, sub_mode="char")
    assert len(compressed) > 0

    fake2 = FakeApiClient(text)  # decoder gets same predictions
    restored = decompress(compressed, api_client=fake2)
    assert restored == text


def test_online_char_roundtrip_mismatch():
    """Verify lossless round-trip even when LLM prediction is wrong."""
    from zippedtext.compressor import compress, decompress

    text = "你好世界！"
    # LLM predicts different text — compression still works, just less efficient
    fake = FakeApiClient("完全不同的文本内容")

    compressed = compress(text, mode="online", api_client=fake, sub_mode="char")
    fake2 = FakeApiClient("完全不同的文本内容")
    restored = decompress(compressed, api_client=fake2)
    assert restored == text


def test_online_char_roundtrip_english():
    """English text round-trip with online mode."""
    from zippedtext.compressor import compress, decompress

    text = "Hello, world! This is a test."
    fake = FakeApiClient("Hello, world! This is a test.")

    compressed = compress(text, mode="online", api_client=fake, sub_mode="char")
    fake2 = FakeApiClient("Hello, world! This is a test.")
    restored = decompress(compressed, api_client=fake2)
    assert restored == text


def test_online_char_roundtrip_mixed():
    """Mixed Chinese/English text round-trip."""
    from zippedtext.compressor import compress, decompress

    text = "Python是一种编程语言，version 3.12"
    fake = FakeApiClient("Python是一种编程语言，version 3.12")

    compressed = compress(text, mode="online", api_client=fake, sub_mode="char")
    fake2 = FakeApiClient("Python是一种编程语言，version 3.12")
    restored = decompress(compressed, api_client=fake2)
    assert restored == text


# ---------------------------------------------------------------------------
# Test token-level online round-trip
# ---------------------------------------------------------------------------


def test_online_token_roundtrip_match():
    """Token-level: text matches generated tokens perfectly."""
    from zippedtext.compressor import compress, decompress

    text = "这是一段测试文本"
    fake = FakeApiClient(text)

    compressed = compress(text, mode="online", api_client=fake, sub_mode="token")
    assert len(compressed) > 0

    fake2 = FakeApiClient(text)
    restored = decompress(compressed, api_client=fake2)
    assert restored == text


def test_online_token_roundtrip_partial_mismatch():
    """Token-level: text partially matches, then diverges."""
    from zippedtext.compressor import compress, decompress

    text = "这是一段不同的文本"
    # Only "这是一段" matches, rest diverges
    fake = FakeApiClient("这是一段测试内容用于比较")

    compressed = compress(text, mode="online", api_client=fake, sub_mode="token")
    fake2 = FakeApiClient("这是一段测试内容用于比较")
    restored = decompress(compressed, api_client=fake2)
    assert restored == text


# ---------------------------------------------------------------------------
# Test chunk boundary handling
# ---------------------------------------------------------------------------


def test_chunk_boundary():
    """Verify correct behavior when text spans multiple API chunks."""
    from zippedtext.compressor import compress, decompress

    # Use a very small chunk size to force multiple API calls.
    # Must patch both llm module AND compressor module (separate bindings).
    text = "你好世界测试文本压缩算法验证"
    fake = FakeApiClient("你好世界测试文本压缩算法验证")

    import zippedtext.compressor as comp_mod
    import zippedtext.predictor.llm as llm_mod
    orig_llm = llm_mod.CHUNK_CHARS
    orig_comp = comp_mod.CHUNK_CHARS

    llm_mod.CHUNK_CHARS = 4
    comp_mod.CHUNK_CHARS = 4

    try:
        compressed = compress(text, mode="online", api_client=fake, sub_mode="char")
        fake2 = FakeApiClient("你好世界测试文本压缩算法验证")
        restored = decompress(compressed, api_client=fake2)
        assert restored == text
        # Should have made multiple API calls
        assert fake.call_count >= 2
    finally:
        llm_mod.CHUNK_CHARS = orig_llm
        comp_mod.CHUNK_CHARS = orig_comp


# ---------------------------------------------------------------------------
# Test find_token_rank
# ---------------------------------------------------------------------------


def test_find_token_rank():
    """Verify token rank lookup."""
    from zippedtext.predictor.llm import _find_token_rank

    alts = [("hello", -0.1), ("world", -0.5), ("test", -1.0)]
    assert _find_token_rank("hello", alts) == 0
    assert _find_token_rank("world", alts) == 1
    assert _find_token_rank("test", alts) == 2
    assert _find_token_rank("missing", alts) == -1


# ---------------------------------------------------------------------------
# Test offline mode still works through compress/decompress API
# ---------------------------------------------------------------------------


def test_offline_mode_unchanged():
    """Verify offline mode is unaffected by online mode additions."""
    from zippedtext.compressor import compress, decompress

    text = "自然语言处理"
    compressed = compress(text, mode="offline")
    restored = decompress(compressed)
    assert restored == text


# Fake property for call counting
FakeApiClient.call_count = property(lambda self: self._call_count)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
