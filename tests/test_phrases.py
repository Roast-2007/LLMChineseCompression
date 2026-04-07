"""Tests for phrase-level encoding."""

import pytest

from zippedtext.predictor.phrases import (
    PhraseTable,
    build_phrase_table,
    greedy_phrase_match,
)
from zippedtext.compressor import compress, decompress


class TestPhraseTable:
    def test_serialize_roundtrip(self):
        pt = PhraseTable(phrases=("你好", "世界", "人工智能"))
        data = pt.serialize()
        pt2 = PhraseTable.deserialize(data)
        assert pt2.phrases == pt.phrases

    def test_empty_table(self):
        pt = PhraseTable(phrases=())
        data = pt.serialize()
        pt2 = PhraseTable.deserialize(data)
        assert pt2.phrases == ()

    def test_single_phrase(self):
        pt = PhraseTable(phrases=("hello",))
        data = pt.serialize()
        pt2 = PhraseTable.deserialize(data)
        assert pt2.phrases == ("hello",)


class TestBuildPhraseTable:
    def test_finds_repeated_phrases(self):
        text = "人工智能是未来。人工智能改变世界。人工智能很重要。"
        pt = build_phrase_table(text, min_freq=2)
        phrase_strs = set(pt.phrases)
        assert "人工智能" in phrase_strs

    def test_respects_min_freq(self):
        text = "abc abc abc xyz"
        pt = build_phrase_table(text, min_freq=3)
        # "abc" appears 3 times, should be found
        # "xyz" appears once, should not
        for p in pt.phrases:
            assert "xyz" not in p

    def test_empty_text(self):
        pt = build_phrase_table("")
        assert pt.phrases == ()

    def test_short_text_no_phrases(self):
        pt = build_phrase_table("ab", min_freq=3)
        assert pt.phrases == ()


class TestGreedyMatch:
    def test_longest_match(self):
        phrases = frozenset(["人工", "人工智能"])
        result = greedy_phrase_match("人工智能技术", 0, phrases)
        assert result == "人工智能"

    def test_no_match(self):
        phrases = frozenset(["hello"])
        result = greedy_phrase_match("world", 0, phrases)
        assert result is None

    def test_match_at_offset(self):
        phrases = frozenset(["智能"])
        result = greedy_phrase_match("人工智能", 2, phrases)
        assert result == "智能"

    def test_boundary(self):
        phrases = frozenset(["ab"])
        result = greedy_phrase_match("a", 0, phrases)
        assert result is None


class TestPhraseRoundtrip:
    """End-to-end: compress with phrases, decompress, verify lossless."""

    def test_chinese_with_phrases(self):
        text = "人工智能是未来的发展方向。人工智能改变世界。人工智能很重要，人工智能是核心技术。"
        data = compress(text, use_phrases=True, use_priors=False)
        restored = decompress(data)
        assert restored == text

    def test_english_with_phrases(self):
        text = (
            "the quick brown fox jumps over the lazy dog. "
            "the quick brown fox runs fast. the quick brown fox sleeps."
        )
        data = compress(text, use_phrases=True, use_priors=False)
        restored = decompress(data)
        assert restored == text

    def test_mixed_with_phrases(self):
        text = "深度学习模型训练需要大量数据。深度学习模型在各个领域应用广泛。深度学习模型持续进化。"
        data = compress(text, use_phrases=True, use_priors=False)
        restored = decompress(data)
        assert restored == text

    def test_short_text_no_phrases(self):
        """Short texts should not use phrases (< 50 chars threshold)."""
        text = "你好世界"
        data = compress(text, use_phrases=True, use_priors=False)
        restored = decompress(data)
        assert restored == text

    def test_phrases_disabled(self):
        text = "人工智能是未来。人工智能很重要。人工智能改变世界。人工智能是核心技术。"
        data = compress(text, use_phrases=False, use_priors=False)
        restored = decompress(data)
        assert restored == text
