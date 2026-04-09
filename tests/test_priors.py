"""Tests for Chinese character frequency priors."""

import pytest

from zippedtext.predictor.priors import CHINESE_CHAR_FREQS, get_chinese_priors
from zippedtext.predictor.adaptive import AdaptivePredictor, ESCAPE_ID
from zippedtext.compressor import compress, decompress


class TestPriorsData:
    def test_returns_dict(self):
        priors = get_chinese_priors()
        assert isinstance(priors, dict)
        assert len(priors) > 100

    def test_frequencies_positive(self):
        for ch, freq in CHINESE_CHAR_FREQS.items():
            assert freq > 0, f"char {ch!r} has non-positive freq {freq}"

    def test_frequencies_sum_reasonable(self):
        total = sum(CHINESE_CHAR_FREQS.values())
        # Should sum to roughly 0.5-1.0 (top chars, not all)
        assert 0.3 < total < 1.5

    def test_common_chars_present(self):
        priors = get_chinese_priors()
        for ch in "的一是不了人我在有他":
            assert ch in priors, f"common char {ch!r} missing"

    def test_returns_copy(self):
        a = get_chinese_priors()
        b = get_chinese_priors()
        assert a is not b  # different dict objects


    def test_has_exactly_3000_unique_chars(self):
        assert len(CHINESE_CHAR_FREQS) == 3000
        assert len(set(CHINESE_CHAR_FREQS)) == 3000


class TestPredictorWithPriors:
    def test_warm_start_knows_common_chars(self):
        priors = get_chinese_priors()
        pred = AdaptivePredictor(priors=priors)
        assert pred.has_char("的")
        assert pred.has_char("是")
        assert pred.vocab_size() > 100

    def test_escape_not_needed_for_prior_chars(self):
        priors = get_chinese_priors()
        pred = AdaptivePredictor(priors=priors)
        # "的" should already be in vocab
        sid = pred.char_to_id("的")
        assert sid != ESCAPE_ID

    def test_unknown_chars_still_work(self):
        priors = get_chinese_priors()
        pred = AdaptivePredictor(priors=priors)
        # A rare character not in priors
        assert not pred.has_char("鑫")
        sid = pred.add_char("鑫")
        assert pred.has_char("鑫")


class TestCompressWithPriors:
    def test_roundtrip_with_priors(self):
        text = "深度学习是人工智能的核心技术，它改变了整个世界的发展方向。"
        data = compress(text, use_priors=True, use_phrases=False)
        restored = decompress(data)
        assert restored == text

    def test_roundtrip_without_priors(self):
        text = "这是一个没有先验的测试。"
        data = compress(text, use_priors=False, use_phrases=False)
        restored = decompress(data)
        assert restored == text

    def test_priors_flag_in_header(self):
        from zippedtext.format import FLAG_HAS_PRIORS, read_file
        text = "先验标志测试文本内容。"
        data_with = compress(text, use_priors=True, use_phrases=False)
        data_without = compress(text, use_priors=False, use_phrases=False)

        h_with, _, _ = read_file(data_with)
        h_without, _, _ = read_file(data_without)

        assert h_with.flags & FLAG_HAS_PRIORS
        assert not (h_without.flags & FLAG_HAS_PRIORS)

    def test_max_order_5(self):
        text = "测试高阶上下文模型，需要较长的文本才能体现效果，所以这里写了更多的内容。"
        data = compress(text, use_priors=False, use_phrases=False, max_order=5)
        restored = decompress(data)
        assert restored == text
