"""N-gram predictor for offline mode compression.

Builds character-level unigram + bigram probability tables.
Used as the local model for offline decompression.
"""

from __future__ import annotations

import json
import math
from collections import Counter

from .base import Predictor

SMOOTHING = 1e-6  # Laplace smoothing floor


class NgramPredictor(Predictor):
    """Character-level unigram + bigram predictor with Laplace smoothing."""

    def __init__(self, vocab_size: int) -> None:
        self._vocab_size = vocab_size
        self._unigram: list[float] = [1.0 / vocab_size] * vocab_size
        self._bigram: dict[int, list[float]] = {}

    def vocab_size(self) -> int:
        return self._vocab_size

    def reset(self) -> None:
        pass  # stateless predictions

    def predict(self, context: list[int]) -> list[float]:
        if context and context[-1] in self._bigram:
            return self._bigram[context[-1]]
        return self._unigram

    @classmethod
    def from_token_ids(cls, token_ids: list[int], vocab_size: int) -> NgramPredictor:
        """Build predictor from a sequence of token IDs."""
        pred = cls(vocab_size)

        # Unigram frequencies
        counts = Counter(token_ids)
        total = len(token_ids)
        if total > 0:
            pred._unigram = [
                (counts.get(i, 0) + SMOOTHING) / (total + SMOOTHING * vocab_size)
                for i in range(vocab_size)
            ]

        # Bigram frequencies
        bigram_counts: dict[int, Counter] = {}
        for i in range(len(token_ids) - 1):
            prev = token_ids[i]
            curr = token_ids[i + 1]
            if prev not in bigram_counts:
                bigram_counts[prev] = Counter()
            bigram_counts[prev][curr] += 1

        for prev, counts in bigram_counts.items():
            total = sum(counts.values())
            pred._bigram[prev] = [
                (counts.get(i, 0) + SMOOTHING) / (total + SMOOTHING * vocab_size)
                for i in range(vocab_size)
            ]

        return pred

    @classmethod
    def from_llm_analysis(cls, analysis: dict, vocab_size: int, char_to_id: dict[str, int]) -> NgramPredictor:
        """Build predictor from LLM analysis results (offline mode)."""
        pred = cls(vocab_size)

        # Apply char frequencies from LLM analysis
        char_freqs = analysis.get("char_frequencies", {})
        if char_freqs:
            total_freq = sum(char_freqs.values())
            if total_freq > 0:
                for ch, freq in char_freqs.items():
                    if ch in char_to_id:
                        pred._unigram[char_to_id[ch]] = freq / total_freq
                # Normalize
                s = sum(pred._unigram)
                pred._unigram = [p / s for p in pred._unigram]

        # Apply bigram frequencies
        bigrams = analysis.get("top_bigrams", [])
        for pair_str, freq in bigrams:
            if len(pair_str) == 2:
                ch1, ch2 = pair_str[0], pair_str[1]
                if ch1 in char_to_id and ch2 in char_to_id:
                    id1, id2 = char_to_id[ch1], char_to_id[ch2]
                    if id1 not in pred._bigram:
                        pred._bigram[id1] = list(pred._unigram)
                    pred._bigram[id1][id2] += freq
                    # Re-normalize this row
                    s = sum(pred._bigram[id1])
                    pred._bigram[id1] = [p / s for p in pred._bigram[id1]]

        return pred

    def serialize(self) -> bytes:
        """Serialize to JSON bytes for storage in file header."""
        data = {
            "vocab_size": self._vocab_size,
            "unigram": self._unigram,
            "bigram": {str(k): v for k, v in self._bigram.items()},
        }
        return json.dumps(data, separators=(",", ":")).encode("utf-8")

    @classmethod
    def deserialize(cls, data: bytes) -> NgramPredictor:
        """Deserialize from JSON bytes."""
        obj = json.loads(data.decode("utf-8"))
        pred = cls(obj["vocab_size"])
        pred._unigram = obj["unigram"]
        pred._bigram = {int(k): v for k, v in obj["bigram"].items()}
        return pred
