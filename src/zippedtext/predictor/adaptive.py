"""Adaptive PPM-style predictor with escape-based dynamic vocabulary.

Key design: no static vocabulary is needed. Characters are added to the
active vocab on first occurrence via an ESCAPE mechanism:
  - Known char → encode directly (cheap, benefits from context model)
  - New char → encode ESCAPE + raw Unicode codepoint (expensive, but only once)

Both encoder and decoder maintain identical state, so no model/vocab
needs to be stored in the compressed file.
"""

from __future__ import annotations

from .base import Predictor

ALPHA = 0.005       # smoothing floor
MAX_ORDER = 4       # max context depth
ESCAPE_ID = 0       # reserved symbol ID for escape


class AdaptivePredictor(Predictor):
    """Multi-order adaptive predictor with dynamic vocabulary via escape coding."""

    def __init__(self) -> None:
        # Dynamic vocabulary: char → symbol_id (0 is reserved for ESCAPE)
        self._char_to_id: dict[str, int] = {}
        self._id_to_char: list[str] = ["<ESC>"]  # index 0 = escape
        self._current_vocab_size = 1  # starts with just ESCAPE

        # Adaptive counts per context
        self._unigram_counts: list[float] = [ALPHA]  # just ESCAPE initially
        self._unigram_total: float = ALPHA
        self._context_counts: dict[tuple[int, ...], list[float]] = {}
        self._context_totals: dict[tuple[int, ...], float] = {}
        self._history: list[int] = []

    def vocab_size(self) -> int:
        return self._current_vocab_size

    def reset(self) -> None:
        self.__init__()

    def has_char(self, ch: str) -> bool:
        return ch in self._char_to_id

    def char_to_id(self, ch: str) -> int:
        return self._char_to_id[ch]

    def id_to_char(self, sid: int) -> str:
        return self._id_to_char[sid]

    def add_char(self, ch: str) -> int:
        """Add a new character to the vocabulary. Returns its new ID."""
        sid = self._current_vocab_size
        self._char_to_id[ch] = sid
        self._id_to_char.append(ch)
        self._current_vocab_size += 1

        # Extend all probability arrays
        self._unigram_counts.append(ALPHA)
        self._unigram_total += ALPHA
        for ctx in self._context_counts:
            self._context_counts[ctx].append(ALPHA)
            self._context_totals[ctx] += ALPHA

        return sid

    def predict(self, context: list[int]) -> list[float]:
        """Return probability distribution over current vocab (including ESCAPE)."""
        vs = self._current_vocab_size

        # Collect distributions from available orders
        distributions: list[tuple[list[float], float]] = []

        for order in range(min(MAX_ORDER, len(self._history)), 0, -1):
            ctx = tuple(self._history[-order:])
            if ctx in self._context_counts:
                total = self._context_totals[ctx]
                counts = self._context_counts[ctx]
                obs = total - ALPHA * vs
                weight = min(obs / (obs + 5.0), 0.9) if obs > 0 else 0.0
                if weight > 0:
                    distributions.append(
                        ([c / total for c in counts], weight)
                    )

        # Unigram base
        uni_total = self._unigram_total
        uni_probs = [c / uni_total for c in self._unigram_counts]

        if not distributions:
            return uni_probs

        result = [0.0] * vs
        remaining = 1.0
        for probs, weight in distributions:
            w = remaining * weight
            for i in range(vs):
                result[i] += w * probs[i]
            remaining -= w
        for i in range(vs):
            result[i] += remaining * uni_probs[i]

        return result

    def update(self, symbol: int) -> None:
        """Update model after observing a symbol."""
        self._unigram_counts[symbol] += 1
        self._unigram_total += 1

        for order in range(1, min(MAX_ORDER, len(self._history)) + 1):
            ctx = tuple(self._history[-order:])
            if ctx not in self._context_counts:
                self._context_counts[ctx] = [ALPHA] * self._current_vocab_size
                self._context_totals[ctx] = ALPHA * self._current_vocab_size
            self._context_counts[ctx][symbol] += 1
            self._context_totals[ctx] += 1

        self._history.append(symbol)
        if len(self._history) > MAX_ORDER:
            self._history.pop(0)
