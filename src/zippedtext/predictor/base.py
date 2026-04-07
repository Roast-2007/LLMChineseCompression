"""Abstract base class for next-character predictors."""

from __future__ import annotations

from abc import ABC, abstractmethod


class Predictor(ABC):
    """Provides P(next_char | context) for arithmetic coding."""

    @abstractmethod
    def predict(self, context: list[int]) -> list[float]:
        """Return probability distribution over vocab given context (token IDs).

        Returns a list of length vocab_size where probs[i] = P(next = i | context).
        Must sum to ~1.0.
        """

    @abstractmethod
    def vocab_size(self) -> int:
        """Return the vocabulary size."""

    @abstractmethod
    def reset(self) -> None:
        """Reset any internal state."""
