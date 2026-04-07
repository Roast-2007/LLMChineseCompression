"""LLM-based predictors for online mode compression.

Two approaches:
  1. LlmCharPredictor — character-level: uses LLM-generated text to boost
     character probabilities in the adaptive PPM distribution.
  2. LlmTokenPredictor — token-level: uses API token logprobs directly,
     matching generated tokens against actual text, with character-level
     fallback for mismatches.

Both predictors maintain a cache of predictions from the last API call and
refresh automatically when the cache is exhausted.  Encoder and decoder
MUST make identical API calls at identical positions to stay in sync.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..api_client import ChunkResult, DeepSeekClient

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

CHAR_BOOST = 20.0        # multiplicative boost for LLM-predicted character
CHUNK_CHARS = 20          # make a new API call every ~N characters
MAX_CONTEXT_LEN = 3000    # max context chars sent to API (trim older text)
MIN_PROB_FLOOR = 1e-8     # minimum probability for any character

# Token-level constants
CATCH_ALL_VOCAB = 100_000  # virtual vocab size for non-top-k tokens


# ---------------------------------------------------------------------------
# LlmCharPredictor — character-level online mode
# ---------------------------------------------------------------------------

class LlmCharPredictor:
    """Boosts adaptive PPM distributions using LLM text predictions.

    Algorithm:
      1. Call API to generate continuation text from current context.
      2. Extract the predicted character at each position.
      3. When encoding/decoding character *i*, multiply the predicted
         character's PPM probability by CHAR_BOOST, then re-normalize.
      4. **Stop boosting** after the first mismatch in a chunk — subsequent
         predictions are unreliable once context diverges from generation.
      5. When cache is exhausted, make a new API call with updated context.

    Both encoder and decoder feed the *actual* text (so far) as context,
    ensuring deterministic API calls at deterministic positions.  They
    independently track the mismatch flag (since they see the same chars).
    """

    def __init__(self, api_client: DeepSeekClient, chunk_chars: int | None = None) -> None:
        self._api = api_client
        self._chunk_chars = chunk_chars if chunk_chars is not None else CHUNK_CHARS

        # Cache: predicted characters for current chunk
        self._predicted_chars: str = ""
        self._chunk_start: int = 0   # char position where current chunk begins
        self._pos: int = 0           # current char position in the full text

        # Accumulated actual text (for API context)
        self._actual_text: str = ""

        # Stop-on-mismatch: once the actual text diverges from the
        # prediction, stop boosting for the rest of this chunk.
        self._chunk_diverged: bool = False

    @property
    def model_name(self) -> str:
        return self._api.last_model_id or self._api.model

    def feed_char(self, ch: str) -> None:
        """Called after encoding/decoding a character to update state.

        Also checks whether the prediction at this position was correct.
        If not, marks the current chunk as diverged so boost_distribution
        stops boosting for the remainder of the chunk.
        """
        predicted = self._get_predicted_char()
        if predicted is not None and predicted != ch:
            self._chunk_diverged = True
        self._actual_text += ch
        self._pos += 1

    def boost_distribution(
        self,
        ppm_probs: list[float],
        char_to_id: dict[str, int],
    ) -> list[float]:
        """Return a modified probability distribution that boosts the LLM
        prediction (if available) in the PPM distribution.

        Returns ppm_probs unchanged when:
          - No prediction available
          - Prediction diverged from actual text (stop-on-mismatch)
          - Predicted char not yet in vocabulary
        """
        if self._chunk_diverged:
            return ppm_probs

        predicted_char = self._get_predicted_char()
        if predicted_char is None:
            return ppm_probs

        if predicted_char not in char_to_id:
            return ppm_probs

        boosted_id = char_to_id[predicted_char]
        return _boost_prob(ppm_probs, boosted_id, CHAR_BOOST)

    def ensure_cache(self) -> None:
        """Refresh the prediction cache if needed (start of chunk boundary).

        Called at deterministic positions so encoder/decoder stay in sync.
        Must be called BEFORE boost_distribution for each character.
        Refreshes when: cache exhausted OR chunk size limit reached.
        """
        offset_in_chunk = self._pos - self._chunk_start
        if offset_in_chunk >= len(self._predicted_chars) or offset_in_chunk >= self._chunk_chars:
            self._refresh_cache()

    # ------------------------------------------------------------------

    def _get_predicted_char(self) -> str | None:
        offset = self._pos - self._chunk_start
        if 0 <= offset < len(self._predicted_chars):
            return self._predicted_chars[offset]
        return None

    def _refresh_cache(self) -> None:
        context = self._actual_text
        if len(context) > MAX_CONTEXT_LEN:
            context = context[-MAX_CONTEXT_LEN:]

        result = self._api.generate_continuation(context, max_tokens=200)
        self._predicted_chars = result.generated_text
        self._chunk_start = self._pos
        self._chunk_diverged = False  # reset for new chunk


# ---------------------------------------------------------------------------
# LlmTokenPredictor — token-level online mode
# ---------------------------------------------------------------------------

@dataclass
class TokenMatch:
    """Result of matching a generated token against actual text."""

    matched: bool
    token_text: str
    token_rank: int             # rank in top-k (0 = top-1), -1 if not found
    token_logprob: float
    top_alternatives: list[tuple[str, float]]
    chars_consumed: int         # how many chars of actual text this covers


class LlmTokenPredictor:
    """Token-level predictor that matches API-generated tokens against text.

    Algorithm:
      1. Call API to generate continuation.
      2. Walk through generated tokens, comparing each token's text against
         the actual text at the current position.
      3. MATCH → the actual text starts with this token's text.
         MISMATCH → the actual text diverges; remaining tokens are invalid.
      4. For matched tokens, build a CDF from top-k logprobs.
      5. For mismatched regions, signal the caller to fall back to
         character-level encoding.

    Encoder and decoder maintain identical state because both feed the
    same actual/decoded text and make API calls at identical positions.
    """

    def __init__(self, api_client: DeepSeekClient) -> None:
        self._api = api_client

        # Cache from last API call
        self._chunk: ChunkResult | None = None
        self._token_idx: int = 0        # index into chunk.tokens
        self._chunk_start: int = 0      # char position where chunk begins
        self._pos: int = 0              # current char position
        self._diverged: bool = False    # True after first mismatch in chunk

        self._actual_text: str = ""

    @property
    def model_name(self) -> str:
        return self._api.last_model_id or self._api.model

    def feed_chars(self, text: str) -> None:
        """Called after encoding/decoding characters to update state."""
        self._actual_text += text
        self._pos += len(text)

    def needs_refresh(self) -> bool:
        """True when we need a new API call (cache empty or diverged)."""
        if self._chunk is None:
            return True
        if self._diverged:
            return True
        if self._token_idx >= len(self._chunk.tokens):
            return True
        return False

    def refresh_cache(self) -> None:
        """Make a new API call with current context."""
        context = self._actual_text
        if len(context) > MAX_CONTEXT_LEN:
            context = context[-MAX_CONTEXT_LEN:]

        self._chunk = self._api.generate_continuation(context, max_tokens=200)
        self._token_idx = 0
        self._chunk_start = self._pos
        self._diverged = False

    def try_match_next_token(self, actual_text_remaining: str) -> TokenMatch | None:
        """Try to match the next generated token against actual text.

        Returns None if cache is empty/diverged (caller should refresh or
        fall back to character-level).
        """
        if self._chunk is None or self._diverged:
            return None
        if self._token_idx >= len(self._chunk.tokens):
            return None

        gen_token = self._chunk.tokens[self._token_idx]
        token_text = gen_token.text

        if actual_text_remaining.startswith(token_text):
            # MATCH — find rank of this token in top alternatives
            rank = _find_token_rank(token_text, gen_token.top_alternatives)
            self._token_idx += 1
            return TokenMatch(
                matched=True,
                token_text=token_text,
                token_rank=rank,
                token_logprob=gen_token.logprob,
                top_alternatives=gen_token.top_alternatives,
                chars_consumed=len(token_text),
            )

        # Check if actual text matches any top alternative
        for alt_text, alt_logprob in gen_token.top_alternatives:
            if actual_text_remaining.startswith(alt_text):
                rank = _find_token_rank(alt_text, gen_token.top_alternatives)
                self._token_idx += 1
                # Mark diverged — subsequent token predictions are invalid
                self._diverged = True
                return TokenMatch(
                    matched=True,
                    token_text=alt_text,
                    token_rank=rank,
                    token_logprob=alt_logprob,
                    top_alternatives=gen_token.top_alternatives,
                    chars_consumed=len(alt_text),
                )

        # No match at all — diverge
        self._diverged = True
        return None

    def build_token_probs(
        self,
        top_alternatives: list[tuple[str, float]],
    ) -> tuple[list[float], list[str]]:
        """Build a probability distribution over tokens from top-k logprobs.

        Returns (probs, token_strings) where probs[i] corresponds to
        token_strings[i].  An extra "catch-all" entry is appended for
        tokens not in the top-k.
        """
        real_alts: list[tuple[str, float]] = []
        total_real = 0.0
        for tok, lp in top_alternatives:
            p = math.exp(lp) if lp > -100 else 0.0  # ignore -9999 entries
            if p > MIN_PROB_FLOOR:
                real_alts.append((tok, p))
                total_real += p

        if not real_alts:
            # No useful alternatives — return uniform over catch-all
            return [1.0], ["<CATCHALL>"]

        remainder = max(1.0 - total_real, MIN_PROB_FLOOR)
        token_strings = [tok for tok, _ in real_alts] + ["<CATCHALL>"]
        probs = [p for _, p in real_alts] + [remainder]

        # Normalize
        s = sum(probs)
        probs = [p / s for p in probs]
        return probs, token_strings


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _boost_prob(probs: list[float], boost_id: int, boost_factor: float) -> list[float]:
    """Multiply probs[boost_id] by boost_factor, then re-normalize."""
    result = list(probs)
    result[boost_id] = max(result[boost_id] * boost_factor, MIN_PROB_FLOOR)
    total = sum(result)
    if total <= 0:
        return probs
    return [p / total for p in result]


def _find_token_rank(token_text: str, alternatives: list[tuple[str, float]]) -> int:
    """Find the rank (0-indexed) of token_text in the alternatives list."""
    for i, (alt_text, _) in enumerate(alternatives):
        if alt_text == token_text:
            return i
    return -1
