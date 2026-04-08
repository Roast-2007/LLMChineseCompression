"""LLM-based predictors for online mode compression.

Two approaches:
  1. LlmCharPredictor — character-level: uses LLM-generated text to boost
     character probabilities in the adaptive PPM distribution.
  2. LlmTokenPredictor — token-level: uses API token logprobs directly,
     matching generated tokens against actual text, with character-level
     fallback for mismatches.

v0.3.1 performance optimisations
---------------------------------
* CHUNK_CHARS raised from 20 → 200 (~10× fewer API calls).
* Refresh positions are strictly deterministic (every CHUNK_CHARS chars).
  Characters beyond the generated text in each chunk get pure PPM (no boost).
* **Prediction cache**: during compression, all LLM responses are collected
  and stored in the .ztxt file (zstd-compressed).  During decompression the
  cached predictions are used directly — NO API calls, NO non-determinism,
  near-instant decompression.
* Parameters (chunk_chars, max_tokens) are stored in the .ztxt model_data
  section so encoder and decoder always agree.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..api_client import ChunkResult, DeepSeekClient

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

CHAR_BOOST = 20.0          # multiplicative boost for LLM-predicted character
CHUNK_CHARS = 200           # characters per chunk before forced API refresh
MAX_TOKENS_DEFAULT = 100    # tokens per API call (v0.3.1: reduced for speed)
MAX_CONTEXT_LEN = 3000      # max context chars sent to API (trim older text)
MIN_PROB_FLOOR = 1e-8       # minimum probability for any character

# Token-level constants
CATCH_ALL_VOCAB = 100_000   # virtual vocab size for non-top-k tokens


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
      4. **Stop boosting** after the first mismatch in a chunk.
      5. Refresh happens at fixed intervals (every ``chunk_chars`` chars).

    v0.3.1 prediction cache:
      - ``prediction_cache`` is a list of generated-text strings, one per
        chunk.  When provided (decompression), predictions are read from
        the cache instead of calling the API.
      - ``collect_cache`` controls whether to collect predictions during
        encoding (compression).
    """

    def __init__(
        self,
        api_client: DeepSeekClient | None = None,
        chunk_chars: int | None = None,
        max_tokens: int | None = None,
        prediction_cache: list[str] | None = None,
        collect_cache: bool = False,
    ) -> None:
        self._api = api_client
        self._chunk_chars = chunk_chars if chunk_chars is not None else CHUNK_CHARS
        self._max_tokens = max_tokens if max_tokens is not None else MAX_TOKENS_DEFAULT

        # Cache: predicted characters for current chunk
        self._predicted_chars: str = ""
        self._chunk_start: int = 0
        self._pos: int = 0
        self._initialized: bool = False

        # Accumulated actual text (for API context)
        self._actual_text: str = ""
        self._chunk_diverged: bool = False

        # --- Prediction cache ---
        self._prediction_cache = prediction_cache  # read-only cache (decompression)
        self._cache_idx: int = 0
        self._collect_cache = collect_cache
        self._collected: list[str] = []  # written during compression

    @property
    def model_name(self) -> str:
        if self._api:
            return self._api.last_model_id or self._api.model
        return ""

    @property
    def collected_predictions(self) -> list[str]:
        """Predictions collected during compression (for embedding in file)."""
        return self._collected

    def feed_char(self, ch: str) -> None:
        """Called after encoding/decoding a character to update state."""
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
        prediction (if available) in the PPM distribution."""
        if self._chunk_diverged:
            return ppm_probs
        predicted_char = self._get_predicted_char()
        if predicted_char is None or predicted_char not in char_to_id:
            return ppm_probs
        boosted_id = char_to_id[predicted_char]
        return _boost_prob(ppm_probs, boosted_id, CHAR_BOOST)

    def ensure_cache(self) -> None:
        """Refresh the prediction cache if needed."""
        if not self._initialized:
            self._refresh_cache()
            self._initialized = True
        elif self._pos - self._chunk_start >= self._chunk_chars:
            self._refresh_cache()

    def cleanup(self) -> None:
        """No-op (kept for interface compatibility)."""

    # ------------------------------------------------------------------

    def _refresh_cache(self) -> None:
        """Load the next chunk's predictions — from cache or API."""
        if self._prediction_cache is not None:
            # Decompression path: read from embedded cache
            if self._cache_idx < len(self._prediction_cache):
                self._predicted_chars = self._prediction_cache[self._cache_idx]
                self._cache_idx += 1
            else:
                self._predicted_chars = ""
        else:
            # Compression path: call API
            ctx = self._actual_text
            if len(ctx) > MAX_CONTEXT_LEN:
                ctx = ctx[-MAX_CONTEXT_LEN:]
            result = self._api.generate_continuation(ctx, self._max_tokens)
            self._predicted_chars = result.generated_text
            if self._collect_cache:
                self._collected.append(self._predicted_chars)

        self._chunk_start = self._pos
        self._chunk_diverged = False

    def _get_predicted_char(self) -> str | None:
        offset = self._pos - self._chunk_start
        if 0 <= offset < len(self._predicted_chars):
            return self._predicted_chars[offset]
        return None


# ---------------------------------------------------------------------------
# LlmTokenPredictor — token-level online mode
# ---------------------------------------------------------------------------

@dataclass
class TokenMatch:
    """Result of matching a generated token against actual text."""
    matched: bool
    token_text: str
    token_rank: int
    token_logprob: float
    top_alternatives: list[tuple[str, float]]
    chars_consumed: int


class LlmTokenPredictor:
    """Token-level predictor that matches API-generated tokens against text.

    v0.3.1: accepts ``max_tokens`` to tune generation length.  Supports
    prediction cache for API-free decompression (stores full ChunkResult
    generated_text + token logprobs).
    """

    def __init__(
        self,
        api_client: DeepSeekClient | None = None,
        chunk_chars: int | None = None,
        max_tokens: int | None = None,
        full_text: str | None = None,
        prediction_cache: list[ChunkResult] | None = None,
        collect_cache: bool = False,
    ) -> None:
        self._api = api_client
        self._max_tokens = max_tokens if max_tokens is not None else MAX_TOKENS_DEFAULT

        self._chunk: ChunkResult | None = None
        self._token_idx: int = 0
        self._chunk_start: int = 0
        self._pos: int = 0
        self._diverged: bool = False
        self._actual_text: str = ""

        # --- Prediction cache ---
        self._prediction_cache = prediction_cache
        self._cache_idx: int = 0
        self._collect_cache = collect_cache
        self._collected: list[ChunkResult] = []

    @property
    def model_name(self) -> str:
        if self._api:
            return self._api.last_model_id or self._api.model
        return ""

    @property
    def collected_predictions(self) -> list:
        return self._collected

    def feed_chars(self, text: str) -> None:
        self._actual_text += text
        self._pos += len(text)

    def needs_refresh(self) -> bool:
        if self._chunk is None:
            return True
        if self._diverged:
            return True
        if self._token_idx >= len(self._chunk.tokens):
            return True
        return False

    def refresh_cache(self) -> None:
        """Make a new API call or read from cache."""
        if self._prediction_cache is not None:
            if self._cache_idx < len(self._prediction_cache):
                self._chunk = self._prediction_cache[self._cache_idx]
                self._cache_idx += 1
            else:
                self._chunk = None
        else:
            ctx = self._actual_text
            if len(ctx) > MAX_CONTEXT_LEN:
                ctx = ctx[-MAX_CONTEXT_LEN:]
            self._chunk = self._api.generate_continuation(ctx, self._max_tokens)
            if self._collect_cache:
                self._collected.append(self._chunk)

        self._token_idx = 0
        self._chunk_start = self._pos
        self._diverged = False

    def try_match_next_token(self, actual_text_remaining: str) -> TokenMatch | None:
        if self._chunk is None or self._diverged:
            return None
        if self._token_idx >= len(self._chunk.tokens):
            return None

        gen_token = self._chunk.tokens[self._token_idx]
        token_text = gen_token.text

        if actual_text_remaining.startswith(token_text):
            rank = _find_token_rank(token_text, gen_token.top_alternatives)
            self._token_idx += 1
            return TokenMatch(
                matched=True, token_text=token_text, token_rank=rank,
                token_logprob=gen_token.logprob,
                top_alternatives=gen_token.top_alternatives,
                chars_consumed=len(token_text),
            )

        for alt_text, alt_logprob in gen_token.top_alternatives:
            if actual_text_remaining.startswith(alt_text):
                rank = _find_token_rank(alt_text, gen_token.top_alternatives)
                self._token_idx += 1
                self._diverged = True
                return TokenMatch(
                    matched=True, token_text=alt_text, token_rank=rank,
                    token_logprob=alt_logprob,
                    top_alternatives=gen_token.top_alternatives,
                    chars_consumed=len(alt_text),
                )

        self._diverged = True
        return None

    def build_token_probs(
        self, top_alternatives: list[tuple[str, float]],
    ) -> tuple[list[float], list[str]]:
        real_alts: list[tuple[str, float]] = []
        total_real = 0.0
        for tok, lp in top_alternatives:
            p = math.exp(lp) if lp > -100 else 0.0
            if p > MIN_PROB_FLOOR:
                real_alts.append((tok, p))
                total_real += p
        if not real_alts:
            return [1.0], ["<CATCHALL>"]
        remainder = max(1.0 - total_real, MIN_PROB_FLOOR)
        token_strings = [tok for tok, _ in real_alts] + ["<CATCHALL>"]
        probs = [p for _, p in real_alts] + [remainder]
        s = sum(probs)
        return [p / s for p in probs], token_strings

    def cleanup(self) -> None:
        pass


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _boost_prob(probs: list[float], boost_id: int, boost_factor: float) -> list[float]:
    result = list(probs)
    result[boost_id] = max(result[boost_id] * boost_factor, MIN_PROB_FLOOR)
    total = sum(result)
    if total <= 0:
        return probs
    return [p / total for p in result]


def _find_token_rank(token_text: str, alternatives: list[tuple[str, float]]) -> int:
    for i, (alt_text, _) in enumerate(alternatives):
        if alt_text == token_text:
            return i
    return -1
