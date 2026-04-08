"""Encoding routines for all compression modes.

Extracted from compressor.py — contains the encoding half of every
compress pipeline (offline, online-char, online-token).
"""

from __future__ import annotations

import io

from .arithmetic import ArithmeticEncoder, probs_to_cdf
from .bitstream import BitOutputStream
from .cdf_utils import uniform_cdf
from .predictor.adaptive import ESCAPE_ID, AdaptivePredictor

# Unicode codepoint range constants
RANGE_ASCII = 0
RANGE_CJK = 1
RANGE_OTHER = 2
NUM_RANGES = 3

ASCII_START, ASCII_END = 0x20, 0x7E
CJK_START, CJK_END = 0x3000, 0x9FFF

# Token-level online mode tags
TAG_TOKEN = 1
TAG_CHARS = 0
MAX_FALLBACK_LEN = 64


# ------------------------------------------------------------------
# Offline encode
# ------------------------------------------------------------------

def encode(
    text: str,
    on_progress=None,
    priors: dict[str, float] | None = None,
    max_order: int = 4,
) -> bytes:
    """Encode text using adaptive arithmetic coding with escape-based vocab."""
    buf = io.BytesIO()
    out = BitOutputStream(buf)
    encoder = ArithmeticEncoder(out)
    predictor = AdaptivePredictor(priors=priors, max_order=max_order)

    for i, ch in enumerate(text):
        if predictor.has_char(ch):
            sid = predictor.char_to_id(ch)
            probs = predictor.predict([])
            cdf = probs_to_cdf(probs, predictor.vocab_size())
            encoder.encode(cdf, sid)
            predictor.update(sid)
        else:
            probs = predictor.predict([])
            cdf = probs_to_cdf(probs, predictor.vocab_size())
            encoder.encode(cdf, ESCAPE_ID)
            predictor.update(ESCAPE_ID)
            _encode_codepoint(encoder, ord(ch))
            sid = predictor.add_char(ch)

        if on_progress and (i % 200 == 0 or i == len(text) - 1):
            on_progress(i + 1, len(text))

    encoder.finish()
    return buf.getvalue()


# ------------------------------------------------------------------
# Online char-level encode
# ------------------------------------------------------------------

def encode_online_char(
    text: str,
    api_client,
    on_progress=None,
    priors: dict[str, float] | None = None,
    max_order: int = 4,
    chunk_chars: int | None = None,
    max_tokens: int | None = None,
) -> tuple[bytes, list[str]]:
    """Encode text using adaptive PPM boosted by LLM character predictions.

    Returns ``(compressed_body, prediction_cache)`` where prediction_cache
    is the list of generated texts to embed in the .ztxt file for
    API-free decompression (v0.3.1).
    """
    from .predictor.llm import LlmCharPredictor

    buf = io.BytesIO()
    out = BitOutputStream(buf)
    encoder = ArithmeticEncoder(out)
    predictor = AdaptivePredictor(priors=priors, max_order=max_order)
    llm = LlmCharPredictor(
        api_client,
        chunk_chars=chunk_chars,
        max_tokens=max_tokens,
        collect_cache=True,
    )

    for i, ch in enumerate(text):
        llm.ensure_cache()

        if predictor.has_char(ch):
            sid = predictor.char_to_id(ch)
            ppm_probs = predictor.predict([])
            probs = llm.boost_distribution(ppm_probs, predictor._char_to_id)
            cdf = probs_to_cdf(probs, predictor.vocab_size())
            encoder.encode(cdf, sid)
            predictor.update(sid)
        else:
            ppm_probs = predictor.predict([])
            probs = llm.boost_distribution(ppm_probs, predictor._char_to_id)
            cdf = probs_to_cdf(probs, predictor.vocab_size())
            encoder.encode(cdf, ESCAPE_ID)
            predictor.update(ESCAPE_ID)
            _encode_codepoint(encoder, ord(ch))
            sid = predictor.add_char(ch)

        llm.feed_char(ch)

        if on_progress and (i % 200 == 0 or i == len(text) - 1):
            on_progress(i + 1, len(text))

    encoder.finish()
    return buf.getvalue(), llm.collected_predictions


# ------------------------------------------------------------------
# Online token-level encode
# ------------------------------------------------------------------

def encode_online_token(
    text: str,
    api_client,
    on_progress=None,
    priors: dict[str, float] | None = None,
    max_order: int = 4,
    chunk_chars: int | None = None,
    max_tokens: int | None = None,
) -> tuple[bytes, list]:
    """Token-level online encoding.

    Returns ``(compressed_body, prediction_cache)`` for legacy token-mode
    decompression without live API calls.
    """
    from .predictor.llm import LlmTokenPredictor

    buf = io.BytesIO()
    out = BitOutputStream(buf)
    encoder = ArithmeticEncoder(out)
    predictor = AdaptivePredictor(priors=priors, max_order=max_order)
    llm = LlmTokenPredictor(
        api_client,
        max_tokens=max_tokens,
        collect_cache=True,
    )

    pos = 0
    try:
        while pos < len(text):
            if llm.needs_refresh():
                llm.refresh_cache()

            remaining = text[pos:]
            match = llm.try_match_next_token(remaining)

            if match is not None and match.matched:
                _encode_tag(encoder, TAG_TOKEN)

                probs, token_strings = llm.build_token_probs(match.top_alternatives)
                rank = -1
                for idx, ts in enumerate(token_strings):
                    if ts == match.token_text:
                        rank = idx
                        break
                if rank == -1:
                    rank = len(token_strings) - 1

                cdf = probs_to_cdf(probs, len(probs))
                encoder.encode(cdf, rank)

                for ch in match.token_text:
                    _update_predictor_for_char(predictor, ch)

                llm.feed_chars(match.token_text)
                if on_progress:
                    on_progress(min(pos + match.chars_consumed, len(text)), len(text))
                pos += match.chars_consumed
            else:
                fallback_end = min(pos + MAX_FALLBACK_LEN, len(text))

                _encode_tag(encoder, TAG_CHARS)
                n_chars = fallback_end - pos
                count_cdf = uniform_cdf(MAX_FALLBACK_LEN)
                encoder.encode(count_cdf, n_chars - 1)

                for j in range(pos, fallback_end):
                    ch = text[j]
                    if predictor.has_char(ch):
                        sid = predictor.char_to_id(ch)
                        probs = predictor.predict([])
                        cdf = probs_to_cdf(probs, predictor.vocab_size())
                        encoder.encode(cdf, sid)
                        predictor.update(sid)
                    else:
                        probs = predictor.predict([])
                        cdf = probs_to_cdf(probs, predictor.vocab_size())
                        encoder.encode(cdf, ESCAPE_ID)
                        predictor.update(ESCAPE_ID)
                        _encode_codepoint(encoder, ord(ch))
                        predictor.add_char(ch)

                llm.feed_chars(text[pos:fallback_end])
                if on_progress:
                    on_progress(min(fallback_end, len(text)), len(text))
                pos = fallback_end
    finally:
        llm.cleanup()

    encoder.finish()
    return buf.getvalue(), llm.collected_predictions


# ------------------------------------------------------------------
# Shared encoding helpers
# ------------------------------------------------------------------

def _encode_codepoint(encoder: ArithmeticEncoder, cp: int) -> None:
    """Encode a Unicode codepoint using range-based coding."""
    if ASCII_START <= cp <= ASCII_END:
        range_id = RANGE_ASCII
        offset = cp - ASCII_START
        range_size = ASCII_END - ASCII_START + 1
    elif CJK_START <= cp <= CJK_END:
        range_id = RANGE_CJK
        offset = cp - CJK_START
        range_size = CJK_END - CJK_START + 1
    else:
        range_id = RANGE_OTHER
        offset = cp
        range_size = 0x110000

    range_cdf = uniform_cdf(NUM_RANGES)
    encoder.encode(range_cdf, range_id)

    offset_cdf = uniform_cdf(range_size)
    encoder.encode(offset_cdf, offset)


def _encode_tag(encoder: ArithmeticEncoder, tag: int) -> None:
    """Encode a 1-bit tag (TOKEN=1 / CHARS=0) via arithmetic coding."""
    cdf = uniform_cdf(2)
    encoder.encode(cdf, tag)


def _update_predictor_for_char(predictor: AdaptivePredictor, ch: str) -> None:
    """Update adaptive predictor for a character (add if new)."""
    if predictor.has_char(ch):
        sid = predictor.char_to_id(ch)
        predictor.update(sid)
    else:
        predictor.update(ESCAPE_ID)
        predictor.add_char(ch)


# ------------------------------------------------------------------
# Phrase-aware offline encode
# ------------------------------------------------------------------

def encode_with_phrases(
    text: str,
    phrase_set: frozenset[str],
    on_progress=None,
    priors: dict[str, float] | None = None,
    max_order: int = 4,
) -> bytes:
    """Encode text using phrase-level + character-level adaptive coding.

    Phrases pre-added to the predictor vocabulary are encoded as single
    symbols.  At each position, a greedy longest-match is attempted.
    """
    from .predictor.phrases import greedy_phrase_match

    buf = io.BytesIO()
    out = BitOutputStream(buf)
    encoder = ArithmeticEncoder(out)
    predictor = AdaptivePredictor(priors=priors, max_order=max_order)

    # Pre-add all phrases to the predictor vocabulary.
    # Both encoder and decoder must do this in the same order.
    for phrase in sorted(phrase_set):
        predictor.add_phrase(phrase)

    pos = 0
    while pos < len(text):
        # Try to match a phrase at this position
        matched = greedy_phrase_match(text, pos, phrase_set)
        if matched is not None:
            # Encode phrase as a single symbol
            sid = predictor.char_to_id(matched)
            probs = predictor.predict([])
            cdf = probs_to_cdf(probs, predictor.vocab_size())
            encoder.encode(cdf, sid)
            predictor.update(sid)
            pos += len(matched)
        else:
            ch = text[pos]
            if predictor.has_char(ch):
                sid = predictor.char_to_id(ch)
                probs = predictor.predict([])
                cdf = probs_to_cdf(probs, predictor.vocab_size())
                encoder.encode(cdf, sid)
                predictor.update(sid)
            else:
                probs = predictor.predict([])
                cdf = probs_to_cdf(probs, predictor.vocab_size())
                encoder.encode(cdf, ESCAPE_ID)
                predictor.update(ESCAPE_ID)
                _encode_codepoint(encoder, ord(ch))
                predictor.add_char(ch)
            pos += 1

        if on_progress and (pos % 200 == 0 or pos == len(text)):
            on_progress(pos, len(text))

    encoder.finish()
    return buf.getvalue()
