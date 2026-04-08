"""Decoding routines for all decompression modes.

Extracted from compressor.py — contains the decoding half of every
decompress pipeline (offline, online-char, online-token).
"""

from __future__ import annotations

from .arithmetic import ArithmeticDecoder, probs_to_cdf
from .bitstream import BitInputStream
from .cdf_utils import uniform_cdf
from .predictor.adaptive import ESCAPE_ID, AdaptivePredictor

# Unicode codepoint range constants (must match encoder.py)
RANGE_ASCII = 0
RANGE_CJK = 1
RANGE_OTHER = 2
NUM_RANGES = 3

ASCII_START, ASCII_END = 0x20, 0x7E
CJK_START, CJK_END = 0x3000, 0x9FFF

# Token-level online mode tags (must match encoder.py)
TAG_TOKEN = 1
TAG_CHARS = 0
MAX_FALLBACK_LEN = 64


# ------------------------------------------------------------------
# Offline decode
# ------------------------------------------------------------------

def decode(
    compressed: bytes,
    char_count: int,
    on_progress=None,
    priors: dict[str, float] | None = None,
    max_order: int = 4,
) -> str:
    """Decode text — mirrors the encoder exactly."""
    inp = BitInputStream(compressed)
    decoder = ArithmeticDecoder(inp)
    predictor = AdaptivePredictor(priors=priors, max_order=max_order)

    chars: list[str] = []
    for i in range(char_count):
        probs = predictor.predict([])
        cdf = probs_to_cdf(probs, predictor.vocab_size())
        sid = decoder.decode(cdf)

        if sid == ESCAPE_ID:
            predictor.update(ESCAPE_ID)
            cp = _decode_codepoint(decoder)
            ch = chr(cp)
            predictor.add_char(ch)
        else:
            ch = predictor.id_to_char(sid)
            predictor.update(sid)

        chars.append(ch)

        if on_progress and (i % 200 == 0 or i == char_count - 1):
            on_progress(i + 1, char_count)

    return "".join(chars)


# ------------------------------------------------------------------
# Online char-level decode
# ------------------------------------------------------------------

def decode_online_char(
    compressed: bytes,
    char_count: int,
    api_client,
    on_progress=None,
    priors: dict[str, float] | None = None,
    max_order: int = 4,
    chunk_chars: int | None = None,
    max_tokens: int | None = None,
    prediction_cache: list[str] | None = None,
) -> str:
    """Decode text — mirrors encode_online_char exactly.

    v0.3.1: uses prediction_cache for instant API-free decompression.
    Falls back to API calls if no cache is embedded.
    """
    from .predictor.llm import LlmCharPredictor

    inp = BitInputStream(compressed)
    decoder = ArithmeticDecoder(inp)
    predictor = AdaptivePredictor(priors=priors, max_order=max_order)
    llm = LlmCharPredictor(
        api_client=api_client,
        chunk_chars=chunk_chars,
        max_tokens=max_tokens,
        prediction_cache=prediction_cache,
    )

    chars: list[str] = []
    try:
        for i in range(char_count):
            llm.ensure_cache()

            ppm_probs = predictor.predict([])
            probs = llm.boost_distribution(ppm_probs, predictor._char_to_id)
            cdf = probs_to_cdf(probs, predictor.vocab_size())
            sid = decoder.decode(cdf)

            if sid == ESCAPE_ID:
                predictor.update(ESCAPE_ID)
                cp = _decode_codepoint(decoder)
                ch = chr(cp)
                predictor.add_char(ch)
            else:
                ch = predictor.id_to_char(sid)
                predictor.update(sid)

            chars.append(ch)
            llm.feed_char(ch)

            if on_progress and (i % 200 == 0 or i == char_count - 1):
                on_progress(i + 1, char_count)
    finally:
        llm.cleanup()

    return "".join(chars)


# ------------------------------------------------------------------
# Online token-level decode
# ------------------------------------------------------------------

def decode_online_token(
    compressed: bytes,
    char_count: int,
    api_client,
    on_progress=None,
    priors: dict[str, float] | None = None,
    max_order: int = 4,
    chunk_chars: int | None = None,
    max_tokens: int | None = None,
    token_cache: list | None = None,
) -> str:
    """Decode token-level online encoding — mirrors encode_online_token.

    v0.3.1: uses token_cache for instant API-free decompression.
    """
    from .predictor.llm import LlmTokenPredictor

    inp = BitInputStream(compressed)
    decoder = ArithmeticDecoder(inp)
    predictor = AdaptivePredictor(priors=priors, max_order=max_order)
    llm = LlmTokenPredictor(
        api_client=api_client,
        max_tokens=max_tokens,
        prediction_cache=token_cache,
    )

    chars: list[str] = []

    try:
        while len(chars) < char_count:
            if llm.needs_refresh():
                llm.refresh_cache()

            tag = _decode_tag(decoder)

            if tag == TAG_TOKEN:
                match = llm.try_match_next_token(
                    _peek_generated_text(llm)
                )

                if match is None:
                    raise ValueError("Token decode desync: expected match but got None")

                probs, token_strings = llm.build_token_probs(match.top_alternatives)
                cdf = probs_to_cdf(probs, len(probs))
                rank = decoder.decode(cdf)

                if rank < len(token_strings) - 1:
                    token_text = token_strings[rank]
                else:
                    raise ValueError("Token decode: unexpected catch-all in matched token")

                for ch in token_text:
                    _update_predictor_for_char(predictor, ch)
                    chars.append(ch)

                llm.feed_chars(token_text)
            else:
                count_cdf = uniform_cdf(MAX_FALLBACK_LEN)
                n_chars = decoder.decode(count_cdf) + 1

                for _ in range(n_chars):
                    if len(chars) >= char_count:
                        break
                    probs = predictor.predict([])
                    cdf = probs_to_cdf(probs, predictor.vocab_size())
                    sid = decoder.decode(cdf)

                    if sid == ESCAPE_ID:
                        predictor.update(ESCAPE_ID)
                        cp = _decode_codepoint(decoder)
                        ch = chr(cp)
                        predictor.add_char(ch)
                    else:
                        ch = predictor.id_to_char(sid)
                        predictor.update(sid)
                    chars.append(ch)

                llm.feed_chars("".join(chars[-(n_chars):]))

            if on_progress and (len(chars) % 200 < 10 or len(chars) >= char_count):
                on_progress(min(len(chars), char_count), char_count)
    finally:
        llm.cleanup()

    return "".join(chars[:char_count])


# ------------------------------------------------------------------
# Shared decoding helpers
# ------------------------------------------------------------------

def _decode_codepoint(decoder: ArithmeticDecoder) -> int:
    """Decode a Unicode codepoint — mirrors _encode_codepoint."""
    range_cdf = uniform_cdf(NUM_RANGES)
    range_id = decoder.decode(range_cdf)

    if range_id == RANGE_ASCII:
        offset_cdf = uniform_cdf(ASCII_END - ASCII_START + 1)
        offset = decoder.decode(offset_cdf)
        return ASCII_START + offset
    elif range_id == RANGE_CJK:
        offset_cdf = uniform_cdf(CJK_END - CJK_START + 1)
        offset = decoder.decode(offset_cdf)
        return CJK_START + offset
    else:
        offset_cdf = uniform_cdf(0x110000)
        return decoder.decode(offset_cdf)


def _decode_tag(decoder: ArithmeticDecoder) -> int:
    """Decode a 1-bit tag."""
    cdf = uniform_cdf(2)
    return decoder.decode(cdf)


def _update_predictor_for_char(predictor: AdaptivePredictor, ch: str) -> None:
    """Update adaptive predictor for a character (add if new)."""
    if predictor.has_char(ch):
        sid = predictor.char_to_id(ch)
        predictor.update(sid)
    else:
        predictor.update(ESCAPE_ID)
        predictor.add_char(ch)


# ------------------------------------------------------------------
# Phrase-aware offline decode
# ------------------------------------------------------------------

def decode_with_phrases(
    compressed: bytes,
    char_count: int,
    phrase_set: frozenset[str],
    on_progress=None,
    priors: dict[str, float] | None = None,
    max_order: int = 4,
) -> str:
    """Decode text with phrase-level symbols — mirrors encode_with_phrases."""
    inp = BitInputStream(compressed)
    decoder = ArithmeticDecoder(inp)
    predictor = AdaptivePredictor(priors=priors, max_order=max_order)

    # Pre-add phrases in same order as encoder
    for phrase in sorted(phrase_set):
        predictor.add_phrase(phrase)

    chars: list[str] = []
    while len(chars) < char_count:
        probs = predictor.predict([])
        cdf = probs_to_cdf(probs, predictor.vocab_size())
        sid = decoder.decode(cdf)

        if sid == ESCAPE_ID:
            predictor.update(ESCAPE_ID)
            cp = _decode_codepoint(decoder)
            ch = chr(cp)
            predictor.add_char(ch)
            chars.append(ch)
        else:
            symbol = predictor.id_to_char(sid)
            predictor.update(sid)
            chars.extend(symbol)  # multi-char phrase or single char

        if on_progress and (len(chars) % 200 < 10 or len(chars) >= char_count):
            on_progress(min(len(chars), char_count), char_count)

    return "".join(chars[:char_count])


def _peek_generated_text(llm) -> str:
    """Get the remaining generated text from the LLM token predictor's cache."""
    if llm._chunk is None:
        return ""
    idx = llm._token_idx
    if idx >= len(llm._chunk.tokens):
        return ""
    return "".join(t.text for t in llm._chunk.tokens[idx:])
