"""Main compression/decompression orchestrator.

Uses an adaptive PPM predictor with escape-based dynamic vocabulary.
No vocab or model is stored in the compressed file — both encoder and
decoder reconstruct identical state from the bitstream.

New characters are encoded as ESCAPE + Unicode codepoint.
Known characters are encoded directly via the adaptive model.

Online mode adds LLM predictions to boost compression:
  - char sub-mode: boosts PPM character probabilities with LLM predictions
  - token sub-mode: encodes matching tokens cheaply, falls back to char-level
"""

from __future__ import annotations

import io
import json
import math
import struct

from .arithmetic import ArithmeticDecoder, ArithmeticEncoder, probs_to_cdf
from .bitstream import BitInputStream, BitOutputStream
from .format import (
    Header,
    MODE_OFFLINE,
    MODE_ONLINE,
    MODEL_DEEPSEEK_CHAT,
    MODEL_DEEPSEEK_REASONER,
    compute_crc32,
    read_file,
    write_file,
)
from .predictor.adaptive import ESCAPE_ID, AdaptivePredictor

MODEL_IDS = {
    "deepseek-chat": MODEL_DEEPSEEK_CHAT,
    "deepseek-reasoner": MODEL_DEEPSEEK_REASONER,
}

# Unicode codepoint encoding constants
# We encode codepoints in 3 ranges for efficiency:
#   Range 0: ASCII printable (0x20-0x7E) — 95 symbols, ~7 bits
#   Range 1: CJK Unified (0x3000-0x9FFF) — 28672 symbols, ~15 bits
#   Range 2: Other (full 21-bit codepoint)
RANGE_ASCII = 0
RANGE_CJK = 1
RANGE_OTHER = 2
NUM_RANGES = 3

ASCII_START, ASCII_END = 0x20, 0x7E
CJK_START, CJK_END = 0x3000, 0x9FFF

# Online mode sub-modes (stored in model_data)
SUB_MODE_CHAR = 0x00
SUB_MODE_TOKEN = 0x01


def compress(
    text: str,
    mode: str = "offline",
    api_client=None,
    model_name: str = "deepseek-chat",
    sub_mode: str = "char",
    on_progress=None,
) -> bytes:
    """Compress text to .ztxt format."""
    if not text:
        raise ValueError("empty text")

    text_bytes = text.encode("utf-8")
    crc = compute_crc32(text_bytes)

    if mode == "online" and api_client is not None:
        sub_mode_byte = SUB_MODE_TOKEN if sub_mode == "token" else SUB_MODE_CHAR
        if sub_mode == "token":
            compressed_body = _encode_online_token(text, api_client, on_progress)
        else:
            compressed_body = _encode_online_char(text, api_client, on_progress)

        # Store sub-mode + model name in model_data
        model_name_actual = api_client.last_model_id or model_name
        model_data = _pack_online_model_data(sub_mode_byte, model_name_actual)

        header = Header(
            mode=MODE_ONLINE,
            model_id=MODEL_IDS.get(model_name, MODEL_DEEPSEEK_CHAT),
            token_count=len(text),
            original_bytes=len(text_bytes),
            crc32=crc,
            model_data_len=len(model_data),
        )
        return write_file(header, model_data, compressed_body)
    else:
        compressed_body = _encode(text, on_progress)
        header = Header(
            mode=MODE_OFFLINE,
            model_id=MODEL_IDS.get(model_name, MODEL_DEEPSEEK_CHAT),
            token_count=len(text),
            original_bytes=len(text_bytes),
            crc32=crc,
            model_data_len=0,
        )
        return write_file(header, b"", compressed_body)


def decompress(
    data: bytes,
    api_client=None,
    on_progress=None,
) -> str:
    """Decompress .ztxt data back to original text."""
    header, model_data, compressed_body = read_file(data)

    if header.mode == MODE_ONLINE:
        if api_client is None:
            raise ValueError(
                "Online mode file requires an API client for decompression. "
                "Provide --api-key to decompress."
            )
        sub_mode_byte, _ = _unpack_online_model_data(model_data)
        if sub_mode_byte == SUB_MODE_TOKEN:
            text = _decode_online_token(
                compressed_body, header.token_count, api_client, on_progress,
            )
        else:
            text = _decode_online_char(
                compressed_body, header.token_count, api_client, on_progress,
            )
    else:
        text = _decode(compressed_body, header.token_count, on_progress)

    actual_crc = compute_crc32(text.encode("utf-8"))
    if actual_crc != header.crc32:
        raise ValueError(
            f"CRC32 mismatch: expected {header.crc32:#010x}, got {actual_crc:#010x}. "
            "File may be corrupted."
        )
    return text


# -----------------------------------------------------------------------
# Offline encode/decode (unchanged)
# -----------------------------------------------------------------------

def _encode(text: str, on_progress=None) -> bytes:
    """Encode text using adaptive arithmetic coding with escape-based vocab."""
    buf = io.BytesIO()
    out = BitOutputStream(buf)
    encoder = ArithmeticEncoder(out)
    predictor = AdaptivePredictor()

    for i, ch in enumerate(text):
        if predictor.has_char(ch):
            # Known character — encode its symbol ID
            sid = predictor.char_to_id(ch)
            probs = predictor.predict([])
            cdf = probs_to_cdf(probs, predictor.vocab_size())
            encoder.encode(cdf, sid)
            predictor.update(sid)
        else:
            # New character — encode ESCAPE + codepoint
            probs = predictor.predict([])
            cdf = probs_to_cdf(probs, predictor.vocab_size())
            encoder.encode(cdf, ESCAPE_ID)
            predictor.update(ESCAPE_ID)

            # Encode the Unicode codepoint
            _encode_codepoint(encoder, ord(ch))

            # Add character to vocab
            sid = predictor.add_char(ch)

        if on_progress and (i % 200 == 0 or i == len(text) - 1):
            on_progress(i + 1, len(text))

    encoder.finish()
    return buf.getvalue()


def _decode(compressed: bytes, char_count: int, on_progress=None) -> str:
    """Decode text — mirrors the encoder exactly."""
    inp = BitInputStream(compressed)
    decoder = ArithmeticDecoder(inp)
    predictor = AdaptivePredictor()

    chars: list[str] = []
    for i in range(char_count):
        probs = predictor.predict([])
        cdf = probs_to_cdf(probs, predictor.vocab_size())
        sid = decoder.decode(cdf)

        if sid == ESCAPE_ID:
            # New character — decode codepoint
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


# -----------------------------------------------------------------------
# Online char-level encode/decode
# -----------------------------------------------------------------------

def _encode_online_char(text: str, api_client, on_progress=None) -> bytes:
    """Encode text using adaptive PPM boosted by LLM character predictions.

    Same structure as _encode(), but the PPM probability distribution is
    modified at each step: the character predicted by the LLM gets a
    multiplicative boost before CDF construction.

    Encoder and decoder make identical API calls at identical positions
    because both use the actual text as context.
    """
    from .predictor.llm import LlmCharPredictor

    buf = io.BytesIO()
    out = BitOutputStream(buf)
    encoder = ArithmeticEncoder(out)
    predictor = AdaptivePredictor()
    llm = LlmCharPredictor(api_client)

    for i, ch in enumerate(text):
        # Ensure LLM cache is fresh (deterministic call point)
        llm.ensure_cache()

        if predictor.has_char(ch):
            sid = predictor.char_to_id(ch)
            ppm_probs = predictor.predict([])
            # Boost the LLM-predicted character
            probs = llm.boost_distribution(ppm_probs, predictor._char_to_id)
            cdf = probs_to_cdf(probs, predictor.vocab_size())
            encoder.encode(cdf, sid)
            predictor.update(sid)
        else:
            # New character — same escape mechanism as offline
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
    return buf.getvalue()


def _decode_online_char(
    compressed: bytes, char_count: int, api_client, on_progress=None,
) -> str:
    """Decode text — mirrors _encode_online_char exactly."""
    from .predictor.llm import LlmCharPredictor

    inp = BitInputStream(compressed)
    decoder = ArithmeticDecoder(inp)
    predictor = AdaptivePredictor()
    llm = LlmCharPredictor(api_client)

    chars: list[str] = []
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

    return "".join(chars)


# -----------------------------------------------------------------------
# Online token-level encode/decode
# -----------------------------------------------------------------------

# Token-level online mode encodes a sequence of "segments".
# Each segment is prefixed by a 1-bit tag:
#   TAG_TOKEN (1) = this segment is a matched token, encoded via top-k CDF
#   TAG_CHARS (0) = this segment is N character-level encoded chars (fallback)
# After TAG_CHARS, we encode the char count (uniform over MAX_FALLBACK_LEN)
# then N characters using the offline PPM encoder.

TAG_TOKEN = 1
TAG_CHARS = 0
MAX_FALLBACK_LEN = 64  # max chars in a single fallback segment

# Special: the very first bit of the stream encodes the total number of
# token-matched chars so we can reconstruct char counts during decode.


def _encode_online_token(text: str, api_client, on_progress=None) -> bytes:
    """Token-level online encoding.

    Walk through the text, trying to match LLM-generated tokens.
    Matched tokens are encoded cheaply via their rank in the top-k CDF.
    Unmatched regions fall back to character-level PPM encoding.
    """
    from .predictor.llm import LlmTokenPredictor

    buf = io.BytesIO()
    out = BitOutputStream(buf)
    encoder = ArithmeticEncoder(out)
    predictor = AdaptivePredictor()
    llm = LlmTokenPredictor(api_client)

    pos = 0
    while pos < len(text):
        # Ensure we have a fresh API prediction
        if llm.needs_refresh():
            llm.refresh_cache()

        remaining = text[pos:]
        match = llm.try_match_next_token(remaining)

        if match is not None and match.matched:
            # Encode TAG_TOKEN
            _encode_tag(encoder, TAG_TOKEN)

            # Encode which token (rank in top-k CDF)
            probs, token_strings = llm.build_token_probs(match.top_alternatives)
            rank = -1
            for idx, ts in enumerate(token_strings):
                if ts == match.token_text:
                    rank = idx
                    break
            if rank == -1:
                # Actual token not in filtered alternatives → catch-all
                rank = len(token_strings) - 1

            cdf = probs_to_cdf(probs, len(probs))
            encoder.encode(cdf, rank)

            # Still update the adaptive predictor for each char
            for ch in match.token_text:
                _update_predictor_for_char(predictor, ch)

            llm.feed_chars(match.token_text)
            if on_progress:
                on_progress(min(pos + match.chars_consumed, len(text)), len(text))
            pos += match.chars_consumed
        else:
            # Fallback: encode a run of chars using PPM
            # Determine how many chars to encode before trying token match again
            fallback_end = min(pos + MAX_FALLBACK_LEN, len(text))

            # Encode TAG_CHARS + char count
            _encode_tag(encoder, TAG_CHARS)
            n_chars = fallback_end - pos
            count_cdf = _uniform_cdf(MAX_FALLBACK_LEN)
            encoder.encode(count_cdf, n_chars - 1)  # 0-indexed

            # Encode each char with PPM (same as offline)
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

    encoder.finish()
    return buf.getvalue()


def _decode_online_token(
    compressed: bytes, char_count: int, api_client, on_progress=None,
) -> str:
    """Decode token-level online encoding — mirrors _encode_online_token."""
    from .predictor.llm import LlmTokenPredictor

    inp = BitInputStream(compressed)
    decoder = ArithmeticDecoder(inp)
    predictor = AdaptivePredictor()
    llm = LlmTokenPredictor(api_client)

    chars: list[str] = []

    while len(chars) < char_count:
        if llm.needs_refresh():
            llm.refresh_cache()

        tag = _decode_tag(decoder)

        if tag == TAG_TOKEN:
            # Decode token match: get the token text
            remaining_for_match = ""  # decoder doesn't know actual text yet
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
                # catch-all — this shouldn't happen for matched tokens
                raise ValueError("Token decode: unexpected catch-all in matched token")

            for ch in token_text:
                _update_predictor_for_char(predictor, ch)
                chars.append(ch)

            llm.feed_chars(token_text)
        else:
            # Fallback: decode char count + chars
            count_cdf = _uniform_cdf(MAX_FALLBACK_LEN)
            n_chars = decoder.decode(count_cdf) + 1  # 1-indexed

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

    return "".join(chars[:char_count])


# -----------------------------------------------------------------------
# Shared helpers
# -----------------------------------------------------------------------

def _encode_codepoint(encoder: ArithmeticEncoder, cp: int) -> None:
    """Encode a Unicode codepoint using range-based coding.

    Step 1: Encode which range (ASCII / CJK / Other) — 3 symbols
    Step 2: Encode offset within that range — uniform distribution
    """
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
        range_size = 0x110000  # full Unicode range

    # Step 1: encode range selector (uniform over 3 choices)
    range_cdf = _uniform_cdf(NUM_RANGES)
    encoder.encode(range_cdf, range_id)

    # Step 2: encode offset within range (uniform)
    offset_cdf = _uniform_cdf(range_size)
    encoder.encode(offset_cdf, offset)


def _decode_codepoint(decoder: ArithmeticDecoder) -> int:
    """Decode a Unicode codepoint — mirrors _encode_codepoint."""
    # Step 1: decode range
    range_cdf = _uniform_cdf(NUM_RANGES)
    range_id = decoder.decode(range_cdf)

    # Step 2: decode offset
    if range_id == RANGE_ASCII:
        offset_cdf = _uniform_cdf(ASCII_END - ASCII_START + 1)
        offset = decoder.decode(offset_cdf)
        return ASCII_START + offset
    elif range_id == RANGE_CJK:
        offset_cdf = _uniform_cdf(CJK_END - CJK_START + 1)
        offset = decoder.decode(offset_cdf)
        return CJK_START + offset
    else:
        offset_cdf = _uniform_cdf(0x110000)
        return decoder.decode(offset_cdf)


def _encode_tag(encoder: ArithmeticEncoder, tag: int) -> None:
    """Encode a 1-bit tag (TOKEN=1 / CHARS=0) via arithmetic coding."""
    cdf = _uniform_cdf(2)
    encoder.encode(cdf, tag)


def _decode_tag(decoder: ArithmeticDecoder) -> int:
    """Decode a 1-bit tag."""
    cdf = _uniform_cdf(2)
    return decoder.decode(cdf)


def _update_predictor_for_char(predictor: AdaptivePredictor, ch: str) -> None:
    """Update adaptive predictor for a character (add if new)."""
    if predictor.has_char(ch):
        sid = predictor.char_to_id(ch)
        predictor.update(sid)
    else:
        probs = predictor.predict([])
        # We don't encode ESCAPE here (token-level handled it), but
        # we still need to update the predictor so it stays in sync
        predictor.update(ESCAPE_ID)
        predictor.add_char(ch)


def _peek_generated_text(llm) -> str:
    """Get the remaining generated text from the LLM token predictor's cache."""
    if llm._chunk is None:
        return ""
    idx = llm._token_idx
    if idx >= len(llm._chunk.tokens):
        return ""
    return "".join(t.text for t in llm._chunk.tokens[idx:])


def _pack_online_model_data(sub_mode: int, model_name: str) -> bytes:
    """Pack sub-mode byte + model name into model_data section."""
    name_bytes = model_name.encode("utf-8")
    return struct.pack("<B", sub_mode) + name_bytes + b"\x00"


def _unpack_online_model_data(data: bytes) -> tuple[int, str]:
    """Unpack sub-mode byte + model name from model_data section."""
    if not data:
        return SUB_MODE_CHAR, ""
    sub_mode = data[0]
    name_bytes = data[1:]
    if b"\x00" in name_bytes:
        name_bytes = name_bytes[:name_bytes.index(b"\x00")]
    return sub_mode, name_bytes.decode("utf-8")


def _uniform_cdf(n: int) -> list[int]:
    """Build a uniform CDF for n symbols. Cached for common sizes."""
    return _UNIFORM_CDF_CACHE.get(n) or _build_uniform_cdf(n)


_UNIFORM_CDF_CACHE: dict[int, list[int]] = {}


def _build_uniform_cdf(n: int) -> list[int]:
    from .arithmetic import TOTAL
    step = TOTAL // n
    cdf = [i * step for i in range(n)]
    cdf.append(TOTAL)
    # Fix rounding: last interval absorbs remainder
    _UNIFORM_CDF_CACHE[n] = cdf
    return cdf
