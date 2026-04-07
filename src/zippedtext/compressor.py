"""Main compression/decompression orchestrator.

Uses an adaptive PPM predictor with escape-based dynamic vocabulary.
No vocab or model is stored in the compressed file — both encoder and
decoder reconstruct identical state from the bitstream.

New characters are encoded as ESCAPE + Unicode codepoint.
Known characters are encoded directly via the adaptive model.
"""

from __future__ import annotations

import io
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


def compress(
    text: str,
    mode: str = "offline",
    api_client=None,
    model_name: str = "deepseek-chat",
    on_progress=None,
) -> bytes:
    """Compress text to .ztxt format."""
    if not text:
        raise ValueError("empty text")

    text_bytes = text.encode("utf-8")
    crc = compute_crc32(text_bytes)

    compressed_body = _encode(text, on_progress)

    # No model_data needed — adaptive model reconstructs during decode
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
    header, _, compressed_body = read_file(data)

    text = _decode(compressed_body, header.token_count, on_progress)

    actual_crc = compute_crc32(text.encode("utf-8"))
    if actual_crc != header.crc32:
        raise ValueError(
            f"CRC32 mismatch: expected {header.crc32:#010x}, got {actual_crc:#010x}. "
            "File may be corrupted."
        )
    return text


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
