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

import struct

from .decoder import (
    decode,
    decode_online_char,
    decode_online_token,
    decode_with_phrases,
)
from .encoder import (
    encode,
    encode_online_char,
    encode_online_token,
    encode_with_phrases,
)
from .format import (
    DEFAULT_MAX_ORDER,
    FLAG_HAS_PRIORS,
    FLAG_PHRASE_ENCODING,
    Header,
    MODE_CODEGEN,
    MODE_OFFLINE,
    MODE_ONLINE,
    MODEL_DEEPSEEK_CHAT,
    MODEL_DEEPSEEK_REASONER,
    compute_crc32,
    read_file,
    read_file_v2,
    write_file,
)

# Online mode sub-modes (stored in model_data)
SUB_MODE_CHAR = 0x00
SUB_MODE_TOKEN = 0x01


def _get_priors() -> dict[str, float]:
    """Load Chinese character frequency priors if available."""
    try:
        from .predictor.priors import get_chinese_priors
        return get_chinese_priors()
    except ImportError:
        return {}


def compress(
    text: str,
    mode: str = "offline",
    api_client=None,
    model_name: str = "deepseek-chat",
    sub_mode: str = "char",
    on_progress=None,
    use_priors: bool = True,
    max_order: int = DEFAULT_MAX_ORDER,
    use_phrases: bool = True,
) -> bytes:
    """Compress text to .ztxt format."""
    if not text:
        raise ValueError("empty text")

    text_bytes = text.encode("utf-8")
    crc = compute_crc32(text_bytes)

    priors = _get_priors() if use_priors else {}
    flags = FLAG_HAS_PRIORS if priors else 0

    enc_kwargs: dict = {"priors": priors or None, "max_order": max_order}

    # Build phrase table (offline mode only, text must be long enough)
    phrase_table_bytes = b""
    phrase_set: frozenset[str] = frozenset()
    if use_phrases and mode == "offline" and len(text) >= 50:
        from .predictor.phrases import PhraseTable, build_phrase_table
        pt = build_phrase_table(text)
        if pt.phrases:
            phrase_set = frozenset(pt.phrases)
            phrase_table_bytes = pt.serialize()
            flags |= FLAG_PHRASE_ENCODING

    if mode == "codegen" and api_client is not None:
        # Code generation mode: LLM identifies code-representable segments
        from .codegen import (
            CodegenManifest,
            analyze_for_codegen,
            apply_codegen,
            restore_codegen,
        )
        manifest = analyze_for_codegen(text, api_client)
        if manifest.segments:
            modified_text = apply_codegen(text, manifest)
            manifest_bytes = manifest.serialize()
        else:
            modified_text = text
            manifest_bytes = CodegenManifest(segments=()).serialize()

        compressed_body = encode(modified_text, on_progress, **enc_kwargs)

        model_name_actual = api_client.last_model_id or model_name
        model_data = manifest_bytes + _pack_online_model_data(0x00, model_name_actual)

        header = Header(
            mode=MODE_CODEGEN,
            model_id=0,
            token_count=len(modified_text),
            original_bytes=len(text_bytes),
            crc32=crc,
            model_data_len=len(model_data),
            flags=flags,
            max_order=max_order,
        )
        return write_file(header, model_data, compressed_body)
    elif mode == "online" and api_client is not None:
        sub_mode_byte = SUB_MODE_TOKEN if sub_mode == "token" else SUB_MODE_CHAR
        if sub_mode == "token":
            compressed_body = encode_online_token(
                text, api_client, on_progress, **enc_kwargs,
            )
        else:
            compressed_body = encode_online_char(
                text, api_client, on_progress, **enc_kwargs,
            )

        model_name_actual = api_client.last_model_id or model_name
        model_data = _pack_online_model_data(sub_mode_byte, model_name_actual)

        header = Header(
            mode=MODE_ONLINE,
            model_id=0,
            token_count=len(text),
            original_bytes=len(text_bytes),
            crc32=crc,
            model_data_len=len(model_data),
            flags=flags,
            max_order=max_order,
        )
        return write_file(header, model_data, compressed_body)
    elif phrase_set:
        # Offline with phrases
        compressed_body = encode_with_phrases(
            text, phrase_set, on_progress, **enc_kwargs,
        )
        header = Header(
            mode=MODE_OFFLINE,
            model_id=0,
            token_count=len(text),
            original_bytes=len(text_bytes),
            crc32=crc,
            model_data_len=0,
            flags=flags,
            max_order=max_order,
            phrase_table_len=len(phrase_table_bytes),
        )
        return write_file(header, b"", compressed_body, phrase_table=phrase_table_bytes)
    else:
        # Offline without phrases
        compressed_body = encode(text, on_progress, **enc_kwargs)
        header = Header(
            mode=MODE_OFFLINE,
            model_id=0,
            token_count=len(text),
            original_bytes=len(text_bytes),
            crc32=crc,
            model_data_len=0,
            flags=flags,
            max_order=max_order,
        )
        return write_file(header, b"", compressed_body)


def decompress(
    data: bytes,
    api_client=None,
    on_progress=None,
) -> str:
    """Decompress .ztxt data back to original text."""
    # Reconstruct priors from flags
    header, model_data, compressed_body = read_file(data)

    priors: dict[str, float] | None = None
    if header.flags & FLAG_HAS_PRIORS:
        priors = _get_priors() or None

    dec_kwargs: dict = {"priors": priors, "max_order": header.max_order}

    if header.mode == MODE_CODEGEN:
        # Code generation mode: decode body, then restore codegen segments
        from .codegen import CodegenManifest, restore_codegen
        manifest = CodegenManifest.deserialize(model_data)
        # model_data contains manifest + model name; manifest.deserialize
        # reads its own length prefix, remaining bytes are model info
        modified_text = decode(
            compressed_body, header.token_count, on_progress, **dec_kwargs,
        )
        text = restore_codegen(modified_text, manifest)
    elif header.mode == MODE_ONLINE:
        if api_client is None:
            raise ValueError(
                "Online mode file requires an API client for decompression. "
                "Provide --api-key to decompress."
            )
        sub_mode_byte, _ = _unpack_online_model_data(model_data)
        if sub_mode_byte == SUB_MODE_TOKEN:
            text = decode_online_token(
                compressed_body, header.token_count, api_client,
                on_progress, **dec_kwargs,
            )
        else:
            text = decode_online_char(
                compressed_body, header.token_count, api_client,
                on_progress, **dec_kwargs,
            )
    elif header.flags & FLAG_PHRASE_ENCODING:
        # Offline with phrases — need to read phrase table from v2 format
        header_v2, model_data_v2, phrase_table_bytes, body = read_file_v2(data)
        from .predictor.phrases import PhraseTable
        pt = PhraseTable.deserialize(phrase_table_bytes)
        phrase_set = frozenset(pt.phrases)
        text = decode_with_phrases(
            body, header.token_count, phrase_set, on_progress, **dec_kwargs,
        )
    else:
        text = decode(
            compressed_body, header.token_count, on_progress, **dec_kwargs,
        )

    actual_crc = compute_crc32(text.encode("utf-8"))
    if actual_crc != header.crc32:
        raise ValueError(
            f"CRC32 mismatch: expected {header.crc32:#010x}, got {actual_crc:#010x}. "
            "File may be corrupted."
        )
    return text


# ------------------------------------------------------------------
# model_data packing helpers
# ------------------------------------------------------------------

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
