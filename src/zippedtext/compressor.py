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
from .predictor.llm import CHUNK_CHARS, MAX_TOKENS_DEFAULT

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


def _offline_compress(
    text: str,
    text_bytes: bytes,
    crc: int,
    priors: dict[str, float],
    flags: int,
    max_order: int,
    phrase_set: frozenset[str],
    phrase_table_bytes: bytes,
    on_progress=None,
) -> bytes:
    """Build an offline .ztxt (used as fallback when online+cache is bigger)."""
    enc_kwargs: dict = {"priors": priors or None, "max_order": max_order}
    if phrase_set:
        compressed_body = encode_with_phrases(
            text, phrase_set, on_progress, **enc_kwargs,
        )
        header = Header(
            mode=MODE_OFFLINE, model_id=0,
            token_count=len(text), original_bytes=len(text_bytes),
            crc32=crc, model_data_len=0, flags=flags,
            max_order=max_order, phrase_table_len=len(phrase_table_bytes),
        )
        return write_file(header, b"", compressed_body, phrase_table=phrase_table_bytes)
    else:
        compressed_body = encode(text, on_progress, **enc_kwargs)
        header = Header(
            mode=MODE_OFFLINE, model_id=0,
            token_count=len(text), original_bytes=len(text_bytes),
            crc32=crc, model_data_len=0, flags=flags,
            max_order=max_order,
        )
        return write_file(header, b"", compressed_body)


def compress(
    text: str,
    mode: str = "offline",
    api_client=None,
    model_name: str = "deepseek-chat",
    sub_mode: str = "char",
    on_progress=None,
    use_priors: bool = True,
    max_order: int = DEFAULT_MAX_ORDER,
    use_phrases: bool = False,
) -> bytes:
    """Compress text to .ztxt format."""
    if not text:
        raise ValueError("empty text")

    text_bytes = text.encode("utf-8")
    crc = compute_crc32(text_bytes)

    priors = _get_priors() if use_priors else {}
    flags = FLAG_HAS_PRIORS if priors else 0

    enc_kwargs: dict = {"priors": priors or None, "max_order": max_order}

    # Build phrase table (offline mode only, text must be long enough).
    # Only use phrases if the estimated savings exceed the table overhead.
    phrase_table_bytes = b""
    phrase_set: frozenset[str] = frozenset()
    if use_phrases and mode == "offline" and len(text) >= 200:
        from .predictor.phrases import PhraseTable, build_phrase_table
        pt = build_phrase_table(text, min_freq=4)
        if pt.phrases:
            table_bytes = pt.serialize()
            # Estimate savings: each phrase match saves roughly
            # (phrase_len - 1) * avg_bits_per_char / 8 bytes.
            # Only include phrases if table cost < estimated savings.
            est_savings = sum(
                (len(p) - 1) * 1.0  # ~1 byte per extra char saved
                for p in pt.phrases
                for i in range(len(text) - len(p) + 1)
                if text[i:i + len(p)] == p
            )
            if est_savings > len(table_bytes) * 2:
                phrase_set = frozenset(pt.phrases)
                phrase_table_bytes = table_bytes
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
        chunk_chars = CHUNK_CHARS
        max_tokens = MAX_TOKENS_DEFAULT
        if sub_mode == "token":
            compressed_body, pred_cache = encode_online_token(
                text, api_client, on_progress, **enc_kwargs,
                chunk_chars=chunk_chars, max_tokens=max_tokens,
            )
        else:
            compressed_body, pred_cache = encode_online_char(
                text, api_client, on_progress, **enc_kwargs,
                chunk_chars=chunk_chars, max_tokens=max_tokens,
            )

        model_name_actual = api_client.last_model_id or model_name
        # Truncate char-mode predictions to chunk_chars to save space
        if sub_mode_byte == SUB_MODE_CHAR:
            truncated = [p[:chunk_chars] for p in pred_cache]
        else:
            truncated = None
        model_data = _pack_online_model_data(
            sub_mode_byte, model_name_actual, chunk_chars, max_tokens,
            prediction_cache=truncated if sub_mode_byte == SUB_MODE_CHAR else None,
            token_cache=pred_cache if sub_mode_byte == SUB_MODE_TOKEN else None,
        )

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
        online_result = write_file(header, model_data, compressed_body)

        # Smart fallback: if online+cache is larger than offline, use offline
        offline_result = _offline_compress(
            text, text_bytes, crc, priors, flags, max_order, phrase_set,
            phrase_table_bytes, on_progress=None,
        )
        if len(online_result) <= len(offline_result):
            return online_result
        return offline_result
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
        modified_text = decode(
            compressed_body, header.token_count, on_progress, **dec_kwargs,
        )
        text = restore_codegen(modified_text, manifest)
    elif header.mode == MODE_ONLINE:
        sub_mode_byte, _, chunk_chars, max_tokens, pred_cache, tok_cache = (
            _unpack_online_model_data(model_data)
        )
        has_cache = pred_cache is not None or tok_cache is not None
        if api_client is None and not has_cache:
            raise ValueError(
                "Online mode file requires an API client for decompression. "
                "Provide --api-key to decompress, or use v0.3.1+ files "
                "which embed prediction cache."
            )
        if sub_mode_byte == SUB_MODE_TOKEN:
            text = decode_online_token(
                compressed_body, header.token_count, api_client,
                on_progress, **dec_kwargs,
                chunk_chars=chunk_chars, max_tokens=max_tokens,
                token_cache=tok_cache,
            )
        else:
            text = decode_online_char(
                compressed_body, header.token_count, api_client,
                on_progress, **dec_kwargs,
                chunk_chars=chunk_chars, max_tokens=max_tokens,
                prediction_cache=pred_cache,
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

def _pack_online_model_data(
    sub_mode: int,
    model_name: str,
    chunk_chars: int = CHUNK_CHARS,
    max_tokens: int = MAX_TOKENS_DEFAULT,
    prediction_cache: list[str] | None = None,
    token_cache: list | None = None,
) -> bytes:
    """Pack sub-mode + model name + params + prediction cache into model_data.

    v0.3.1 embeds the LLM prediction cache so decompression is API-free.
    The cache is zstd-compressed after the chunk params.  Old decoders
    that stop at the null byte safely ignore the extra bytes.
    """
    import zstandard

    name_bytes = model_name.encode("utf-8")
    base = (
        struct.pack("<B", sub_mode)
        + name_bytes
        + b"\x00"
        + struct.pack("<HH", chunk_chars, max_tokens)
    )

    # Char-mode prediction cache: list of generated-text strings
    if prediction_cache is not None:
        parts = [struct.pack("<H", len(prediction_cache))]
        for pred_text in prediction_cache:
            encoded = pred_text.encode("utf-8")
            parts.append(struct.pack("<I", len(encoded)))
            parts.append(encoded)
        raw = b"".join(parts)
        compressed_cache = zstandard.ZstdCompressor(level=19).compress(raw)
        return base + struct.pack("<BI", 0x01, len(compressed_cache)) + compressed_cache

    # Token-mode cache: serialize ChunkResult objects
    if token_cache is not None:
        import json
        entries = []
        for chunk_result in token_cache:
            entry = {
                "text": chunk_result.generated_text,
                "tokens": [
                    {
                        "text": t.text,
                        "logprob": t.logprob,
                        "top_alternatives": t.top_alternatives,
                        "char_offset": t.char_offset,
                    }
                    for t in chunk_result.tokens
                ],
            }
            entries.append(entry)
        raw = json.dumps(entries, ensure_ascii=False).encode("utf-8")
        compressed_cache = zstandard.ZstdCompressor(level=19).compress(raw)
        return base + struct.pack("<BI", 0x02, len(compressed_cache)) + compressed_cache

    # No cache
    return base


def _unpack_online_model_data(
    data: bytes,
) -> tuple[int, str, int, int, list[str] | None, list | None]:
    """Unpack sub-mode, model name, chunk_chars, max_tokens, prediction caches.

    Returns ``(sub_mode, model_name, chunk_chars, max_tokens, char_cache, token_cache)``.
    """
    import zstandard

    if not data:
        return SUB_MODE_CHAR, "", 20, 200, None, None
    sub_mode = data[0]
    rest = data[1:]
    null_idx = rest.index(b"\x00") if b"\x00" in rest else len(rest)
    model_name = rest[:null_idx].decode("utf-8")

    extra = rest[null_idx + 1:] if null_idx < len(rest) else b""
    if len(extra) >= 4:
        chunk_chars, max_tokens = struct.unpack("<HH", extra[:4])
        extra = extra[4:]
    else:
        return sub_mode, model_name, 20, 200, None, None

    # Check for prediction cache
    char_cache = None
    token_cache = None
    if len(extra) >= 5:
        cache_type = extra[0]
        cache_len = struct.unpack("<I", extra[1:5])[0]
        cache_data = extra[5:5 + cache_len]

        if cache_type == 0x01 and cache_data:
            # Char-mode cache
            raw = zstandard.ZstdDecompressor().decompress(cache_data)
            offset = 0
            n = struct.unpack("<H", raw[offset:offset + 2])[0]
            offset += 2
            char_cache = []
            for _ in range(n):
                text_len = struct.unpack("<I", raw[offset:offset + 4])[0]
                offset += 4
                char_cache.append(raw[offset:offset + text_len].decode("utf-8"))
                offset += text_len

        elif cache_type == 0x02 and cache_data:
            # Token-mode cache
            import json
            from .api_client import ChunkResult, GeneratedToken
            raw = zstandard.ZstdDecompressor().decompress(cache_data)
            entries = json.loads(raw.decode("utf-8"))
            token_cache = []
            for entry in entries:
                tokens = []
                char_offset = 0
                for t in entry["tokens"]:
                    tokens.append(GeneratedToken(
                        text=t["text"],
                        logprob=t["logprob"],
                        top_alternatives=[tuple(a) for a in t["top_alternatives"]],
                        char_offset=t["char_offset"],
                    ))
                token_cache.append(ChunkResult(
                    generated_text=entry["text"],
                    tokens=tokens,
                    model="",
                ))

    return sub_mode, model_name, chunk_chars, max_tokens, char_cache, token_cache
