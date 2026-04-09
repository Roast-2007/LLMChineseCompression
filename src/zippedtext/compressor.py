"""Main compression/decompression orchestrator.

Uses an adaptive PPM predictor with escape-based dynamic vocabulary.
No vocab or model is stored in the compressed file — both encoder and
decoder reconstruct identical state from the bitstream.

New characters are encoded as ESCAPE + Unicode codepoint.
Known characters are encoded directly via the adaptive model.

Online mode now has two families:
  - structured (default): LLM-assisted analysis + routing + compact side info
  - legacy char/token: the original prediction-cache based online modes
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
    SECTION_ANALYSIS,
    SECTION_PHRASE_TABLE,
    SECTION_SEGMENTS,
    SECTION_STATS,
    SECTION_TEMPLATES,
    VERSION_V3,
    compute_crc32,
    read_file,
    read_file_v2,
    read_file_v3,
    write_file,
    write_file_v3,
)
from .online_manifest import (
    AnalysisManifest,
    ROUTE_PHRASE,
    ROUTE_TEMPLATE,
    SegmentRecord,
    StructuredOnlineStats,
    deserialize_segment_records,
    serialize_segment_records,
)
from .predictor.llm import CHUNK_CHARS, MAX_TOKENS_DEFAULT
from .router import route_segments
from .segment import split_text_segments
from .sideinfo_codec import make_section, section_stored_size
from .template_codec import TemplateCatalog, build_template_catalog, decode_template_segment
from .term_dictionary import build_structured_phrase_table

# Online mode sub-modes (stored in model_data)
SUB_MODE_STRUCTURED = 0xFF
SUB_MODE_CHAR = 0x00
SUB_MODE_TOKEN = 0x01


def _get_priors() -> dict[str, float]:
    """Load Chinese character frequency priors if available."""
    try:
        from .predictor.priors import get_chinese_priors
        return get_chinese_priors()
    except ImportError:
        return {}


def _merge_priors(
    base_priors: dict[str, float] | None,
    manifest_priors: dict[str, float] | None,
    manifest_weight: float = 0.35,
) -> dict[str, float] | None:
    if not base_priors and not manifest_priors:
        return None
    if not manifest_priors:
        return dict(base_priors) if base_priors else None
    if not base_priors:
        return dict(manifest_priors)

    merged = {
        ch: (base_priors.get(ch, 0.0) * (1.0 - manifest_weight))
        + (manifest_priors.get(ch, 0.0) * manifest_weight)
        for ch in set(base_priors) | set(manifest_priors)
    }
    total = sum(merged.values())
    if total <= 0:
        return dict(base_priors)
    return {
        ch: value / total
        for ch, value in merged.items()
        if value > 0
    }


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
    """Build an offline .ztxt (used as fallback when online is bigger)."""
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
    sub_mode: str = "structured",
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

    phrase_table_bytes = b""
    phrase_set: frozenset[str] = frozenset()
    if use_phrases and mode == "offline" and len(text) >= 200:
        from .predictor.phrases import build_phrase_table
        pt = build_phrase_table(text, min_freq=4)
        if pt.phrases:
            table_bytes = pt.serialize()
            est_savings = sum(
                (len(p) - 1) * 1.0
                for p in pt.phrases
                for i in range(len(text) - len(p) + 1)
                if text[i:i + len(p)] == p
            )
            if est_savings > len(table_bytes) * 2:
                phrase_set = frozenset(pt.phrases)
                phrase_table_bytes = table_bytes
                flags |= FLAG_PHRASE_ENCODING

    offline_result = _offline_compress(
        text,
        text_bytes,
        crc,
        priors,
        flags,
        max_order,
        phrase_set,
        phrase_table_bytes,
        on_progress=None,
    )

    if mode == "codegen" and api_client is not None:
        from .codegen import CodegenManifest, analyze_for_codegen, apply_codegen

        manifest = analyze_for_codegen(text, api_client)
        if manifest.segments:
            modified_text = apply_codegen(text, manifest)
            manifest_bytes = manifest.serialize()
        else:
            modified_text = text
            manifest_bytes = CodegenManifest(segments=()).serialize()

        compressed_body = encode(modified_text, on_progress, priors=priors or None, max_order=max_order)
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

    if mode == "online" and api_client is not None and sub_mode == "structured":
        structured_result = _structured_online_compress(
            text=text,
            text_bytes=text_bytes,
            crc=crc,
            api_client=api_client,
            model_name=model_name,
            priors=priors,
            flags=flags,
            max_order=max_order,
        )
        if len(structured_result) <= len(offline_result):
            return structured_result
        return offline_result

    if mode == "online" and api_client is not None:
        sub_mode_byte = SUB_MODE_TOKEN if sub_mode == "token" else SUB_MODE_CHAR
        chunk_chars = CHUNK_CHARS
        max_tokens = MAX_TOKENS_DEFAULT
        if sub_mode == "token":
            compressed_body, pred_cache = encode_online_token(
                text, api_client, on_progress, priors=priors or None, max_order=max_order,
                chunk_chars=chunk_chars, max_tokens=max_tokens,
            )
        else:
            compressed_body, pred_cache = encode_online_char(
                text, api_client, on_progress, priors=priors or None, max_order=max_order,
                chunk_chars=chunk_chars, max_tokens=max_tokens,
            )

        model_name_actual = api_client.last_model_id or model_name
        truncated = [p[:chunk_chars] for p in pred_cache] if sub_mode_byte == SUB_MODE_CHAR else None
        model_data = _pack_online_model_data(
            sub_mode_byte,
            model_name_actual,
            chunk_chars,
            max_tokens,
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
        if len(online_result) <= len(offline_result):
            return online_result
        return offline_result

    if phrase_set:
        compressed_body = encode_with_phrases(
            text, phrase_set, on_progress, priors=priors or None, max_order=max_order,
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

    compressed_body = encode(text, on_progress, priors=priors or None, max_order=max_order)
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
    header, model_data, compressed_body = read_file(data)

    priors: dict[str, float] | None = None
    if header.flags & FLAG_HAS_PRIORS:
        priors = _get_priors() or None

    dec_kwargs: dict = {"priors": priors, "max_order": header.max_order}

    if header.mode == MODE_CODEGEN:
        from .codegen import CodegenManifest, restore_codegen

        manifest = CodegenManifest.deserialize(model_data)
        modified_text = decode(compressed_body, header.token_count, on_progress, **dec_kwargs)
        text = restore_codegen(modified_text, manifest)
    elif header.mode == MODE_ONLINE and header.version == VERSION_V3:
        text = _structured_online_decompress(data, priors=priors, max_order=header.max_order, on_progress=on_progress)
    elif header.mode == MODE_ONLINE:
        sub_mode_byte, _, chunk_chars, max_tokens, pred_cache, tok_cache = _unpack_online_model_data(model_data)
        has_cache = pred_cache is not None or tok_cache is not None
        if api_client is None and not has_cache:
            raise ValueError(
                "Online mode file requires an API client for decompression. "
                "Provide --api-key to decompress, or use cache-embedded files.",
            )
        if sub_mode_byte == SUB_MODE_TOKEN:
            text = decode_online_token(
                compressed_body,
                header.token_count,
                api_client,
                on_progress,
                **dec_kwargs,
                chunk_chars=chunk_chars,
                max_tokens=max_tokens,
                token_cache=tok_cache,
            )
        else:
            text = decode_online_char(
                compressed_body,
                header.token_count,
                api_client,
                on_progress,
                **dec_kwargs,
                chunk_chars=chunk_chars,
                max_tokens=max_tokens,
                prediction_cache=pred_cache,
            )
    elif header.flags & FLAG_PHRASE_ENCODING:
        _, _, phrase_table_bytes, body = read_file_v2(data)
        from .predictor.phrases import PhraseTable

        pt = PhraseTable.deserialize(phrase_table_bytes)
        phrase_set = frozenset(pt.phrases)
        text = decode_with_phrases(body, header.token_count, phrase_set, on_progress, **dec_kwargs)
    else:
        text = decode(compressed_body, header.token_count, on_progress, **dec_kwargs)

    actual_crc = compute_crc32(text.encode("utf-8"))
    if actual_crc != header.crc32:
        raise ValueError(
            f"CRC32 mismatch: expected {header.crc32:#010x}, got {actual_crc:#010x}. "
            "File may be corrupted.",
        )
    return text


def _structured_online_compress(
    text: str,
    text_bytes: bytes,
    crc: int,
    api_client,
    model_name: str,
    priors: dict[str, float],
    flags: int,
    max_order: int,
) -> bytes:
    analysis = api_client.analyze_text(text)
    stored_analysis = analysis.for_storage()
    effective_priors = _merge_priors(priors or None, stored_analysis.to_prior_map())
    phrase_table = build_structured_phrase_table(text, analysis)
    phrase_set = frozenset(phrase_table.phrases)
    phrase_table_bytes = phrase_table.serialize() if phrase_table.phrases else b""
    segments = split_text_segments(text, analysis)
    template_catalog = build_template_catalog(segments, text, analysis)
    template_bytes = template_catalog.serialize() if template_catalog.entries else b""
    route_summary = route_segments(
        text=text,
        segments=segments,
        phrase_set=phrase_set,
        priors=effective_priors,
        max_order=max_order,
        template_catalog=template_catalog,
        analysis=analysis,
        phrase_section_cost=len(phrase_table_bytes),
        template_section_cost=len(template_bytes),
    )
    payload_parts: list[bytes] = []
    records: list[SegmentRecord] = []
    payload_bytes = 0
    for routed in route_summary.routed_segments:
        payload_parts.append(routed.payload)
        payload_len = len(routed.payload)
        payload_bytes += payload_len
        records.append(
            SegmentRecord(
                kind=routed.segment.kind,
                route=routed.route,
                char_count=routed.segment.char_count,
                payload_len=payload_len,
                original_bytes=routed.original_bytes,
                encoded_bytes=routed.encoded_bytes,
                residual_bytes=routed.residual_bytes,
                estimated_gain_bytes=routed.estimated_gain_bytes,
            )
        )
    analysis_bytes = stored_analysis.serialize()
    segment_bytes = serialize_segment_records(tuple(records))

    analysis_section = make_section(analysis_bytes, prefer_compression=True)
    phrase_section = make_section(phrase_table_bytes)
    template_section = make_section(template_bytes)
    segment_section = make_section(segment_bytes, prefer_compression=True)

    stats = StructuredOnlineStats(
        segment_count=len(records),
        phrase_count=len(phrase_table.phrases),
        template_count=len(template_catalog.entries),
        template_hit_count=route_summary.template_hit_count,
        typed_slot_count=route_summary.typed_slot_count,
        typed_template_count=route_summary.typed_template_count,
        residual_bytes=route_summary.residual_bytes,
        analysis_bytes=section_stored_size(analysis_section),
        dictionary_bytes=section_stored_size(phrase_section),
        templates_bytes=section_stored_size(template_section),
        segments_bytes=section_stored_size(segment_section),
        payload_bytes=payload_bytes,
        route_counts=route_summary.route_counts,
        reason_counts=route_summary.reason_counts,
        template_family_counts=route_summary.template_family_counts,
        estimated_gain_bytes=route_summary.estimated_gain_bytes,
        literal_payload_bytes=route_summary.literal_payload_bytes,
    )
    stats_section = make_section(stats.serialize(), prefer_compression=True)
    sections = {
        SECTION_ANALYSIS: analysis_section,
        SECTION_PHRASE_TABLE: phrase_section,
        SECTION_TEMPLATES: template_section,
        SECTION_SEGMENTS: segment_section,
        SECTION_STATS: stats_section,
    }
    structured_flags = flags | (FLAG_PHRASE_ENCODING if phrase_table.phrases else 0)
    header = Header(
        mode=MODE_ONLINE,
        model_id=0,
        token_count=len(text),
        original_bytes=len(text_bytes),
        crc32=crc,
        model_data_len=0,
        flags=structured_flags,
        max_order=max_order,
        version=VERSION_V3,
    )
    return write_file_v3(header, sections, b"".join(payload_parts))


def _structured_online_decompress(
    data: bytes,
    priors: dict[str, float] | None,
    max_order: int,
    on_progress=None,
) -> str:
    _, sections, body = read_file_v3(data)
    from .predictor.phrases import PhraseTable

    phrase_table = PhraseTable.deserialize(sections.get(SECTION_PHRASE_TABLE, make_section(b"")).data)
    phrase_set = frozenset(phrase_table.phrases)
    template_catalog = TemplateCatalog.deserialize(sections.get(SECTION_TEMPLATES, make_section(b"")).data)
    analysis_section = sections.get(SECTION_ANALYSIS, make_section(b""))
    total_chars = sum(record.char_count for record in deserialize_segment_records(sections.get(SECTION_SEGMENTS, make_section(b"")).data))
    analysis = AnalysisManifest.deserialize(analysis_section.data, text_len=total_chars)
    effective_priors = _merge_priors(priors, analysis.to_prior_map())
    records = deserialize_segment_records(sections.get(SECTION_SEGMENTS, make_section(b"")).data)
    pieces: list[str] = []
    offset = 0
    produced = 0
    total = sum(record.char_count for record in records)
    for record in records:
        payload = body[offset:offset + record.payload_len]
        offset += record.payload_len
        if record.route == ROUTE_PHRASE:
            text = decode_with_phrases(
                payload,
                record.char_count,
                phrase_set,
                None,
                priors=effective_priors,
                max_order=max_order,
            )
        elif record.route == ROUTE_TEMPLATE:
            text = decode_template_segment(
                payload,
                template_catalog,
                phrase_set,
                record.char_count,
                priors=effective_priors,
                max_order=max_order,
                analysis=analysis,
            )
        else:
            text = decode(
                payload,
                record.char_count,
                None,
                priors=effective_priors,
                max_order=max_order,
            )
        pieces.append(text)
        produced += record.char_count
        if on_progress:
            on_progress(min(produced, total), total)
    return "".join(pieces)


# ------------------------------------------------------------------
# model_data packing helpers for legacy online modes
# ------------------------------------------------------------------


def _pack_online_model_data(
    sub_mode: int,
    model_name: str,
    chunk_chars: int = CHUNK_CHARS,
    max_tokens: int = MAX_TOKENS_DEFAULT,
    prediction_cache: list[str] | None = None,
    token_cache: list | None = None,
) -> bytes:
    """Pack sub-mode + model name + params + prediction cache into model_data."""
    import zstandard

    name_bytes = model_name.encode("utf-8")
    base = (
        struct.pack("<B", sub_mode)
        + name_bytes
        + b"\x00"
        + struct.pack("<HH", chunk_chars, max_tokens)
    )

    if prediction_cache is not None:
        parts = [struct.pack("<H", len(prediction_cache))]
        for pred_text in prediction_cache:
            encoded = pred_text.encode("utf-8")
            parts.append(struct.pack("<I", len(encoded)))
            parts.append(encoded)
        raw = b"".join(parts)
        compressed_cache = zstandard.ZstdCompressor(level=19).compress(raw)
        return base + struct.pack("<BI", 0x01, len(compressed_cache)) + compressed_cache

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

    return base


def _unpack_online_model_data(
    data: bytes,
) -> tuple[int, str, int, int, list[str] | None, list | None]:
    """Unpack legacy online model_data fields."""
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

    char_cache = None
    token_cache = None
    if len(extra) >= 5:
        cache_type = extra[0]
        cache_len = struct.unpack("<I", extra[1:5])[0]
        cache_data = extra[5:5 + cache_len]

        if cache_type == 0x01 and cache_data:
            raw = zstandard.ZstdDecompressor().decompress(cache_data)
            offset = 0
            count = struct.unpack("<H", raw[offset:offset + 2])[0]
            offset += 2
            char_cache = []
            for _ in range(count):
                text_len = struct.unpack("<I", raw[offset:offset + 4])[0]
                offset += 4
                char_cache.append(raw[offset:offset + text_len].decode("utf-8"))
                offset += text_len
        elif cache_type == 0x02 and cache_data:
            import json
            from .api_client import ChunkResult, GeneratedToken

            raw = zstandard.ZstdDecompressor().decompress(cache_data)
            entries = json.loads(raw.decode("utf-8"))
            token_cache = []
            for entry in entries:
                tokens = []
                for token_payload in entry["tokens"]:
                    tokens.append(
                        GeneratedToken(
                            text=token_payload["text"],
                            logprob=token_payload["logprob"],
                            top_alternatives=[tuple(item) for item in token_payload["top_alternatives"]],
                            char_offset=token_payload["char_offset"],
                        )
                    )
                token_cache.append(
                    ChunkResult(
                        generated_text=entry["text"],
                        tokens=tokens,
                        model="",
                    )
                )

    return sub_mode, model_name, chunk_chars, max_tokens, char_cache, token_cache
