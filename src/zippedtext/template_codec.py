from __future__ import annotations

import struct
from dataclasses import dataclass

from .decoder import decode, decode_with_phrases
from .online_manifest import AnalysisManifest, ROUTE_LITERAL, ROUTE_PHRASE, ROUTE_TEMPLATE
from .predictor.phrases import PhraseTable
from .residual import EncodedResidual, deserialize_string_tuple, encode_residual_segments, serialize_string_tuple

_TEMPLATE_MAGIC = b"TPL1"
_TEMPLATE_HEADER_FMT = "<4sHH"
_TEMPLATE_HEADER_SIZE = struct.calcsize(_TEMPLATE_HEADER_FMT)

_TEMPLATE_KIND_KEY_VALUE = "key_value"
_TEMPLATE_KIND_LIST_PREFIX = "list_prefix"
_TEMPLATE_KIND_TABLE_ROW = "table_row"

_TEMPLATE_KIND_TO_ID = {
    _TEMPLATE_KIND_KEY_VALUE: 0,
    _TEMPLATE_KIND_LIST_PREFIX: 1,
    _TEMPLATE_KIND_TABLE_ROW: 2,
}
_TEMPLATE_ID_TO_KIND = {value: key for key, value in _TEMPLATE_KIND_TO_ID.items()}
_TEMPLATE_ENABLED_KINDS = frozenset({"prose", "list", "table", "config", "mixed"})


@dataclass(frozen=True)
class TemplateMatch:
    template_kind: str
    skeleton: str
    slot_values: tuple[str, ...]
    residual_spans: tuple[tuple[int, int], ...]
    confidence: float


@dataclass(frozen=True)
class EncodedTemplatePayload:
    template_kind: str
    template_index: int
    slot_values: tuple[str, ...]
    residual: EncodedResidual
    residual_text_ranges: tuple[tuple[int, int], ...]
    payload: bytes

    @property
    def encoded_bytes(self) -> int:
        return len(self.payload)

    @property
    def residual_bytes(self) -> int:
        return self.residual.total_payload_bytes


@dataclass(frozen=True)
class TemplateCatalog:
    entries: tuple[tuple[str, str], ...]

    def serialize(self) -> bytes:
        parts = [struct.pack(_TEMPLATE_HEADER_FMT, _TEMPLATE_MAGIC, len(self.entries), 0)]
        for template_kind, skeleton in self.entries:
            parts.append(struct.pack("<B", _TEMPLATE_KIND_TO_ID[template_kind]))
            parts.append(serialize_string_tuple((skeleton,)))
        return b"".join(parts)

    @classmethod
    def deserialize(cls, data: bytes) -> "TemplateCatalog":
        if not data:
            return cls(entries=())
        if len(data) < _TEMPLATE_HEADER_SIZE:
            raise ValueError("invalid template catalog")
        magic, count, _reserved = struct.unpack(_TEMPLATE_HEADER_FMT, data[:_TEMPLATE_HEADER_SIZE])
        if magic != _TEMPLATE_MAGIC:
            raise ValueError("invalid template catalog")
        offset = _TEMPLATE_HEADER_SIZE
        entries: list[tuple[str, str]] = []
        for _ in range(count):
            if offset + 1 > len(data):
                raise ValueError("invalid template catalog")
            kind_id = data[offset]
            offset += 1
            if offset + 2 > len(data):
                raise ValueError("invalid template catalog")
            tuple_length = struct.unpack("<H", data[offset:offset + 2])[0]
            tuple_end = offset + 2
            for _i in range(tuple_length):
                if tuple_end + 2 > len(data):
                    raise ValueError("invalid template catalog")
                item_len = struct.unpack("<H", data[tuple_end:tuple_end + 2])[0]
                tuple_end += 2 + item_len
            skeleton_tuple = deserialize_string_tuple(data[offset:tuple_end])
            offset = tuple_end
            skeleton = skeleton_tuple[0] if skeleton_tuple else ""
            entries.append((_TEMPLATE_ID_TO_KIND.get(kind_id, _TEMPLATE_KIND_KEY_VALUE), skeleton))
        if offset != len(data):
            raise ValueError("invalid template catalog")
        return cls(entries=tuple(entries))


def build_template_catalog(
    segments: tuple,
    text: str,
    analysis: AnalysisManifest,
) -> TemplateCatalog:
    entries: list[tuple[str, str]] = []
    seen: set[tuple[str, str]] = set()
    for segment in segments:
        if segment.kind not in _TEMPLATE_ENABLED_KINDS:
            continue
        segment_text = text[segment.start:segment.end]
        match = detect_template(segment_text, analysis)
        if not match:
            continue
        entry = (match.template_kind, match.skeleton)
        if entry in seen:
            continue
        seen.add(entry)
        entries.append(entry)
    return TemplateCatalog(entries=tuple(entries))


def detect_template(text: str, analysis: AnalysisManifest | None = None) -> TemplateMatch | None:
    stripped = text.strip()
    if not stripped:
        return None
    key_value = _match_key_value(stripped)
    if key_value:
        return key_value
    list_prefix = _match_list_prefix(stripped)
    if list_prefix:
        return list_prefix
    table_row = _match_table_row(stripped)
    if table_row:
        return table_row
    return None


def encode_template_segment(
    text: str,
    match: TemplateMatch,
    template_index: int,
    phrase_set: frozenset[str],
    priors: dict[str, float] | None,
    max_order: int,
) -> EncodedTemplatePayload:
    residual = encode_residual_segments(
        text,
        match.residual_spans,
        phrase_set,
        priors,
        max_order,
    )
    payload = b"".join(
        [
            struct.pack("<H", template_index),
            serialize_string_tuple(match.slot_values),
            residual.serialize(),
        ]
    )
    return EncodedTemplatePayload(
        template_kind=match.template_kind,
        template_index=template_index,
        slot_values=match.slot_values,
        residual=residual,
        residual_text_ranges=match.residual_spans,
        payload=payload,
    )


def decode_template_segment(
    payload: bytes,
    catalog: TemplateCatalog,
    phrase_set: frozenset[str],
    char_count: int,
    priors: dict[str, float] | None,
    max_order: int,
) -> str:
    if len(payload) < 2:
        raise ValueError("invalid template payload")
    template_index = struct.unpack("<H", payload[:2])[0]
    if template_index >= len(catalog.entries):
        raise ValueError("invalid template payload")
    slot_values, offset = _decode_string_tuple_with_offset(payload, 2)
    residual = EncodedResidual.deserialize(payload[offset:])
    template_kind, skeleton = catalog.entries[template_index]
    residual_text = _decode_residual_text(
        residual,
        phrase_set,
        priors=priors,
        max_order=max_order,
    )
    restored = _render_template(template_kind, skeleton, slot_values, residual_text)
    if len(restored) != char_count:
        raise ValueError("invalid template payload")
    return restored


def _decode_residual_text(
    residual: EncodedResidual,
    phrase_set: frozenset[str],
    *,
    priors: dict[str, float] | None,
    max_order: int,
) -> tuple[str, ...]:
    items: list[str] = []
    for segment in residual.segments:
        char_count = max(segment.original_end - segment.original_start, 0)
        if segment.route == ROUTE_PHRASE:
            restored = decode_with_phrases(
                segment.payload,
                char_count,
                phrase_set,
                priors=priors,
                max_order=max_order,
            )
        else:
            restored = decode(
                segment.payload,
                char_count,
                priors=priors,
                max_order=max_order,
            )
        items.append(restored)
    return tuple(items)


def _decode_string_tuple_with_offset(data: bytes, offset: int) -> tuple[tuple[str, ...], int]:
    if offset + 2 > len(data):
        raise ValueError("invalid template payload")
    count = struct.unpack("<H", data[offset:offset + 2])[0]
    cursor = offset + 2
    items: list[str] = []
    for _ in range(count):
        if cursor + 2 > len(data):
            raise ValueError("invalid template payload")
        length = struct.unpack("<H", data[cursor:cursor + 2])[0]
        cursor += 2
        end = cursor + length
        if end > len(data):
            raise ValueError("invalid template payload")
        items.append(data[cursor:end].decode("utf-8"))
        cursor = end
    return tuple(items), cursor


def _render_template(
    template_kind: str,
    skeleton: str,
    slot_values: tuple[str, ...],
    residual_text: tuple[str, ...],
) -> str:
    if template_kind == _TEMPLATE_KIND_KEY_VALUE:
        if len(slot_values) < 2:
            raise ValueError("invalid key-value template")
        suffix = residual_text[0] if residual_text else ""
        return f"{slot_values[0]}{skeleton}{slot_values[1]}{suffix}"
    if template_kind == _TEMPLATE_KIND_LIST_PREFIX:
        suffix = residual_text[0] if residual_text else ""
        value = slot_values[0] if slot_values else ""
        return f"{skeleton}{value}{suffix}"
    if template_kind == _TEMPLATE_KIND_TABLE_ROW:
        suffix = residual_text[0] if residual_text else ""
        values = " | ".join(slot_values)
        return f"| {values} |{suffix}"
    raise ValueError("invalid template kind")


def _match_key_value(text: str) -> TemplateMatch | None:
    separators = (": ", "：", " = ", "=")
    for separator in separators:
        if separator not in text:
            continue
        key, value = text.split(separator, 1)
        key = key.strip()
        value = value.strip()
        if not key or not value:
            continue
        if len(key) > 40 or len(value) < 2:
            continue
        suffix = ""
        base_value = value
        if "（" in value and value.endswith("）"):
            base_value, suffix = value.rsplit("（", 1)
            suffix = "（" + suffix
            base_value = base_value.rstrip()
        elif " (" in value and value.endswith(")"):
            base_value, suffix = value.rsplit(" (", 1)
            suffix = " (" + suffix
            base_value = base_value.rstrip()
        residual_spans = ()
        if suffix:
            start = text.rfind(suffix)
            residual_spans = ((start, len(text)),)
        return TemplateMatch(
            template_kind=_TEMPLATE_KIND_KEY_VALUE,
            skeleton=separator,
            slot_values=(key, base_value),
            residual_spans=residual_spans,
            confidence=0.92,
        )
    return None


def _match_list_prefix(text: str) -> TemplateMatch | None:
    prefixes = ("- ", "* ", "• ")
    for prefix in prefixes:
        if text.startswith(prefix) and len(text) > len(prefix) + 3:
            return TemplateMatch(
                template_kind=_TEMPLATE_KIND_LIST_PREFIX,
                skeleton=prefix,
                slot_values=(text[len(prefix):],),
                residual_spans=(),
                confidence=0.8,
            )
    if len(text) > 4 and text[0].isdigit() and text[1:3] in (". ", ") "):
        prefix = text[:3]
        return TemplateMatch(
            template_kind=_TEMPLATE_KIND_LIST_PREFIX,
            skeleton=prefix,
            slot_values=(text[3:],),
            residual_spans=(),
            confidence=0.78,
        )
    return None


def _match_table_row(text: str) -> TemplateMatch | None:
    if text.count("|") < 2:
        return None
    stripped = text.strip()
    if not stripped.startswith("|") or not stripped.endswith("|"):
        return None
    parts = [item.strip() for item in stripped.strip("|").split("|")]
    if len(parts) < 2 or any(not item for item in parts):
        return None
    return TemplateMatch(
        template_kind=_TEMPLATE_KIND_TABLE_ROW,
        skeleton="|",
        slot_values=tuple(parts),
        residual_spans=(),
        confidence=0.82,
    )


def template_route_allowed(segment_kind: str) -> bool:
    return segment_kind in _TEMPLATE_ENABLED_KINDS
