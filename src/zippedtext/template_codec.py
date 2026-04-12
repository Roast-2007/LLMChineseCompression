from __future__ import annotations

import re
import struct
from collections import Counter
from dataclasses import dataclass, replace

from .decoder import decode, decode_with_phrases
from .online_manifest import AnalysisManifest, ROUTE_PHRASE
from .residual import EncodedResidual, deserialize_string_tuple, encode_residual_segments, serialize_string_tuple

_TEMPLATE_MAGIC = b"TPL1"
_TEMPLATE_HEADER_FMT = "<4sHH"
_TEMPLATE_HEADER_SIZE = struct.calcsize(_TEMPLATE_HEADER_FMT)
_TYPED_SLOT_MAGIC = b"TSL1"
_TYPED_SLOT_HEADER_FMT = "<4sH"
_TYPED_SLOT_HEADER_SIZE = struct.calcsize(_TYPED_SLOT_HEADER_FMT)

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
_NUMBERED_LIST_RE = re.compile(r"^(\d+[.)]\s+)(.+)$")
_SENTENCE_PUNCTUATION = frozenset("。！？!?；;")
_IDENTIFIER_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_.-]{1,127}$")
_VERSION_RE = re.compile(
    r"^(?P<prefix>[A-Za-z]*)(?P<body>\d+(?P<sep>[._-])\d+(?:(?P=sep)\d+){0,5})(?P<suffix>[A-Za-z0-9-]*)$"
)
_NUMBER_WITH_UNIT_RE = re.compile(
    r"^(?P<number>[+-]?\d+(?:\.\d+)?)(?P<space>\s?)(?P<unit>[A-Za-z%°μµ/][A-Za-z0-9%°μµ/_-]{0,15})$"
)
_URL_PREFIXES = (
    "https://",
    "http://",
    "ftp://",
    "file://",
    "/",
    "./",
    "../",
    "~/",
    "C:/",
    "C:\\",
)
_COMMON_UNITS = (
    "%",
    "ms",
    "s",
    "sec",
    "min",
    "h",
    "day",
    "days",
    "B",
    "KB",
    "MB",
    "GB",
    "TB",
    "KiB",
    "MiB",
    "GiB",
    "Hz",
    "kHz",
    "MHz",
    "GHz",
    "px",
    "dpi",
    "℃",
    "°C",
    "°F",
)
_SLOT_TYPE_TO_ID = {
    "string": 0,
    "identifier": 1,
    "version": 2,
    "path_or_url": 3,
    "enum": 4,
    "number_with_unit": 5,
}
_ID_TO_SLOT_TYPE = {value: key for key, value in _SLOT_TYPE_TO_ID.items()}
_ENUM_FIELD_HINTS = (
    "mode",
    "status",
    "state",
    "level",
    "flag",
    "type",
    "format",
)


@dataclass(frozen=True)
class TemplateMatch:
    template_kind: str
    skeleton: str
    slot_values: tuple[str, ...]
    residual_spans: tuple[tuple[int, int], ...]
    confidence: float
    slot_types: tuple[str, ...] = ()
    slot_fields: tuple[str, ...] = ()
    slot_enum_candidates: tuple[tuple[str, ...], ...] = ()


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
    counts: Counter[tuple[str, str]] = Counter()
    confidences: dict[tuple[str, str], float] = {}
    ordered: list[tuple[str, str]] = []
    seen: set[tuple[str, str]] = set()

    for segment in segments:
        if segment.kind not in _TEMPLATE_ENABLED_KINDS:
            continue
        segment_text = text[segment.start:segment.end]
        match = detect_template(segment_text, analysis)
        if not match:
            continue
        entry = (match.template_kind, match.skeleton)
        counts[entry] += 1
        confidences[entry] = max(confidences.get(entry, 0.0), match.confidence)
        if entry not in seen:
            seen.add(entry)
            ordered.append(entry)

    entries: list[tuple[str, str]] = []
    for entry in ordered:
        hinted = template_kind_is_hinted(entry[0], analysis)
        if counts[entry] >= 2 or (hinted and confidences.get(entry, 0.0) >= 0.9):
            entries.append(entry)
    return TemplateCatalog(entries=tuple(entries))


def detect_template(text: str, analysis: AnalysisManifest | None = None) -> TemplateMatch | None:
    stripped = text.strip()
    if not stripped:
        return None

    candidates = [
        _match_key_value(stripped),
        _match_list_prefix(stripped),
        _match_table_row(stripped),
    ]
    adjusted = [
        _apply_template_hint_score(_annotate_slot_metadata(candidate, analysis), stripped, analysis)
        for candidate in candidates
        if candidate is not None
    ]
    if not adjusted:
        return None
    return max(adjusted, key=_template_sort_key)


def template_kind_is_hinted(template_kind: str, analysis: AnalysisManifest | None) -> bool:
    return analysis is not None and template_kind in analysis.template_hints


def template_confidence_threshold(
    match: TemplateMatch,
    analysis: AnalysisManifest | None,
) -> float:
    threshold = 0.75
    if template_kind_is_hinted(match.template_kind, analysis):
        threshold -= 0.08
    elif analysis and analysis.template_hints:
        threshold += 0.03
    if match.template_kind == _TEMPLATE_KIND_KEY_VALUE and match.slot_values:
        if len(match.slot_values[0]) > 48:
            threshold += 0.04
    if any(slot_type != "string" for slot_type in match.slot_types):
        threshold -= 0.02
    return min(max(threshold, 0.6), 0.9)


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
            _serialize_typed_slots(match),
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
    analysis: AnalysisManifest | None = None,
) -> str:
    if len(payload) < 2:
        raise ValueError("invalid template payload")
    template_index = struct.unpack("<H", payload[:2])[0]
    if template_index >= len(catalog.entries):
        raise ValueError("invalid template payload")
    template_kind, skeleton = catalog.entries[template_index]
    if payload[2:2 + len(_TYPED_SLOT_MAGIC)] == _TYPED_SLOT_MAGIC:
        slot_values, offset = _decode_typed_slots_with_offset(
            payload,
            2,
            template_kind,
            skeleton,
            analysis,
        )
    else:
        slot_values, offset = _decode_string_tuple_with_offset(payload, 2)
    residual = EncodedResidual.deserialize(payload[offset:])
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
        if not slot_values:
            raise ValueError("invalid key-value template")
        suffix = residual_text[0] if residual_text else ""
        return f"{skeleton}{slot_values[0]}{suffix}"
    if template_kind == _TEMPLATE_KIND_LIST_PREFIX:
        suffix = residual_text[0] if residual_text else ""
        value = slot_values[0] if slot_values else ""
        return f"{skeleton}{value}{suffix}"
    if template_kind == _TEMPLATE_KIND_TABLE_ROW:
        suffix = residual_text[0] if residual_text else ""
        return skeleton.format(*slot_values) + suffix
    raise ValueError("invalid template kind")


def _apply_template_hint_score(
    match: TemplateMatch,
    text: str,
    analysis: AnalysisManifest | None,
) -> TemplateMatch:
    confidence = match.confidence
    if template_kind_is_hinted(match.template_kind, analysis):
        confidence += 0.08
    elif analysis and analysis.template_hints:
        confidence -= 0.03
    if match.template_kind == _TEMPLATE_KIND_KEY_VALUE and len(text) > 120:
        confidence -= 0.08
    if any(slot_type != "string" for slot_type in match.slot_types):
        confidence += 0.02
    return replace(match, confidence=min(max(confidence, 0.0), 0.99))


def _template_sort_key(match: TemplateMatch) -> tuple[float, int, int]:
    priority = {
        _TEMPLATE_KIND_TABLE_ROW: 2,
        _TEMPLATE_KIND_KEY_VALUE: 1,
        _TEMPLATE_KIND_LIST_PREFIX: 0,
    }.get(match.template_kind, 0)
    typed_score = sum(1 for slot_type in match.slot_types if slot_type != "string")
    return (match.confidence, typed_score, priority, len(match.skeleton))


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
        if len(key) > 48 or len(value) < 2:
            continue
        if any(ch in _SENTENCE_PUNCTUATION for ch in key):
            continue
        if key.count(" ") > 6:
            continue

        suffix = ""
        base_value = value
        # Reject nested parentheses to avoid ambiguous split
        if "（" in value and value.endswith("）"):
            inner = value[len(value.index("（")) + 1:-1]
            if "（" not in inner and "）" not in inner:
                base_value, suffix = value.rsplit("（", 1)
                suffix = "（" + suffix
                base_value = base_value.rstrip()
            # else: leave value as-is, no suffix extraction
        elif " (" in value and value.endswith(")"):
            inner = value[value.index(" (") + 2:-1]
            if "(" not in inner and ")" not in inner:
                base_value, suffix = value.rsplit(" (", 1)
                suffix = " (" + suffix
                base_value = base_value.rstrip()
        if not base_value:
            continue

        confidence = 0.82
        if len(key) <= 24:
            confidence += 0.05
        if separator in (": ", " = "):
            confidence += 0.02
        if len(base_value) > 64:
            confidence -= 0.08
        if sum(base_value.count(mark) for mark in "。！？!?") >= 2:
            confidence -= 0.08

        residual_spans: tuple[tuple[int, int], ...] = ()
        if suffix:
            start = text.rfind(suffix)
            residual_spans = ((start, len(text)),)
        return TemplateMatch(
            template_kind=_TEMPLATE_KIND_KEY_VALUE,
            skeleton=f"{key}{separator}",
            slot_values=(base_value,),
            residual_spans=residual_spans,
            confidence=confidence,
            slot_fields=(_normalize_field_key(key),),
            slot_enum_candidates=((),),
        )
    return None


def _match_list_prefix(text: str) -> TemplateMatch | None:
    prefixes = ("- ", "* ", "• ")
    for prefix in prefixes:
        if text.startswith(prefix) and len(text) > len(prefix) + 2:
            confidence = 0.8 if len(text) <= 80 else 0.74
            return TemplateMatch(
                template_kind=_TEMPLATE_KIND_LIST_PREFIX,
                skeleton=prefix,
                slot_values=(text[len(prefix):],),
                residual_spans=(),
                confidence=confidence,
                slot_fields=("",),
                slot_enum_candidates=((),),
            )

    numbered = _NUMBERED_LIST_RE.match(text)
    if not numbered:
        return None
    prefix, value = numbered.groups()
    confidence = 0.79 if len(value) <= 80 else 0.73
    return TemplateMatch(
        template_kind=_TEMPLATE_KIND_LIST_PREFIX,
        skeleton=prefix,
        slot_values=(value,),
        residual_spans=(),
        confidence=confidence,
        slot_fields=("",),
        slot_enum_candidates=((),),
    )


def _match_table_row(text: str) -> TemplateMatch | None:
    stripped = text.strip()
    if stripped.startswith("|") and stripped.endswith("|") and stripped.count("|") >= 2:
        raw_cells = stripped[1:-1].split("|")
        slot_values = tuple(cell.strip() for cell in raw_cells)
        if len(slot_values) >= 2 and all(value for value in slot_values):
            skeleton_parts = ["|"]
            for index, raw_cell in enumerate(raw_cells):
                leading = raw_cell[:len(raw_cell) - len(raw_cell.lstrip())]
                trailing = raw_cell[len(raw_cell.rstrip()):]
                skeleton_parts.append(f"{_escape_braces(leading)}{{{index}}}{_escape_braces(trailing)}|")
            return TemplateMatch(
                template_kind=_TEMPLATE_KIND_TABLE_ROW,
                skeleton="".join(skeleton_parts),
                slot_values=slot_values,
                residual_spans=(),
                confidence=0.84,
                slot_fields=tuple("" for _ in slot_values),
                slot_enum_candidates=tuple(() for _ in slot_values),
            )

    if "\t" not in stripped:
        return None
    raw_cells = stripped.split("\t")
    slot_values = tuple(cell.strip() for cell in raw_cells)
    if len(slot_values) < 2 or any(not value for value in slot_values):
        return None
    skeleton_parts: list[str] = []
    for index, raw_cell in enumerate(raw_cells):
        leading = raw_cell[:len(raw_cell) - len(raw_cell.lstrip())]
        trailing = raw_cell[len(raw_cell.rstrip()):]
        skeleton_parts.append(f"{_escape_braces(leading)}{{{index}}}{_escape_braces(trailing)}")
    return TemplateMatch(
        template_kind=_TEMPLATE_KIND_TABLE_ROW,
        skeleton="\t".join(skeleton_parts),
        slot_values=slot_values,
        residual_spans=(),
        confidence=0.8,
        slot_fields=tuple("" for _ in slot_values),
        slot_enum_candidates=tuple(() for _ in slot_values),
    )


def _annotate_slot_metadata(
    match: TemplateMatch,
    analysis: AnalysisManifest | None,
) -> TemplateMatch:
    fields = list(match.slot_fields) if match.slot_fields else [""] * len(match.slot_values)
    while len(fields) < len(match.slot_values):
        fields.append("")

    slot_types: list[str] = []
    enum_candidates_by_slot: list[tuple[str, ...]] = []
    for index, value in enumerate(match.slot_values):
        field = fields[index]
        enum_candidates = ()
        hinted_type = "string"
        if analysis is not None:
            slot_hint = analysis.slot_hint_for(match.template_kind, index, field)
            schema = analysis.field_schema_for(field) if field else None
            if slot_hint is not None:
                hinted_type = slot_hint.slot_type
                if slot_hint.enum_candidates:
                    enum_candidates = slot_hint.enum_candidates
            elif schema is not None:
                hinted_type = schema.slot_type
                if schema.enum_candidates:
                    enum_candidates = schema.enum_candidates
            elif field:
                enum_candidates = analysis.enum_candidates_for(field)
        slot_type = hinted_type
        if slot_type == "string":
            slot_type = _infer_slot_type(value, field, enum_candidates)
        elif slot_type == "enum" and not enum_candidates and analysis is not None and field:
            enum_candidates = analysis.enum_candidates_for(field)
        slot_types.append(slot_type)
        enum_candidates_by_slot.append(enum_candidates)

    return replace(
        match,
        slot_types=tuple(slot_types),
        slot_fields=tuple(fields),
        slot_enum_candidates=tuple(enum_candidates_by_slot),
    )


def _infer_slot_type(value: str, field: str, enum_candidates: tuple[str, ...]) -> str:
    stripped = value.strip()
    if enum_candidates and stripped in enum_candidates:
        return "enum"
    if _looks_like_version(stripped):
        return "version"
    if _looks_like_path_or_url(stripped):
        return "path_or_url"
    if _looks_like_number_with_unit(stripped):
        return "number_with_unit"
    if _looks_like_identifier(stripped):
        return "identifier"
    normalized_field = field.lower()
    if normalized_field.endswith(_ENUM_FIELD_HINTS) or normalized_field in _ENUM_FIELD_HINTS:
        return "enum"
    return "string"


def _serialize_typed_slots(match: TemplateMatch) -> bytes:
    slot_types = match.slot_types or tuple("string" for _ in match.slot_values)
    enum_candidates = match.slot_enum_candidates or tuple(() for _ in match.slot_values)
    parts = [struct.pack(_TYPED_SLOT_HEADER_FMT, _TYPED_SLOT_MAGIC, len(match.slot_values))]
    for index, value in enumerate(match.slot_values):
        slot_type = slot_types[index] if index < len(slot_types) else "string"
        enum_values = enum_candidates[index] if index < len(enum_candidates) else ()
        slot_payload = _encode_slot_value(slot_type, value, enum_values)
        parts.append(_pack_slot_payload(slot_type, slot_payload))
    return b"".join(parts)


def _decode_typed_slots_with_offset(
    data: bytes,
    offset: int,
    template_kind: str,
    skeleton: str,
    analysis: AnalysisManifest | None,
) -> tuple[tuple[str, ...], int]:
    end = offset + _TYPED_SLOT_HEADER_SIZE
    if end > len(data):
        raise ValueError("invalid template payload")
    magic, count = struct.unpack(_TYPED_SLOT_HEADER_FMT, data[offset:end])
    if magic != _TYPED_SLOT_MAGIC:
        raise ValueError("invalid template payload")
    cursor = end
    slot_values: list[str] = []
    slot_fields = _default_slot_fields(template_kind, skeleton, count)
    for index in range(count):
        slot_type, slot_payload, cursor = _unpack_slot_payload(data, cursor)
        field = slot_fields[index] if index < len(slot_fields) else ""
        enum_candidates = ()
        if analysis is not None:
            slot_hint = analysis.slot_hint_for(template_kind, index, field)
            if slot_hint is not None and slot_hint.enum_candidates:
                enum_candidates = slot_hint.enum_candidates
            elif field:
                enum_candidates = analysis.enum_candidates_for(field)
        slot_values.append(_decode_slot_value(slot_type, slot_payload, field, enum_candidates))
    return tuple(slot_values), cursor


def _pack_slot_payload(slot_type: str, payload: bytes) -> bytes:
    type_id = _SLOT_TYPE_TO_ID.get(slot_type, 0)
    if len(payload) < 0xFF:
        return struct.pack("<BB", type_id, len(payload)) + payload
    return struct.pack("<BBH", type_id, 0xFF, len(payload)) + payload


def _unpack_slot_payload(data: bytes, offset: int) -> tuple[str, bytes, int]:
    if offset + 2 > len(data):
        raise ValueError("invalid template payload")
    type_id = data[offset]
    short_len = data[offset + 1]
    offset += 2
    if short_len == 0xFF:
        if offset + 2 > len(data):
            raise ValueError("invalid template payload")
        payload_len = struct.unpack("<H", data[offset:offset + 2])[0]
        offset += 2
    else:
        payload_len = short_len
    end = offset + payload_len
    if end > len(data):
        raise ValueError("invalid template payload")
    return _ID_TO_SLOT_TYPE.get(type_id, "string"), data[offset:end], end


def _encode_slot_value(slot_type: str, value: str, enum_candidates: tuple[str, ...]) -> bytes:
    if slot_type == "enum":
        return _encode_enum_slot(value, enum_candidates)
    if slot_type == "version":
        return _encode_version_slot(value)
    if slot_type == "path_or_url":
        return _encode_path_or_url_slot(value)
    if slot_type == "number_with_unit":
        return _encode_number_with_unit_slot(value)
    return value.encode("utf-8")


def _decode_slot_value(slot_type: str, payload: bytes, field: str, enum_candidates: tuple[str, ...]) -> str:
    if slot_type == "enum":
        return _decode_enum_slot(payload, enum_candidates)
    if slot_type == "version":
        return _decode_version_slot(payload)
    if slot_type == "path_or_url":
        return _decode_path_or_url_slot(payload)
    if slot_type == "number_with_unit":
        return _decode_number_with_unit_slot(payload)
    return payload.decode("utf-8")


def _encode_enum_slot(value: str, enum_candidates: tuple[str, ...]) -> bytes:
    if value in enum_candidates and len(enum_candidates) < 0xFF:
        return b"\x01" + bytes([enum_candidates.index(value)])
    return b"\x00" + value.encode("utf-8")


def _decode_enum_slot(payload: bytes, enum_candidates: tuple[str, ...]) -> str:
    if not payload:
        return ""
    if payload[0] == 0x00:
        return payload[1:].decode("utf-8")
    if payload[0] == 0x01 and len(payload) >= 2:
        index = payload[1]
        if index < len(enum_candidates):
            return enum_candidates[index]
        raise ValueError("invalid enum slot payload")
    raise ValueError("invalid enum slot payload")


def _encode_version_slot(value: str) -> bytes:
    match = _VERSION_RE.match(value)
    if not match:
        return b"\x00" + value.encode("utf-8")
    prefix = match.group("prefix")
    body = match.group("body")
    separator = match.group("sep")
    suffix = match.group("suffix")
    parts = body.split(separator)
    try:
        encoded_numbers = b"".join(_encode_varint(int(part)) for part in parts)
    except ValueError:
        return b"\x00" + value.encode("utf-8")
    if len(prefix.encode("utf-8")) >= 0xFF or len(suffix.encode("utf-8")) >= 0xFF:
        return b"\x00" + value.encode("utf-8")
    return b"".join(
        [
            b"\x01",
            bytes([len(prefix.encode("utf-8"))]),
            prefix.encode("utf-8"),
            separator.encode("utf-8"),
            bytes([len(parts)]),
            encoded_numbers,
            bytes([len(suffix.encode("utf-8"))]),
            suffix.encode("utf-8"),
        ]
    )


def _decode_version_slot(payload: bytes) -> str:
    if not payload:
        return ""
    if payload[0] == 0x00:
        return payload[1:].decode("utf-8")
    if payload[0] != 0x01 or len(payload) < 4:
        raise ValueError("invalid version slot payload")
    offset = 1
    prefix_len = payload[offset]
    offset += 1
    prefix = payload[offset:offset + prefix_len].decode("utf-8")
    offset += prefix_len
    separator = payload[offset:offset + 1].decode("utf-8")
    offset += 1
    part_count = payload[offset]
    offset += 1
    parts: list[str] = []
    for _ in range(part_count):
        value, offset = _decode_varint(payload, offset)
        parts.append(str(value))
    if offset >= len(payload):
        raise ValueError("invalid version slot payload")
    suffix_len = payload[offset]
    offset += 1
    suffix = payload[offset:offset + suffix_len].decode("utf-8")
    offset += suffix_len
    if offset != len(payload):
        raise ValueError("invalid version slot payload")
    return prefix + separator.join(parts) + suffix


def _encode_path_or_url_slot(value: str) -> bytes:
    for index, prefix in enumerate(_URL_PREFIXES):
        if value.startswith(prefix):
            return bytes([index]) + value[len(prefix):].encode("utf-8")
    return b"\xff" + value.encode("utf-8")


def _decode_path_or_url_slot(payload: bytes) -> str:
    if not payload:
        return ""
    prefix_index = payload[0]
    tail = payload[1:].decode("utf-8")
    if prefix_index == 0xFF:
        return tail
    if prefix_index >= len(_URL_PREFIXES):
        raise ValueError("invalid path/url slot payload")
    return _URL_PREFIXES[prefix_index] + tail


def _encode_number_with_unit_slot(value: str) -> bytes:
    match = _NUMBER_WITH_UNIT_RE.match(value)
    if not match:
        return b"\x00" + value.encode("utf-8")
    number = match.group("number").encode("utf-8")
    unit = match.group("unit")
    spaced = b"\x01" if match.group("space") else b"\x00"
    if len(number) >= 0xFF:
        return b"\x00" + value.encode("utf-8")
    if unit in _COMMON_UNITS:
        unit_bytes = bytes([_COMMON_UNITS.index(unit)])
        return b"\x01" + spaced + bytes([len(number)]) + number + unit_bytes
    encoded_unit = unit.encode("utf-8")
    if len(encoded_unit) >= 0xFF:
        return b"\x00" + value.encode("utf-8")
    return b"\x02" + spaced + bytes([len(number)]) + number + bytes([len(encoded_unit)]) + encoded_unit


def _decode_number_with_unit_slot(payload: bytes) -> str:
    if not payload:
        return ""
    mode = payload[0]
    if mode == 0x00:
        return payload[1:].decode("utf-8")
    if len(payload) < 4:
        raise ValueError("invalid number_with_unit slot payload")
    spaced = payload[1] == 0x01
    number_len = payload[2]
    number_start = 3
    number_end = number_start + number_len
    if number_end > len(payload):
        raise ValueError("invalid number_with_unit slot payload")
    number = payload[number_start:number_end].decode("utf-8")
    if mode == 0x01:
        if number_end >= len(payload):
            raise ValueError("invalid number_with_unit slot payload")
        unit_index = payload[number_end]
        if unit_index >= len(_COMMON_UNITS):
            raise ValueError("invalid number_with_unit slot payload")
        unit = _COMMON_UNITS[unit_index]
        return number + (" " if spaced else "") + unit
    if mode == 0x02:
        if number_end >= len(payload):
            raise ValueError("invalid number_with_unit slot payload")
        unit_len = payload[number_end]
        unit_start = number_end + 1
        unit_end = unit_start + unit_len
        if unit_end > len(payload):
            raise ValueError("invalid number_with_unit slot payload")
        unit = payload[unit_start:unit_end].decode("utf-8")
        return number + (" " if spaced else "") + unit
    raise ValueError("invalid number_with_unit slot payload")


def _encode_varint(value: int) -> bytes:
    if value < 0:
        raise ValueError("varint must be non-negative")
    parts: list[int] = []
    current = value
    while True:
        byte = current & 0x7F
        current >>= 7
        if current:
            parts.append(byte | 0x80)
        else:
            parts.append(byte)
            break
    return bytes(parts)


def _decode_varint(data: bytes, offset: int) -> tuple[int, int]:
    value = 0
    shift = 0
    cursor = offset
    while cursor < len(data):
        byte = data[cursor]
        cursor += 1
        value |= (byte & 0x7F) << shift
        if byte < 0x80:
            return value, cursor
        shift += 7
    raise ValueError("invalid varint")


def _default_slot_fields(template_kind: str, skeleton: str, count: int) -> tuple[str, ...]:
    if template_kind == _TEMPLATE_KIND_KEY_VALUE and count == 1:
        key = skeleton.rstrip(" :=")
        return (_normalize_field_key(key),)
    return tuple("" for _ in range(count))


def _normalize_field_key(value: str) -> str:
    return value.strip().lower().replace(" ", "_").replace("-", "_")[:96]


def _looks_like_identifier(value: str) -> bool:
    return bool(_IDENTIFIER_RE.fullmatch(value))


def _looks_like_version(value: str) -> bool:
    return bool(_VERSION_RE.fullmatch(value))


def _looks_like_path_or_url(value: str) -> bool:
    return value.startswith(_URL_PREFIXES) or "://" in value or value.count("/") >= 2 or value.count("\\") >= 2


def _looks_like_number_with_unit(value: str) -> bool:
    return bool(_NUMBER_WITH_UNIT_RE.fullmatch(value))


def _escape_braces(text: str) -> str:
    return text.replace("{", "{{").replace("}", "}}")


def template_route_allowed(segment_kind: str) -> bool:
    return segment_kind in _TEMPLATE_ENABLED_KINDS
