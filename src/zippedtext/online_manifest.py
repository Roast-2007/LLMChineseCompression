from __future__ import annotations

import json
import struct
from dataclasses import dataclass

VALID_LANGS = frozenset({"zh", "en", "num", "mixed"})
ROUTE_LITERAL = "literal"
ROUTE_PHRASE = "phrase"
ROUTE_TEMPLATE = "template"
VALID_ROUTES = frozenset({ROUTE_LITERAL, ROUTE_PHRASE, ROUTE_TEMPLATE})
VALID_KINDS = frozenset({"prose", "list", "table", "code", "config", "numeric", "mixed"})
VALID_SLOT_TYPES = frozenset(
    {
        "string",
        "identifier",
        "version",
        "path_or_url",
        "enum",
        "number_with_unit",
    }
)
VALID_TEMPLATE_KINDS = frozenset({"key_value", "list_prefix", "table_row", "record"})

_KIND_TO_ID = {
    "prose": 0,
    "list": 1,
    "table": 2,
    "code": 3,
    "config": 4,
    "numeric": 5,
    "mixed": 6,
}
_ID_TO_KIND = {value: key for key, value in _KIND_TO_ID.items()}
_ROUTE_TO_ID = {
    ROUTE_LITERAL: 0,
    ROUTE_PHRASE: 1,
    ROUTE_TEMPLATE: 2,
}
_ID_TO_ROUTE = {value: key for key, value in _ROUTE_TO_ID.items()}
_SEGMENT_RECORD_MAGIC = b"SRB1"
_SEGMENT_RECORD_FMT = "<BBHIIIIIi"
_SEGMENT_RECORD_SIZE = struct.calcsize(_SEGMENT_RECORD_FMT)
_STATS_MAGIC = b"STS1"
_STATS_HEADER_FMT = "<4sIIIIIIIIIIIIiI"
_STATS_HEADER_SIZE = struct.calcsize(_STATS_HEADER_FMT)


@dataclass(frozen=True)
class LanguageHint:
    start: int
    end: int
    lang: str

    def to_dict(self) -> dict[str, int | str]:
        return {"start": self.start, "end": self.end, "lang": self.lang}


@dataclass(frozen=True)
class BlockFamilyHint:
    kind: str
    start: int
    end: int
    family: str

    def to_dict(self) -> dict[str, int | str]:
        return {
            "kind": self.kind,
            "start": self.start,
            "end": self.end,
            "family": self.family,
        }


@dataclass(frozen=True)
class FieldSchemaHint:
    field: str
    slot_type: str
    enum_candidates: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, object]:
        payload: dict[str, object] = {
            "field": self.field,
            "slot_type": self.slot_type,
        }
        if self.enum_candidates:
            payload["enum_candidates"] = list(self.enum_candidates)
        return payload


@dataclass(frozen=True)
class SlotHint:
    template_kind: str
    slot_index: int
    slot_type: str
    field: str = ""
    enum_candidates: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, object]:
        payload: dict[str, object] = {
            "template_kind": self.template_kind,
            "slot_index": self.slot_index,
            "slot_type": self.slot_type,
        }
        if self.field:
            payload["field"] = self.field
        if self.enum_candidates:
            payload["enum_candidates"] = list(self.enum_candidates)
        return payload


@dataclass(frozen=True)
class EnumCandidateSet:
    field: str
    values: tuple[str, ...]

    def to_dict(self) -> dict[str, object]:
        return {
            "field": self.field,
            "values": list(self.values),
        }


@dataclass(frozen=True)
class AnalysisManifest:
    char_frequencies: tuple[tuple[str, float], ...] = ()
    top_bigrams: tuple[tuple[str, float], ...] = ()
    phrase_dictionary: tuple[str, ...] = ()
    language_segments: tuple[LanguageHint, ...] = ()
    template_hints: tuple[str, ...] = ()
    document_family: str = ""
    block_families: tuple[BlockFamilyHint, ...] = ()
    field_schemas: tuple[FieldSchemaHint, ...] = ()
    slot_hints: tuple[SlotHint, ...] = ()
    enum_candidates: tuple[EnumCandidateSet, ...] = ()

    @classmethod
    def from_api_payload(
        cls,
        payload: dict | None,
        text_len: int,
    ) -> "AnalysisManifest":
        if not payload:
            return cls()

        char_frequencies = _normalize_char_frequencies(payload.get("char_frequencies"))
        top_bigrams = _normalize_bigrams(payload.get("top_bigrams"))
        phrase_dictionary = _normalize_phrases(payload.get("phrase_dictionary"))
        language_segments = _normalize_language_segments(
            payload.get("language_segments"),
            text_len,
        )
        template_hints = _normalize_template_hints(payload.get("template_hints"))
        document_family = _normalize_document_family(payload.get("document_family"))
        block_families = _normalize_block_families(payload.get("block_families"), text_len)
        field_schemas = _normalize_field_schemas(payload.get("field_schemas"))
        slot_hints = _normalize_slot_hints(payload.get("slot_hints"))
        enum_candidates = _normalize_enum_candidates(payload.get("enum_candidates"))
        return cls(
            char_frequencies=char_frequencies,
            top_bigrams=top_bigrams,
            phrase_dictionary=phrase_dictionary,
            language_segments=language_segments,
            template_hints=template_hints,
            document_family=document_family,
            block_families=block_families,
            field_schemas=field_schemas,
            slot_hints=slot_hints,
            enum_candidates=enum_candidates,
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "char_frequencies": {ch: freq for ch, freq in self.char_frequencies},
            "top_bigrams": [list(item) for item in self.top_bigrams],
            "phrase_dictionary": list(self.phrase_dictionary),
            "language_segments": [hint.to_dict() for hint in self.language_segments],
            "template_hints": list(self.template_hints),
            "document_family": self.document_family,
            "block_families": [hint.to_dict() for hint in self.block_families],
            "field_schemas": [hint.to_dict() for hint in self.field_schemas],
            "slot_hints": [hint.to_dict() for hint in self.slot_hints],
            "enum_candidates": [hint.to_dict() for hint in self.enum_candidates],
        }

    def serialize(self) -> bytes:
        return json.dumps(
            self.to_dict(),
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")

    @classmethod
    def deserialize(cls, data: bytes, text_len: int) -> "AnalysisManifest":
        if not data:
            return cls()
        payload = json.loads(data.decode("utf-8"))
        return cls.from_api_payload(payload, text_len)

    def to_prior_map(self, limit: int = 256) -> dict[str, float]:
        limited = self.char_frequencies[:limit]
        total = sum(freq for _, freq in limited)
        if total <= 0:
            return {}
        return {ch: freq / total for ch, freq in limited}

    def for_storage(
        self,
        *,
        max_char_frequencies: int = 96,
        max_bigrams: int = 48,
        max_phrases: int = 64,
        max_language_segments: int = 32,
        max_block_families: int = 32,
        max_field_schemas: int = 32,
        max_slot_hints: int = 64,
        max_enum_candidates: int = 32,
    ) -> "AnalysisManifest":
        return AnalysisManifest(
            char_frequencies=self.char_frequencies[:max_char_frequencies],
            top_bigrams=self.top_bigrams[:max_bigrams],
            phrase_dictionary=self.phrase_dictionary[:max_phrases],
            language_segments=self.language_segments[:max_language_segments],
            template_hints=self.template_hints,
            document_family=self.document_family,
            block_families=self.block_families[:max_block_families],
            field_schemas=self.field_schemas[:max_field_schemas],
            slot_hints=self.slot_hints[:max_slot_hints],
            enum_candidates=self.enum_candidates[:max_enum_candidates],
        )

    def slot_hint_for(
        self,
        template_kind: str,
        slot_index: int,
        field: str = "",
    ) -> SlotHint | None:
        normalized_field = _normalize_field_name(field)
        fallback: SlotHint | None = None
        for hint in self.slot_hints:
            if hint.template_kind != template_kind or hint.slot_index != slot_index:
                continue
            if normalized_field and hint.field == normalized_field:
                return hint
            if not fallback:
                fallback = hint
        return fallback

    def field_schema_for(self, field: str) -> FieldSchemaHint | None:
        normalized = _normalize_field_name(field)
        if not normalized:
            return None
        for schema in self.field_schemas:
            if schema.field == normalized:
                return schema
        return None

    def enum_candidates_for(self, field: str) -> tuple[str, ...]:
        schema = self.field_schema_for(field)
        if schema and schema.enum_candidates:
            return schema.enum_candidates
        normalized = _normalize_field_name(field)
        if not normalized:
            return ()
        for candidate in self.enum_candidates:
            if candidate.field == normalized:
                return candidate.values
        return ()


@dataclass(frozen=True)
class SegmentRecord:
    kind: str
    route: str
    char_count: int
    payload_len: int
    original_bytes: int
    encoded_bytes: int
    residual_bytes: int = 0
    estimated_gain_bytes: int = 0

    def to_dict(self) -> dict[str, int | str]:
        return {
            "kind": self.kind,
            "route": self.route,
            "char_count": self.char_count,
            "payload_len": self.payload_len,
            "original_bytes": self.original_bytes,
            "encoded_bytes": self.encoded_bytes,
            "residual_bytes": self.residual_bytes,
            "estimated_gain_bytes": self.estimated_gain_bytes,
        }

    @classmethod
    def from_dict(cls, payload: dict) -> "SegmentRecord":
        kind = str(payload.get("kind", "prose"))
        route = str(payload.get("route", ROUTE_LITERAL))
        if kind not in VALID_KINDS:
            kind = "mixed"
        if route not in VALID_ROUTES:
            route = ROUTE_LITERAL
        return cls(
            kind=kind,
            route=route,
            char_count=max(int(payload.get("char_count", 0)), 0),
            payload_len=max(int(payload.get("payload_len", 0)), 0),
            original_bytes=max(int(payload.get("original_bytes", 0)), 0),
            encoded_bytes=max(int(payload.get("encoded_bytes", 0)), 0),
            residual_bytes=max(int(payload.get("residual_bytes", 0)), 0),
            estimated_gain_bytes=int(payload.get("estimated_gain_bytes", 0)),
        )


@dataclass(frozen=True)
class StructuredOnlineStats:
    segment_count: int = 0
    phrase_count: int = 0
    template_count: int = 0
    template_hit_count: int = 0
    typed_slot_count: int = 0
    typed_template_count: int = 0
    residual_bytes: int = 0
    analysis_bytes: int = 0
    dictionary_bytes: int = 0
    templates_bytes: int = 0
    segments_bytes: int = 0
    payload_bytes: int = 0
    route_counts: tuple[tuple[str, int], ...] = ()
    reason_counts: tuple[tuple[str, int], ...] = ()
    template_family_counts: tuple[tuple[str, int], ...] = ()
    estimated_gain_bytes: int = 0
    literal_payload_bytes: int = 0
    fallback_reason: str = ""

    @property
    def side_info_bytes(self) -> int:
        return (
            self.analysis_bytes
            + self.dictionary_bytes
            + self.templates_bytes
            + self.segments_bytes
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "segment_count": self.segment_count,
            "phrase_count": self.phrase_count,
            "template_count": self.template_count,
            "template_hit_count": self.template_hit_count,
            "typed_slot_count": self.typed_slot_count,
            "typed_template_count": self.typed_template_count,
            "residual_bytes": self.residual_bytes,
            "analysis_bytes": self.analysis_bytes,
            "dictionary_bytes": self.dictionary_bytes,
            "templates_bytes": self.templates_bytes,
            "segments_bytes": self.segments_bytes,
            "payload_bytes": self.payload_bytes,
            "route_counts": [
                {"route": route, "count": count}
                for route, count in self.route_counts
            ],
            "reason_counts": [
                {"reason": reason, "count": count}
                for reason, count in self.reason_counts
            ],
            "template_family_counts": [
                {"family": family, "count": count}
                for family, count in self.template_family_counts
            ],
            "estimated_gain_bytes": self.estimated_gain_bytes,
            "literal_payload_bytes": self.literal_payload_bytes,
            "fallback_reason": self.fallback_reason,
        }

    def serialize(self) -> bytes:
        header = struct.pack(
            _STATS_HEADER_FMT,
            _STATS_MAGIC,
            self.segment_count,
            self.phrase_count,
            self.template_count,
            self.template_hit_count,
            self.residual_bytes,
            self.analysis_bytes,
            self.dictionary_bytes,
            self.templates_bytes,
            self.segments_bytes,
            self.payload_bytes,
            len(self.route_counts),
            len(self.reason_counts),
            self.estimated_gain_bytes,
            self.literal_payload_bytes,
        )
        parts = [header]
        parts.append(struct.pack("<II", self.typed_slot_count, self.typed_template_count))
        parts.append(struct.pack("<H", len(self.template_family_counts)))
        for family, count in self.template_family_counts:
            parts.append(_pack_string(family))
            parts.append(struct.pack("<I", max(count, 0)))
        for route, count in self.route_counts:
            route_id = _ROUTE_TO_ID.get(route, _ROUTE_TO_ID[ROUTE_LITERAL])
            parts.append(struct.pack("<BI", route_id, max(count, 0)))
        for reason, count in self.reason_counts:
            parts.append(_pack_string(reason))
            parts.append(struct.pack("<I", max(count, 0)))
        parts.append(_pack_string(self.fallback_reason))
        return b"".join(parts)

    @classmethod
    def deserialize(cls, data: bytes | object) -> "StructuredOnlineStats":
        payload_bytes = _coerce_section_bytes(data)
        if not payload_bytes:
            return cls()
        if payload_bytes.startswith(_STATS_MAGIC):
            return _deserialize_stats_binary(payload_bytes)
        payload = json.loads(payload_bytes.decode("utf-8"))
        route_counts_raw = payload.get("route_counts", [])
        route_counts = tuple(
            (str(item.get("route", ROUTE_LITERAL)), max(int(item.get("count", 0)), 0))
            for item in route_counts_raw
        )
        reason_counts_raw = payload.get("reason_counts", [])
        reason_counts = tuple(
            (str(item.get("reason", "")), max(int(item.get("count", 0)), 0))
            for item in reason_counts_raw
            if str(item.get("reason", ""))
        )
        template_family_counts_raw = payload.get("template_family_counts", [])
        template_family_counts = tuple(
            (str(item.get("family", "")), max(int(item.get("count", 0)), 0))
            for item in template_family_counts_raw
            if str(item.get("family", ""))
        )
        return cls(
            segment_count=max(int(payload.get("segment_count", 0)), 0),
            phrase_count=max(int(payload.get("phrase_count", 0)), 0),
            template_count=max(int(payload.get("template_count", 0)), 0),
            template_hit_count=max(int(payload.get("template_hit_count", 0)), 0),
            typed_slot_count=max(int(payload.get("typed_slot_count", 0)), 0),
            typed_template_count=max(int(payload.get("typed_template_count", 0)), 0),
            residual_bytes=max(int(payload.get("residual_bytes", 0)), 0),
            analysis_bytes=max(int(payload.get("analysis_bytes", 0)), 0),
            dictionary_bytes=max(int(payload.get("dictionary_bytes", 0)), 0),
            templates_bytes=max(int(payload.get("templates_bytes", 0)), 0),
            segments_bytes=max(int(payload.get("segments_bytes", 0)), 0),
            payload_bytes=max(int(payload.get("payload_bytes", 0)), 0),
            route_counts=route_counts,
            reason_counts=reason_counts,
            template_family_counts=template_family_counts,
            estimated_gain_bytes=int(payload.get("estimated_gain_bytes", 0)),
            literal_payload_bytes=max(int(payload.get("literal_payload_bytes", 0)), 0),
            fallback_reason=str(payload.get("fallback_reason", "")),
        )


def serialize_segment_records(records: tuple[SegmentRecord, ...]) -> bytes:
    parts = [_SEGMENT_RECORD_MAGIC, struct.pack("<H", len(records))]
    for record in records:
        kind_id = _KIND_TO_ID.get(record.kind, _KIND_TO_ID["mixed"])
        route_id = _ROUTE_TO_ID.get(record.route, _ROUTE_TO_ID[ROUTE_LITERAL])
        parts.append(
            struct.pack(
                _SEGMENT_RECORD_FMT,
                kind_id,
                route_id,
                0,
                record.char_count,
                record.payload_len,
                record.original_bytes,
                record.encoded_bytes,
                record.residual_bytes,
                record.estimated_gain_bytes,
            )
        )
    return b"".join(parts)


def deserialize_segment_records(data: bytes | object) -> tuple[SegmentRecord, ...]:
    payload_bytes = _coerce_section_bytes(data)
    if not payload_bytes:
        return ()
    if payload_bytes.startswith(_SEGMENT_RECORD_MAGIC):
        return _deserialize_segment_records_binary(payload_bytes)
    payload = json.loads(payload_bytes.decode("utf-8"))
    if not isinstance(payload, list):
        raise ValueError("segment records must be a list")
    return tuple(SegmentRecord.from_dict(item) for item in payload)


def _deserialize_segment_records_binary(data: bytes) -> tuple[SegmentRecord, ...]:
    if len(data) < len(_SEGMENT_RECORD_MAGIC) + 2:
        raise ValueError("invalid segment records: truncated header")
    count = struct.unpack("<H", data[len(_SEGMENT_RECORD_MAGIC):len(_SEGMENT_RECORD_MAGIC) + 2])[0]
    offset = len(_SEGMENT_RECORD_MAGIC) + 2
    records: list[SegmentRecord] = []
    for _ in range(count):
        end = offset + _SEGMENT_RECORD_SIZE
        if end > len(data):
            raise ValueError("invalid segment records: truncated record")
        (
            kind_id,
            route_id,
            _reserved,
            char_count,
            payload_len,
            original_bytes,
            encoded_bytes,
            residual_bytes,
            estimated_gain_bytes,
        ) = struct.unpack(_SEGMENT_RECORD_FMT, data[offset:end])
        records.append(
            SegmentRecord(
                kind=_ID_TO_KIND.get(kind_id, "mixed"),
                route=_ID_TO_ROUTE.get(route_id, ROUTE_LITERAL),
                char_count=char_count,
                payload_len=payload_len,
                original_bytes=original_bytes,
                encoded_bytes=encoded_bytes,
                residual_bytes=residual_bytes,
                estimated_gain_bytes=estimated_gain_bytes,
            )
        )
        offset = end
    if offset != len(data):
        raise ValueError("invalid segment records: trailing bytes")
    return tuple(records)


def _deserialize_stats_binary(data: bytes) -> StructuredOnlineStats:
    try:
        return _deserialize_stats_binary_extended(data)
    except ValueError:
        return _deserialize_stats_binary_legacy(data)


def _deserialize_stats_binary_extended(data: bytes) -> StructuredOnlineStats:
    if len(data) < _STATS_HEADER_SIZE + 10:
        raise ValueError("invalid stats: truncated header")
    (
        magic,
        segment_count,
        phrase_count,
        template_count,
        template_hit_count,
        residual_bytes,
        analysis_bytes,
        dictionary_bytes,
        templates_bytes,
        segments_bytes,
        payload_bytes,
        route_count_len,
        reason_count_len,
        estimated_gain_bytes,
        literal_payload_bytes,
    ) = struct.unpack(_STATS_HEADER_FMT, data[:_STATS_HEADER_SIZE])
    if magic != _STATS_MAGIC:
        raise ValueError("invalid stats: bad magic")
    offset = _STATS_HEADER_SIZE
    typed_slot_count, typed_template_count = struct.unpack("<II", data[offset:offset + 8])
    offset += 8
    family_count_len = struct.unpack("<H", data[offset:offset + 2])[0]
    offset += 2
    template_family_counts: list[tuple[str, int]] = []
    for _ in range(family_count_len):
        family, offset = _unpack_string(data, offset, "invalid stats: truncated template family")
        end = offset + 4
        if end > len(data):
            raise ValueError("invalid stats: truncated template family count")
        count = struct.unpack("<I", data[offset:end])[0]
        offset = end
        if family:
            template_family_counts.append((family, count))
    route_counts: list[tuple[str, int]] = []
    for _ in range(route_count_len):
        end = offset + 5
        if end > len(data):
            raise ValueError("invalid stats: truncated route count")
        route_id, count = struct.unpack("<BI", data[offset:end])
        route_counts.append((_ID_TO_ROUTE.get(route_id, ROUTE_LITERAL), count))
        offset = end
    reason_counts: list[tuple[str, int]] = []
    for _ in range(reason_count_len):
        reason, offset = _unpack_string(data, offset, "invalid stats: truncated reason")
        end = offset + 4
        if end > len(data):
            raise ValueError("invalid stats: truncated reason count")
        count = struct.unpack("<I", data[offset:end])[0]
        offset = end
        if reason:
            reason_counts.append((reason, count))
    fallback_reason, offset = _unpack_string(
        data,
        offset,
        "invalid stats: truncated fallback reason",
    )
    if offset != len(data):
        raise ValueError("invalid stats: trailing bytes")
    return StructuredOnlineStats(
        segment_count=segment_count,
        phrase_count=phrase_count,
        template_count=template_count,
        template_hit_count=template_hit_count,
        typed_slot_count=typed_slot_count,
        typed_template_count=typed_template_count,
        residual_bytes=residual_bytes,
        analysis_bytes=analysis_bytes,
        dictionary_bytes=dictionary_bytes,
        templates_bytes=templates_bytes,
        segments_bytes=segments_bytes,
        payload_bytes=payload_bytes,
        route_counts=tuple(route_counts),
        reason_counts=tuple(reason_counts),
        template_family_counts=tuple(template_family_counts),
        estimated_gain_bytes=estimated_gain_bytes,
        literal_payload_bytes=literal_payload_bytes,
        fallback_reason=fallback_reason,
    )


def _deserialize_stats_binary_legacy(data: bytes) -> StructuredOnlineStats:
    if len(data) < _STATS_HEADER_SIZE:
        raise ValueError("invalid stats: truncated header")
    (
        magic,
        segment_count,
        phrase_count,
        template_count,
        template_hit_count,
        residual_bytes,
        analysis_bytes,
        dictionary_bytes,
        templates_bytes,
        segments_bytes,
        payload_bytes,
        route_count_len,
        reason_count_len,
        estimated_gain_bytes,
        literal_payload_bytes,
    ) = struct.unpack(_STATS_HEADER_FMT, data[:_STATS_HEADER_SIZE])
    if magic != _STATS_MAGIC:
        raise ValueError("invalid stats: bad magic")
    offset = _STATS_HEADER_SIZE
    route_counts: list[tuple[str, int]] = []
    for _ in range(route_count_len):
        end = offset + 5
        if end > len(data):
            raise ValueError("invalid stats: truncated route count")
        route_id, count = struct.unpack("<BI", data[offset:end])
        route_counts.append((_ID_TO_ROUTE.get(route_id, ROUTE_LITERAL), count))
        offset = end
    reason_counts: list[tuple[str, int]] = []
    for _ in range(reason_count_len):
        reason, offset = _unpack_string(data, offset, "invalid stats: truncated reason")
        end = offset + 4
        if end > len(data):
            raise ValueError("invalid stats: truncated reason count")
        count = struct.unpack("<I", data[offset:end])[0]
        offset = end
        if reason:
            reason_counts.append((reason, count))
    fallback_reason, offset = _unpack_string(
        data,
        offset,
        "invalid stats: truncated fallback reason",
    )
    if offset != len(data):
        raise ValueError("invalid stats: trailing bytes")
    return StructuredOnlineStats(
        segment_count=segment_count,
        phrase_count=phrase_count,
        template_count=template_count,
        template_hit_count=template_hit_count,
        residual_bytes=residual_bytes,
        analysis_bytes=analysis_bytes,
        dictionary_bytes=dictionary_bytes,
        templates_bytes=templates_bytes,
        segments_bytes=segments_bytes,
        payload_bytes=payload_bytes,
        route_counts=tuple(route_counts),
        reason_counts=tuple(reason_counts),
        estimated_gain_bytes=estimated_gain_bytes,
        literal_payload_bytes=literal_payload_bytes,
        fallback_reason=fallback_reason,
    )


def _pack_string(value: str) -> bytes:
    encoded = value.encode("utf-8")
    return struct.pack("<H", len(encoded)) + encoded


def _unpack_string(data: bytes, offset: int, error_message: str) -> tuple[str, int]:
    if offset + 2 > len(data):
        raise ValueError(error_message)
    length = struct.unpack("<H", data[offset:offset + 2])[0]
    offset += 2
    end = offset + length
    if end > len(data):
        raise ValueError(error_message)
    return data[offset:end].decode("utf-8"), end


def _coerce_section_bytes(data: bytes | object) -> bytes:
    if isinstance(data, bytes):
        return data
    if hasattr(data, "data"):
        payload = getattr(data, "data")
        if isinstance(payload, bytes):
            return payload
    if data is None:
        return b""
    raise TypeError("expected bytes-like section payload")


def _normalize_document_family(value: object) -> str:
    if not isinstance(value, str):
        return ""
    normalized = value.strip().lower().replace(" ", "_")
    return normalized[:64]


def _normalize_char_frequencies(value: object) -> tuple[tuple[str, float], ...]:
    if not isinstance(value, dict):
        return ()
    items: list[tuple[str, float]] = []
    for ch, freq in value.items():
        if not isinstance(ch, str) or len(ch) != 1:
            continue
        try:
            normalized = float(freq)
        except (TypeError, ValueError):
            continue
        if normalized <= 0:
            continue
        items.append((ch, normalized))
    items.sort(key=lambda item: (-item[1], item[0]))
    return tuple(items[:512])


def _normalize_bigrams(value: object) -> tuple[tuple[str, float], ...]:
    if not isinstance(value, list):
        return ()
    result: list[tuple[str, float]] = []
    for item in value:
        if not isinstance(item, (list, tuple)) or len(item) != 2:
            continue
        text, freq = item
        if not isinstance(text, str) or len(text) < 2:
            continue
        try:
            normalized = float(freq)
        except (TypeError, ValueError):
            continue
        if normalized <= 0:
            continue
        result.append((text, normalized))
    result.sort(key=lambda item: (-item[1], item[0]))
    return tuple(result[:256])


def _normalize_phrases(value: object) -> tuple[str, ...]:
    phrases: list[str] = []
    if isinstance(value, list):
        for item in value:
            if isinstance(item, (list, tuple)) and item:
                phrase = str(item[0])
            else:
                phrase = str(item)
            phrase = phrase.strip()
            if len(phrase) < 2 or len(phrase) > 64:
                continue
            phrases.append(phrase)
    ordered: list[str] = []
    seen: set[str] = set()
    for phrase in phrases:
        if phrase in seen:
            continue
        seen.add(phrase)
        ordered.append(phrase)
    return tuple(ordered[:256])


def _normalize_language_segments(
    value: object,
    text_len: int,
) -> tuple[LanguageHint, ...]:
    if not isinstance(value, list):
        return ()
    hints: list[LanguageHint] = []
    for item in value:
        if not isinstance(item, dict):
            continue
        try:
            start = max(int(item.get("start", 0)), 0)
            end = min(int(item.get("end", 0)), text_len)
        except (TypeError, ValueError):
            continue
        lang = str(item.get("lang", "mixed"))
        if start >= end:
            continue
        if lang not in VALID_LANGS:
            lang = "mixed"
        hints.append(LanguageHint(start=start, end=end, lang=lang))
    hints.sort(key=lambda hint: (hint.start, hint.end, hint.lang))
    result: list[LanguageHint] = []
    cursor = -1
    for hint in hints:
        if hint.start < cursor:
            continue
        result.append(hint)
        cursor = hint.end
    return tuple(result)


def _normalize_template_hints(value: object) -> tuple[str, ...]:
    if not isinstance(value, list):
        return ()
    ordered: list[str] = []
    seen: set[str] = set()
    for item in value:
        hint = str(item).strip().lower()
        if hint not in VALID_TEMPLATE_KINDS or hint in seen:
            continue
        seen.add(hint)
        ordered.append(hint)
    return tuple(ordered[:32])


def _normalize_block_families(
    value: object,
    text_len: int,
) -> tuple[BlockFamilyHint, ...]:
    if not isinstance(value, list):
        return ()
    hints: list[BlockFamilyHint] = []
    for item in value:
        if not isinstance(item, dict):
            continue
        try:
            start = max(int(item.get("start", 0)), 0)
            end = min(int(item.get("end", 0)), text_len)
        except (TypeError, ValueError):
            continue
        if start >= end:
            continue
        kind = str(item.get("kind", "mixed")).strip().lower()
        if kind not in VALID_KINDS:
            kind = "mixed"
        family = _normalize_document_family(item.get("family"))
        if not family:
            continue
        hints.append(BlockFamilyHint(kind=kind, start=start, end=end, family=family))
    hints.sort(key=lambda hint: (hint.start, hint.end, hint.kind, hint.family))
    return tuple(hints[:128])


def _normalize_field_schemas(value: object) -> tuple[FieldSchemaHint, ...]:
    if not isinstance(value, list):
        return ()
    hints: list[FieldSchemaHint] = []
    seen: set[tuple[str, str]] = set()
    for item in value:
        if not isinstance(item, dict):
            continue
        field = _normalize_field_name(item.get("field"))
        slot_type = _normalize_slot_type(item.get("slot_type"))
        enum_candidates = _normalize_string_list(item.get("enum_candidates"), limit=32)
        if not field:
            continue
        if slot_type == "string" and enum_candidates:
            slot_type = "enum"
        key = (field, slot_type)
        if key in seen:
            continue
        seen.add(key)
        hints.append(
            FieldSchemaHint(
                field=field,
                slot_type=slot_type,
                enum_candidates=enum_candidates,
            )
        )
    return tuple(hints[:128])


def _normalize_slot_hints(value: object) -> tuple[SlotHint, ...]:
    if not isinstance(value, list):
        return ()
    hints: list[SlotHint] = []
    seen: set[tuple[str, int, str]] = set()
    for item in value:
        if not isinstance(item, dict):
            continue
        template_kind = str(item.get("template_kind", "")).strip().lower()
        if template_kind not in VALID_TEMPLATE_KINDS:
            continue
        try:
            slot_index = int(item.get("slot_index", -1))
        except (TypeError, ValueError):
            continue
        if slot_index < 0:
            continue
        field = _normalize_field_name(item.get("field"))
        slot_type = _normalize_slot_type(item.get("slot_type"))
        enum_candidates = _normalize_string_list(item.get("enum_candidates"), limit=32)
        if slot_type == "string" and enum_candidates:
            slot_type = "enum"
        key = (template_kind, slot_index, field)
        if key in seen:
            continue
        seen.add(key)
        hints.append(
            SlotHint(
                template_kind=template_kind,
                slot_index=slot_index,
                slot_type=slot_type,
                field=field,
                enum_candidates=enum_candidates,
            )
        )
    return tuple(hints[:256])


def _normalize_enum_candidates(value: object) -> tuple[EnumCandidateSet, ...]:
    entries: list[EnumCandidateSet] = []
    seen: set[str] = set()
    if isinstance(value, dict):
        iterable = [
            {"field": field, "values": values}
            for field, values in value.items()
        ]
    elif isinstance(value, list):
        iterable = value
    else:
        return ()
    for item in iterable:
        if not isinstance(item, dict):
            continue
        field = _normalize_field_name(item.get("field"))
        values = _normalize_string_list(item.get("values", item.get("enum_candidates")), limit=64)
        if not field or not values or field in seen:
            continue
        seen.add(field)
        entries.append(EnumCandidateSet(field=field, values=values))
    return tuple(entries[:128])


def _normalize_field_name(value: object) -> str:
    if not isinstance(value, str):
        return ""
    normalized = value.strip().lower().replace(" ", "_")
    return normalized[:96]


def _normalize_slot_type(value: object) -> str:
    if not isinstance(value, str):
        return "string"
    normalized = value.strip().lower()
    return normalized if normalized in VALID_SLOT_TYPES else "string"


def _normalize_string_list(value: object, *, limit: int) -> tuple[str, ...]:
    if not isinstance(value, list):
        return ()
    ordered: list[str] = []
    seen: set[str] = set()
    for item in value:
        candidate = str(item).strip()
        if not candidate or candidate in seen:
            continue
        seen.add(candidate)
        ordered.append(candidate)
    return tuple(ordered[:limit])
