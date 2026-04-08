from __future__ import annotations

import json
from dataclasses import dataclass

VALID_LANGS = frozenset({"zh", "en", "num", "mixed"})
ROUTE_LITERAL = "literal"
ROUTE_PHRASE = "phrase"
VALID_ROUTES = frozenset({ROUTE_LITERAL, ROUTE_PHRASE})
VALID_KINDS = frozenset({"prose", "list", "table", "code", "config", "numeric", "mixed"})


@dataclass(frozen=True)
class LanguageHint:
    start: int
    end: int
    lang: str

    def to_dict(self) -> dict[str, int | str]:
        return {"start": self.start, "end": self.end, "lang": self.lang}


@dataclass(frozen=True)
class AnalysisManifest:
    char_frequencies: tuple[tuple[str, float], ...] = ()
    top_bigrams: tuple[tuple[str, float], ...] = ()
    phrase_dictionary: tuple[str, ...] = ()
    language_segments: tuple[LanguageHint, ...] = ()

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
        return cls(
            char_frequencies=char_frequencies,
            top_bigrams=top_bigrams,
            phrase_dictionary=phrase_dictionary,
            language_segments=language_segments,
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "char_frequencies": {
                ch: freq
                for ch, freq in self.char_frequencies
            },
            "top_bigrams": [list(item) for item in self.top_bigrams],
            "phrase_dictionary": list(self.phrase_dictionary),
            "language_segments": [hint.to_dict() for hint in self.language_segments],
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


@dataclass(frozen=True)
class SegmentRecord:
    kind: str
    route: str
    char_count: int
    payload_len: int
    original_bytes: int
    encoded_bytes: int

    def to_dict(self) -> dict[str, int | str]:
        return {
            "kind": self.kind,
            "route": self.route,
            "char_count": self.char_count,
            "payload_len": self.payload_len,
            "original_bytes": self.original_bytes,
            "encoded_bytes": self.encoded_bytes,
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
        )


@dataclass(frozen=True)
class StructuredOnlineStats:
    segment_count: int
    phrase_count: int
    analysis_bytes: int
    dictionary_bytes: int
    segments_bytes: int
    payload_bytes: int
    route_counts: tuple[tuple[str, int], ...]

    def to_dict(self) -> dict[str, object]:
        return {
            "segment_count": self.segment_count,
            "phrase_count": self.phrase_count,
            "analysis_bytes": self.analysis_bytes,
            "dictionary_bytes": self.dictionary_bytes,
            "segments_bytes": self.segments_bytes,
            "payload_bytes": self.payload_bytes,
            "route_counts": [
                {"route": route, "count": count}
                for route, count in self.route_counts
            ],
        }

    def serialize(self) -> bytes:
        return json.dumps(
            self.to_dict(),
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")

    @classmethod
    def deserialize(cls, data: bytes) -> "StructuredOnlineStats":
        if not data:
            return cls(0, 0, 0, 0, 0, 0, ())
        payload = json.loads(data.decode("utf-8"))
        route_counts_raw = payload.get("route_counts", [])
        route_counts = tuple(
            (str(item.get("route", ROUTE_LITERAL)), max(int(item.get("count", 0)), 0))
            for item in route_counts_raw
        )
        return cls(
            segment_count=max(int(payload.get("segment_count", 0)), 0),
            phrase_count=max(int(payload.get("phrase_count", 0)), 0),
            analysis_bytes=max(int(payload.get("analysis_bytes", 0)), 0),
            dictionary_bytes=max(int(payload.get("dictionary_bytes", 0)), 0),
            segments_bytes=max(int(payload.get("segments_bytes", 0)), 0),
            payload_bytes=max(int(payload.get("payload_bytes", 0)), 0),
            route_counts=route_counts,
        )


def serialize_segment_records(records: tuple[SegmentRecord, ...]) -> bytes:
    return json.dumps(
        [record.to_dict() for record in records],
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")


def deserialize_segment_records(data: bytes) -> tuple[SegmentRecord, ...]:
    if not data:
        return ()
    payload = json.loads(data.decode("utf-8"))
    if not isinstance(payload, list):
        raise ValueError("segment records must be a list")
    return tuple(SegmentRecord.from_dict(item) for item in payload)


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
            if len(phrase) < 2 or len(phrase) > 32:
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
