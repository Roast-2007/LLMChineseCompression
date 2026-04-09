from __future__ import annotations

import re
from dataclasses import dataclass

from .online_manifest import AnalysisManifest

_PARAGRAPH_BREAK_RE = re.compile(r"(?:\r?\n){2,}")
_LINE_BREAK_RE = re.compile(r"\r?\n")
_STRUCTURED_LINE_KINDS = frozenset({"list", "table", "config"})


@dataclass(frozen=True)
class TextSegment:
    start: int
    end: int
    kind: str

    @property
    def char_count(self) -> int:
        return self.end - self.start


def split_text_segments(
    text: str,
    analysis: AnalysisManifest | None = None,
    max_chars: int = 260,
) -> tuple[TextSegment, ...]:
    if not text:
        return ()

    paragraph_ranges = _split_paragraphs(text)
    segments: list[TextSegment] = []
    for start, end in paragraph_ranges:
        block = text[start:end]
        if not block:
            continue
        if not block.strip():
            segments.append(TextSegment(start=start, end=end, kind="mixed"))
            continue
        kind = _classify_block(block, analysis, start, end)
        if kind in _STRUCTURED_LINE_KINDS and _has_multiple_lines(block):
            segments.extend(_split_structured_lines(text, start, end, kind))
            continue
        if kind == "prose" and (end - start) > max_chars:
            segments.extend(_split_long_prose(text, start, end, max_chars))
            continue
        segments.append(TextSegment(start=start, end=end, kind=kind))
    return tuple(segment for segment in segments if segment.char_count > 0)


def _split_paragraphs(text: str) -> list[tuple[int, int]]:
    parts: list[tuple[int, int]] = []
    cursor = 0
    for match in _PARAGRAPH_BREAK_RE.finditer(text):
        start, end = match.span()
        if cursor < start:
            parts.append((cursor, start))
        parts.append((start, end))
        cursor = end
    if cursor < len(text):
        parts.append((cursor, len(text)))
    return parts or [(0, len(text))]


def _split_structured_lines(
    text: str,
    start: int,
    end: int,
    kind: str,
) -> list[TextSegment]:
    segments: list[TextSegment] = []
    cursor = start
    block = text[start:end]
    for match in _LINE_BREAK_RE.finditer(block):
        line_end = start + match.start()
        sep_end = start + match.end()
        if cursor < line_end:
            segments.append(TextSegment(start=cursor, end=line_end, kind=kind))
        segments.append(TextSegment(start=line_end, end=sep_end, kind="mixed"))
        cursor = sep_end
    if cursor < end:
        segments.append(TextSegment(start=cursor, end=end, kind=kind))
    return segments


def _split_long_prose(
    text: str,
    start: int,
    end: int,
    max_chars: int,
) -> list[TextSegment]:
    result: list[TextSegment] = []
    cursor = start
    while cursor < end:
        chunk_end = min(cursor + max_chars, end)
        if chunk_end < end:
            split_at = _find_sentence_boundary(text, cursor, chunk_end, end)
            if split_at > cursor:
                chunk_end = split_at
        result.append(TextSegment(start=cursor, end=chunk_end, kind="prose"))
        cursor = chunk_end
    return result


def _find_sentence_boundary(text: str, start: int, soft_end: int, hard_end: int) -> int:
    best = -1
    for idx in range(soft_end, min(hard_end, soft_end + 80)):
        if text[idx - 1:idx] in "。！？!?；;,.，":
            best = idx
            break
    return best if best > start else soft_end


def _classify_block(
    block: str,
    analysis: AnalysisManifest | None,
    start: int,
    end: int,
) -> str:
    stripped = block.strip()
    if not stripped:
        return "mixed"

    lines = [line for line in stripped.splitlines() if line.strip()]
    if len(lines) >= 2 and sum(_is_list_like(line) for line in lines) >= max(2, len(lines) // 2):
        return "list"
    if len(lines) >= 2 and sum(_is_table_like(line) for line in lines) >= max(2, len(lines) // 2):
        return "table"
    if _looks_like_code(stripped):
        return "code"
    if _looks_like_config(stripped):
        return "config"
    if _digit_ratio(stripped) >= 0.30:
        return "numeric"
    if analysis and _dominant_lang(analysis, start, end) == "num":
        return "numeric"
    return "prose"


def _has_multiple_lines(text: str) -> bool:
    return bool(_LINE_BREAK_RE.search(text))


def _is_list_like(line: str) -> bool:
    line = line.lstrip()
    return line.startswith(("- ", "* ", "• ")) or bool(re.match(r"\d+[.)]\s+", line))


def _is_table_like(line: str) -> bool:
    stripped = line.strip()
    if stripped.startswith("|") and stripped.endswith("|") and stripped.count("|") >= 2:
        return True
    cells = [item for item in stripped.split("\t") if item]
    return len(cells) >= 2


def _looks_like_code(text: str) -> bool:
    code_tokens = ("{", "}", "();", "def ", "class ", "=>", "::", "[]", "()")
    return any(token in text for token in code_tokens)


def _looks_like_config(text: str) -> bool:
    return (
        text.startswith("{")
        or text.startswith("[")
        or text.count(":") >= 3
        or text.count("=") >= 3
    )


def _digit_ratio(text: str) -> float:
    if not text:
        return 0.0
    digits = sum(ch.isdigit() for ch in text)
    return digits / len(text)


def _dominant_lang(analysis: AnalysisManifest, start: int, end: int) -> str:
    counts: dict[str, int] = {}
    for hint in analysis.language_segments:
        overlap_start = max(start, hint.start)
        overlap_end = min(end, hint.end)
        if overlap_start >= overlap_end:
            continue
        counts[hint.lang] = counts.get(hint.lang, 0) + (overlap_end - overlap_start)
    if not counts:
        return "mixed"
    return max(counts.items(), key=lambda item: (item[1], item[0]))[0]
