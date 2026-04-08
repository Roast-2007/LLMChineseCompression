from __future__ import annotations

from dataclasses import dataclass

from .online_manifest import AnalysisManifest


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
        kind = _classify_block(block, analysis)
        if kind == "prose" and (end - start) > max_chars:
            segments.extend(_split_long_prose(text, start, end, max_chars))
            continue
        segments.append(TextSegment(start=start, end=end, kind=kind))
    return tuple(segments)


def _split_paragraphs(text: str) -> list[tuple[int, int]]:
    parts: list[tuple[int, int]] = []
    start = 0
    pos = 0
    while pos < len(text):
        if text.startswith("\n\n", pos):
            if start < pos:
                parts.append((start, pos))
            pos += 2
            start = pos
            continue
        pos += 1
    if start < len(text):
        parts.append((start, len(text)))
    return parts or [(0, len(text))]


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
    if analysis and _dominant_lang(analysis, block) == "num":
        return "numeric"
    return "prose"


def _is_list_like(line: str) -> bool:
    line = line.lstrip()
    return line.startswith(("- ", "* ", "• ")) or (
        len(line) > 2 and line[0].isdigit() and line[1] in ".)"
    )


def _is_table_like(line: str) -> bool:
    return line.count("|") >= 2 or line.count("\t") >= 1


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


def _dominant_lang(analysis: AnalysisManifest, block: str) -> str:
    counts: dict[str, int] = {}
    for hint in analysis.language_segments:
        counts[hint.lang] = counts.get(hint.lang, 0) + (hint.end - hint.start)
    if not counts:
        return "mixed"
    return max(counts.items(), key=lambda item: (item[1], item[0]))[0]
