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


@dataclass(frozen=True)
class RecordGroup:
    """A group of consecutive structured segments belonging to the same logical record."""
    kind: str
    segment_indices: tuple[int, ...]
    family: str
    text_span: tuple[int, int]  # (start_char, end_char) in original text

    @property
    def segment_count(self) -> int:
        return len(self.segment_indices)


def group_record_groups(
    segments: tuple[TextSegment, ...],
    text: str,
    analysis: AnalysisManifest | None = None,
) -> tuple[RecordGroup, ...]:
    """Cluster consecutive structured segments of the same kind into RecordGroups.

    Only segments with kind in _STRUCTURED_LINE_KINDS (list, table, config) are grouped.
    Segments separated by "mixed" separator segments (newlines) within the same block
    are still considered consecutive for grouping purposes.

    Only groups with 2+ segments are emitted; lone segments remain ungrouped.
    The family name is derived from analysis.block_families if available,
    otherwise inferred from the segment kind.
    """
    if len(segments) < 2:
        return ()

    groups: list[RecordGroup] = []
    current_run: list[int] = []  # indices of segments in the current run
    current_kind: str | None = None

    for i, seg in enumerate(segments):
        if seg.kind in _STRUCTURED_LINE_KINDS:
            if current_kind is None or seg.kind == current_kind:
                current_run.append(i)
                current_kind = seg.kind
            else:
                # Flush previous run
                if len(current_run) >= 2:
                    groups.append(_build_record_group(current_run, current_kind, segments, text, analysis))
                current_run = [i]
                current_kind = seg.kind
        else:
            # Non-structured segment (prose/mixed/code/numeric)
            # "mixed" separators between structured segments don't break the run
            if seg.kind == "mixed" and current_run:
                # Check if there's a structured segment after this mixed one
                # Peek ahead to see if next non-mixed segment matches
                continue  # Don't flush yet, wait for next structured segment
            else:
                # Different kind or non-mixed separator -> flush
                if len(current_run) >= 2:
                    groups.append(_build_record_group(current_run, current_kind, segments, text, analysis))
                current_run = []
                current_kind = None

    # Flush final run
    if len(current_run) >= 2:
        groups.append(_build_record_group(current_run, current_kind, segments, text, analysis))

    return tuple(groups)


def _build_record_group(
    indices: list[int],
    kind: str,
    segments: tuple[TextSegment, ...],
    text: str,
    analysis: AnalysisManifest | None,
) -> RecordGroup:
    """Build a RecordGroup from a run of segment indices."""
    first = segments[indices[0]]
    last = segments[indices[-1]]
    text_span = (first.start, last.end)

    # Derive family name from analysis.block_families if available
    family = _derive_family(analysis, text_span, kind)
    return RecordGroup(
        kind=kind,
        segment_indices=tuple(indices),
        family=family,
        text_span=text_span,
    )


def _derive_family(
    analysis: AnalysisManifest | None,
    text_span: tuple[int, int],
    kind: str,
) -> str:
    """Derive a family name for a record group from analysis or kind."""
    if analysis and analysis.block_families:
        start, end = text_span
        for block_family in analysis.block_families:
            # Check if the block family overlaps with this text span
            overlap_start = max(start, block_family.start)
            overlap_end = min(end, block_family.end)
            if overlap_start < overlap_end:
                return block_family.family
    # Fallback: derive from kind
    return f"{kind}_block"


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
    if text.startswith("{") or text.startswith("["):
        return True
    lines = text.splitlines()
    config_like_lines = 0
    for line in lines:
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or stripped.startswith("//"):
            continue
        # key = value or key: value patterns typical of config files
        if re.match(r'^[A-Za-z_.\-/]+\s*[=:]\s*\S', stripped):
            config_like_lines += 1
    return config_like_lines >= 2 or text.count(":") >= 3 and text.count("=") >= 1 or text.count("=") >= 3


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
