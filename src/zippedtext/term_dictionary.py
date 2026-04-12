from __future__ import annotations

from collections import Counter

from .online_manifest import AnalysisManifest
from .predictor.phrases import PhraseTable, build_phrase_table


def build_structured_phrase_table(
    text: str,
    analysis: AnalysisManifest,
    max_phrases: int = 128,
    min_freq: int = 2,
) -> PhraseTable:
    heuristic = build_phrase_table(
        text,
        max_phrases=max_phrases,
        min_freq=max(min_freq, 3),
    )
    counter: Counter[str] = Counter(heuristic.phrases)

    # Single-pass occurrence counting for all candidate phrases
    candidate_phrases = set(analysis.phrase_dictionary) | set(heuristic.phrases)
    candidate_phrases = {p for p in candidate_phrases if len(p) >= 2}
    occurrence_counts = _count_all_phrase_occurrences(text, candidate_phrases)

    for phrase in analysis.phrase_dictionary:
        if phrase in occurrence_counts and occurrence_counts[phrase] > 0:
            counter[phrase] += occurrence_counts[phrase] + 2

    bigram_weights = dict(analysis.top_bigrams)
    char_weights = dict(analysis.char_frequencies)
    scored: list[tuple[str, int, float]] = []
    for phrase, freq in counter.items():
        if freq < min_freq:
            continue
        occurrence_count = occurrence_counts.get(phrase, 0)
        if occurrence_count <= 0:
            continue
        score = float(freq * max(len(phrase) - 1, 1))
        score += occurrence_count * 0.5
        score += _phrase_bigram_bonus(phrase, bigram_weights)
        score += _phrase_char_bonus(phrase, char_weights)
        if phrase in analysis.phrase_dictionary:
            score += 1.5
        scored.append((phrase, occurrence_count, score))
    scored.sort(key=lambda item: (-item[2], -item[1], -len(item[0]), item[0]))

    selected: list[str] = []
    for phrase, _occurrence_count, _score in scored:
        if len(selected) >= max_phrases:
            break
        if any(phrase != chosen and phrase in chosen for chosen in selected):
            continue
        selected.append(phrase)
    return PhraseTable(phrases=tuple(selected))


def _count_all_phrase_occurrences(
    text: str,
    phrases: set[str],
) -> dict[str, int]:
    """Count occurrences of all phrases in a single pass over the text.

    Uses a sliding window approach: for each position, check if any phrase
    starts here. This avoids O(n * m * k) repeated full-text scans.
    """
    counts: dict[str, int] = {p: 0 for p in phrases}
    if not phrases:
        return counts
    max_len = max(len(p) for p in phrases)
    phrase_set = frozenset(phrases)
    for start in range(len(text)):
        for length in range(2, min(max_len + 1, len(text) - start + 1)):
            candidate = text[start:start + length]
            if candidate in phrase_set:
                counts[candidate] += 1
    return counts


def _count_phrase_occurrences(text: str, phrase: str) -> int:
    """Legacy single-phrase counter (kept for backward compat)."""
    count = 0
    start = 0
    while True:
        idx = text.find(phrase, start)
        if idx == -1:
            break
        count += 1
        start = idx + 1
    return count


def _phrase_bigram_bonus(phrase: str, bigram_weights: dict[str, float]) -> float:
    if len(phrase) < 2 or not bigram_weights:
        return 0.0
    bonus = 0.0
    for index in range(len(phrase) - 1):
        bonus += bigram_weights.get(phrase[index:index + 2], 0.0)
    return bonus


def _phrase_char_bonus(phrase: str, char_weights: dict[str, float]) -> float:
    if not char_weights:
        return 0.0
    return sum(char_weights.get(ch, 0.0) for ch in phrase) * 0.25
