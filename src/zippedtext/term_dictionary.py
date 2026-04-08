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

    for phrase in analysis.phrase_dictionary:
        if phrase in text and len(phrase) >= 2:
            counter[phrase] += _count_phrase_occurrences(text, phrase) + 2

    scored: list[tuple[str, int, int]] = []
    for phrase, freq in counter.items():
        if freq < min_freq:
            continue
        score = freq * max(len(phrase) - 1, 1)
        scored.append((phrase, freq, score))
    scored.sort(key=lambda item: (-item[2], -len(item[0]), item[0]))

    selected: list[str] = []
    for phrase, _freq, _score in scored:
        if len(selected) >= max_phrases:
            break
        if any(phrase != chosen and phrase in chosen for chosen in selected):
            continue
        selected.append(phrase)
    return PhraseTable(phrases=tuple(selected))


def _count_phrase_occurrences(text: str, phrase: str) -> int:
    count = 0
    start = 0
    while True:
        idx = text.find(phrase, start)
        if idx == -1:
            break
        count += 1
        start = idx + 1
    return count
