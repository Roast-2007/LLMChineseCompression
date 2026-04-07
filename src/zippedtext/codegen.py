"""Code generation mode — compress text segments as executable code snippets.

The LLM identifies portions of text that can be represented by compact
Python expressions (patterns, repetitions, sequences, etc.).  Those
portions are replaced by code snippets in the compressed file; the
remaining text goes through normal PPM compression.

Security: all code is executed in a restricted sandbox with no imports,
no builtins, and a 1-second timeout.
"""

from __future__ import annotations

import json
import struct
import threading
from dataclasses import dataclass


# ------------------------------------------------------------------
# Sandbox
# ------------------------------------------------------------------

_SAFE_NAMESPACE: dict = {
    "str": str,
    "int": int,
    "float": float,
    "chr": chr,
    "ord": ord,
    "range": range,
    "len": len,
    "list": list,
    "tuple": tuple,
    "True": True,
    "False": False,
    "None": None,
}

_FORBIDDEN_PATTERNS = (
    "import", "__", "exec", "eval", "open", "os.", "sys.",
    "subprocess", "compile", "globals", "locals", "getattr",
    "setattr", "delattr", "vars", "dir",
)

_EVAL_TIMEOUT = 1.0  # seconds


def safe_eval(code: str) -> str:
    """Evaluate *code* in a restricted sandbox. Returns result as string.

    Raises ``ValueError`` if the code is rejected or times out.
    """
    # Pre-check for dangerous patterns
    for pat in _FORBIDDEN_PATTERNS:
        if pat in code:
            raise ValueError(f"forbidden pattern in code: {pat!r}")

    result_box: list = []
    error_box: list = []

    def _run() -> None:
        try:
            # Safe namespace must be in globals for generator expressions
            # to access it (Python scoping: genexprs create a new scope
            # that only sees globals, not locals).
            safe_globals = {"__builtins__": {}, **_SAFE_NAMESPACE}
            result = eval(code, safe_globals)
            result_box.append(str(result))
        except Exception as e:
            error_box.append(e)

    thread = threading.Thread(target=_run, daemon=True)
    thread.start()
    thread.join(timeout=_EVAL_TIMEOUT)

    if thread.is_alive():
        raise ValueError("code execution timed out")
    if error_box:
        raise ValueError(f"code execution failed: {error_box[0]}")
    if not result_box:
        raise ValueError("code produced no result")
    return result_box[0]


# ------------------------------------------------------------------
# Manifest
# ------------------------------------------------------------------

@dataclass(frozen=True)
class CodegenSegment:
    """A text region replaceable by code."""
    start: int
    end: int
    code: str
    output: str  # expected result of eval(code)


@dataclass(frozen=True)
class CodegenManifest:
    """Collection of code-replaceable segments."""
    segments: tuple[CodegenSegment, ...]

    def serialize(self) -> bytes:
        """Pack manifest into bytes."""
        items = [
            {"s": seg.start, "e": seg.end, "c": seg.code}
            for seg in self.segments
        ]
        payload = json.dumps(items, ensure_ascii=False, separators=(",", ":"))
        encoded = payload.encode("utf-8")
        return struct.pack("<I", len(encoded)) + encoded

    @classmethod
    def deserialize(cls, data: bytes) -> "CodegenManifest":
        """Reconstruct manifest from bytes."""
        if len(data) < 4:
            return cls(segments=())
        payload_len = struct.unpack("<I", data[:4])[0]
        payload = data[4:4 + payload_len].decode("utf-8")
        items = json.loads(payload)
        segments = []
        for item in items:
            code = item["c"]
            output = safe_eval(code)
            segments.append(CodegenSegment(
                start=item["s"],
                end=item["e"],
                code=code,
                output=output,
            ))
        return cls(segments=tuple(segments))


# ------------------------------------------------------------------
# LLM analysis
# ------------------------------------------------------------------

_CODEGEN_PROMPT = """\
分析以下文本，找出可以用简短 Python 表达式生成的文本片段。

规则：
1. 只使用 str, int, chr, ord, range, len 等基本函数
2. 不使用 import、open 等危险操作
3. 代码必须能生成与原文完全一致的字符串
4. 只选择能显著缩短的片段（代码长度 < 原文长度的 50%）

返回 JSON 数组，每个元素包含：
- "start": 起始字符位置（从0开始）
- "end": 结束字符位置（不含）
- "code": Python 表达式，eval 后得到目标字符串

只返回 JSON 数组，不要任何解释。如果没有合适的片段，返回空数组 []。

文本：
{text}"""


def analyze_for_codegen(text: str, api_client) -> CodegenManifest:
    """Use an LLM to identify code-representable text segments."""
    prompt = _CODEGEN_PROMPT.format(text=text[:4000])

    try:
        response = api_client._client.chat.completions.create(
            model=api_client.model,
            messages=[
                {"role": "system", "content": "You are a code analysis assistant. Return only valid JSON."},
                {"role": "user", "content": prompt},
            ],
            max_tokens=2000,
            temperature=0,
        )
    except Exception:
        return CodegenManifest(segments=())

    raw = response.choices[0].message.content.strip()
    # Strip markdown code fences if present
    if raw.startswith("```"):
        raw = raw.split("\n", 1)[1].rsplit("```", 1)[0].strip()

    try:
        items = json.loads(raw)
    except json.JSONDecodeError:
        return CodegenManifest(segments=())

    if not isinstance(items, list):
        return CodegenManifest(segments=())

    # Validate each segment
    validated: list[CodegenSegment] = []
    for item in items:
        try:
            start = int(item["start"])
            end = int(item["end"])
            code = str(item["code"])

            if start < 0 or end > len(text) or start >= end:
                continue
            if len(code) >= (end - start):
                continue  # code not shorter than original

            output = safe_eval(code)
            if output == text[start:end]:
                validated.append(CodegenSegment(
                    start=start, end=end, code=code, output=output,
                ))
        except (KeyError, ValueError, TypeError):
            continue

    # Remove overlapping segments (keep higher-saving ones)
    validated.sort(key=lambda s: len(s.output) - len(s.code), reverse=True)
    used_ranges: list[tuple[int, int]] = []
    final: list[CodegenSegment] = []
    for seg in validated:
        overlaps = any(
            not (seg.end <= rs or seg.start >= re)
            for rs, re in used_ranges
        )
        if not overlaps:
            final.append(seg)
            used_ranges.append((seg.start, seg.end))

    final.sort(key=lambda s: s.start)
    return CodegenManifest(segments=tuple(final))


# ------------------------------------------------------------------
# Text manipulation for encode/decode
# ------------------------------------------------------------------

# Sentinel character used to mark codegen replacement points.
# Using a Private Use Area codepoint that won't appear in normal text.
CODEGEN_SENTINEL = "\uE000"


def apply_codegen(text: str, manifest: CodegenManifest) -> str:
    """Replace codegen segments with sentinel markers.

    Each sentinel is followed by a 1-char index (chr(index)) so the
    decoder knows which manifest entry to substitute.
    """
    if not manifest.segments:
        return text

    parts: list[str] = []
    prev_end = 0
    for i, seg in enumerate(manifest.segments):
        parts.append(text[prev_end:seg.start])
        parts.append(CODEGEN_SENTINEL + chr(i))
        prev_end = seg.end
    parts.append(text[prev_end:])
    return "".join(parts)


def restore_codegen(text: str, manifest: CodegenManifest) -> str:
    """Replace sentinel markers with evaluated code outputs."""
    if not manifest.segments:
        return text

    parts: list[str] = []
    i = 0
    while i < len(text):
        if text[i] == CODEGEN_SENTINEL and i + 1 < len(text):
            idx = ord(text[i + 1])
            if idx < len(manifest.segments):
                parts.append(manifest.segments[idx].output)
            i += 2
        else:
            parts.append(text[i])
            i += 1
    return "".join(parts)
