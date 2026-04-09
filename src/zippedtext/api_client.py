"""DeepSeek API client for obtaining next-token logprobs."""

from __future__ import annotations

import json
import math
import os
import time
from dataclasses import dataclass

from openai import OpenAI

from .online_manifest import AnalysisManifest


@dataclass(frozen=True)
class TokenLogprob:
    """Log-probability info for a single generated token."""

    token: str
    logprob: float
    top_alternatives: list[tuple[str, float]]


@dataclass(frozen=True)
class GeneratedToken:
    """A generated token with logprobs and its position in the generated text."""

    text: str
    logprob: float
    top_alternatives: list[tuple[str, float]]
    char_offset: int


@dataclass(frozen=True)
class ChunkResult:
    """Result of a single API continuation call."""

    generated_text: str
    tokens: list[GeneratedToken]
    model: str


class ApiClient:
    """Wraps any OpenAI-compatible API for logprobs retrieval."""

    MAX_RETRIES = 3
    RETRY_BASE_DELAY = 1.0

    def __init__(
        self,
        api_key: str | None = None,
        model: str = "deepseek-chat",
        base_url: str = "https://api.deepseek.com",
    ) -> None:
        self.model = model
        self._client = OpenAI(
            api_key=api_key or os.environ.get("DEEPSEEK_API_KEY", ""),
            base_url=base_url,
            timeout=60.0,
        )
        self.last_model_id: str = ""

    def generate_continuation(
        self,
        context: str,
        max_tokens: int = 200,
        max_top_logprobs: int = 20,
    ) -> ChunkResult:
        """Generate continuation text with token-level logprobs."""
        messages = self._build_continuation_messages(context)

        for attempt in range(self.MAX_RETRIES):
            try:
                response = self._client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    max_tokens=max_tokens,
                    temperature=0,
                    seed=42,
                    logprobs=True,
                    top_logprobs=max_top_logprobs,
                )
                break
            except Exception:
                if attempt == self.MAX_RETRIES - 1:
                    raise
                time.sleep(self.RETRY_BASE_DELAY * (2 ** attempt))

        choice = response.choices[0]
        self.last_model_id = response.model or self.model

        if not choice.logprobs or not choice.logprobs.content:
            raise RuntimeError("API did not return logprobs")

        tokens: list[GeneratedToken] = []
        char_offset = 0
        for token_info in choice.logprobs.content:
            alternatives = [
                (alt.token, alt.logprob)
                for alt in (token_info.top_logprobs or [])
            ]
            tokens.append(
                GeneratedToken(
                    text=token_info.token,
                    logprob=token_info.logprob,
                    top_alternatives=alternatives,
                    char_offset=char_offset,
                )
            )
            char_offset += len(token_info.token)

        generated_text = "".join(t.text for t in tokens)
        return ChunkResult(
            generated_text=generated_text,
            tokens=tokens,
            model=self.last_model_id,
        )

    def get_logprobs_for_text(
        self,
        context: str,
        continuation: str,
        max_top_logprobs: int = 20,
    ) -> list[TokenLogprob]:
        """Get logprobs for generating `continuation` given `context`."""
        messages = self._build_continuation_messages(context)

        response = self._client.chat.completions.create(
            model=self.model,
            messages=messages,
            max_tokens=max(len(continuation) * 2, 100),
            temperature=0,
            logprobs=True,
            top_logprobs=max_top_logprobs,
        )

        choice = response.choices[0]
        if not choice.logprobs or not choice.logprobs.content:
            raise RuntimeError("API did not return logprobs")

        result = []
        for token_info in choice.logprobs.content:
            alternatives = [
                (alt.token, alt.logprob)
                for alt in (token_info.top_logprobs or [])
            ]
            result.append(
                TokenLogprob(
                    token=token_info.token,
                    logprob=token_info.logprob,
                    top_alternatives=alternatives,
                )
            )
        return result

    def analyze_text(self, text: str) -> AnalysisManifest:
        """Use the LLM once to build a structured compression manifest."""
        prompt = f"""分析以下文本，返回一个JSON对象用于文本压缩优化。JSON格式如下：
{{
  "char_frequencies": {{"字符": 频率, ...}},
  "top_bigrams": [["字符对", 频率], ...],
  "phrase_dictionary": [["常用短语", 编号], ...],
  "language_segments": [{{"start": 起始位置, "end": 结束位置, "lang": "zh"|"en"|"num"}}],
  "template_hints": ["key_value", "list_prefix", "table_row"]
}}

你的目标不是泛泛总结文本，而是给压缩器提供“可复用、可离散化、可在解码期本地恢复”的高价值提示。
请优先关注以下高结构模式：
- API 文档中的参数说明行、字段说明行、返回值说明行
- 配置 / 环境变量 / key-value 行
- 列表项、编号项、枚举行
- 表格行、TSV/制表符分隔行
- 中英术语对照或名称:说明这类行级模式

要求：
- char_frequencies 最多返回 64 项，只保留真正高频且有压缩价值的字符
- top_bigrams 最多返回 32 项，只返回文本中真实高频、可复用的字符对
- phrase_dictionary 最多返回 32 项，优先返回文本中重复出现或明显可复用的术语/短语
- language_segments 只保留高价值片段，位置尽量准确
- template_hints 只返回高置信度命中，不要为了覆盖面乱猜
- 如果文本主要是普通 prose，没有明显模板，就让 template_hints 为空数组
- 只返回 JSON，不要任何解释

文本内容：
{text[:4000]}"""

        response = self._client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": "You are a text analysis assistant. Return only valid JSON."},
                {"role": "user", "content": prompt},
            ],
            max_tokens=4000,
            temperature=0,
        )
        raw = response.choices[0].message.content.strip()
        if raw.startswith("```"):
            raw = raw.split("\n", 1)[1].rsplit("```", 1)[0].strip()
        self.last_model_id = response.model or self.model
        payload = _safe_parse_analysis_payload(raw)
        return AnalysisManifest.from_api_payload(payload, len(text))

    def list_models(self) -> list[str]:
        """Fetch available model IDs from the API provider."""
        try:
            response = self._client.models.list()
            ids = sorted(m.id for m in response.data)
            return ids
        except Exception:
            return []

    def _build_continuation_messages(self, context: str) -> list[dict]:
        """Build messages for text continuation with logprobs."""
        if context:
            return [
                {
                    "role": "system",
                    "content": (
                        "你是一个文本续写助手。请严格按照原文的风格、语言和内容"
                        "自然地续写文本。尽量保持与原文一致的表达方式。"
                    ),
                },
                {
                    "role": "user",
                    "content": f"请续写以下文本：\n\n{context}",
                },
            ]
        return [
            {
                "role": "system",
                "content": "你是一个文本续写助手。",
            },
            {
                "role": "user",
                "content": "请开始写一段文本。",
            },
        ]

    def _build_messages(self, context: str, expected_len: int) -> list[dict]:
        return self._build_continuation_messages(context)


def _safe_parse_analysis_payload(raw: str) -> dict | None:
    try:
        payload = json.loads(raw)
        return payload if isinstance(payload, dict) else None
    except json.JSONDecodeError:
        start = raw.find("{")
        end = raw.rfind("}")
        if start == -1 or end == -1 or end <= start:
            return None
        snippet = raw[start:end + 1]
        try:
            payload = json.loads(snippet)
            return payload if isinstance(payload, dict) else None
        except json.JSONDecodeError:
            return None


DeepSeekClient = ApiClient
