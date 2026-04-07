"""DeepSeek API client for obtaining next-token logprobs."""

from __future__ import annotations

import math
import os
from dataclasses import dataclass

from openai import OpenAI


@dataclass
class TokenLogprob:
    """Log-probability info for a single generated token."""

    token: str
    logprob: float
    top_alternatives: list[tuple[str, float]]  # (token_str, logprob)


class DeepSeekClient:
    """Wraps the DeepSeek API (OpenAI-compatible) for logprobs retrieval."""

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
        )

    def get_logprobs_for_text(
        self,
        context: str,
        continuation: str,
        max_top_logprobs: int = 20,
    ) -> list[TokenLogprob]:
        """Get logprobs for generating `continuation` given `context`.

        Strategy: Ask the model to continue the text. We use temperature=0
        for determinism. The model may not reproduce `continuation` exactly,
        so we collect whatever logprobs the API returns.

        Returns one TokenLogprob per generated token.
        """
        messages = self._build_messages(context, len(continuation))

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
            result.append(TokenLogprob(
                token=token_info.token,
                logprob=token_info.logprob,
                top_alternatives=alternatives,
            ))
        return result

    def analyze_text(self, text: str) -> dict:
        """Use LLM to analyze text and generate an optimized compression model.

        Used by offline mode: one API call to generate character frequencies,
        bigrams, phrase dictionary, and segment hints.
        """
        prompt = f"""分析以下文本，返回一个JSON对象用于文本压缩优化。JSON格式如下：
{{
  "char_frequencies": {{"字符": 频率, ...}},  // 出现频率最高的前200个字符及其相对频率(0-1)
  "top_bigrams": [["字符对", 频率], ...],  // 前100个最常见的双字符组合
  "phrase_dictionary": [["常用短语", 编号], ...],  // 文本中重复出现的短语(2-8字)，最多50个
  "language_segments": [{{"start": 起始位置, "end": 结束位置, "lang": "zh"|"en"|"num"}}]
}}

只返回JSON，不要任何解释。

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
        import json
        raw = response.choices[0].message.content.strip()
        # Strip markdown code fences if present
        if raw.startswith("```"):
            raw = raw.split("\n", 1)[1].rsplit("```", 1)[0].strip()
        return json.loads(raw)

    def _build_messages(self, context: str, expected_len: int) -> list[dict]:
        if context:
            return [
                {
                    "role": "system",
                    "content": "你是一个文本续写助手。请根据上下文自然地续写文本，保持原文的风格和语言。",
                },
                {
                    "role": "user",
                    "content": f"请续写以下文本（约{expected_len}个字符）：\n\n{context}",
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
