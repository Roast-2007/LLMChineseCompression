"""DeepSeek API client for obtaining next-token logprobs."""

from __future__ import annotations

import math
import os
import time
from dataclasses import dataclass, field

from openai import OpenAI


@dataclass(frozen=True)
class TokenLogprob:
    """Log-probability info for a single generated token."""

    token: str
    logprob: float
    top_alternatives: list[tuple[str, float]]  # (token_str, logprob)


@dataclass(frozen=True)
class GeneratedToken:
    """A generated token with logprobs and its position in the generated text."""

    text: str
    logprob: float
    top_alternatives: list[tuple[str, float]]  # (token_str, logprob)
    char_offset: int  # starting character position in generated text


@dataclass(frozen=True)
class ChunkResult:
    """Result of a single API continuation call."""

    generated_text: str
    tokens: list[GeneratedToken]
    model: str


class DeepSeekClient:
    """Wraps the DeepSeek API (OpenAI-compatible) for logprobs retrieval."""

    MAX_RETRIES = 3
    RETRY_BASE_DELAY = 1.0  # seconds

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
        self.last_model_id: str = ""  # set after each API call

    def generate_continuation(
        self,
        context: str,
        max_tokens: int = 200,
        max_top_logprobs: int = 20,
    ) -> ChunkResult:
        """Generate continuation text with token-level logprobs.

        Calls the API with the given context and returns generated tokens,
        each annotated with logprobs and character offset in the generated
        text.  Uses temperature=0 and a fixed seed for deterministic output
        — encoder and decoder MUST get identical predictions.

        Retries on transient failures.
        """
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
            tokens.append(GeneratedToken(
                text=token_info.token,
                logprob=token_info.logprob,
                top_alternatives=alternatives,
                char_offset=char_offset,
            ))
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
        """Get logprobs for generating `continuation` given `context`.

        Strategy: Ask the model to continue the text. We use temperature=0
        for determinism. The model may not reproduce `continuation` exactly,
        so we collect whatever logprobs the API returns.

        Returns one TokenLogprob per generated token.
        """
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

    # Keep legacy alias for backwards compatibility
    def _build_messages(self, context: str, expected_len: int) -> list[dict]:
        return self._build_continuation_messages(context)
