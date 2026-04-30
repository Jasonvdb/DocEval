"""OpenAI provider — Chat Completions with streaming + usage capture."""
from __future__ import annotations

import os
import time

from openai import AsyncOpenAI

from doceval.providers.base import Provider
from doceval.schemas import GenerationResult, ModelSpec


class OpenAIProvider(Provider):
    def __init__(self) -> None:
        self.client = AsyncOpenAI(api_key=os.environ["OPENAI_API_KEY"])

    async def generate(
        self,
        model: ModelSpec,
        system: str,
        user: str,
        *,
        task_id: str,
        trial: int,
        max_output_tokens: int = 4096,
    ) -> GenerationResult:
        start = time.monotonic()
        ttft: float | None = None
        text_parts: list[str] = []
        prompt_tokens = completion_tokens = 0

        try:
            stream = await self.client.chat.completions.create(
                model=model.name,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
                max_completion_tokens=max_output_tokens,
                stream=True,
                stream_options={"include_usage": True},
            )
            async for chunk in stream:
                if chunk.choices:
                    delta = chunk.choices[0].delta
                    content = getattr(delta, "content", None)
                    if content:
                        if ttft is None:
                            ttft = (time.monotonic() - start) * 1000
                        text_parts.append(content)
                if getattr(chunk, "usage", None):
                    prompt_tokens = chunk.usage.prompt_tokens
                    completion_tokens = chunk.usage.completion_tokens
        except Exception as e:
            return GenerationResult(
                task_id=task_id,
                model=model.name,
                trial=trial,
                error=f"{type(e).__name__}: {e}",
                total_ms=(time.monotonic() - start) * 1000,
            )

        text = "".join(text_parts)
        total_ms = (time.monotonic() - start) * 1000
        return GenerationResult(
            task_id=task_id,
            model=model.name,
            trial=trial,
            text=text,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            ttft_ms=ttft,
            total_ms=total_ms,
            cost_usd=model.cost(prompt_tokens, completion_tokens),
        )
