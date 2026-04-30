"""Anthropic provider — streams completions, captures TTFT and token counts."""
from __future__ import annotations

import os
import time

from anthropic import AsyncAnthropic

from doceval.providers.base import Provider
from doceval.schemas import GenerationResult, ModelSpec


class AnthropicProvider(Provider):
    def __init__(self) -> None:
        self.client = AsyncAnthropic(api_key=os.environ["ANTHROPIC_API_KEY"])

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
            async with self.client.messages.stream(
                model=model.name,
                max_tokens=max_output_tokens,
                system=system,
                messages=[{"role": "user", "content": user}],
            ) as stream:
                async for chunk in stream.text_stream:
                    if chunk:
                        if ttft is None:
                            ttft = (time.monotonic() - start) * 1000
                        text_parts.append(chunk)
                final = await stream.get_final_message()
                prompt_tokens = final.usage.input_tokens
                completion_tokens = final.usage.output_tokens
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
