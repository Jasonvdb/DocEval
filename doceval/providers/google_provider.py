"""Google provider — google-genai async streaming."""
from __future__ import annotations

import os
import time

from google import genai
from google.genai import types

from doceval.providers.base import Provider
from doceval.schemas import GenerationResult, ModelSpec


class GoogleProvider(Provider):
    def __init__(self) -> None:
        self.client = genai.Client(api_key=os.environ["GOOGLE_API_KEY"])

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
            stream = await self.client.aio.models.generate_content_stream(
                model=model.name,
                contents=user,
                config=types.GenerateContentConfig(
                    system_instruction=system,
                    max_output_tokens=max_output_tokens,
                ),
            )
            async for chunk in stream:
                chunk_text = getattr(chunk, "text", None)
                if chunk_text:
                    if ttft is None:
                        ttft = (time.monotonic() - start) * 1000
                    text_parts.append(chunk_text)
                usage = getattr(chunk, "usage_metadata", None)
                if usage:
                    prompt_tokens = usage.prompt_token_count or prompt_tokens
                    completion_tokens = usage.candidates_token_count or completion_tokens
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
