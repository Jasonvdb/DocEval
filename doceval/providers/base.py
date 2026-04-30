"""Provider abstraction. Each implementation calls one vendor's API and returns
a GenerationResult, capturing tokens, TTFT, and total wall-clock time.
"""
from __future__ import annotations

from abc import ABC, abstractmethod

from doceval.schemas import GenerationResult, ModelSpec


class Provider(ABC):
    @abstractmethod
    async def generate(
        self,
        model: ModelSpec,
        system: str,
        user: str,
        *,
        task_id: str,
        trial: int,
        max_output_tokens: int = 4096,
    ) -> GenerationResult: ...


def get_provider(name: str) -> Provider:
    if name == "anthropic":
        from doceval.providers.anthropic_provider import AnthropicProvider
        return AnthropicProvider()
    if name == "openai":
        from doceval.providers.openai_provider import OpenAIProvider
        return OpenAIProvider()
    if name == "google":
        from doceval.providers.google_provider import GoogleProvider
        return GoogleProvider()
    raise ValueError(f"Unknown provider: {name}")
