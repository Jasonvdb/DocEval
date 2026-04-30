"""Central registry of supported models, prices, and env handling.

Adding a new model is a one-line addition to MODEL_REGISTRY.
"""
from __future__ import annotations

import os

from dotenv import load_dotenv

from doceval.schemas import ModelSpec

load_dotenv()


# Prices in USD per 1,000,000 tokens. Verified April 2026 from official pricing pages.
# Update here when prices change.
MODEL_REGISTRY: list[ModelSpec] = [
    # Anthropic
    ModelSpec(
        name="claude-opus-4-7",
        provider="anthropic",
        input_price_per_mtok=5.0,
        output_price_per_mtok=25.0,
        context_window=200_000,
    ),
    ModelSpec(
        name="claude-opus-4-6",
        provider="anthropic",
        input_price_per_mtok=5.0,
        output_price_per_mtok=25.0,
        context_window=200_000,
    ),
    ModelSpec(
        name="claude-sonnet-4-6",
        provider="anthropic",
        input_price_per_mtok=3.0,
        output_price_per_mtok=15.0,
        context_window=1_000_000,
    ),
    ModelSpec(
        name="claude-haiku-4-5",
        provider="anthropic",
        input_price_per_mtok=1.0,
        output_price_per_mtok=5.0,
        context_window=200_000,
    ),
    # OpenAI
    ModelSpec(
        name="gpt-5.5",
        provider="openai",
        input_price_per_mtok=5.0,
        output_price_per_mtok=30.0,
        context_window=1_000_000,
    ),
    ModelSpec(
        name="gpt-5.5-pro",
        provider="openai",
        input_price_per_mtok=30.0,
        output_price_per_mtok=180.0,
        context_window=1_000_000,
    ),
    ModelSpec(
        name="gpt-5.4-mini",
        provider="openai",
        input_price_per_mtok=0.75,
        output_price_per_mtok=4.50,
        context_window=400_000,
    ),
    ModelSpec(
        name="gpt-5.4-nano",
        provider="openai",
        input_price_per_mtok=0.20,
        output_price_per_mtok=1.25,
        context_window=400_000,
    ),
    # Google
    ModelSpec(
        name="gemini-3.1-pro",
        provider="google",
        input_price_per_mtok=2.0,
        output_price_per_mtok=12.0,
        context_window=1_000_000,
    ),
    ModelSpec(
        name="gemini-3-flash",
        provider="google",
        input_price_per_mtok=0.50,
        output_price_per_mtok=3.0,
        context_window=1_000_000,
    ),
    ModelSpec(
        name="gemini-3.1-flash-lite",
        provider="google",
        input_price_per_mtok=0.25,
        output_price_per_mtok=1.50,
        context_window=1_000_000,
    ),
    ModelSpec(
        name="gemini-2.5-pro",
        provider="google",
        input_price_per_mtok=1.25,
        output_price_per_mtok=10.0,
        context_window=1_000_000,
    ),
]

MODELS_BY_NAME: dict[str, ModelSpec] = {m.name: m for m in MODEL_REGISTRY}

PROVIDER_ENV_KEYS = {
    "anthropic": "ANTHROPIC_API_KEY",
    "openai": "OPENAI_API_KEY",
    "google": "GOOGLE_API_KEY",
}


def available_providers() -> set[str]:
    """Providers with a non-empty API key in the environment."""
    return {p for p, k in PROVIDER_ENV_KEYS.items() if os.environ.get(k)}


def available_models(filter_names: list[str] | None = None) -> list[ModelSpec]:
    """Return models whose provider has a key set, optionally filtered by name."""
    avail = available_providers()
    models = [m for m in MODEL_REGISTRY if m.provider in avail]
    if filter_names:
        wanted = set(filter_names)
        unknown = wanted - set(MODELS_BY_NAME)
        if unknown:
            raise ValueError(f"Unknown model(s): {sorted(unknown)}")
        models = [m for m in models if m.name in wanted]
    return models


def skipped_providers() -> list[str]:
    return sorted(set(PROVIDER_ENV_KEYS) - available_providers())
