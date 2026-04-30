"""Pydantic models for tasks, generations, judgments, and run manifests."""
from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field

TaskKind = Literal["generation", "editing"]
Provider = Literal["anthropic", "openai", "google"]


class ModelSpec(BaseModel):
    name: str
    provider: Provider
    input_price_per_mtok: float
    output_price_per_mtok: float
    context_window: int

    def cost(self, prompt_tokens: int, completion_tokens: int) -> float:
        return (
            prompt_tokens * self.input_price_per_mtok
            + completion_tokens * self.output_price_per_mtok
        ) / 1_000_000


class Task(BaseModel):
    id: str
    kind: TaskKind
    category: str
    system: str
    prompt: str
    source: str | None = None  # editing tasks only — original document to edit


class GenerationResult(BaseModel):
    task_id: str
    model: str
    trial: int
    text: str = ""
    prompt_tokens: int = 0
    completion_tokens: int = 0
    ttft_ms: float | None = None
    total_ms: float | None = None
    cost_usd: float = 0.0
    error: str | None = None

    @property
    def ok(self) -> bool:
        return self.error is None and bool(self.text.strip())


CRITERIA_GENERATION = [
    "legal_accuracy",
    "completeness",
    "faithfulness",
    "formatting",
    "clarity",
]
CRITERIA_EDITING = [
    "legal_accuracy",
    "change_discipline",
    "faithfulness",
    "formatting",
    "clarity",
]


def criteria_for(kind: TaskKind) -> list[str]:
    return CRITERIA_GENERATION if kind == "generation" else CRITERIA_EDITING


class Judgment(BaseModel):
    task_id: str
    model_a: str  # canonical pair member (alphabetically first)
    model_b: str
    judge: str
    label_order: Literal["AB", "BA"]  # whether model_a was shown as Output 1 or 2
    criteria_winners: dict[str, Literal["a", "b", "tie"]] = Field(default_factory=dict)
    overall_winner: Literal["a", "b", "tie"] | None = None
    raw_response: str = ""
    error: str | None = None

    @property
    def ok(self) -> bool:
        return self.error is None and self.overall_winner is not None


class RunManifest(BaseModel):
    run_id: str
    started_at: datetime
    finished_at: datetime | None = None
    models: list[str]
    trials: int
    kinds: list[TaskKind]
    task_ids: list[str]
    fixture_hashes: dict[str, str]
    skipped_providers: list[str] = Field(default_factory=list)
