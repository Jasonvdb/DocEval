"""Generation orchestrator. Fans out task × model × trial calls with bounded
per-provider concurrency, persists each result to disk as it completes.
"""
from __future__ import annotations

import asyncio
from collections.abc import Iterable

from rich.console import Console
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    TextColumn,
    TimeElapsedColumn,
)

from doceval.providers import get_provider
from doceval.schemas import GenerationResult, ModelSpec, Task
from doceval.storage import write_generation

PER_PROVIDER_CONCURRENCY = 4
MAX_OUTPUT_TOKENS = 4096


def _build_user_message(task: Task) -> str:
    if task.kind == "editing":
        return (
            f"{task.prompt}\n\n"
            f"--- ORIGINAL DOCUMENT ---\n{task.source}\n--- END ORIGINAL ---"
        )
    return task.prompt


async def run_generations(
    tasks: Iterable[Task],
    models: list[ModelSpec],
    trials: int,
    run_id: str,
    *,
    console: Console | None = None,
) -> list[GenerationResult]:
    console = console or Console()
    tasks = list(tasks)
    providers = {m.provider: get_provider(m.provider) for m in models}
    semaphores = {p: asyncio.Semaphore(PER_PROVIDER_CONCURRENCY) for p in providers}

    units: list[tuple[Task, ModelSpec, int]] = [
        (t, m, trial) for t in tasks for m in models for trial in range(trials)
    ]

    results: list[GenerationResult] = []

    with Progress(
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        MofNCompleteColumn(),
        TimeElapsedColumn(),
        console=console,
        transient=False,
    ) as progress:
        bar = progress.add_task("[cyan]Generating", total=len(units))

        async def run_unit(task: Task, model: ModelSpec, trial: int) -> GenerationResult:
            async with semaphores[model.provider]:
                gen = await providers[model.provider].generate(
                    model,
                    system=task.system,
                    user=_build_user_message(task),
                    task_id=task.id,
                    trial=trial,
                    max_output_tokens=MAX_OUTPUT_TOKENS,
                )
            write_generation(run_id, gen)
            progress.update(bar, advance=1)
            return gen

        coros = [run_unit(t, m, trial) for t, m, trial in units]
        for fut in asyncio.as_completed(coros):
            results.append(await fut)

    return results
