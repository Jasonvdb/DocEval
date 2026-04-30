"""Pairwise tournament judging with anonymized, randomized outputs.

For each task, every pair of models is judged by all other available models.
A model never judges a pair containing its own output (self-preference control).
"""
from __future__ import annotations

import asyncio
import json
import random
import re
from collections import defaultdict
from collections.abc import Iterable
from typing import Literal

from rich.console import Console
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    TextColumn,
    TimeElapsedColumn,
)

from doceval.providers import get_provider
from doceval.runner import PER_PROVIDER_CONCURRENCY
from doceval.schemas import (
    GenerationResult,
    Judgment,
    ModelSpec,
    Task,
    criteria_for,
)
from doceval.storage import write_judgment

JUDGE_MAX_TOKENS = 1024


CRITERION_DEFINITIONS = {
    "legal_accuracy": "Are legal terms used correctly? Are referenced statutes, doctrines, and standard provisions accurate and appropriate?",
    "completeness": "Does the output cover everything the instruction asked for, with no missing required elements?",
    "faithfulness": "Does the output follow the specific instructions (parties, jurisdiction, terms, structure) precisely?",
    "formatting": "Is the document well-structured, properly numbered/sectioned, and ready to use without reformatting?",
    "clarity": "Is the language clear, professional, and free of ambiguity?",
    "change_discipline": "Did the edit modify only what the instruction targeted, preserving unchanged content verbatim and not introducing unrelated changes?",
}


def _criterion_block(kind) -> str:
    lines = []
    for c in criteria_for(kind):
        lines.append(f"  - {c}: {CRITERION_DEFINITIONS[c]}")
    return "\n".join(lines)


def _build_judge_prompt(task: Task, output_1: str, output_2: str) -> tuple[str, str]:
    crit_block = _criterion_block(task.kind)
    crit_keys = criteria_for(task.kind)
    json_template = (
        "{\n"
        '  "criteria": {\n'
        + ",\n".join(f'    "{c}": "1" | "2" | "tie"' for c in crit_keys)
        + "\n  },\n"
        '  "overall": "1" | "2" | "tie",\n'
        '  "rationale": "one or two sentences"\n'
        "}"
    )

    system = (
        "You are an experienced commercial attorney evaluating legal document drafts. "
        "Be impartial and rigorous. The two outputs were produced by different drafters; "
        "you do not know which. Respond with a single JSON object and nothing else."
    )

    src_block = ""
    if task.kind == "editing":
        src_block = f"\n\nORIGINAL DOCUMENT BEING EDITED:\n{task.source}\n"

    user = f"""You are comparing two anonymous drafts produced for the same legal task.

TASK INSTRUCTION:
{task.prompt}{src_block}

OUTPUT 1:
{output_1}

OUTPUT 2:
{output_2}

Evaluate each output against the following criteria. For each, decide whether
Output 1, Output 2, or Tie is better:
{crit_block}

Then choose an overall winner taking all criteria into account.

Respond in this exact JSON format and nothing else:
{json_template}
"""
    return system, user


_JSON_RE = re.compile(r"\{.*\}", re.DOTALL)


def _parse_verdict(raw: str, kind) -> tuple[dict[str, str] | None, str | None]:
    """Return (criteria_winners, overall_winner) parsed from the judge's reply,
    or (None, None) if unparseable. Values are 'a'/'b'/'tie' (caller swaps for label_order)."""
    m = _JSON_RE.search(raw)
    if not m:
        return None, None
    try:
        data = json.loads(m.group(0))
    except json.JSONDecodeError:
        return None, None

    def normalize(v) -> str | None:
        if not isinstance(v, str):
            return None
        v = v.strip().lower()
        if v in ("1", "output 1"):
            return "1"
        if v in ("2", "output 2"):
            return "2"
        if v == "tie":
            return "tie"
        return None

    overall = normalize(data.get("overall"))
    if overall is None:
        return None, None

    raw_crits = data.get("criteria") or {}
    crits: dict[str, str] = {}
    for c in criteria_for(kind):
        n = normalize(raw_crits.get(c))
        if n is None:
            n = "tie"
        crits[c] = n
    return crits, overall


def _swap_for_label_order(verdict: str, label_order: Literal["AB", "BA"]) -> Literal["a", "b", "tie"]:
    """Convert '1'/'2'/'tie' (judge's view) back to 'a'/'b'/'tie' (canonical pair view)."""
    if verdict == "tie":
        return "tie"
    if label_order == "AB":
        return "a" if verdict == "1" else "b"
    return "b" if verdict == "1" else "a"


def _select_representative(gens: list[GenerationResult]) -> GenerationResult | None:
    """Pick the representative trial per model: the successful one with median length.
    Returns None if no successful trial exists."""
    ok = [g for g in gens if g.ok]
    if not ok:
        return None
    ok.sort(key=lambda g: len(g.text))
    return ok[len(ok) // 2]


def select_representatives(
    gens: list[GenerationResult],
    tasks: list[Task],
) -> dict[tuple[str, str], GenerationResult]:
    """Map (task_id, model) -> representative GenerationResult."""
    by_key: dict[tuple[str, str], list[GenerationResult]] = defaultdict(list)
    for g in gens:
        by_key[(g.task_id, g.model)].append(g)
    reps: dict[tuple[str, str], GenerationResult] = {}
    for key, group in by_key.items():
        rep = _select_representative(group)
        if rep is not None:
            reps[key] = rep
    return reps


async def run_tournament(
    tasks: Iterable[Task],
    models: list[ModelSpec],
    generations: list[GenerationResult],
    run_id: str,
    *,
    console: Console | None = None,
    rng_seed: int | None = None,
) -> list[Judgment]:
    """Run pairwise tournament. Returns all Judgments (also written to disk)."""
    console = console or Console()
    rng = random.Random(rng_seed)
    tasks = list(tasks)
    reps = select_representatives(generations, tasks)

    providers = {m.provider: get_provider(m.provider) for m in models}
    semaphores = {p: asyncio.Semaphore(PER_PROVIDER_CONCURRENCY) for p in providers}

    units: list[tuple[Task, str, str, ModelSpec, Literal["AB", "BA"]]] = []
    for task in tasks:
        models_with_output = sorted(
            m.name for m in models if (task.id, m.name) in reps
        )
        for i, a in enumerate(models_with_output):
            for b in models_with_output[i + 1 :]:
                judges = [m for m in models if m.name not in (a, b)]
                for judge in judges:
                    label_order: Literal["AB", "BA"] = rng.choice(["AB", "BA"])
                    units.append((task, a, b, judge, label_order))

    judgments: list[Judgment] = []

    if not units:
        return judgments

    with Progress(
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        MofNCompleteColumn(),
        TimeElapsedColumn(),
        console=console,
        transient=False,
    ) as progress:
        bar = progress.add_task("[magenta]Judging", total=len(units))

        async def judge_one(
            task: Task,
            a: str,
            b: str,
            judge_spec: ModelSpec,
            label_order: Literal["AB", "BA"],
        ) -> Judgment:
            rep_a = reps[(task.id, a)]
            rep_b = reps[(task.id, b)]
            if label_order == "AB":
                out1, out2 = rep_a.text, rep_b.text
            else:
                out1, out2 = rep_b.text, rep_a.text
            sys_prompt, user_prompt = _build_judge_prompt(task, out1, out2)

            j = Judgment(
                task_id=task.id,
                model_a=a,
                model_b=b,
                judge=judge_spec.name,
                label_order=label_order,
            )
            try:
                async with semaphores[judge_spec.provider]:
                    gen = await providers[judge_spec.provider].generate(
                        judge_spec,
                        system=sys_prompt,
                        user=user_prompt,
                        task_id=f"judge::{task.id}",
                        trial=0,
                        max_output_tokens=JUDGE_MAX_TOKENS,
                    )
                if gen.error:
                    j.error = gen.error
                else:
                    j.raw_response = gen.text
                    crits_raw, overall_raw = _parse_verdict(gen.text, task.kind)
                    if overall_raw is None:
                        j.error = "unparseable_judgment"
                    else:
                        j.criteria_winners = {
                            c: _swap_for_label_order(v, label_order)
                            for c, v in (crits_raw or {}).items()
                        }
                        j.overall_winner = _swap_for_label_order(overall_raw, label_order)
            except Exception as e:
                j.error = f"{type(e).__name__}: {e}"

            write_judgment(run_id, j)
            progress.update(bar, advance=1)
            return j

        coros = [judge_one(*u) for u in units]
        for fut in asyncio.as_completed(coros):
            judgments.append(await fut)

    return judgments
