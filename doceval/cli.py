"""DocEval CLI — typer-based commands."""
from __future__ import annotations

import asyncio
from datetime import datetime

import typer
from rich.console import Console
from rich.table import Table

from doceval.config import (
    MODEL_REGISTRY,
    available_models,
    available_providers,
    skipped_providers,
)
from doceval.judge import run_tournament
from doceval.report import render_from_run_id
from doceval.runner import MAX_OUTPUT_TOKENS, run_generations
from doceval.schemas import RunManifest, TaskKind
from doceval.storage import (
    ensure_run_dirs,
    list_runs,
    new_run_id,
    read_manifest,
    write_manifest,
)
from doceval.tasks.loader import fixture_hashes, load_tasks

app = typer.Typer(help="Evaluate LLMs on legal document generation and editing.", no_args_is_help=True)
console = Console()


def _parse_csv(value: str | None) -> list[str] | None:
    if not value:
        return None
    return [s.strip() for s in value.split(",") if s.strip()]


def _parse_kinds(value: str | None) -> list[TaskKind]:
    if not value:
        return ["generation", "editing"]
    parsed = _parse_csv(value) or []
    valid: list[TaskKind] = []
    for k in parsed:
        if k in ("generation", "gen"):
            valid.append("generation")
        elif k in ("editing", "edit"):
            valid.append("editing")
        else:
            raise typer.BadParameter(f"Unknown kind: {k}. Use 'generation' or 'editing'.")
    return valid


def _approx_tokens(text: str) -> int:
    return max(1, len(text) // 4)


@app.command()
def run(
    models: str = typer.Option(None, help="Comma-separated model names. Default: all available."),
    kinds: str = typer.Option(None, help="Comma-separated task kinds: generation,editing."),
    tasks: str = typer.Option(None, help="Comma-separated task ids. Default: all in selected kinds."),
    exclude_providers: str = typer.Option(None, "--exclude-providers", help="Comma-separated providers to skip: anthropic,openai,google."),
    trials: int = typer.Option(3, help="Trials per (task, model). Higher = lower variance, higher cost."),
    yes: bool = typer.Option(False, "--yes", "-y", help="Skip confirmation prompt."),
) -> None:
    """Run a full evaluation: generate, judge, score, report."""
    kind_list = _parse_kinds(kinds)
    excluded = set(_parse_csv(exclude_providers) or [])
    selected = [m for m in available_models(_parse_csv(models)) if m.provider not in excluded]
    skipped = skipped_providers()
    task_filter = _parse_csv(tasks)

    if skipped:
        console.print(
            f"[yellow]Skipping providers without API keys: {', '.join(skipped)}[/yellow]"
        )
    if excluded:
        console.print(f"[yellow]Excluding providers (--exclude-providers): {', '.join(sorted(excluded))}[/yellow]")
    if len(selected) < 2:
        console.print(
            "[red]Need at least 2 models with valid API keys to run a tournament.[/red]"
        )
        console.print(
            f"Available providers: {sorted(available_providers()) or 'none'}"
        )
        raise typer.Exit(1)

    tasks = load_tasks(kind_list)
    if task_filter:
        wanted = set(task_filter)
        unknown = wanted - {t.id for t in tasks}
        if unknown:
            console.print(f"[red]Unknown task id(s): {sorted(unknown)}[/red]")
            raise typer.Exit(1)
        tasks = [t for t in tasks if t.id in wanted]
    if not tasks:
        console.print("[red]No tasks loaded.[/red]")
        raise typer.Exit(1)

    n_pairs_per_task = len(selected) * (len(selected) - 1) // 2
    n_judges_per_pair = max(0, len(selected) - 2)
    judge_calls = len(tasks) * n_pairs_per_task * n_judges_per_pair
    gen_calls = len(tasks) * len(selected) * trials

    console.print()
    console.print(f"Models ({len(selected)}): {', '.join(m.name for m in selected)}")
    console.print(f"Tasks ({len(tasks)}): {', '.join(t.id for t in tasks)}")
    console.print(f"Trials per (task, model): {trials}")
    console.print(f"Generation calls: {gen_calls}    Judge calls: {judge_calls}")
    if not yes and not typer.confirm("\nProceed?", default=True):
        raise typer.Exit(0)

    run_id = new_run_id()
    ensure_run_dirs(run_id)
    manifest = RunManifest(
        run_id=run_id,
        started_at=datetime.now(),
        models=[m.name for m in selected],
        trials=trials,
        kinds=kind_list,
        task_ids=[t.id for t in tasks],
        fixture_hashes=fixture_hashes(),
        skipped_providers=skipped,
    )
    write_manifest(manifest)
    console.print(f"\n[bold]Run id:[/bold] {run_id}\n")

    generations = asyncio.run(
        run_generations(tasks, selected, trials, run_id, console=console)
    )

    failed = [g for g in generations if not g.ok]
    if failed:
        console.print(
            f"\n[yellow]{len(failed)} generation calls failed (logged to disk; "
            f"models with all-failed trials excluded from judging).[/yellow]"
        )

    console.print()
    judgments = asyncio.run(
        run_tournament(tasks, selected, generations, run_id, console=console, rng_seed=42)
    )
    bad = [j for j in judgments if not j.ok]
    if bad:
        console.print(
            f"[yellow]{len(bad)} judgments failed or were unparseable.[/yellow]"
        )

    manifest.finished_at = datetime.now()
    write_manifest(manifest)

    console.print()
    render_from_run_id(run_id, console=console)
    console.print()
    console.print(f"[dim]Artifacts written to results/{run_id}/[/dim]")


@app.command(name="list")
def list_cmd() -> None:
    """List past runs."""
    runs = list_runs()
    if not runs:
        console.print("No runs yet.")
        return
    t = Table(title="Past runs", header_style="bold")
    t.add_column("Run id")
    t.add_column("Started")
    t.add_column("Models")
    t.add_column("Tasks")
    for run_id in runs:
        try:
            m = read_manifest(run_id)
            t.add_row(
                run_id,
                f"{m.started_at:%Y-%m-%d %H:%M}",
                str(len(m.models)),
                str(len(m.task_ids)),
            )
        except Exception:
            t.add_row(run_id, "?", "?", "?")
    console.print(t)


@app.command()
def report(run_id: str = typer.Argument(..., help="Run id (e.g. 20260430-184530)")) -> None:
    """Re-render the report for a past run from saved artifacts."""
    render_from_run_id(run_id, console=console)


@app.command(name="cost-estimate")
def cost_estimate(
    models: str = typer.Option(None, help="Comma-separated model names."),
    kinds: str = typer.Option(None, help="Comma-separated task kinds."),
    tasks: str = typer.Option(None, help="Comma-separated task ids."),
    exclude_providers: str = typer.Option(None, "--exclude-providers", help="Comma-separated providers to skip: anthropic,openai,google."),
    trials: int = typer.Option(3, help="Trials per (task, model)."),
) -> None:
    """Approximate $ cost of a run without making API calls. Useful before launching big runs."""
    kind_list = _parse_kinds(kinds)
    excluded = set(_parse_csv(exclude_providers) or [])
    selected = [m for m in available_models(_parse_csv(models)) if m.provider not in excluded]
    if len(selected) < 2:
        console.print("[red]Need at least 2 models with valid API keys.[/red]")
        raise typer.Exit(1)
    task_filter = _parse_csv(tasks)
    tasks = load_tasks(kind_list)
    if task_filter:
        wanted = set(task_filter)
        unknown = wanted - {t.id for t in tasks}
        if unknown:
            console.print(f"[red]Unknown task id(s): {sorted(unknown)}[/red]")
            raise typer.Exit(1)
        tasks = [t for t in tasks if t.id in wanted]

    # Approx output sizes (heuristic)
    AVG_GEN_OUTPUT_TOKENS = min(MAX_OUTPUT_TOKENS, 800)
    AVG_JUDGE_INPUT_OVERHEAD = 400  # system + criteria block + JSON template
    AVG_JUDGE_OUTPUT_TOKENS = 200
    AVG_OUTPUT_LEN_CHARS = AVG_GEN_OUTPUT_TOKENS * 4

    total = 0.0
    rows = []

    for m in selected:
        gen_cost = 0.0
        for t in tasks:
            in_tokens = _approx_tokens(t.system + t.prompt + (t.source or ""))
            gen_cost += m.cost(in_tokens, AVG_GEN_OUTPUT_TOKENS) * trials
        judge_cost = 0.0
        # This model serves as judge for: pairs not involving it × per-task pair count
        n_other = len(selected) - 1
        n_pairs_judged = n_other * (n_other - 1) // 2  # pairs among other models
        for t in tasks:
            judge_in = _approx_tokens(t.system + t.prompt + (t.source or "")) + 2 * AVG_GEN_OUTPUT_TOKENS + AVG_JUDGE_INPUT_OVERHEAD
            judge_cost += m.cost(judge_in, AVG_JUDGE_OUTPUT_TOKENS) * n_pairs_judged
        rows.append((m.name, gen_cost, judge_cost, gen_cost + judge_cost))
        total += gen_cost + judge_cost

    t = Table(title="Estimated cost (approximate)", header_style="bold")
    t.add_column("Model", style="bold")
    t.add_column("Generation $", justify="right")
    t.add_column("Judging $", justify="right")
    t.add_column("Total $", justify="right")
    rows.sort(key=lambda r: -r[3])
    for name, g, j, tot in rows:
        t.add_row(name, f"${g:.3f}", f"${j:.3f}", f"${tot:.3f}")
    console.print(t)
    console.print(f"\n[bold]Estimated grand total: ${total:.2f}[/bold]")
    console.print(
        "[dim]Estimate uses ~800 output tokens/generation and ~200 output tokens/judgment. "
        "Real costs vary with response length.[/dim]"
    )


@app.command(name="models")
def models_cmd() -> None:
    """List the model registry and which providers have API keys."""
    avail = available_providers()
    t = Table(title="Model registry", header_style="bold")
    t.add_column("Model", style="bold")
    t.add_column("Provider")
    t.add_column("Input $/Mtok", justify="right")
    t.add_column("Output $/Mtok", justify="right")
    t.add_column("Available")
    for m in MODEL_REGISTRY:
        ok = m.provider in avail
        t.add_row(
            m.name,
            m.provider,
            f"${m.input_price_per_mtok:.2f}",
            f"${m.output_price_per_mtok:.2f}",
            "[green]yes[/green]" if ok else "[red]no key[/red]",
        )
    console.print(t)


if __name__ == "__main__":
    app()
