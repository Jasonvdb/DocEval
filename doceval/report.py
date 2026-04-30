"""Rich-rendered terminal report from a completed run."""
from __future__ import annotations

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from doceval.schemas import GenerationResult, Judgment, RunManifest, TaskKind, criteria_for
from doceval.scoring import (
    PerformanceEntry,
    QualityEntry,
    best_value_pick,
    performance_stats,
    quality_for_kind,
)
from doceval.storage import read_generations, read_judgments, read_manifest
from doceval.tasks.loader import load_tasks


def _fmt_ms(ms: float | None) -> str:
    if ms is None:
        return "—"
    if ms >= 1000:
        return f"{ms/1000:.2f}s"
    return f"{ms:.0f}ms"


def _fmt_cost(usd: float) -> str:
    if usd == 0:
        return "—"
    if usd < 0.001:
        return f"${usd*1000:.2f}m"  # millicents
    if usd < 0.01:
        return f"${usd*100:.2f}¢"
    return f"${usd:.4f}"


def _fmt_pct(x: float) -> str:
    return f"{x*100:.0f}%"


def _quality_table(entries: list[QualityEntry], kind: TaskKind) -> Table:
    t = Table(
        title=f"Quality leaderboard — {kind} tasks",
        title_style="bold cyan",
        show_lines=False,
        header_style="bold",
    )
    t.add_column("Rank", justify="right")
    t.add_column("Model", style="bold")
    t.add_column("BT score", justify="right")
    t.add_column("Win rate", justify="right")
    t.add_column("W–L–T", justify="right")
    for i, e in enumerate(entries, 1):
        t.add_row(
            str(i),
            e.model,
            f"{e.bt_score:+.2f}",
            _fmt_pct(e.overall_winrate),
            f"{e.wins}–{e.losses}–{e.ties}",
        )
    return t


def _criteria_table(entries: list[QualityEntry], kind: TaskKind) -> Table:
    crits = criteria_for(kind)
    t = Table(
        title=f"Per-criterion win rates — {kind}",
        title_style="bold cyan",
        header_style="bold",
    )
    t.add_column("Model", style="bold")
    for c in crits:
        t.add_column(c.replace("_", " "), justify="right")
    for e in entries:
        t.add_row(e.model, *[_fmt_pct(e.criterion_winrates.get(c, 0.0)) for c in crits])
    return t


def _performance_table(perf: dict[str, PerformanceEntry], order: list[str]) -> Table:
    t = Table(
        title="Performance per model (averaged across all tasks and trials)",
        title_style="bold cyan",
        header_style="bold",
    )
    t.add_column("Model", style="bold")
    t.add_column("Avg cost", justify="right")
    t.add_column("TTFT", justify="right")
    t.add_column("Total", justify="right")
    t.add_column("± stdev", justify="right")
    t.add_column("Tokens (in/out)", justify="right")
    t.add_column("Fail rate", justify="right")
    for model in order:
        p = perf.get(model)
        if not p:
            continue
        t.add_row(
            model,
            _fmt_cost(p.avg_cost_usd),
            _fmt_ms(p.avg_ttft_ms),
            _fmt_ms(p.avg_total_ms),
            _fmt_ms(p.stdev_total_ms),
            f"{p.avg_prompt_tokens:.0f}/{p.avg_completion_tokens:.0f}",
            _fmt_pct(p.failure_rate) if p.failure_rate > 0 else "—",
        )
    return t


def render_report(
    manifest: RunManifest,
    generations: list[GenerationResult],
    judgments: list[Judgment],
    *,
    console: Console | None = None,
) -> None:
    console = console or Console()
    tasks = load_tasks()
    task_kinds = {t.id: t.kind for t in tasks}

    console.rule(f"[bold]DocEval run {manifest.run_id}[/bold]")
    console.print(
        f"Started: {manifest.started_at:%Y-%m-%d %H:%M:%S}    "
        f"Models: {len(manifest.models)}    "
        f"Tasks: {len(manifest.task_ids)}    "
        f"Trials: {manifest.trials}"
    )
    if manifest.skipped_providers:
        console.print(
            f"[yellow]Skipped providers (no API key): "
            f"{', '.join(manifest.skipped_providers)}[/yellow]"
        )

    perf = performance_stats(generations)
    quality_by_kind: dict[TaskKind, list[QualityEntry]] = {}
    for kind in manifest.kinds:
        entries = quality_for_kind(judgments, kind, task_kinds)
        quality_by_kind[kind] = entries
        if not entries:
            console.print(f"\n[yellow]No judgments for kind '{kind}'[/yellow]")
            continue
        console.print()
        console.print(_quality_table(entries, kind))
        console.print()
        console.print(_criteria_table(entries, kind))

    console.print()
    console.print(_performance_table(perf, manifest.models))

    # Best-value pick uses combined-kind quality
    all_quality = [e for entries in quality_by_kind.values() for e in entries]
    if all_quality and perf:
        # Average BT scores across kinds for each model
        bt_avg: dict[str, list[float]] = {}
        for e in all_quality:
            bt_avg.setdefault(e.model, []).append(e.bt_score)
        avg_entries = [
            QualityEntry(model=m, bt_score=sum(s) / len(s)) for m, s in bt_avg.items()
        ]
        pick = best_value_pick(avg_entries, perf)
        if pick:
            top = max(avg_entries, key=lambda e: e.bt_score).model
            cheapest = min(
                (m for m in perf if perf[m].n_success > 0),
                key=lambda m: perf[m].avg_cost_usd,
                default=None,
            )
            text = Text()
            text.append("Highest quality: ", style="bold")
            text.append(f"{top}\n", style="cyan")
            text.append("Best value (quality per dollar): ", style="bold")
            text.append(f"{pick}\n", style="green")
            if cheapest:
                text.append("Cheapest: ", style="bold")
                text.append(f"{cheapest} ({_fmt_cost(perf[cheapest].avg_cost_usd)}/task)", style="dim")
            console.print()
            console.print(Panel(text, title="Recommendations", border_style="green"))


def render_from_run_id(run_id: str, *, console: Console | None = None) -> None:
    manifest = read_manifest(run_id)
    generations = read_generations(run_id)
    judgments = read_judgments(run_id)
    render_report(manifest, generations, judgments, console=console)
