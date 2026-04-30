"""Aggregate generations and judgments into per-model scores.

Quality scores come from a Bradley-Terry fit over pairwise judgments. Cost,
latency, and token stats come from the GenerationResult records.
"""
from __future__ import annotations

import statistics
from collections import defaultdict
from dataclasses import dataclass, field

import choix

from doceval.schemas import GenerationResult, Judgment, TaskKind, criteria_for


@dataclass
class QualityEntry:
    model: str
    bt_score: float = 0.0
    overall_winrate: float = 0.0
    criterion_winrates: dict[str, float] = field(default_factory=dict)
    pairs_judged: int = 0  # number of judgments involving this model
    wins: int = 0
    losses: int = 0
    ties: int = 0


@dataclass
class PerformanceEntry:
    model: str
    n_attempts: int = 0
    n_success: int = 0
    avg_cost_usd: float = 0.0
    avg_ttft_ms: float | None = None
    avg_total_ms: float | None = None
    stdev_total_ms: float | None = None
    avg_prompt_tokens: float = 0.0
    avg_completion_tokens: float = 0.0

    @property
    def failure_rate(self) -> float:
        return 0.0 if self.n_attempts == 0 else 1.0 - self.n_success / self.n_attempts


def _bt_fit(model_list: list[str], pairs: list[tuple[int, int]]) -> list[float]:
    """Fit Bradley-Terry. Returns one parameter per model. Higher = better.
    Falls back to 0 if no decisive pairs (e.g., all ties or no judgments)."""
    if not pairs or len(model_list) < 2:
        return [0.0] * len(model_list)
    params = choix.ilsr_pairwise(len(model_list), pairs, alpha=0.01)
    return [float(p) for p in params]


def quality_for_kind(
    judgments: list[Judgment],
    kind: TaskKind,
    task_kinds_by_id: dict[str, TaskKind],
) -> list[QualityEntry]:
    """Compute Bradley-Terry scores and per-criterion win-rates for one task kind."""
    relevant = [
        j for j in judgments if j.ok and task_kinds_by_id.get(j.task_id) == kind
    ]
    if not relevant:
        return []

    models = sorted({m for j in relevant for m in (j.model_a, j.model_b)})
    idx = {m: i for i, m in enumerate(models)}

    overall_pairs: list[tuple[int, int]] = []
    crit_pairs: dict[str, list[tuple[int, int]]] = defaultdict(list)

    wins = defaultdict(int)
    losses = defaultdict(int)
    ties = defaultdict(int)
    pairs_judged = defaultdict(int)
    crit_w = defaultdict(lambda: defaultdict(int))
    crit_l = defaultdict(lambda: defaultdict(int))

    for j in relevant:
        a_i, b_i = idx[j.model_a], idx[j.model_b]
        pairs_judged[j.model_a] += 1
        pairs_judged[j.model_b] += 1

        if j.overall_winner == "a":
            overall_pairs.append((a_i, b_i))
            wins[j.model_a] += 1
            losses[j.model_b] += 1
        elif j.overall_winner == "b":
            overall_pairs.append((b_i, a_i))
            wins[j.model_b] += 1
            losses[j.model_a] += 1
        else:
            ties[j.model_a] += 1
            ties[j.model_b] += 1

        for c in criteria_for(kind):
            v = j.criteria_winners.get(c)
            if v == "a":
                crit_pairs[c].append((a_i, b_i))
                crit_w[c][j.model_a] += 1
                crit_l[c][j.model_b] += 1
            elif v == "b":
                crit_pairs[c].append((b_i, a_i))
                crit_w[c][j.model_b] += 1
                crit_l[c][j.model_a] += 1

    bt_scores = _bt_fit(models, overall_pairs)

    entries: list[QualityEntry] = []
    for i, m in enumerate(models):
        decisive = wins[m] + losses[m]
        winrate = wins[m] / decisive if decisive else 0.0
        crit_rates: dict[str, float] = {}
        for c in criteria_for(kind):
            d = crit_w[c][m] + crit_l[c][m]
            crit_rates[c] = (crit_w[c][m] / d) if d else 0.0
        entries.append(
            QualityEntry(
                model=m,
                bt_score=bt_scores[i],
                overall_winrate=winrate,
                criterion_winrates=crit_rates,
                pairs_judged=pairs_judged[m],
                wins=wins[m],
                losses=losses[m],
                ties=ties[m],
            )
        )
    entries.sort(key=lambda e: e.bt_score, reverse=True)
    return entries


def performance_stats(generations: list[GenerationResult]) -> dict[str, PerformanceEntry]:
    by_model: dict[str, list[GenerationResult]] = defaultdict(list)
    for g in generations:
        by_model[g.model].append(g)

    out: dict[str, PerformanceEntry] = {}
    for model, gens in by_model.items():
        ok = [g for g in gens if g.ok]
        if not ok:
            out[model] = PerformanceEntry(model=model, n_attempts=len(gens))
            continue
        ttfts = [g.ttft_ms for g in ok if g.ttft_ms is not None]
        totals = [g.total_ms for g in ok if g.total_ms is not None]
        out[model] = PerformanceEntry(
            model=model,
            n_attempts=len(gens),
            n_success=len(ok),
            avg_cost_usd=statistics.mean(g.cost_usd for g in ok),
            avg_ttft_ms=statistics.mean(ttfts) if ttfts else None,
            avg_total_ms=statistics.mean(totals) if totals else None,
            stdev_total_ms=statistics.stdev(totals) if len(totals) >= 2 else None,
            avg_prompt_tokens=statistics.mean(g.prompt_tokens for g in ok),
            avg_completion_tokens=statistics.mean(g.completion_tokens for g in ok),
        )
    return out


def best_value_pick(
    quality: list[QualityEntry],
    perf: dict[str, PerformanceEntry],
) -> str | None:
    """Pick the model with the best BT-score-per-dollar among those with positive
    BT score. Returns the model name or None if no candidates."""
    candidates = [
        e for e in quality if e.model in perf and perf[e.model].avg_cost_usd > 0
    ]
    if not candidates:
        return None
    # Shift BT scores so min becomes 0; avoids negative ratios skewing the pick.
    min_bt = min(e.bt_score for e in candidates)
    best = max(
        candidates,
        key=lambda e: (e.bt_score - min_bt) / perf[e.model].avg_cost_usd,
    )
    return best.model
