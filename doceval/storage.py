"""On-disk storage of run artifacts. Layout:

    results/
        <run_id>/
            manifest.json
            generations/<task_id>__<model>__<trial>.json
            judgments/<task_id>__<model_a>__<model_b>__<judge>.json

All raw responses are persisted so a run can be re-judged or re-reported later
without re-generating.
"""
from __future__ import annotations

from datetime import datetime
from pathlib import Path

from doceval.schemas import GenerationResult, Judgment, RunManifest

RESULTS_ROOT = Path("results")


def new_run_id() -> str:
    return datetime.now().strftime("%Y%m%d-%H%M%S")


def run_dir(run_id: str) -> Path:
    return RESULTS_ROOT / run_id


def ensure_run_dirs(run_id: str) -> Path:
    base = run_dir(run_id)
    (base / "generations").mkdir(parents=True, exist_ok=True)
    (base / "judgments").mkdir(parents=True, exist_ok=True)
    return base


def list_runs() -> list[str]:
    if not RESULTS_ROOT.exists():
        return []
    return sorted(p.name for p in RESULTS_ROOT.iterdir() if p.is_dir())


def write_manifest(manifest: RunManifest) -> None:
    path = run_dir(manifest.run_id) / "manifest.json"
    path.write_text(manifest.model_dump_json(indent=2))


def read_manifest(run_id: str) -> RunManifest:
    path = run_dir(run_id) / "manifest.json"
    return RunManifest.model_validate_json(path.read_text())


def _safe(s: str) -> str:
    return s.replace("/", "_").replace(" ", "_")


def write_generation(run_id: str, gen: GenerationResult) -> None:
    fname = f"{_safe(gen.task_id)}__{_safe(gen.model)}__{gen.trial}.json"
    path = run_dir(run_id) / "generations" / fname
    path.write_text(gen.model_dump_json(indent=2))


def read_generations(run_id: str) -> list[GenerationResult]:
    base = run_dir(run_id) / "generations"
    if not base.exists():
        return []
    return [GenerationResult.model_validate_json(p.read_text()) for p in sorted(base.glob("*.json"))]


def write_judgment(run_id: str, j: Judgment) -> None:
    fname = f"{_safe(j.task_id)}__{_safe(j.model_a)}__{_safe(j.model_b)}__{_safe(j.judge)}.json"
    path = run_dir(run_id) / "judgments" / fname
    path.write_text(j.model_dump_json(indent=2))


def read_judgments(run_id: str) -> list[Judgment]:
    base = run_dir(run_id) / "judgments"
    if not base.exists():
        return []
    return [Judgment.model_validate_json(p.read_text()) for p in sorted(base.glob("*.json"))]
