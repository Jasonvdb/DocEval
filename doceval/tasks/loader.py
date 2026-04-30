"""Load and validate task fixtures from YAML."""
from __future__ import annotations

import hashlib
from pathlib import Path

import yaml

from doceval.schemas import Task, TaskKind

FIXTURES_DIR = Path(__file__).parent / "fixtures"


def load_tasks(kinds: list[TaskKind] | None = None) -> list[Task]:
    """Load all tasks from fixtures, optionally filtered by kind."""
    tasks: list[Task] = []
    for path in sorted(FIXTURES_DIR.glob("*.yaml")):
        with path.open() as f:
            entries = yaml.safe_load(f) or []
        for raw in entries:
            tasks.append(Task(**raw))

    if kinds:
        tasks = [t for t in tasks if t.kind in kinds]

    seen: set[str] = set()
    for t in tasks:
        if t.id in seen:
            raise ValueError(f"Duplicate task id: {t.id}")
        seen.add(t.id)
        if t.kind == "editing" and not t.source:
            raise ValueError(f"Editing task '{t.id}' missing source field")
    return tasks


def fixture_hashes() -> dict[str, str]:
    """SHA-256 of each fixture file — recorded in run manifest for reproducibility."""
    return {
        path.name: hashlib.sha256(path.read_bytes()).hexdigest()[:16]
        for path in sorted(FIXTURES_DIR.glob("*.yaml"))
    }
