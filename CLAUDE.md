# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

This project uses `uv` for everything. There is no `pip install` / `requirements.txt`.

```bash
uv sync                                       # install / update dependencies (creates .venv/)
uv run doceval --help                         # show CLI
uv run doceval cost-estimate                  # dry-run cost preview, no API calls
uv run doceval run [--models ...] [--kinds ...] [--trials N] [-y]
uv run doceval report <run_id>                # re-render report from results/<run_id>/
uv run doceval list                           # list past runs
uv run doceval models                         # show registry + which providers have keys

uv run pytest                                 # full test suite (4 tests, ~10s)
uv run pytest tests/test_scoring.py::test_dominant_winner_ranks_first -v   # single test
```

There is no separate lint/format step configured.

## Architecture

The pipeline is **generate → judge → score → report**. Each stage owns one module and persists its outputs so the next stage can run independently.

```
runner.py         judge.py            scoring.py         report.py
   │                 │                    │                  │
   ▼                 ▼                    ▼                  ▼
GenerationResult  Judgment           QualityEntry,         rich Tables /
   │                 │              PerformanceEntry         Panels
   │                 │                    │
   ▼                 ▼                    ▲
results/<run_id>/generations/  ──┐        │
results/<run_id>/judgments/    ──┴────────┘
results/<run_id>/manifest.json
```

Key invariant: **a completed run is immutable**. `doceval report <run_id>` re-renders from disk; `scoring.py` and `report.py` are pure transforms over `read_generations()` + `read_judgments()`. To change report formatting or scoring math, you don't need to re-run any API calls.

### Module ownership

- `config.py` — `MODEL_REGISTRY` (single source of truth for prices, providers, context windows). Adding a new model is one entry here. `available_models()` filters by which env keys are set.
- `schemas.py` — all Pydantic types. Note: `criteria_for(kind)` returns different rubrics for generation vs editing (editing replaces `completeness` with `change_discipline`).
- `providers/base.py` — `Provider.generate()` returns a `GenerationResult` and **never raises**. Errors are captured into `GenerationResult.error`. Three concrete implementations (`anthropic_provider.py`, `openai_provider.py`, `google_provider.py`) all stream and capture TTFT on the first non-empty content chunk.
- `runner.py` — fan-out with `asyncio.Semaphore` per provider (`PER_PROVIDER_CONCURRENCY=4`). Editing tasks have their `source` document spliced into the user message via `_build_user_message`.
- `judge.py` — pairwise tournament. Three subtle bits to preserve:
  1. **Self-preference exclusion** — a model is never a judge for a pair containing its own output (`judges = [m for m in models if m.name not in (a, b)]`).
  2. **Label randomization** — order of Output 1 / Output 2 is randomized per judgment via `label_order ∈ {"AB", "BA"}`. The judge sees anonymous outputs; we record the order so we can swap verdicts back to the canonical (model_a, model_b) view via `_swap_for_label_order()`.
  3. **Representative selection** — `select_representatives()` picks the median-length successful trial per (task, model) so judging operates on one output per model per task even when `--trials > 1`.
- `scoring.py` — Bradley-Terry fit via `choix.ilsr_pairwise`. Input format is `(winner_idx, loser_idx)` tuples; ties are dropped from the BT fit but counted in W–L–T totals. `quality_for_kind` runs separately per task kind because rubrics differ.
- `storage.py` — flat JSON layout under `results/<run_id>/`. Filenames encode keys (`<task_id>__<model>__<trial>.json`) so the `read_*` functions just glob and parse.
- `tasks/loader.py` — validates fixture YAML; editing tasks must have a `source` field. Computes SHA-256 hashes of fixture files into the run manifest for reproducibility.
- `cli.py` — `typer` commands. The `run` command writes the manifest twice: once at start (so partial runs are inspectable) and once on completion with `finished_at`.

### Provider abstraction

All three providers conform to the same async signature and return the same shape. They differ only in:
- How they construct messages (system + user)
- Which streaming API they use
- How they read usage tokens off the stream (Anthropic: from final message; OpenAI: from final chunk with `stream_options={"include_usage": True}`; Google: from `usage_metadata` on each chunk)

To add a new provider, implement `Provider.generate()` in `providers/<name>_provider.py` and wire it into `providers/base.py:get_provider()`.

### What goes stale

- **Prices in `MODEL_REGISTRY`.** Verified April 2026. Re-check before relying on `cost-estimate` or "best value" reports months later.
- **Model names.** Vendors deprecate model IDs (e.g., Gemini 2.0 Flash sunset 2026-06-01). When a model 404s, the harness records the error and excludes it from the tournament — but the registry entry should be removed/replaced.

### Cost characteristics

Judging is O(N²) over models and is the dominant cost — `--trials` only scales generation cost. The `cost-estimate` command precomputes the bill; trust it before launching anything with the full lineup (~$140 by default).

## Conventions

- **Commits:** never include `Co-Authored-By` trailers (user is sole author).
- **Comments:** the codebase deliberately has very few. The module docstrings explain the "why" of each stage; method bodies are expected to be self-explanatory. Don't add running commentary.
- **No fallbacks or feature flags.** Missing API keys are the one runtime branch; everything else assumes happy-path types.
