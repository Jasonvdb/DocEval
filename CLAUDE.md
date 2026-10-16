# DocEval

Evaluation harness that compares LLMs on document generation and editing. Pipeline: generate → judge → score → report; each stage persists its output under `results/<run_id>/`.

## Commands and cost

- Use `uv` for everything (`uv sync`, `uv run doceval ...`, `uv run pytest`). There is no `pip install` or `requirements.txt`.
- `uv run doceval run` makes paid API calls. Judging is O(N²) over models and is the dominant cost; `--trials` only scales generation. Run `uv run doceval cost-estimate` (no API calls) first and trust it before launching anything with the full lineup (~$140 by default).

## Invariants

- A completed run is immutable. `doceval report <run_id>` re-renders from disk; `scoring.py` and `report.py` are pure transforms over `read_generations()` + `read_judgments()`. Changing report formatting or scoring math never needs new API calls.
- `Provider.generate()` (`providers/base.py`) never raises; errors go into `GenerationResult.error`.
- `judge.py` has three subtle bits to preserve:
  1. Self-preference exclusion: a model never judges a pair containing its own output.
  2. Label randomization: Output 1 / Output 2 order is randomized per judgment (`label_order`), and `_swap_for_label_order()` maps verdicts back to the canonical (model_a, model_b) view.
  3. Representative selection: `select_representatives()` picks the median-length successful trial per (task, model), so judging uses one output per model per task even with `--trials > 1`.
- Editing tasks use a different rubric: `criteria_for(kind)` in `schemas.py` replaces `completeness` with `change_discipline`.

## What goes stale

- Prices in `MODEL_REGISTRY` (`doceval/config.py`, the single source of truth for models) were verified April 2026. Re-check them before relying on `cost-estimate` or "best value" reports.
- Vendors deprecate model IDs. A model that 404s is recorded as an error and excluded from the tournament, but its registry entry should be removed or replaced.

## Conventions

- The codebase deliberately has very few comments; module docstrings explain the "why". Don't add running commentary.
- No fallbacks or feature flags. Missing API keys are the one runtime branch; everything else assumes happy-path types.
