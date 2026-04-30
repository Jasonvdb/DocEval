# DocEval

A reproducible harness for finding the best LLM for legal document generation and editing — across Claude, OpenAI, and Gemini — measured on quality, cost, and speed.

## Why this exists

Picking a model for a legal product isn't just "which is smartest." It's a trade-off between **quality** (does it draft accurate, complete, well-formed legal text?), **cost** (you're shipping millions of tokens), and **latency** (users wait for output). New models ship every few weeks and prices move. DocEval lets you re-run the same eval on the latest lineup with one command and get a defensible answer.

It runs the same drafting and editing tasks across every available model, then has the models blindly judge each other's outputs in a pairwise tournament. A Bradley-Terry fit turns the matches into a single ranking. Output: a terminal leaderboard plus per-criterion breakdowns, cost/latency numbers, and a "best value" pick.

## Quick start

```bash
# 1. Install dependencies (uv handles the Python toolchain)
uv sync

# 2. Add API keys
cp .env.example .env
# edit .env and fill in any of: ANTHROPIC_API_KEY, OPENAI_API_KEY, GOOGLE_API_KEY
# missing keys → that provider is skipped, the run continues

# 3. Preview cost before running anything
uv run doceval cost-estimate

# 4. Run a cheap smoke test (~$0.10)
uv run doceval run --models claude-haiku-4-5,gpt-5.4-nano,gemini-3.1-flash-lite --trials 1

# 5. Full run
uv run doceval run

# 6. Re-render the report from saved artifacts later
uv run doceval list
uv run doceval report 20260430-184530
```

## Architecture

```
            ┌─────────────────┐
            │  Task fixtures  │  YAML — generation.yaml, editing.yaml
            │   (legal docs)  │
            └────────┬────────┘
                     │
                     ▼
            ┌─────────────────┐
            │     Runner      │  Fan out task × model × trial calls
            │  (async, with   │  with bounded per-provider concurrency.
            │   streaming)    │  Captures TTFT, total time, tokens, cost.
            └────────┬────────┘
                     │
                     ▼
            ┌─────────────────┐         results/<run_id>/
            │     Storage     │ ──────► generations/*.json
            │  (raw on disk)  │         judgments/*.json
            └────────┬────────┘         manifest.json
                     │
                     ▼
            ┌─────────────────┐
            │     Judge       │  Pairwise tournament. For every pair (A,B),
            │ (blind pairwise │  every other model judges them with anonymized,
            │   tournament)   │  randomized labels and a 5-criterion rubric.
            └────────┬────────┘
                     │
                     ▼
            ┌─────────────────┐
            │     Scoring     │  Bradley-Terry fit per task kind.
            │  (Bradley-Terry │  Per-criterion win rates.
            │   + perf stats) │  Latency / cost / token aggregates.
            └────────┬────────┘
                     │
                     ▼
            ┌─────────────────┐
            │     Report      │  rich-rendered terminal leaderboard,
            │   (rich tables) │  per-criterion breakdown, cost/latency
            │                 │  table, best-value pick.
            └─────────────────┘
```

The pipeline runs in four stages, each owns one module:

1. **Generate** (`runner.py`) — for each task × model × trial (default 3), call the provider over a streaming connection. Capture text, prompt/completion tokens, time-to-first-token, and total wall-clock time. Compute cost from the model's $/Mtok rates. Persist every result to `results/<run_id>/generations/`.
2. **Judge** (`judge.py`) — for each task, pick a representative output per model (median-length successful trial). Form every unordered pair of models. For each pair, send the two outputs anonymized as "Output 1" / "Output 2" (order randomized) to every *other* model with a rubric. Persist verdicts to `results/<run_id>/judgments/`.
3. **Score** (`scoring.py`) — fit a Bradley-Terry model over the pairwise verdicts (using the `choix` library) to produce one quality score per model per task kind. Compute per-criterion win rates and aggregate generation stats (cost, latency, tokens, failure rate).
4. **Report** (`report.py`) — render leaderboard, per-criterion table, performance table, and a recommendations panel via `rich`.

Everything is read-only after the run finishes. `doceval report <run_id>` re-renders the report from disk; you can also re-judge or re-score later without re-generating.

## How judging works

Naive LLM-as-judge has a well-known flaw: models prefer their own outputs. DocEval controls for this with three layers:

1. **Self-preference exclusion.** A model never judges a pair containing its own output. With a typical lineup of ~12 models, every pair still has ~10 independent judges.
2. **Anonymized, randomized labels.** Outputs are presented as "Output 1" / "Output 2" with the order randomized per judge call. The label order is recorded so we can detect positional bias if it ever shows up in the data.
3. **Pairwise > N-way ranking.** Pairwise comparisons are far more reliable than asking a judge to rank N outputs at once. Bradley-Terry then turns the per-pair vote tallies into a globally consistent ranking, even when the matchups are sparse or noisy.

The rubric for **generation** tasks scores: `legal_accuracy`, `completeness`, `faithfulness` (to the instructions), `formatting`, and `clarity`. **Editing** tasks swap `completeness` for `change_discipline` (did the model only modify what was asked, leaving everything else verbatim?). Judges return both per-criterion verdicts and an overall winner; the overall winner drives the BT fit, the per-criterion verdicts drive the breakdown table.

## Adding a new model

One line in `doceval/config.py`:

```python
ModelSpec(
    name="my-new-model",
    provider="anthropic",            # or "openai" or "google"
    input_price_per_mtok=4.0,
    output_price_per_mtok=20.0,
    context_window=200_000,
),
```

The provider abstraction handles everything else. Re-run; the new model joins the tournament and the leaderboard.

To add a brand-new provider, implement `Provider.generate()` in `doceval/providers/` and register it in `providers/base.py:get_provider()`.

## Adding a new task

Append to `doceval/tasks/fixtures/generation.yaml` or `editing.yaml`:

```yaml
- id: gen-my-task              # must be unique
  kind: generation             # or "editing"
  category: contract           # free-form label; appears in reports later
  system: |
    You are an experienced commercial attorney...
  prompt: |
    Draft a ...
  # editing tasks also need:
  source: |
    [original text the model is asked to revise]
```

Fixture file hashes are recorded in the run manifest, so changing tasks invalidates direct comparisons with prior runs (by design — different tasks, different scores).

## Cost notes

The full default lineup is expensive because pairwise judging scales as O(N²) — every pair of models is judged by every *other* model. With 12 models and 13 tasks at 3 trials, expect roughly:

| Configuration | Approx total |
|---|---|
| All 12 models, 3 trials (default) | **~$140** (gpt-5.5-pro alone is ~$80) |
| All except gpt-5.5-pro, 3 trials | ~$60 |
| Mid-tier only (exclude both Opus + gpt-5.5-pro), 3 trials | ~$30 |
| Cheap tier only (haiku, mini, nano, flash, flash-lite), 3 trials | ~$5–10 |

The biggest lever is **which models you include**, not trial count — judging is the dominant cost and is independent of trials.

```bash
# Cheap-tier only — useful for a fast iteration baseline
uv run doceval run --models claude-haiku-4-5,gpt-5.4-mini,gpt-5.4-nano,gemini-3-flash,gemini-3.1-flash-lite

# Single trial instead of three (faster, slightly noisier)
uv run doceval run --trials 1

# Generation only, skip editing
uv run doceval run --kinds generation
```

Always run `doceval cost-estimate` first — it shows the per-model bill before any API calls are made.

## Project layout

```
DocEval/
├── .env.example              # Copy to .env, fill in keys
├── .gitignore
├── pyproject.toml            # uv-managed
├── README.md
├── doceval/
│   ├── cli.py                # typer commands: run, report, list, cost-estimate, models
│   ├── config.py             # MODEL_REGISTRY (prices, context windows)
│   ├── schemas.py            # Pydantic: ModelSpec, Task, GenerationResult, Judgment, RunManifest
│   ├── providers/            # One module per vendor, all conform to Provider ABC
│   │   ├── base.py
│   │   ├── anthropic_provider.py
│   │   ├── openai_provider.py
│   │   └── google_provider.py
│   ├── tasks/
│   │   ├── loader.py
│   │   └── fixtures/
│   │       ├── generation.yaml
│   │       └── editing.yaml
│   ├── runner.py             # Generation orchestrator (async)
│   ├── judge.py              # Pairwise tournament
│   ├── scoring.py            # Bradley-Terry + perf aggregates
│   ├── storage.py            # results/<run_id>/ layout
│   └── report.py             # rich-rendered terminal output
├── tests/
│   └── test_scoring.py       # Bradley-Terry sanity tests
└── results/                  # gitignored — one subdir per run
```

## Limitations

- **Judge bias is reduced, not eliminated.** Cross-model judging plus Bradley-Terry gets you most of the way; for high-stakes decisions, spot-check the top-N matchups by reading raw outputs in `results/<run_id>/`.
- **Variance is real.** With `--trials 1` you'll see noticeable run-to-run swings on close models. Use 3+ trials when the answer matters.
- **Quality scores are within-run.** BT scores are relative to the lineup in that run; don't compare absolute scores across runs with different model sets.
- **Hosted models only.** Local / open-source models aren't supported (intentional — different tooling, different cost model).
- **No multi-turn or tool-use eval.** Single-shot drafting and editing only; that's the use case it's built for.

## Commands reference

```bash
doceval run                              # full eval, all available models, all tasks
doceval run --models opus-4-7,gpt-5.5    # filter to specific models
doceval run --kinds generation           # generation only
doceval run --kinds editing              # editing only
doceval run --trials 5                   # more trials = lower variance, higher cost
doceval run -y                           # skip confirmation
doceval cost-estimate                    # dry-run cost preview
doceval list                             # list past runs
doceval report <run_id>                  # re-render report from saved artifacts
doceval models                           # show registered models + which providers have keys
```
