# PRIORI-X

PRIORI-X is a clinician-facing research workbench for probabilistic differential diagnosis, next-best-test reasoning, calibration analysis, and offline policy improvement.

It is designed as an inspectable medical decision-science lab:

- Bayesian first
- evidence and provenance aware
- offline-evaluation first
- safety constrained
- explicitly non-autonomous

## What PRIORI-X Is

PRIORI-X accepts a vignette, note, or structured findings and produces:

- a ranked differential diagnosis with posterior mass
- evidence for and against each hypothesis
- next-best-test recommendations with likelihood ratios, information gain, cost, and safety tradeoffs
- threshold-aware action framing
- uncertainty intervals and calibration signals
- reproducible traces for offline evaluation

## What PRIORI-X Is Not

PRIORI-X is not:

- a clinical autopilot
- an order-entry system
- a treatment execution engine
- a live reinforcement loop over patient traffic
- a substitute for clinician judgment

Live online learning from real patient interactions is intentionally disabled. All prompt, policy, and routing improvements must happen inside the offline benchmark sandbox.

## Safety Model

PRIORI-X ships with conservative defaults:

- explicit UI and API disclaimers
- inspectable reasoning and provenance badges
- contradiction and duplication checks
- redaction-aware trace handling
- hard safety veto support in the reward model
- configurable safety modes: `conservative`, `standard`, `research`

## Repository Layout

```text
apps/
  api/                FastAPI surface for programmatic access
  research_console/   Streamlit workbench for interactive research
src/
  core/               Deterministic Bayesian and decision-theory engine
  evidence/           Local evidence registries and provenance
  llm/                Prompting, validation, routing, and structured outputs
  agent/              Orchestration, traces, reward model, offline rollouts
  datasets/           Benchmark dataset adapters and task normalization
  eval/               Benchmark runner, metrics, and reports
  ui/                 Shared viewmodels and chart helpers
  security/           Secrets, audit logging, and PII redaction
  utils/              Config, IDs, JSON helpers, logging, dates
artifacts/
  evals/ reports/ traces/ models/
tests/
  unit/ integration/ golden/ synthetic/
```

## Quick Start

1. Create a Python 3.11+ environment.
2. Install the package in editable mode.
3. Copy `.env.example` to `.env`.
4. Keep `PRIORI_ALLOW_LIVE_LLM=false` unless you are explicitly evaluating an external model offline.

```bash
python -m venv .venv
. .venv/Scripts/activate
pip install -e .[dev]
copy .env.example .env
python -m pytest
uvicorn apps.api.main:app --reload
streamlit run apps/research_console/app.py
```

## Benchmarks

PRIORI-X normalizes several task families into a shared research schema:

- `diagnosis_mcq`
- `diagnosis_open`
- `next_best_test`
- `triage`
- `evidence_verification`

Supported adapters include MedMCQA, MedQA, PubMedQA, FindZebra, and an optional MIMIC-like adapter gated behind explicit credentials and local documentation.

## Offline Self-Improvement

Agent Lightning integration is used as an offline adapter only.

The workflow is:

1. Run benchmark cases.
2. Collect traces, structured outputs, safety signals, and reward components.
3. Compare prompts, policies, and routing heuristics.
4. Promote only candidate policies that pass safety gates.

No hidden online updates are permitted.

## Running Evaluations

```bash
python -m eval.benchmark_runner
```

Generated artifacts are written to `artifacts/evals`, `artifacts/traces`, and `artifacts/reports`.

## API Surface

- `GET /api/health` for environment and readiness metadata
- `POST /api/analyze` for a structured research report from note text
- `POST /api/benchmark/sample` for a sample offline benchmark sweep

## Research Console

The Streamlit workbench includes:

- case intake
- diagnostic cockpit
- next-best-test table with stewardship fields
- safety and provenance view
- calibration panel
- JSON audit trail

## Known Limitations

- The shipped evidence catalogs are intentionally small starter registries and should be expanded with institution-approved sources before serious research use.
- External model integration is optional and disabled by default.
- The initial dataset adapters focus on normalization and offline evaluation, not direct dataset redistribution.
- The research console prioritizes inspectability over production deployment concerns.

## Development Guardrails

- no hardcoded API keys or tokens
- no `.env` commits
- secret scanning in pre-commit
- deterministic seeds for benchmark runs
- explicit provenance tags for medical claims where possible
