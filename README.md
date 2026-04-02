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

For a practical operator guide with section-by-section advice, prompt-improvement workflow, and “what to use when,” see [docs/HANDBOOK.md](C:/Users/msmsh/Downloads/bayesmodern/docs/HANDBOOK.md).

1. Create a Python 3.11+ environment.
2. Install the package in editable mode.
3. Copy `.env.example` to `.env`.
4. Keep `PRIORI_ALLOW_LIVE_LLM=false` unless you are explicitly evaluating an external model offline.
5. The default OpenAI snapshot in this repo is `gpt-5.4-nano-2026-03-17` for parser, reasoner, and verifier routing when live offline eval is enabled.

```bash
python -m venv .venv
. .venv/Scripts/activate
pip install -e .[dev,datasets,lightning]
copy .env.example .env
python -m pytest
python run_priori_x.py --check
python run_priori_x.py
uvicorn apps.api.main:app --reload
streamlit run apps/research_console/app.py
```

The easiest integrated launch path from a normal VS Code terminal is:

```bash
python run_priori_x.py
```

That starts the FastAPI surface and the Streamlit workbench together. If you want a preflight check without opening the app, use:

```bash
python run_priori_x.py --check
```

## Benchmarks

PRIORI-X normalizes several task families into a shared research schema:

- `diagnosis_mcq`
- `diagnosis_open`
- `next_best_test`
- `triage`
- `evidence_verification`
- `generation_audit`

Supported adapters include MedMCQA, MedQA, PubMedQA, FindZebra, MedVAL-Bench, and an optional MIMIC-like adapter gated behind explicit credentials and local documentation.

Current Hugging Face dataset targets:

- `openlifescienceai/medmcqa`
- `augtoma/medqa_usmle`
- `qiaojin/PubMedQA` using `pqa_labeled`
- `findzebra/case-reports`

Lightning feedback curricula can now blend these sources into a single offline training bundle for Microsoft Agent Lightning export and offline policy search.

Credentialed local dataset targets include:

- `medval_bench` for physician-graded medical generation audit from a single local CSV
- `mietic` for ED triage
- `n2c2_2018_track2` for medication safety
- `reviewed_cases` for your own clinician-reviewed, de-identified failure cases and gold labels

## Offline Self-Improvement

Microsoft Agent Lightning integration is used as an offline adapter only.

The workflow is:

1. Run benchmark cases.
2. Collect traces, structured outputs, safety signals, and reward components.
3. Export an Agent Lightning sandbox bundle with train/validation tasks, transitions, traces, and a baseline prompt template.
4. Compare prompts, deterministic reasoning policies, and routing heuristics.
5. Promote only candidate policies that pass safety gates.

No hidden online updates are permitted.

PRIORI-X now maintains a local prompt registry. The active prompt is used by default for benchmark runs, the API, and the research console. Microsoft Agent Lightning improves that prompt offline, then PRIORI-X evaluates the candidate on held-out validation cases and only promotes it if safety and quality do not regress.

The safe long-term loop is:

1. use the app
2. convert interesting failures into de-identified reviewed cases
3. add them to `reviewed_cases`
4. run offline prompt improvement against public QA plus reviewed cases
5. promote only prompts that clear the held-out gate

On Windows, PRIORI-X exports and evaluates everything locally, but native Agent Lightning prompt optimization still requires Linux or WSL2 with an installed distro. If `wsl --status` works but no distro is installed, the CLI and UI will now tell you explicitly to run `wsl --install Ubuntu`.

## Running Evaluations

```bash
python -m eval.benchmark_runner
python -m eval.experiment_cli status
python -m eval.experiment_cli list-prompts
python -m eval.experiment_cli list-presets
python -m eval.experiment_cli list-curricula
python -m eval.experiment_cli list-policies
python -m eval.experiment_cli run-dataset-rollout medmcqa --train-limit 8 --validation-limit 4
python -m eval.experiment_cli run-preset-rollout core_diagnostic_lab
python -m eval.experiment_cli run-preset-rollout generation_audit_lab
python -m eval.experiment_cli run-curriculum-rollout broad_medical_feedback_lab --train-cap-per-component 1 --validation-cap-per-component 1
python -m eval.experiment_cli optimize-dataset-policy medmcqa --train-limit 8 --validation-limit 4
python -m eval.experiment_cli optimize-curriculum-policy broad_medical_feedback_lab --train-cap-per-component 2 --validation-cap-per-component 1
python -m eval.experiment_cli train-curriculum-prompt continuous_improvement_feedback_lab --train-cap-per-component 8 --validation-cap-per-component 4
python -m eval.experiment_cli auto-improve --train-cap-per-component 8 --validation-cap-per-component 4
python -m eval.experiment_cli list-experiments
```

The broad public curriculum is the fastest way to get MedMCQA, MedQA, PubMedQA, and FindZebra into one Lightning-compatible bundle. Use small per-component caps for a quick morning pass, then increase them for longer offline optimization runs and policy search.

If you have the PhysioNet MedVAL-Bench CSV locally, set `PRIORI_MEDVAL_BENCH_PATH` and use `generation_audit_lab` or `physician_audit_feedback_lab` to improve the verifier side of PRIORI-X. The loader partitions the single CSV deterministically and balances limited runs across MedVAL task groups.

If you want the app to improve from your own review work, set `PRIORI_REVIEWED_CASES_PATH` to a JSONL/JSON/CSV export matching the template in `artifacts/reviewed_cases/reviewed_cases.template.jsonl`. The `continuous_improvement_feedback_lab` curriculum will mix those reviewed cases with public QA automatically when the file is present.

If native Lightning training is blocked on Windows, install a distro first:

```powershell
wsl --install Ubuntu
```

Then rerun the same `train-curriculum-prompt` or `auto-improve` command from inside WSL in the repo directory.

For a Windows-first morning start, you can use either:

```powershell
python run_priori_x.py
```

Or, if you prefer the PowerShell wrapper:

```powershell
powershell -ExecutionPolicy Bypass -File scripts\start_priori_x.ps1
```

The optional warmup script still exists if you want bundled smoke-test artifacts before external datasets finish loading:

```powershell
powershell -ExecutionPolicy Bypass -File scripts\warmup_priori_x.ps1
```

Generated artifacts are written to `artifacts/evals`, `artifacts/traces`, and `artifacts/reports`.

To build a benchmark-driven rollout with an Agent Lightning sandbox bundle, use the offline rollout entry points in `src/agent/offline_rollout.py`. The rollout writes:

- `benchmark_report.md`
- `benchmark_summary.json`
- `benchmark_traces.json`
- `lightning_train_tasks.jsonl`
- `lightning_validation_tasks.jsonl`
- `lightning_transitions.jsonl`
- `lightning_bundle_manifest.json`
- `experiment_summary.json`

## API Surface

- `GET /api/health` for environment and readiness metadata
- `POST /api/analyze` for a structured research report from note text
- `POST /api/benchmark/sample` for a sample offline benchmark sweep
- `GET /api/research/status` for dataset and Microsoft Agent Lightning runtime metadata
- `GET /api/research/datasets` for the benchmark catalog
- `GET /api/research/curricula` for named multi-dataset Lightning feedback curricula
- `GET /api/research/policies` for deterministic Bayesian policy variants available to the optimizer
- `GET /api/research/prompts` for the active prompt registry
- `GET /api/research/presets` for named specialty benchmark tracks
- `POST /api/research/benchmark/dataset` for an on-demand dataset benchmark summary
- `POST /api/research/rollout/dataset` for an artifact-producing offline rollout
- `POST /api/research/rollout/curriculum` for a multi-dataset Lightning curriculum rollout
- `POST /api/research/rollout/preset` for a named specialty-track rollout
- `POST /api/research/optimize/dataset-policy` for offline policy search on a single dataset
- `POST /api/research/optimize/curriculum-policy` for offline policy search across a Lightning curriculum
- `POST /api/research/prompts/promote` for manual prompt promotion
- `POST /api/research/train/dataset-prompt` for dataset-specific Lightning prompt improvement
- `POST /api/research/train/curriculum-prompt` for curriculum-based Lightning prompt improvement
- `POST /api/research/train/auto-improve` for the default continuous-improvement loop
- `GET /api/research/experiments` for recorded experiment summaries
- `POST /api/research/experiments/compare` for before/after comparison

## Research Console

The Streamlit workbench includes:

- case intake
- diagnostic cockpit
- next-best-test table with stewardship fields
- safety and provenance view
- calibration panel
- research lab controls for dataset rollouts
- Lightning feedback curriculum controls for multi-dataset Hugging Face training bundles
- offline policy search over deterministic Bayesian policies with safety-gated promotion
- prompt registry visibility and active-prompt tracking
- prompt-improvement controls for the continuous offline self-improvement loop
- specialty preset tracks for ED triage, medication safety, rare disease, and evidence verification
- physician-audit preset and curriculum for MedVAL-Bench generation-risk benchmarking
- experiment registry and comparison view
- promotion-gate verdicts that block unsafe prompt/policy regressions
- JSON audit trail

## OpenAI Model Configuration

When `PRIORI_ALLOW_LIVE_LLM=true` and `PRIORI_DEFAULT_MODEL_PROVIDER=openai`, PRIORI-X uses:

- parser: `gpt-5.4-nano-2026-03-17`
- reasoner route: `gpt-5.4-nano-2026-03-17`
- verifier route: `gpt-5.4-nano-2026-03-17`

The live OpenAI path is still offline-evaluation-only. The deterministic Bayesian engine remains the source of truth for posterior math and threshold calculations.

## Known Limitations

- The shipped evidence catalogs are intentionally small starter registries and should be expanded with institution-approved sources before serious research use.
- External model integration is optional and disabled by default.
- The initial dataset adapters focus on normalization and offline evaluation, not direct dataset redistribution.
- Native Microsoft Agent Lightning training is expected to run in Linux or WSL2; the Windows workflow exports a compatible sandbox bundle instead.
- The research console prioritizes inspectability over production deployment concerns.

## Development Guardrails

- no hardcoded API keys or tokens
- no `.env` commits
- secret scanning in pre-commit
- deterministic seeds for benchmark runs
- explicit provenance tags for medical claims where possible
