# AGENTS.md

Read this file first before making changes to PRIORI-X.

## Core Rule

PRIORI-X must reason through reusable structured evidence and deterministic probabilistic/mechanistic logic, not through disease-specific prompt templates.

Do not rebuild this project from scratch.
Do not simplify it into a chatbot.
Do not replace the current hybrid architecture.
Do not move core runtime reasoning into disease-specific prompts.

## Project Identity

PRIORI-X is a clinician-facing, research-only Bayesian decision-support workbench.

It is not:

- a clinical autopilot
- an order-entry system
- a live reinforcement loop over patient traffic

It is:

- inspectable
- offline-evaluation-first
- safety-constrained
- provenance-aware

Main run command:

```bash
python run_priori_x.py
```

Main app surfaces:

- Streamlit research console
- FastAPI backend

## Architecture To Preserve

### 1. Parser / Extraction Step

- Converts vignette or note text into structured context
- Includes offline keyword and synonym logic
- May optionally use live OpenAI parsing only when explicitly enabled
- Handles medications, ADEs, bleeding clues, chest pain clues, and similar structured findings

### 2. Curated Bayesian Engine

- Uses a deterministic hypothesis pool with disease profiles and supporting / contradicting evidence
- Performs Bayesian updates, contextual prior adjustments, explaining-away, calibration labeling, and Monte Carlo uncertainty propagation
- Remains the main trustworthy math layer

### 3. Open-World Reasoning

- Used only when curated reasoning is weak or uncertain
- Open-world proposals are never returned directly as final answers
- Proposed hypotheses and tests must be converted into structured candidates and re-scored by deterministic logic
- The visible reasoning-mode trace showing `curated Bayesian` vs `hybrid open-world` must remain

### 4. Next-Best-Test Engine

- Ranks tests using expected posterior movement, entropy / information gain, threshold movement, stewardship, cost, downstream cost, actionability, urgency, and safety penalties

### 5. Evidence / Provenance Layer

- Curated LR registry with schema, validator, loader, and query layer
- Supports `sourced`, `estimated`, and `seed_only` evidence states
- Must never silently treat `seed_only` rows as sourced
- Provenance badges must remain visible in the app

### 6. Offline Self-Improvement

- Reviewed cases, benchmark datasets, prompt optimization, deterministic policy optimization, and Microsoft Agent Lightning are offline-only
- Never recommend autonomous live learning from user traffic

## Critical Anti-Regression Rule

Do not reintroduce reliance on disease-specific preformatted prompts as the primary reasoning mechanism.

This project previously had a brittle failure mode where reasoning could become too dependent on narrow disease or syndrome prompt templates. That must not return.

Runtime reasoning must remain centered on:

- structured observations
- reusable evidence primitives
- deterministic Bayesian updates
- provenance-aware evidence weighting
- latent-state / mechanism inference
- deterministic re-scoring of hypotheses and tests

## Allowed Uses Of Prompts / LLMs

- optional structured extraction
- optional open-world hypothesis proposal
- offline evaluation and improvement workflows only

## Disallowed Pattern

- routing a case into a disease-specific prompt that contains the main reasoning logic
- relying on syndrome-specific prose templates as the source of diagnostic inference
- implementing prompt-first runtime logic for shock, GI bleed, ACS, AKI, hyponatremia, or similar syndromes

## Preferred Pattern

- build reusable observation-to-evidence mappings
- allow one observation to affect multiple latent states and hypotheses
- synthesize disease-level conclusions from structured evidence and latent states
- ensure reasoning generalizes to mixed, atypical, and previously unseen cases

## Current Development Goal

Make PRIORI-X more clinically realistic by adding a deterministic latent-state / mechanism reasoning layer for overlapping physiology, while preserving the current disease-level Bayesian engine and hybrid research architecture.

Do not replace disease-level reasoning.
Add a new deterministic layer that complements it.

## New Latent-State / Mechanism Layer

Add a deterministic latent-state / mechanism inference layer that can represent overlapping hidden processes such as:

- distributive physiology
- impaired contractility / cardiogenic physiology
- low effective arterial volume / low preload
- venous congestion
- fluid responsiveness
- fluid intolerance
- intrinsic tubular injury
- capillary leak / permeability process
- hemorrhagic tendency / active blood loss
- thrombotic / ischemic tendency
- medication or toxicity effect
- obstructive physiology where relevant

Desired behavior:

- a case can have multiple active latent states at once
- observations can support or contradict several latent states simultaneously
- the system should not force mutually exclusive diagnoses when physiology is mixed
- disease-level outputs should remain, but they should be enriched by mechanistic state estimates

Example style of output:

- mixed cardiogenic and distributive physiology
- high congestion signal with uncertain fluid responsiveness
- posterior differential remains broad, but mechanism layer strongly favors venous congestion plus low forward flow

## Mathematical Design Requirements

1. Preserve the current disease-level Bayesian engine unless narrowly refactoring for compatibility.
2. Add a deterministic latent-state engine with:
   - structured latent states
   - observation-to-state evidence mappings
   - graded belief updates
   - support for soft evidence / uncertain observations
   - provenance and confidence tracking
3. Do not use LLMs to compute the core latent-state math.
4. Keep the latent-state engine inspectable and testable.
5. Continue using Monte Carlo uncertainty propagation where appropriate.
6. Keep provenance explicit for all evidence weights.
7. Prefer reusable evidence / state machinery over disease-specific reasoning branches.

## Interaction With Existing System

- parser outputs continue to feed structured findings
- structured findings now feed both:
  - the disease-level Bayesian engine
  - the latent-state mechanism engine
- the open-world layer remains optional and gated
- open-world proposals may suggest additional hypotheses or tests, but final outputs must still be re-scored by deterministic logic
- next-best-test ranking should be upgraded to consider:
  - disease discrimination
  - latent-state uncertainty reduction
  - threshold crossing
  - safety / stewardship / cost
  - mechanistic information gain
- reporting and UI should surface both:
  - disease-level differential
  - mechanism / physiology state summary

## Causal / Intervention-Aware Upgrade

Add a lightweight deterministic causal reasoning layer for counterfactual and test-value logic.

Do not implement a giant causal inference framework.

For this pass, prefer a structured intervention-effect model that can answer questions such as:

- what latent states would be clarified by PLR with LVOT VTI
- how diuresis response would help distinguish competing mechanisms
- how repeat hemoglobin changes bleeding-related belief
- how troponin delta, ECG change, or bedside echo alter ischemic vs non-ischemic reasoning
- which next tests have the highest expected mechanistic information gain

This causal/intervention layer should improve the next-best-test engine, not replace it.

## Targeted Engineering Tasks

1. Summarize the current architecture and identify the smallest integration points for a new latent-state engine.
2. Propose the minimal set of new schemas and modules needed to add this without breaking the current app.
3. Implement the latent-state engine in a deterministic, inspectable way.
4. Add evidence mappings from structured observations to latent states.
5. Upgrade next-best-test ranking so latent-state uncertainty reduction is part of utility.
6. Update the research console and API outputs so both disease and mechanism reasoning are visible.
7. Preserve backward compatibility where reasonable.
8. Add tests and example cases demonstrating mixed physiology and non-binary reasoning.

Preferred change areas:

- `src/core`: latent-state math, integration with differentials, test ranking
- `src/evidence`: latent-state evidence registry / mappings / provenance
- `src/llm`: structured extraction / output compatibility only as needed
- `src/agent`: only if needed for offline eval / improvement compatibility
- `apps/research_console`: UI additions for mechanism display and reasoning trace
- `apps/api`: response schema additions
- `src/eval`: evaluation hooks for mechanism-layer behavior where appropriate

## Debugging Rule

When debugging wrong answers, explicitly separate possible causes into:

- parser failure
- missing disease hypothesis / evidence path
- missing latent-state mapping
- bad prior / likelihood logic
- next-best-test ranking issue
- open-world expansion behavior
- UI / reporting issue

## What Not To Do

- do not remove the curated disease hypothesis pool
- do not return raw open-world LLM suggestions as final truth
- do not recommend online RL from user traffic
- do not turn the app into a generic conversational assistant
- do not hide uncertainty
- do not hide provenance
- do not hardcode narrow disease prompt templates as the main reasoning mechanism
- do not rewrite the repo broadly unless strictly necessary

## Deliverables For This Development Pass

1. Plain-English summary of current architecture and integration points
2. Highest-risk failure modes in the current design
3. Design for a latent-state / mechanism engine that preserves the current hybrid Bayesian system
4. Concrete module-by-module change plan
5. Initial implementation of the latent-state layer
6. Updated next-best-test logic incorporating mechanistic information gain
7. UI / API updates to display mechanism-layer outputs
8. Regression-safety notes so the current app does not break
9. Example cases showing mixed physiology and non-binary reasoning
10. Concise README or docs update documenting the new reasoning layer

## Execution Constraints

- start by summarizing the current system in plain English
- then list high-risk failure modes
- then propose the smallest high-leverage architecture changes
- then implement in bounded steps
- do not over-plan
- do not rewrite the whole repo
- preserve the research-only, Bayesian-first, provenance-aware architecture
- prefer deterministic fixes over prompt tricks
- keep any prompt changes confined to offline eval / Lightning workflows only

## Validation Expectations

- run relevant tests and checks after code changes
- if tests fail, fix regressions before stopping
- stop after the first coherent working vertical slice and summarize:
  - files changed
  - what works
  - what remains
  - risks or assumptions
