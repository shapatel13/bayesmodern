# PRIORI-X Handbook

This handbook is the practical operator guide for PRIORI-X.

Use it when you want to know:

- where to start
- which section of the app to use
- how to improve prompts safely
- what to ignore until later
- how to turn your own review into better offline performance

## What To Use First

If you are opening PRIORI-X for normal day-to-day use, focus on only these parts at first:

1. `Case Intake`
2. `Diagnostic Cockpit`
3. `Mechanism Layer`
4. `Next Best Test`
5. `Safety & Provenance`
6. `Prompt Improvement` only after you have benchmark data or reviewed cases

You do not need to use every research control on day 1.

## The Fastest Useful Workflow

1. Launch the app with:

```bash
python run_priori_x.py
```

2. Open the Streamlit app at `http://127.0.0.1:8501`.
3. Paste a case into `Case Intake`.
4. Leave `Case Policy` on `v1-deterministic` unless you are deliberately comparing policies.
5. Click `Analyze Case`.
6. Read the answer in this order:
   - `Diagnostic Cockpit`
   - `Mechanism Layer`
   - `Next Best Test`
   - `Safety & Provenance`
   - `Calibration Lab`
7. If the answer is weak or misses something, save that case as a reviewed case for later offline improvement.

## What Each Section Is For

## Case Intake

Use this for:

- a new vignette
- a copied note
- a quick scenario test

Best practice:

- paste a concise but clinically meaningful note
- include timing, vitals, major symptoms, and any already-done tests
- include obvious context like pregnancy, renal disease, anticoagulation, or instability

Do not expect:

- final treatment advice
- autonomous clinical decisions
- perfect parsing of very long unstructured notes

## Case Policy

This changes how conservative or stewardship-focused the deterministic engine is.

Use:

- `v1-deterministic` for your default baseline
- `v1-conservative-safety` when you want stronger safety bias
- `v1-stewardship` when you want more cost/test restraint
- `v1-sensitive-triage` when urgency detection matters most
- `v1-balanced-bayesian` when comparing a less rigid reasoning style

Recommendation:

- stay on `v1-deterministic` for routine use
- only switch policies when doing deliberate comparison work

## Diagnostic Cockpit

This is the main answer surface.

Use it to see:

- the ranked differential
- posterior probabilities
- what moved diagnoses up or down
- whether the model is confident or fragile

Best for:

- understanding the engine’s current hypothesis ranking
- spotting whether the top diagnosis is only barely ahead
- deciding if the reasoning looks clinically plausible before looking at next tests

When to trust it more:

- the top diagnosis is supported by clear findings
- uncertainty is moderate rather than fragile
- the provenance view is not warning about unsupported evidence

When to slow down:

- the top few diagnoses are close together
- the case is high-risk but the posterior is diffuse
- the contradictions or provenance warnings are non-empty

## Mechanism Layer

This is the physiology and hidden-state surface.

Use it to see:

- overlapping latent states rather than a forced single diagnosis
- mixed physiology such as congestion plus low forward flow
- whether fluid responsiveness remains uncertain
- whether bleeding, obstructive, ischemic, or medication-effect signals are dominating

Best for:

- cases where the disease differential remains broad
- mixed shock or mixed cardiorenal cases
- deciding whether the best next test should clarify physiology rather than just one diagnosis

Important:

- this layer complements the disease differential; it does not replace it
- multiple mechanism states can be active at once
- the mechanism layer is deterministic and inspectable, not prompt-first runtime reasoning

## Next Best Test

This is the “what should discriminate next” section.

Use it when:

- the differential is still broad
- you want a cheap/high-yield next step
- you want to avoid overtesting

Read these fields first:

- recommendation name
- rationale
- LR+ / LR-
- stewardship score
- disposition such as `worth_it_now`, `defer`, or `already_answered`

Good use cases:

- prioritizing between several possible tests
- deciding whether a bedside or cheaper test can come before imaging
- checking whether renal risk or radiation risk should push a test down

## Safety & Provenance

This is where you decide whether the answer is usable as research support.

Use it to inspect:

- triage urgency
- threshold action
- provenance badges
- contradictions
- warnings
- extracted medications and adverse events

This section matters most when:

- the case is unstable
- the patient is pregnant
- there is kidney disease
- the patient is on risky medications
- the top recommendation is invasive or expensive

If this section looks bad, do not move on to prompt optimization yet. Fix the benchmark or case labeling first.

## Calibration Lab

Use this when you want to know whether the engine is acting overconfident.

Best use:

- benchmark review
- comparing prompt versions
- spotting false certainty

This section is more important during evaluation than during a quick one-off case run.

## Audit Trail

Use this when you want full inspectability.

Best use:

- debugging
- reviewing a surprising result
- saving examples for later prompt improvement
- verifying exactly what the model or engine produced

## Research Lab

This is the experiment area, not the normal case-analysis area.

Use it when you want:

- benchmark rollouts
- policy comparisons
- prompt improvement
- experiment history

If you only want to analyze a case, you can ignore most of this section at first.

## Manual Rollout

Use this to run one dataset directly.

Good first datasets:

- `medmcqa` for broad medical QA
- `medqa` for harder clinical reasoning
- `pubmedqa` for evidence verification
- `findzebra` for rare disease reasoning

Use this when:

- you want to test a single benchmark family
- you are comparing prompts or policies on one domain

## Specialty Tracks

These are curated presets.

Use:

- `Core Diagnostic Lab` for general medical QA
- `USMLE Reasoning Lab` for harder reasoning
- `Rare Disease Lab` for broad hypothesis generation
- `Evidence Verifier Lab` for evidence-grounding behavior
- `Generation Audit Lab` for MedVAL-style safety and hallucination audit
- `ED Triage Lab` for triage
- `Medication Safety Lab` for ADE and medication extraction

If you are unsure where to start:

- start with `Core Diagnostic Lab`
- then try `Evidence Verifier Lab`
- then `Rare Disease Lab`

## Prompt Improvement

This is where the app gets better, but only offline.

The important rule:

- PRIORI-X should improve from experiments and reviewed cases, not silently from raw usage

Use prompt improvement when:

- you have a benchmark dataset
- you have reviewed failure cases
- you want to test whether a new prompt improves held-out performance

Do not use prompt improvement when:

- you only ran one case and want the model to “learn” immediately
- the failure is clearly bad labeling or a bad benchmark row
- the case still contains PHI that has not been de-identified

## How Prompt Optimization Works

The system uses two loops:

1. Microsoft Agent Lightning improves prompt templates offline
2. PRIORI-X evaluates whether the candidate prompt is actually safer or better on held-out validation cases

Promotion only happens if the held-out gate passes.

That means:

- no silent live updates
- no automatic learning from user traffic
- no promotion without validation

## Best Prompt-Improvement Sequence

1. Start with the active prompt.
2. Run `Core Diagnostic Lab` or `continuous_improvement_feedback_lab`.
3. Review failure cases.
4. Add good reviewed cases to `reviewed_cases`.
5. Rerun prompt training.
6. Check whether safety, contradictions, unsupported claims, and mean reward improved.
7. Only keep the new prompt if the gate promotes it.

## When To Use Each Curriculum

## `broad_medical_feedback_lab`

Use when:

- you want general medical improvement
- you want public Hugging Face datasets only

Best for:

- early broad improvements
- first-pass Lightning work

## `diagnostic_reasoning_feedback_lab`

Use when:

- you mainly care about differential reasoning and distractor handling

## `evidence_rare_feedback_lab`

Use when:

- you want better abstention
- you want better evidence-grounding
- you want rare disease coverage

## `physician_audit_feedback_lab`

Use when:

- you want safer output behavior
- you have MedVAL-Bench locally

## `reviewed_cases_feedback_lab`

Use when:

- you have your own clinician-reviewed local cases
- you want the app to improve on your real failure patterns

## `continuous_improvement_feedback_lab`

Use when:

- you want the best default long-term loop
- you want public QA plus your reviewed cases together

This should usually be your main improvement curriculum.

## How To Add Reviewed Cases

Use the template at:

`artifacts/reviewed_cases/reviewed_cases.template.jsonl`

Put your local reviewed dataset somewhere private and set:

`PRIORI_REVIEWED_CASES_PATH`

Only include:

- de-identified cases
- reviewed gold answers
- reviewed differential or acceptable tests where possible
- final review status

Do not include:

- raw PHI
- unreviewed guesses
- large unfiltered app logs

## Recommended Daily Workflow

## If You Just Want To Use The App

1. Launch app
2. Run case
3. Read cockpit
4. Read next test
5. Check safety/provenance
6. Save interesting misses for review later

## If You Want To Improve The App Weekly

1. Gather reviewed misses
2. De-identify them
3. Add them to `reviewed_cases`
4. Run:

```bash
python -m eval.experiment_cli auto-improve --train-cap-per-component 8 --validation-cap-per-component 4
```

5. Inspect the training summary
6. Keep the candidate only if promoted

## If You Want To Improve One Specific Capability

Use this mapping:

- wrong differentials: `diagnostic_reasoning_feedback_lab`
- weak evidence behavior: `evidence_rare_feedback_lab`
- hallucination or risky output: `physician_audit_feedback_lab`
- local recurring misses: `reviewed_cases_feedback_lab`
- broad general improvement: `continuous_improvement_feedback_lab`

## Windows And WSL

If prompt training says it is blocked, that usually means Windows does not yet have a Linux distro installed for WSL.

Fix with:

```powershell
wsl --install Ubuntu
```

Then run the prompt-training command from inside WSL.

## Most Important Do And Don’t

Do:

- use the app for inspectable reasoning support
- improve it with reviewed offline datasets
- prefer the active prompt unless you are testing
- use held-out validation as the promotion gate

Don’t:

- expect it to learn from one case instantly
- treat it as a clinical autopilot
- promote prompts based only on intuition
- feed raw patient traffic directly into training

## Best Starting Commands

Run the app:

```bash
python run_priori_x.py
```

See what is configured:

```bash
python -m eval.experiment_cli status
python -m eval.experiment_cli list-prompts
```

Run the default improvement loop:

```bash
python -m eval.experiment_cli auto-improve --train-cap-per-component 8 --validation-cap-per-component 4
```
