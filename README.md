# Arbiter

**A fraud decision system, not a fraud classifier.** Given a payment, it decides — by
what the outcome actually costs in real rupees, not an abstract score — whether to let it
through, ask for extra verification, or block it. Built solo for Razorpay's AI Buildathon
2026, Track 02 (AI Risk Manager).

| | |
|---|---|
| **Shipped model** | A **2-model ensemble** (XGBoost + untuned LightGBM, simple-averaged after independent calibration) — a 3rd model (CatBoost) and further tuning were both tried and empirically rejected. [Full evidence trail](docs/experiments.md) |
| **Measured lift** | **+₹1.678 crore** vs no fraud system (95% CI ₹1.510cr–₹1.850cr), **+₹77.03 lakh** vs the industry-default 0.5 cutoff (CI ₹64.9L–₹89.5L), on one untouched test month (92,427 real transactions) — a real, further-confirmed improvement over the single-model baseline (+₹1.54cr / +₹64.3L) |
| **Model quality** | PR-AUC **0.5597** / ROC-AUC **0.9126** — 16.1x random on this dataset's ~3.5% fraud rate |
| **False-positive cost** | **223** genuine customers wrongly blocked, exact cost **₹13.20 lakh** — lower than the single model's 233 / ₹17.97L on both counts, found while optimizing for something else entirely |
| **Honesty check** | 100.16% of the ₹17.355cr headline comes straight from real fraud labels — the modeled step-up component is actually slightly negative, not a rounding artifact |
| **AI-judgment evidence** | XGBoost beats a real LLM (`gpt-oss:20b`, given a fair shot) by **3.65x** on this task, ~6 orders of magnitude faster per call — [full benchmark](docs/experiments.md) |
| **Causal-honesty cost** | Refusing to use future information (unlike the original Kaggle-winning solution) costs only **+0.0066 PR-AUC** — honesty is nearly free here |

Full story, in order, including everything that broke: [`journal/`](journal/build-log.md).

---

## Quickstart

Requires the six trained-model artifacts (not committed to git — see [`artifacts/README.md`](artifacts/README.md) for how to generate and download them from the companion Kaggle notebooks). The shipped model is a 2-model ensemble (XGBoost + LightGBM), so both models' files are required — the engine fails closed if either is missing.

```bash
git clone <this-repo-url>
cd arbiter
python -m venv .venv

# Windows
.venv\Scripts\pip install -r requirements.txt
# Mac/Linux
.venv/bin/pip install -r requirements.txt

# Place model.json, calibrator.json, model_lgb.txt, calibrator_lgb.json,
# feature_manifest.json, sample_transactions.json into artifacts/ — see
# artifacts/README.md for exactly how

# Windows
.venv\Scripts\python scripts\demo_engine.py
# Mac/Linux
.venv/bin/python scripts/demo_engine.py
```

Expected output: `ALL CHECKS PASSED` — idempotency, replay/tamper detection, fail-closed
behavior, both ensemble members' raw scores in the audit record, and a live decision on a
real held-out transaction, all running on plain CPU, no GPU, no training.

**Dashboard:** open [`dashboard.html`](dashboard.html) directly in any browser — double-click
it, or `file://` the path. One self-contained file: PR/cost curves with a live threshold
slider, review queue, audit log, and sensitivity map, all populated with real data captured
from this project's own test runs. No server, no install, no Python — just a file in this
repo. (An earlier version ran locally via Streamlit; retired in favor of this.)

---

## The problem, in money

A merchant's dashboard doesn't have an "accuracy" line — it has a P&L. Fraud that slips
through costs the full transaction amount plus a chargeback dispute fee plus a payment
processing fee that's never refunded. Blocking a genuine customer by mistake costs that
sale *and* probably that customer's future business. Most fraud-detection projects
optimize a classification score and stop there, leaving the actual business decision —
and its cost — as someone else's problem.

Arbiter prices every one of those outcomes in real ₹ and picks whichever action loses the
least money, **per transaction** — not one global threshold, because a ₹500 payment and a
₹50,000 payment don't carry the same risk at the same fraud probability. Full mechanism
and worked numbers: [CLAUDE.md §6](CLAUDE.md#6-architecture).

## Approach

1. **Data**: [IEEE-CIS Fraud Detection](https://www.kaggle.com/c/ieee-fraud-detection) (Kaggle) — 590k real transactions, ~3.5% fraud, real semantic features (card, email, device). Chosen over anonymized alternatives specifically so SHAP explanations are true statements about the world, not noise. Split **temporally** (train ≤ day 120, validate 121–150, test > 150) — never randomly, so nothing is ever scored using information that wouldn't exist yet in production.
2. **Model**: XGBoost, tuned via a 60-trial Optuna search on PR-AUC (not ROC-AUC — the latter is flattered by the ~96.5% negative class), then calibrated with Platt scaling. Isotonic calibration was tried first and rejected — it silently collapsed 91,271 distinct scores to 323, costing real ranking accuracy for no net benefit. Full ladder, every step measured: [`docs/experiments.md`](docs/experiments.md).
3. **Decision**: the calibrated probability feeds a cost model (chargeback fee, processing fee, merchant margin, step-up friction cost — all sourced from Razorpay's own disclosed pricing or cited industry studies, never invented) that computes the expected ₹ value of allow/step-up/block for every transaction and picks the best one.
4. **Explanation**: SHAP contributions for flagged transactions only, turned into a one-sentence narrative by an LLM — which never decides, only describes. Delete the LLM entirely and every decision stays byte-identical; only the prose degrades to a deterministic template. Proven, not just claimed — see [Fallback design](#fallback-design-proven-not-claimed) below.
5. **Serving**: a plain-CPU Python package (`src/`) that loads the trained artifact and scores transactions with no GPU, no retraining, no internet dependency — see [Architecture](#architecture).

## Results

| Policy | Total value, test month |
|---|---|
| No fraud system | ₹15.68 crore |
| Naive 0.5 threshold (industry default) | ₹16.585 crore |
| **Arbiter (2-model ensemble)** | **₹17.355 crore** |

Policy mix: 95.6% allow, 3.0% step-up, 1.4% block. Full metrics (PR-AUC, ROC-AUC,
precision/recall at the operating threshold, the reliability diagram, sensitivity across
margin × chargeback-fee assumptions): [`docs/eval_report.md`](docs/eval_report.md),
[`docs/experiments.md`](docs/experiments.md).

### The two differentiators

**Would an LLM just do this job directly?** Benchmarked head-to-head against XGBoost on
the same 200 held-out transactions: XGBoost PR-AUC 0.5735 vs `gpt-oss:20b` PR-AUC 0.1571
— a real, capable model given a genuinely fair shot (not a strawman), still losing by
3.65x on accuracy and ~6 orders of magnitude on latency (microseconds vs a 6.7s median).
This is the evidence behind "where we chose not to use AI."

**What does refusing to cheat actually cost?** The original Kaggle-winning solution uses
each client's *future* transactions to score a *past* one — legal on a static leaderboard,
impossible in a gateway with a 200ms decision window. Built that leaky version for real
(not just described why it's wrong) and measured the gap: honest causal model 0.5446
PR-AUC vs the full Kaggle-legal reproduction 0.5512 — **a +0.0066 PR-AUC difference**.
Causal honesty is nearly free here. A genuine surprise found along the way: one of the two
leaky techniques (client-mean post-processing) actually *hurt* PR-AUC, for the same reason
isotonic calibration did earlier in the project — see `docs/experiments.md` for the
mechanism.

## Architecture

```
Kaggle notebooks (GPU)          Local repo (CPU-only)
─────────────────────           ─────────────────────
research: EDA, features,   ──▶  src/store.py     client history (causal, online)
Optuna tuning, training         src/features.py  raw txn → model-ready vector
       │                        src/model.py     load artifact, score, version-guard
       ▼                        src/policy.py    calibrated p → allow/step-up/block
 export artifacts:               src/audit.py     append-only log, idempotent, replayable
 model.json, calibrator,        src/explain.py   SHAP, flagged transactions only
 feature_manifest,              src/narrative.py LLM + template fallback
 sample_transactions             src/engine.py    orchestrates all of the above
```

Research happens in notebooks (`notebooks/`, numbered, each a self-contained jupytext
pair — `NN_name.py` source + `NN_name.ipynb` executed with outputs; see
[`notebooks/README.md`](notebooks/README.md) for the stage-by-stage index); the product
is a plain Python package (`src/`) that only ever *loads* a trained artifact — it never
trains anything, never needs a GPU. This split is itself a build-quality signal: a
notebook can't demonstrate graceful degradation, an audit trail, or "would you trust it."

### Fallback design, proven not claimed

| Failure | Behavior | Verified how |
|---|---|---|
| Model artifact missing/corrupt | Fail closed to a conservative rules baseline, never silently allow | `scripts/demo_engine.py` checks 11–13; `tests/test_engine.py` |
| LLM unavailable/times out/returns garbage | Decision unchanged; narrative falls back to a deterministic template | `tests/test_narrative.py` — timeout and garbage-response each their own named test, via dependency injection, no live network call |
| Required feature missing | Impute, flag degraded, widen the step-up band (not just relabeled — a real bug where this mechanism was mathematically incapable of working was caught and fixed, see `journal/`) | `tests/test_policy.py::test_degraded_mode_widens_the_stepup_band` |
| Duplicate/replayed transaction ID | Return the original decision, never re-decide | `scripts/demo_engine.py`, idempotency checks; `tests/test_engine.py` |
| A stored audit record is tampered with | `verify_and_replay()` re-derives the decision from stored inputs and flags a mismatch | `tests/test_audit.py`, `tests/test_engine.py` |

Every one of these was broken on purpose and confirmed to recover correctly — not assumed
from reading the code. Full account of every bug found this way (20, across the project —
including three real defects two rounds of automated AI code review caught after the fact:
a train/serve ddof mismatch on one feature, a malformed-artifact path that crashed instead
of failing closed, and a NaN calibrator coefficient that would have produced a silent
`allow`):
[`journal/`](journal/build-log.md).

## Honest limitations

- **US data, Indian cost model.** IEEE-CIS is US card-not-present e-commerce fraud; the
  *method* transfers, the specific features and thresholds would need Indian data to
  match Razorpay's real UPI-heavy mix.
- **Card-level identity reconstruction was attempted and cut.** Tried to reproduce the
  1st-place solution's client-identity purity (target 96.9/2.9/0.2 pure-legit/pure-fraud/
  mixed); best achieved was 2.11% mixed across 8 tested key configurations, ~11x the
  target. Root-caused to their reliance on an undocumented separate matching script.
  Scoped out rather than shipped on an unreliable identity signal — full investigation in
  `docs/experiments.md`.
- **Step-up outcomes are modeled, not measured.** This dataset has no real record of
  whether a specific step-up challenge actually stopped a fraudster or made a genuine
  customer abandon — those use cited industry population rates. Stated explicitly: the
  modeled component is actually *negative* for the shipped ensemble (−0.16% of the
  headline ₹17.355cr) — more than 100% of the reported value is hard-computed from real
  labels.
- **Cost parameters are assumptions, sourced but not certain.** Chargeback fee and MDR are
  Razorpay's own disclosed pricing; merchant margin and the LTV penalty for a wrongly
  blocked customer are the genuinely uncertain ones, swept in a sensitivity analysis
  rather than reported as fact.
- **A handful of held-out transactions this project's own testing produced legitimately
  probability-less records** (fail-closed decisions made while the model artifact was
  deliberately broken during testing) — visible in the live audit log, correctly labeled,
  not hidden.

## Attribution

This project borrows heavily from the [1st-place IEEE-CIS Fraud Detection solution](https://www.kaggle.com/c/ieee-fraud-detection/discussion/111284) ([technical writeup](https://www.kaggle.com/c/ieee-fraud-detection/discussion/111321))
by Chris Deotte and team, and says so explicitly rather than presenting it as independent
work. **Taken:** the UID construction technique (`card1 + addr1 + D1n`), the card-level
chargeback label semantics, group aggregates as features, time-consistency feature
screening, V-column reduction by NaN structure, and the "unseen clients, not time drift"
diagnosis. **Deliberately diverged:** causal (expanding-window) aggregates instead of
full-history ones, no future-leaking post-processing, PR-AUC over ROC-AUC as the tuning
metric, a cost-minimizing objective instead of AUC-maximizing, a three-way decision output
instead of a score, and a real explainability layer. Component B of this project exists
specifically to measure that divergence's cost, not to claim superiority — their model
beats this one on the raw Kaggle task, by design; different games, see `CLAUDE.md`.

## Repository layout

```
notebooks/    Kaggle research record — EDA, feature engineering, tuning, cost model,
              the Kaggle-legal comparison model. Numbered, self-contained, run on Kaggle.
src/          The product — CPU-only decision engine. store/features/model/policy/
              audit/explain/narrative/engine.
scripts/      demo_engine.py (proves the engine), llm_benchmark.py (the LLM-vs-XGBoost evidence)
tests/        98 pytest tests — idempotency, replay/tamper detection, fail-closed paths (missing
              AND malformed artifacts), both version guards, batch-vs-online feature parity, cost-model boundaries, every LLM fallback mode
artifacts/    Trained model + calibrator + sample data — not committed, see its README
dashboard.html  The dashboard — self-contained, open directly in a browser, no server
docs/         experiments.md (the measured ladder + both evidence experiments), plots,
              docket.html (narrative walkthrough, self-contained, for pitch narration)
journal/      Engineering log — every real bug, found and fixed, written as it happened
CLAUDE.md     Context and rationale — why this track, why this data, what the rubric wants
```

## Setup, in full

1. `python -m venv .venv`, then `.venv\Scripts\pip install -r requirements.txt` (Windows) or `.venv/bin/pip install -r requirements.txt` (Mac/Linux)
2. Generate the artifacts by running `notebooks/06_cost_model_refined.py` in a Kaggle session (competition data attached, GPU on) and downloading the output files — exact list and destinations in [`artifacts/README.md`](artifacts/README.md). (`notebooks/04_cost_model.py` is the full historical record, including completed one-off diagnostics whose findings already live in `docs/experiments.md` — 06 is the consolidated, going-forward version and produces everything still needed.)
3. `python scripts/demo_engine.py` — should print `ALL CHECKS PASSED`
4. `pytest tests/` — 98 tests: idempotency, replay/tamper detection, fail-closed behavior (missing *and* malformed artifacts), both model version guards, batch-vs-online feature-parity golden vectors, the cost model's decision boundaries, and every LLM fallback path (timeout, garbage response, no API key) exercised without needing a live network call. The schema/fail-closed/parity/policy tests run on a bare clone; the rest need `artifacts/` populated (see step 2).
5. Optional: `python scripts/llm_benchmark.py --kaggle-results artifacts/llm_benchmark_kaggle_results.json` to reproduce the LLM-vs-XGBoost comparison
6. The dashboard needs no setup at all — open [`dashboard.html`](dashboard.html) directly in a browser

No GPU, no training, and no API keys are required for steps 1 or 3 — the LLM narrative
layer and the LLM benchmark are the only pieces that call out to a model provider, and
both degrade to a deterministic template/skip cleanly without one.
