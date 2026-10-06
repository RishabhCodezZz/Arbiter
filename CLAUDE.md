# Arbiter project context

## Current scope

Arbiter is a solo, defense-only fraud decision project for card-not-present payments.
It chooses allow, step-up or block from a calibrated probability and an illustrative
INR cost model. Research runs in self-contained Kaggle notebooks; `src/` is the local
CPU inference engine. The delivered scope includes the decision engine and historical
causal-versus-full-history feature comparison.

The current ensemble averages independently Platt-calibrated Optuna-tuned XGBoost
and untuned LightGBM scores on the same feature set. CatBoost and other candidate
models were evaluated but are not runtime dependencies. Model-selection evidence
belongs in [experiments](docs/experiments.md).

Card-level risk state and time-to-detection (Component C) were cut after the UID
investigation. Dispute-evidence drafting (Component D) was never started. Neither is
active scope; alerting, asynchronous explanation backfill and production deployment
are also not delivered capabilities.

## Document responsibilities

- [README](README.md): purpose, quickstart, offline entry and concise limitations.
- [Evaluation report](docs/eval_report.md): authoritative current measurements,
  bootstrap intervals and exception list.
- [Experiments](docs/experiments.md): economic sources, assumptions, selection rules
  and historical results. Keep costs and claimed measurements here and in the report,
  rather than maintaining another table in this file.
- [Architecture](docs/architecture.md): current runtime behavior and audit schema.
- [Artifact guide](artifacts/README.md) and [notebook index](notebooks/README.md):
  required files, producers and regeneration.
- [Journal](journal/build-log.md): sequential history. Earlier entries retain the
  scope, claims and numbers used at that stage; they are not a current-state spec.

## Data and evaluation constraints

Use the labeled IEEE-CIS training data for evaluation: train through day 120,
validation days 121 to 150, temporal test after day 150. The competition's separate
`test_transaction.csv` has no fraud labels. The project's temporal test month was
reused for diagnostics, adaptive comparisons and the final ensemble choice. Do not
call it untouched or interpret a row-bootstrap interval as independent replication.

Keep observed labels, amounts and routing counts separate from retrospective monetary
simulation. Margin, fees, FX, lost-customer penalties and intervention rates are
scenario inputs. The dataset contains no observed step-up response outcomes. Current
sensitivity coverage is margin and chargeback fee only.

The dashboard's two-way allow/block cost curve is a presentation proxy optimized on
the reused test month. The runtime policy compares all three action values per
transaction, using amount-dependent boundaries and a fixed chargeback fee.

Historical diagnostics and single-model calibration evidence must remain labeled.
Do not infer current ensemble calibration quality from the old reliability diagram.
Rerunning the current LLM benchmark uses ensemble scoring and does not reproduce the
historical single-XGBoost comparison; see its qualifications in the evaluation report.

## Runtime agreements

- Look up a transaction ID before scoring, then recheck it when committing. Duplicate
  calls return the stored action and probability with replay metadata.
- Update history after scoring and audit append. The instance-local `RLock` protects
  duplicate-ID commits; feature construction and scoring run unlocked. It does not
  serialize events chronologically or guarantee causality for late arrivals.
- Keep the LLM outside decision logic. SHAP and narrative run synchronously for
  flagged scored cases; explanation failures preserve the chosen action.
- Missing/unusable model files, failed scoring or unrecoverable feature construction
  use the step-up rules fallback with no probability. Degraded feature scoring shrinks
  probability toward uncertainty before comparing action values.
- Replay stored policy inputs and cost parameters. Replay verifies decisions and
  hashes; it does not rerun model inference.
- Preserve model-version guards and artifact fingerprints. Keep credentials out of
  files and Git. The optional expected-checksum guard needs a populated manifest.

## Working agreements

Run commands from the repository root with the environment's interpreter, as shown in
[quickstart](README.md#quickstart). Make claims from inspected outputs and source.
Report unsuccessful experiments and unresolved limitations alongside successful ones.
Use deterministic logic for policy and replay; justify any LLM call by its role.

Update scope and runtime documentation when behavior changes, and update authoritative
results when new evidence exists. Log failures in the journal as they occur; preserve
old entries and append corrections rather than rewriting history. Retain trained
artifacts, result exports and notebook evidence. Runtime state resets lose history and
audit records; caches and optional private notes have separate retention roles in the
[artifact guide](artifacts/README.md#retention).

## Unresolved work

The [exception list](docs/eval_report.md#honest-exception-list) tracks fresh temporal
evaluation, current ensemble calibration, broader economic sensitivity, representative
end-to-end latency measurements, risk-inclusive snapshot exports, stronger audit
integrity, transactional serving state with event-time handling, and checksum-backed
artifact releases. These are follow-ups, not delivered features. C-only recalibration
and exact UID reconstruction also remain unresolved.

## Attribution and research references

The [README attribution](README.md#attribution) describes reuse of the IEEE-CIS
first-place solution. Preserve credit when extending derived features or comparisons.

- [First-place solution](https://www.kaggle.com/c/ieee-fraud-detection/discussion/111284)
- [Technical writeup](https://www.kaggle.com/c/ieee-fraud-detection/discussion/111321)
- [UID detection](https://www.kaggle.com/kyakovlev/ieee-uid-detection-v6)
- [XGBoost reference](https://www.kaggle.com/cdeotte/xgb-fraud-with-magic-0-9600)
- [Transaction-column EDA](https://www.kaggle.com/alijs1/ieee-transaction-columns-reference)
- [V/identity-column EDA](https://www.kaggle.com/cdeotte/eda-for-columns-v-and-id)
