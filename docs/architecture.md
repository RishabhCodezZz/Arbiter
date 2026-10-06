# Architecture

## Research and runtime

Kaggle notebooks build features, fit models/calibrators and export portable artifacts.
[Notebook 06](../notebooks/06_cost_model_refined.py) is the current regeneration entry;
[notebook 04](../notebooks/04_cost_model.py) retains historical diagnostics. The local
Python package loads artifacts for CPU inference and does not train models.

```text
Kaggle notebooks -> model/calibrator/manifest exports -> local Engine
                                      |
                           evaluation exports -> offline dashboard
```

Required files and optional exports are listed in the [artifact guide](../artifacts/README.md).
The [evaluation report](eval_report.md) owns current results; [experiments](experiments.md)
owns parameter provenance and historical comparisons.

## Decision flow

[Engine.decide()](../src/engine.py) validates the request's transaction ID and amount,
then follows this path:

```text
lookup ID under instance RLock -> existing? return stored decision
                 |
          build features from history
                 |
     score XGBoost + LightGBM; calibrate each; average
                 |
      compare allow / step-up / block expected values
                 |
      flagged scored case: synchronous SHAP + narrative
                 |
      measure latency; construct audit record
                 |
     under same RLock: recheck ID -> duplicate? return stored decision
                 |
          append audit -> update/save history -> return
```

The first lookup happens before scoring. At commit, the second lookup catches another
caller that completed the same ID while this caller was scoring. Its new record is
then discarded, preventing a duplicate append or double history update within that
Engine instance. A replay returns the stored action/probability and narrative with
`idempotent_replay=True` and `latency_ms=0`; it does not append another audit record.

Feature construction, scoring and explanation run outside the lock. Updating history
after scoring prevents a transaction from using its own newly added history. It does
not guarantee a chronological feature snapshot for concurrent calls. The lock is local
to one Engine instance; separate instances/processes/hosts are not coordinated.
Late events can see history from later events. There is no event-time ordering guarantee.

## Modules and policy

| Module | Current responsibility |
|---|---|
| [store.py](../src/store.py) | Fine/coarse client aggregates in memory, persisted to JSON after updates |
| [features.py](../src/features.py) | Raw transaction to manifest-ordered features, with degraded construction fallback |
| [model.py](../src/model.py) | Load both models and Platt coefficients; validate schemas/versions; score and average |
| [policy.py](../src/policy.py) | Compare expected INR values for allow, step-up and block |
| [audit.py](../src/audit.py) | JSONL append, ID lookup, hashes and stored-policy replay |
| [explain.py](../src/explain.py) | Top-five averaged SHAP contributions, retaining each member's contribution |
| [narrative.py](../src/narrative.py) | Optional LLM rendering of contributions with deterministic template fallback |
| [engine.py](../src/engine.py) | Request validation, sequencing, fallbacks and instance-local commit protection |

Both models use the same feature vector. XGBoost is Optuna-tuned; LightGBM is untuned.
Each raw score passes through its own Platt calibrator before their probabilities are
simple-averaged. The loader requires both training library versions in the manifest,
fingerprints all five model/manifest files and can check expected manifest checksums.
The current notebook export does not populate those expected checksums.

The policy converts the input amount to INR using the recorded scenario FX rate.
Allow receives genuine margin less processing fees and incurs fraud loss plus a fixed
chargeback fee. Step-up applies assumed genuine completion and fraud-stopping rates;
processing fees apply only to completed payments. Block incurs an assumed lost-margin
and customer-value penalty for genuine transactions. The fixed chargeback fee means
costs are not all proportional to amount: action boundaries depend on both probability
and transaction size. Choose the maximum expected value among all three actions.

Parameters and their limitations are in the
[provenance table](experiments.md#cost-model-parameters-and-provenance). The dashboard's
two-way threshold sweep is a test-optimized presentation proxy, not the runtime policy.

## Explanations, fallbacks and latency

SHAP and narrative run synchronously after the action is chosen, only for non-allow
cases with a successful model score. The explainer averages per-feature contributions
and retains each member's value. This is not an exact decomposition of the averaged
calibrated probability or a causal explanation of fraud. The LLM renders contributions;
its output does not select or revise the action. The narrative request timeout is six
seconds, with possible SDK retry overhead. There is no asynchronous explanation queue,
backfill or alerting integration.

| Condition | Behavior |
|---|---|
| Missing/unusable model, version mismatch or scoring failure | Step-up rules fallback; probabilities absent and action-value dictionary empty |
| Primary feature construction fails | Try degraded features and record fallback; shrink scored probability toward 0.5 to widen the step-up band |
| Both feature builders fail | Step-up rules fallback with minimal features |
| SHAP fails | Record failure and use an empty contribution list; action preserved |
| LLM missing, fails, times out or returns invalid prose | Deterministic narrative template; action preserved |
| Missing ID or invalid present amount | Clear request-boundary `ValueError` |

Recorded `latency_ms` starts after the first ID lookup and stops after explanation,
before audit-record construction, commit-lock acquisition and audit/history writes.
It is not full end-to-end return latency. Representative latency tails and the 200ms
target remain unverified; see the [evaluation report](eval_report.md).

## Audit and persistence

The actual [AuditRecord schema](../src/audit.py) contains:

```text
transaction_id, timestamp, model_version
feature_vector, feature_vector_hash, record_hash
raw_probability, calibrated_probability, cost_params
action, action_values, latency_ms, fallbacks_triggered
idempotent_replay, degraded, shap_contributions, narrative, used_llm
```

`raw_probability` is a dictionary with `xgboost` and `lightgbm` scores, or `None` on
fallback. `calibrated_probability` is their calibrated average. The schema has no
threshold field, LLM prompt or LLM response ID.

The feature hash covers the stored vector. The broader record hash covers transaction
ID, model version, features, raw/calibrated probabilities, cost parameters, action,
`action_values` and degraded status. Timestamp, narrative, SHAP, latency and fallback
metadata are not covered by that broader hash. These are unkeyed SHA-256 hashes:
a writer with file access can edit records and recompute them.

`verify_and_replay()` checks hashes and recomputes action/value outputs from stored
probability, amount, cost parameters and degraded status. It does not recompute model
inference. Probability-less fallback records receive hash checks without policy replay.
Legacy records without a record hash receive only the feature-hash check before replay.

History saves write a sibling temporary JSON file and use `os.replace()` to protect
against partial-write corruption. Audit append and history save are separate operations,
so a failure between them can leave inconsistent state. Atomic replacement is crash
protection for a file, not cross-process transactionality or a durable transaction
across audit and history. Audit lookups scan the JSONL file linearly.

## Offline presentation

[dashboard.html](../dashboard.html) embeds CSS, JavaScript, curve/queue/audit data and
the reliability image. Google Fonts imports have been removed; the page needs no
external dependency or server. Its headline, PR/ROC curves and sensitivity data come
from the current ensemble export. The calibration panel is historical single-XGBoost
evidence, and current ensemble calibration remains unmeasured. Queue/audit panels
retain labeled single-model snapshots rather than a live connection to the engine.
The [docket](docket.html) is a historical walkthrough. Snapshot regeneration needs a
risk-inclusive full-feature export, as tracked in the evaluation report's
[exception list](eval_report.md#honest-exception-list).
