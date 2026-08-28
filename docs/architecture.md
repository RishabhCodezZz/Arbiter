# Architecture

## Where the work happens, and why it's split this way

```
┌─────────────────────────────┐        ┌──────────────────────────────────┐
│   Kaggle notebooks (GPU)     │        │      Local repo (CPU-only)        │
│   research                   │  export │      product                      │
│                               │  ────▶ │                                    │
│  01  EDA + baseline           │ artifacts/  src/    the decision engine    │
│  02  causal features + screen │  model.json         (this doc, below)     │
│  03  V-col reduction, tuning, │  calibrator.json                          │
│      calibration              │  feature_manifest    scripts/  demo +      │
│  04  cost model + export      │  sample_txns          benchmarks           │
│  05  Kaggle-legal comparison  │  llm_benchmark_*                          │
│                               │  dashboard_data      (→ Control Room, ↓)   │
└─────────────────────────────┘        └──────────────────────────────────┘
```

Research happens where the data and the GPU already are — Kaggle mounts the 1.35GB
dataset with no download, gives free GPU time (~30hrs/week), and needs nothing beyond a
browser tab. The product is a plain Python package that only ever **loads** a trained
artifact; it never trains anything, never imports `xgboost`'s training path, never needs a
GPU. This split is deliberate and is itself a build-quality signal: a Kaggle notebook
cannot demonstrate graceful degradation, an audit trail, or "would you trust it" — only a
real, standalone program can.

## The decision engine (`src/`)

```
                    Engine.decide(transaction)
                              │
              ┌───────────────┴───────────────┐
              │  1. Idempotency check FIRST     │  ← audit.lookup(txn_id)
              │     (before any scoring work)   │     existing? return it, unchanged
              └───────────────┬───────────────┘
                              │ not seen before
              ┌───────────────▼───────────────┐
              │  2. Build feature vector        │  ← features.build(txn, store, manifest)
              │     (degrade if it fails)       │     fails? features.build_degraded()
              └───────────────┬───────────────┘
                              │
              ┌───────────────▼───────────────┐
              │  3. Score                       │  ← model.score_df(X)
              │     (fail closed if unavailable)│     no model? policy.rules_baseline()
              └───────────────┬───────────────┘
                              │ calibrated probability
              ┌───────────────▼───────────────┐
              │  4. Decide                      │  ← policy.decide_action(p, amount)
              │     allow / step-up / block      │     max(value_allow, value_stepup,
              │     (per-transaction, not a       │      value_block) — every cost term
              │      global threshold)            │      scales with amount, so this can't
              └───────────────┬───────────────┘     collapse to one cutoff
                              │ if flagged (≠ allow)
              ┌───────────────▼───────────────┐
              │  5. Explain (flagged only)      │  ← explain.Explainer (SHAP, lazy-built)
              │     LLM never decides,           │     narrative.narrate() — template
              │     only renders                 │     fallback if LLM fails/unavailable
              └───────────────┬───────────────┘
                              │
              ┌───────────────▼───────────────┐
              │  6. Write the audit record       │  ← audit.append() — append-only
              └───────────────┬───────────────┘
                              │
              ┌───────────────▼───────────────┐
              │  7. Update history store LAST    │  ← store.update() — AFTER scoring,
              │     (never before scoring)        │     so a transaction never sees its
              └───────────────────────────────┘     own effect on its own score
```

Eight modules, each one job (`explain.py`/`narrative.py` share a row below since they're tightly coupled — SHAP contributions feeding directly into the narrative renderer — but they're two separate files):

| Module | Responsibility |
|---|---|
| `store.py` | Client history — the live analog of the notebook's expanding-window batch aggregates. Updated *after* a decision, never before. |
| `features.py` | Raw transaction → the exact feature vector the model was trained on. Has a `build_degraded()` fallback for missing/malformed input. |
| `model.py` | Loads both artifacts (XGBoost + LightGBM — the shipped model is a 2-model ensemble, simple-averaged after independent Platt calibration), no training. Records both training-time versions in the manifest and refuses to score on a mismatch for either one (`XGBoostVersionMismatchError` / `LightGBMVersionMismatchError`) — fails closed, same path as a missing model, rather than silently producing a wrong probability (see `journal/` for why this exists: a 23x probability discrepancy from a version mismatch, found by testing). |
| `policy.py` | Identical cost formulas to `notebooks/04_cost_model.py` — computes the expected ₹ value of all three actions per transaction, picks the max. |
| `audit.py` | Append-only log. Idempotency (`lookup()` before any scoring). Tamper detection (`verify_and_replay()` re-derives a decision from its own stored inputs and flags a mismatch). |
| `explain.py` / `narrative.py` | SHAP contributions for flagged transactions only, turned into a sentence by an LLM that never decides — only ever called after the action is already locked in. |
| `engine.py` | Orchestrates the above in the exact order shown, and is where every one of CLAUDE.md's invariants (causal, replayable, idempotent, "LLM never decides") is actually enforced — see its own module docstring for the four ordering rules it exists to guarantee. |

## The audit record

Every decision writes one append-only record:

```
transaction_id · timestamp · model_version · feature_vector · feature_vector_hash
record_hash (covers action + probabilities + cost_params + degraded, not just features)
raw_probability · calibrated_probability · cost_params (the exact values in force)
action · action_values (what each of the 3 options was worth, for review)
latency_ms · fallbacks_triggered · idempotent_replay · degraded
shap_contributions · narrative · used_llm
```

**Replayable by construction:** `verify_and_replay()` takes a stored record's own inputs —
including the `cost_params` that were in force when the decision was made, not whatever
`policy.py`'s constants happen to be now — reruns them through the same deterministic policy
formulas, and compares the result to the stored action. It also recomputes both hashes. A
mismatch means the record was edited after the fact or there's a real bug — either way it's
caught, not trusted blindly.

**What this is not:** the hashes are **unkeyed** SHA-256. They make casual or partial
tampering detectable; they do not stop an attacker with write access who also recomputes
the hash. Real payment-grade integrity needs a keyed signature (HMAC/asymmetric), a hash
chain, and append-only external storage — see `docs/eval_report.md` exception list item 12.
Likewise `store.py` is a single-process JSON file: no locking or event-time ordering, so
the causal guarantee holds for in-order, single-threaded processing only (item 13).

## Failure paths — designed, and each one proven by actually breaking it

| Failure | Behavior | How it was verified |
|---|---|---|
| Model artifact missing/corrupt | Fail closed to `rules_baseline()` (step-up everything), never silently allow | `scripts/demo_engine.py` — artifact deleted mid-test, confirmed; `tests/test_engine.py` |
| Model artifact malformed (valid JSON, wrong shape — e.g. `{}` manifest, incomplete calibrator) | Full schema validation → `ModelUnavailableError` → same fail-closed path (was a `KeyError` that crashed engine construction; found in code review) | `tests/test_model.py` — parametrized over 6 malformed manifests + 4 malformed calibrators; `tests/test_model.py::test_engine_survives_a_malformed_manifest` |
| Request missing `TransactionID`, or both feature-build paths fail | Missing ID → clear `ValueError` at the boundary (not a bare `KeyError`); double-build failure → fail closed, logged | `tests/test_engine.py::test_missing_transaction_id_raises_a_clear_error`, `::test_unrecoverable_feature_build_still_fails_closed` |
| A stored `cost_params` / probability / action is edited | `record_hash` recompute mismatch → rejected; replay also uses the *stored* `cost_params` | `tests/test_audit.py::test_tampered_cost_params_are_detected_by_record_hash`, `::test_replay_uses_stored_cost_params_not_current_module_constants` |
| Online feature stats drift from the training definition | Batch-vs-online golden-vector parity test pins `CoarseStats.std()` (ddof=1) and `ClientStats.std()` (ddof=0) to their notebook formulas | `tests/test_store.py` — added after a real ddof=0/ddof=1 skew was caught in code review |
| XGBoost or LightGBM version mismatch | Refuses to score at all (`XGBoostVersionMismatchError` / `LightGBMVersionMismatchError`) | `tests/test_model.py` — a faked mismatch on EITHER model's version independently confirmed to fire, and the explicit override independently confirmed to work, for both; a clean unmodified copy of the real artifacts confirmed to load with no override needed, proving the mismatch tests exercise the version check and not something else |
| LLM unavailable / times out / returns garbage | Decision unaffected; narrative falls back to a deterministic template built from the same SHAP numbers | `tests/test_narrative.py` — timeout and garbage-response each their own named test via dependency injection (no live network call), plus `narrate()`-level fallback tests; response validation rejects 4 classes of bad LLM output |
| Required feature missing | Impute, flag `degraded=True`, **widen** the step-up band (shrink probability toward 0.5, not blend output values — the first design was proven algebraically incapable of moving the decision boundary at all) | Boundary measured directly before/after the fix: confident-allow ceiling moved p<0.0393→p<0.0150; regression-tested in `tests/test_policy.py::test_degraded_mode_widens_the_stepup_band` |
| A degraded decision gets replayed later | `degraded` is stored in the record and threaded through replay, so a legitimate cautious decision doesn't false-flag as tampered | `tests/test_audit.py::test_degraded_flag_is_required_for_correct_replay` |
| Duplicate/replayed transaction ID | Idempotency check happens *before* any scoring; returns the original decision unchanged | `scripts/demo_engine.py` — same ID submitted twice, confirmed byte-identical result, 0ms latency; `tests/test_engine.py` |

**The invariant demonstrated live, not just claimed:** delete the LLM entirely — no key,
network down — and every decision comes out byte-identical to before; only the narrative
degrades to a template. Proven by killing the LLM path during testing and diffing the
decisions, not assumed from reading the code.

## The dashboard — [`dashboard.html`](../dashboard.html)

Originally a local Streamlit app (`dashboard.py`) with a deliberate data split: the
PR curve, cost curve, and sensitivity map used a pre-computed export
(`artifacts/dashboard_data.json`, since the full 92,427-row test month only ever exists
inside a Kaggle session); the Review Queue and Audit Log ran `src/engine.py` **live**.

**Later retired** in favor of a single self-contained HTML file — same real data, but
captured once rather than requiring a local Python install to view. Every dependency
(CSS, JS, the real data arrays, even the reliability-diagram image) is inlined into one
file with no external calls except Google Fonts, so it opens directly in a browser with no
server, no build step, and nothing to install. The PR curve, cost curve (with a
still-fully-interactive client-side threshold slider — real array lookups against the
embedded data, no server needed), and sensitivity map are rendered natively from the same
`dashboard_data.json` export. The Review Queue and Audit Log became honest **snapshots**:
real transactions, real SHAP-derived narratives, real audit records, captured from an
actual `src/engine.py` run and explicitly labeled as captured rather than live — a static
page cannot run Python, and pretending otherwise would violate the same honesty standard
as everything else here.
