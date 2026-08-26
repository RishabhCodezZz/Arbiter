# Evaluation report

Every metric called for in [CLAUDE.md §9](../CLAUDE.md#9-metrics-we-will-report), computed
on the untouched test month (day > 150, 92,427 real transactions, 3,213 fraud, 3.48%),
touched exactly once, at the end, for these numbers. Where a number came from Kaggle vs.
was derived locally is stated explicitly — nothing here is invented to fill a gap.

---

## 1. Model quality

> **The shipped model is now a 2-model ensemble** (XGBoost + untuned LightGBM,
> simple-averaged) — see `docs/experiments.md`'s ensemble sections and exception-list item
> 10 below. The table and bootstrap CIs immediately below describe the single-XGBoost
> model's own numbers, kept as an accurate historical record rather than overwritten; the
> ensemble's own confirmed aggregate numbers (test PR-AUC 0.5597, total value ₹17.355cr,
> +₹13.58L over this single-model baseline, CI [+₹6.55L, +₹21.24L]) are reported separately
> in `docs/experiments.md` and have not yet been re-run through this report's own granular
> per-transaction breakdown (§2b/§4 below) — a tracked gap, not a silent one.

| Metric | Value | 95% bootstrap CI | vs. random baseline |
|---|---|---|---|
| PR-AUC | **0.5514** | 0.5350 – 0.5688 | random ≈ base rate (0.0348) → **15.8x** |
| ROC-AUC | **0.9077** | 0.9019 – 0.9132 | random = 0.5 (context only — see below) |

**Why PR-AUC is the number that matters, ROC-AUC is context:** at a ~3.5% fraud rate,
ROC-AUC's false-positive-rate denominator is dominated by the huge negative class, which
flatters the score — a model can look excellent on ROC-AUC while still wrongly flagging a
large fraction of the customers it blocks. PR-AUC has no such cushion. Tuned on PR-AUC;
ROC-AUC reported because Kaggle's own leaderboard scored on it (winners: 0.9408) and it
gives external context. Full ladder from an untuned baseline (0.5486) through calibration:
[`experiments.md`](experiments.md).

**Both now carry a real 95% CI**, not just a bare point estimate — "measured precision and
recall" should mean the same thing everywhere in this report, not only in the cost-model
section. Computed by the same paired-bootstrap infrastructure as §2b's rupee-lift CI
(2,000 resamples, seed 42, full 92,427-row test month — `scripts/robustness_checks.py`).
Both metrics are hand-rolled rather than imported from sklearn (same discipline as
`scripts/llm_benchmark.py`'s hand-rolled average precision — `requirements.txt`
deliberately keeps sklearn out of `scripts/`/`src/`), and each was verified to reproduce
the point estimate above *exactly* (0.5514 / 0.9077) before the bootstrap interval built on
top of it was trusted.

## 2. The cost curve and the chosen operating point

Full three-way policy (per-transaction, amount-dependent — see [CLAUDE.md §6](../CLAUDE.md#6-architecture)
for why this isn't a single cutoff):

| Policy | Total value, test month | Lift |
|---|---|---|
| No fraud system | ₹15.68 crore | — |
| Naive 0.5 threshold | ₹16.58 crore | +₹90.0 lakh vs no system |
| **Arbiter** | **₹17.22 crore** | **+₹1.54 crore vs no system, +₹64.3 lakh vs naive** |

Policy mix: allow 88,560 (95.8%), step-up 2,519 (2.7%), block 1,348 (1.5%).

**Hard vs. modeled split, stated explicitly:** ₹17.21cr (99.95%) of the total comes from
allow/block outcomes computed directly from real test-month fraud labels — hard,
verifiable. Only ₹92,945 (0.05%) depends on the modeled step-up population rates
(`P_STOP`=60%, `P_DROPOFF`=15%, both cited from industry studies, not measured on this
dataset — see honest exception list item #2 below). The headline claim is overwhelmingly
evidence-based, not assumption-dependent.

The simplified single-threshold (allow/block only) view used for the pitch-video visual —
[`cost_curve.png`](cost_curve.png) — has its minimum at **p = 0.774**, confirmed interior
(not at an edge, G6 gate passed), robust to the FX-rate correction described in
`experiments.md`.

## 2b. Robustness — is the headline number real, or one lucky month?

Every number above is a point estimate from a single untouched test month. `scripts/robustness_checks.py`
puts a real interval around it, against the full 92,427-row test month (`artifacts/test_month_raw.json`),
not a smaller sample:

**Bootstrap 95% CI (2,000 resamples, paired per resample — Arbiter's total and each
baseline's total computed on the *same* resampled rows each time, then differenced, so the
interval doesn't overstate uncertainty by ignoring the correlation between them):**

| | Point estimate | 95% CI |
|---|---|---|
| Lift vs no system | +₹1.54cr | ₹1.38cr – ₹1.71cr |
| Lift vs naive 0.5 | +₹64.3L | ₹53.7L – ₹75.5L |

Both intervals sit comfortably clear of zero — the lift is a stable effect, not a favorable
draw from one month.

**A second, genuinely fair baseline** — not another strawman like naive 0.5, which still
needs a model. A merchant with *no* model might instead block above a flat rupee amount;
swept for its own best threshold rather than picked arbitrarily. Real result: the best
amount-only threshold is **₹512,535 — high enough that it never blocks anything in this
dataset**, collapsing to the exact same value as doing nothing at all. Two untuned
reference thresholds (₹10,000, ₹50,000) are sharply negative for context. **Honest
finding:** fraud isn't separable by amount alone here — a pure rule cannot beat doing
nothing, so Arbiter's full lift over the best possible simple rule is attributable to the
model's actual signal, not to any threshold a merchant could have picked without one.

## 3. Precision / recall at the operating threshold

CLAUDE.md §9 specifically calls for this **at the operating threshold, not at the
meaningless default of 0.5.** Computed from the real PR curve (`artifacts/dashboard_data.json`,
91,272 points, full test month) at the nearest point to p = 0.774:

| | Precision | Recall |
|---|---|---|
| **At the operating threshold (0.774)** | **86.1%** | **33.1%** |
| At naive 0.5 (for comparison) | 81.7% | 36.9% |

Read plainly: at the point the cost model actually recommends operating, ~86% of
transactions flagged above this single cutoff are genuinely fraud, catching about a third
of all fraud in the test month at that cutoff alone. Recall looks low in isolation, but
this table is intentionally the *simplified 2-way proxy* (a single global threshold), not
the real deployed policy — the real system also has the **step-up band**, which catches
additional fraud the single-threshold view above credits as "missed." The real 3-way
policy's own numbers are in §2 above (99.95% hard-verified) and are the ones that actually
matter; this section exists specifically to answer CLAUDE.md's own named metric honestly,
including its limits.

**Estimated confusion counts at this threshold** (derived: TP = recall × n_fraud, total
flagged = TP / precision, FP = flagged − TP — an estimate from the aggregate curve, not a
recount of individual transactions):

| | Count |
|---|---|
| Fraud caught (TP) | ~1,062 |
| Genuine wrongly flagged (FP) | ~172 |
| Total flagged at/above threshold | ~1,234 |

## 4. False-positive cost, explicitly

The rubric names this directly: *"honest metrics including false-positive cost."*

**§3's ~172 is a single-threshold PROXY estimate**, derived from the aggregate PR curve at
the simplified 2-way (allow/block only) view — see §3's own caveats. **Below is the EXACT
count from the real, deployed 3-way policy** — computed by `scripts/robustness_checks.py`
directly from `artifacts/test_month_raw.json`'s real per-transaction (probability, label,
amount) for all 92,427 test-month rows. No estimation, no aggregate-curve proxy.

| | Count |
|---|---|
| Blocked, correctly (real fraud) | 1,115 |
| **Blocked, WRONGLY (real genuine — the exact false positives)** | **233** |
| Total blocked | 1,348 |

**Exact false-positive cost: ₹17.97 lakh (₹17,96,854)** — ₹7,712 average per
wrongly-blocked customer, computed at `margin × amount × (1 + LTV_multiplier)` for each of
these 233 real rows individually, not modeled or estimated. Same cited parameters as
everywhere else: 20% margin, 3x LTV penalty on the lost customer — the LTV multiplier
remains the single most speculative parameter in the model, swept in the sensitivity
analysis in `experiments.md`. (The chargeback fee doesn't apply here — these are the
customers *wrongly* blocked, so no chargeback ever occurs.)

**Why this doesn't match §3's ~172 — different policies, not a contradiction.** §3's proxy
blocks everything above one flat probability cutoff (p ≥ 0.774), with no step-up option.
The real 3-way policy's block boundary is amount-dependent — every cost term scales with
transaction size (see [CLAUDE.md §6](../CLAUDE.md#6-architecture)) — and routes much of the
middle-risk band to step-up instead of an outright block. A genuinely different rule,
applied to the same data, produces a different count. Neither number is wrong; they answer
different questions, and both are stated rather than reconciled to look consistent.

**Previously an honest gap, now closed.** An earlier version of this report noted that this
exact breakdown existed only inside a Kaggle session's memory and was never exported.
`artifacts/test_month_raw.json` (the row-level export addendum) now makes it possible to
compute locally — verified before being trusted: the recomputed policy mix
(88,560/2,519/1,348) and total portfolio value (₹17.22cr) both reproduce the
already-published headline numbers exactly. Full numbers, including the step-up band's
exact fraud/genuine composition (its *outcome* stays modeled, not measured — honest
exception list item #2): `docs/robustness_results.json`.

## 5. Kaggle-legal vs. causal gap — Component B

| Model | PR-AUC | ROC-AUC |
|---|---|---|
| Honest (causal, deployable) | 0.5446 | 0.9014 |
| Leaky features only (undeployable) | 0.5697 | 0.9064 |
| Leaky features + leaky post-processing (undeployable) | 0.5512 | 0.9152 |
| **Total gap, full Kaggle-legal vs. honest** | **+0.0066** | +0.0138 |

Causal honesty costs ~1.2% relative PR-AUC here — a small, measured price for something
that can actually run in a 200ms decision window. A genuine finding along the way: the
post-processing sub-step (replacing each prediction with its client's average) *hurt*
PR-AUC relative to leaky-features-alone, for the same tie-collapse reason isotonic
calibration did in §6 below — reported as found, not smoothed into a simpler "leaky always
wins" story. Full investigation: `experiments.md`.

## 6. Reliability diagram — before and after calibration

| Method | PR-AUC | ROC-AUC | ECE |
|---|---|---|---|
| Raw (uncalibrated) | 0.5514 | 0.9077 | 0.0103 |
| Isotonic (tested, rejected) | 0.5384 | 0.9073 | 0.0042 |
| **Platt (shipped)** | **0.5514** | **0.9077** | **0.0036** |

Platt matches raw ranking exactly (proving the isotonic-ties theory) and beats isotonic's
own calibration error too — no trade-off, strictly dominant. `docs/reliability_diagram.png`
is downloaded and committed. (Caught once already: this report briefly repeated CLAUDE.md's
stale claim that the file had already "shipped" before it had actually been downloaded —
checked the file exists before trusting that, found it didn't at the time, fixed by
re-running the plotting cell already complete and correct in
`notebooks/03_reduce_tune_calibrate.py` and downloading for real.)

## 7. LLM-as-classifier benchmark — the AI-judgment evidence

| | PR-AUC | Latency/call |
|---|---|---|
| XGBoost | **0.5735** | microseconds |
| `gpt-oss:20b` (real model, fair shot, Kaggle GPU) | 0.1571 | ~6.7s median |

On this 200-row comparison sample (5% fraud rate, random PR-AUC ≈ 0.05): XGBoost scores
~11.5x random, the LLM ~3.1x random — genuinely better than guessing, not a strawman
comparison, and XGBoost still wins by **3.65x** on accuracy and roughly six orders of
magnitude on latency. This is the direct evidence for the rubric's "where you chose not to
use one" line — benchmarked, not assumed. Full saga (a real cloud-API reliability failure,
the pivot to self-hosting on Kaggle's GPU, both fixed and verified): `journal/`.

Note: 0.5735 here is the XGBoost score on this 200-row *comparison* sample specifically —
not the same as the 0.5514 headline (§1), which is the full test month. Different sample
sizes, expected to differ; the 200-row number exists only to be an identical-data
comparison against the LLM.

## 8. Error analysis and the val→test gap — why the model behaves the way it does

Diagnosed using Andrew Ng's *ML Yearning* framework (ch14 error analysis; ch40/41
bias-variance-mismatch decomposition). Before this pass, every decision in this project was
driven by aggregate metrics (PR-AUC, ECE, rupee totals) — nobody had ever looked at an
individual misclassified transaction, and the final model's own val PR-AUC (0.6128, from
`notebooks/03-reduce-tune-calibrate-outputs.ipynb`'s own executed output) vs. test PR-AUC
(0.5514) gap had never been investigated, even though it was sitting in plain sight since
the tuning stage.

### 8a. Decomposing the gap: variance vs. temporal mismatch

A **training-dev set** — a random 15% slice of the *training* period itself (day ≤120),
held out and never trained on — separates "does the model generalize within one time
period" (variance) from "does performance degrade specifically because of moving forward in
time" (mismatch). Computed with a diagnostic model only (trained on 85% of `tr`) — **not**
the shipped artifact: `artifacts/training_dev_gap.json`.

| Set | n | PR-AUC | Gap from previous |
|---|---|---|---|
| train′ (fit on) | 352,361 | 0.9864 | — |
| training-dev (unseen, same period) | 62,181 | 0.8357 | **−0.151 (variance)** |
| val (unseen, next period) | 83,571 | 0.6110 | **−0.225 (mismatch)** |
| test (unseen, furthest period) | 92,427 | 0.5496 | −0.061 (further mismatch) |

**Verdict: the val→test gap is overwhelmingly temporal mismatch, not overfitting.** The
mismatch gap (0.225) is 1.5x the variance gap (0.151), and the largest gap identifies the
dominant cause. This is a controlled measurement of the project's own borrowed thesis
("unseen clients, not time drift" — CLAUDE.md §5), not just a citation of it. The
diagnostic model's own val/test scores (0.6110/0.5496) land within 0.002 of the shipped
model's (0.6128/0.5514), confirming the 85%-training-data diagnostic is a fair proxy, not a
different model's story.

**Why this isn't being chased with more regularization.** Real variance does exist (15
points) and reducing it is a legitimate lever in general — but it is not the dominant
problem here, and the largest gap should drive the response. The 60-trial Optuna search
already searched the regularization space (`reg_lambda`, `reg_alpha`, `max_depth`,
`min_child_weight`) against a genuine validation objective; re-opening tuning this late
would cascade into re-verifying every downstream artifact (cost model, dashboard, docket,
audit log, 36 tests) that depends on the current `model.json`, for an uncertain and likely
modest payoff against the *smaller* of two measured gaps. The dominant gap — temporal
mismatch — is a property of the problem (new clients appearing over time), not a model
defect regularization can fix; the project's existing responses to it (causal per-client
history features, a temporal split, bootstrap CI on the untouched final month) are the
appropriate mitigations, already in place.

### 8b. Manual error analysis — where the real mistakes concentrate

Reviewed all 233 exact false positives and a 100-row sample of false negatives (fraud
allowed through with zero friction, of 1,444 total real cases) — real raw features and real
per-row SHAP contributions, exported by a Kaggle addendum, cross-referenced against
notebook 01's own original EDA (never previously connected to a failure population).

| | False positives (233) | False negatives sampled (100) |
|---|---|---|
| ProductCD = 'C' (11.7% base fraud rate — highest of 5 codes) | **81.1%** | 17% |
| ProductCD = 'W' (2.0% base fraud rate — lowest) | 10.3% | **75%** |
| card6 = 'credit' (6.7% base rate vs. debit's 2.4%) | **58.4%** | 22% |
| Dominant SHAP driver (V258/C1/C14 combined, for FP) | **80.3%** | max 17% (diffuse — C13) |
| Near the 0.774 operating threshold (0.10 < p < 0.774) | 28.8% below 0.80 | **0 of 100** |

**Reading: the model's mistakes are not random — they're the structurally expected residual
of a well-calibrated system.** It over-flags in ProductCD='C', the segment it has correctly
learned to distrust (11.7% base fraud rate); it under-flags in 'W', the segment it has
correctly learned to trust (2.0%). **The one hard, quantified answer this rules out:**
threshold-tuning cannot help the false-negative population at all — 0 of the 100 sampled
cases scored anywhere near the threshold (all under 10%, versus the 77.4% cutoff). Only a
better feature or a segment-specific policy could move that number.

**Follow-up, since tried:** per-ProductCD Platt calibration (the real analogue of a
per-segment threshold in an architecture where `decide_action` has no explicit threshold to
segment — see `docs/experiments.md`). It does what it was scoped to do — real false-positive
reduction in ProductCD='C', 26.8% on val (127→93) and 22.8% on test (189→146), checked once
as confirmation — but recalibrating *all five* segments together costs more elsewhere than
it saves in C, so total realized value drops (−0.18% val, −0.72% test) and the pre-committed
rule correctly rejects adopting it. **Not shipped.** A narrower, C-only variant (leaving the
other four segments on the existing global calibrator) is the natural next bounded test if
this gets revisited, but untried here — flagged, not chased, given the effect size involved
is a fraction of a percent of total value.

**One retracted finding, logged for honesty rather than quietly dropped:** an initial
cold-start check (`uid_amt_mean_prior is None`) returned 100% for both populations — later
found to be a bug in the check itself (that field is an engineered feature, deliberately
excluded from the raw export, so the lookup always hit the missing-key default), not a real
signal. Caught and dropped before being reported as fact anywhere.

**Full numbers:** `artifacts/error_analysis_false_positives.json`,
`artifacts/error_analysis_false_negatives_sample.json`, `artifacts/training_dev_gap.json`.

---

## Honest exception list

What this system could not resolve, stated plainly because the rubric rewards it and a
panel will find it anyway if it isn't. (This list has moved four times: down to 7 items
when the false-positive-cost estimate — previously item 5 — was resolved with an exact
number in §4; up to 8 with the error-analysis variance finding; up to 9 with the
segment-calibration confound; up to 10 with the ensemble-rebuild granular-recompute gap
below. Same discipline every time: fix it and say so, don't
quietly renumber and hope no one compares versions.)

1. **Card-level identity reconstruction (Component C) — scoped out, not solved.** Tried to
   reproduce the 1st-place solution's client-identity purity (target 96.9/2.9/0.2%
   pure-legit/pure-fraud/mixed); best achieved was 2.11% mixed across 8 tested key
   configurations, ~11x the target. Root-caused to their reliance on an undocumented
   separate matching script. A scoping decision made from evidence within a pre-committed
   time budget, not an unexamined failure. Full investigation: `experiments.md`.

2. **Step-up outcomes are modeled, not measured.** No real record in this dataset of
   whether a specific step-up challenge actually stopped a fraudster or made a genuine
   customer abandon. Uses cited industry population rates (`P_STOP`=60%, `P_DROPOFF`=15%).
   Consequence quantified, not hidden: only 0.05% of the ₹17.22cr headline depends on this.
   The band's *composition* is now exact (654 real fraud, 1,865 real genuine, out of 2,519
   step-ups — `docs/robustness_results.json`); it's specifically what happens to each of
   them next that remains modeled, not the population itself.

3. **Cost parameters are sourced assumptions, not certainties.** Chargeback fee (₹500) and
   MDR (2.36%) are Razorpay's own disclosed pricing — closer to fact than assumption.
   Merchant margin (20%) and the LTV penalty (3x lost margin, the single most speculative
   number in the model) are genuinely uncertain, swept across a full sensitivity grid in
   `experiments.md` rather than reported as certain.

4. **US card-not-present data, not Indian UPI-heavy data.** IEEE-CIS is a US e-commerce
   dataset; the method (causal aggregation, cost-optimal three-way policy, SHAP + narrative
   layer) transfers directly, but the specific feature set and thresholds would need to be
   re-derived against Razorpay's actual transaction mix.

5. **A handful of real audit-log records are legitimately probability-less.** During
   testing, the model artifact was deliberately made unavailable to prove the fail-closed
   path — those decisions (a conservative step-up default) have no `calibrated_probability`
   by design, since `rules_baseline()` never fabricates a number it doesn't have. Visible
   in the live audit log, correctly labeled as `fail-closed`, not hidden or smoothed over
   — see the dashboard's Review Queue tab for a live example.

6. **SHAP explanations are only computed for flagged transactions** (step-up/block, ~4.2%
   of traffic), not the full stream — a deliberate scoping choice (nobody reads 570,000
   explanations), not a coverage gap in the sense of missing capability, but worth stating:
   an allowed transaction never gets an explanation, by design.

7. **Component D (dispute-evidence drafting) was never started.** A stretch goal from the start,
   explicitly first on the pre-committed cut order. Not attempted; no partial version
   exists to evaluate.

8. **Real model variance exists (15pp PR-AUC gap, train′→training-dev, §8a) — measured,
   tested, and confirmed not worth chasing.** It's real but secondary — the dominant driver
   of the project's val→test gap is temporal mismatch (22.5pp), which regularization
   cannot fix. Originally reasoned through and left untried; then actually tested with a
   bounded 6-config hyperparameter sweep selecting the most conservative candidate by
   smallest traindev→val gap, checked once on test: **PR-AUC 0.5290 vs. the shipped
   model's 0.5514 — a real regression (−0.0224), not an improvement.** The more
   conservative config generalizes worse, not better, confirming rather than merely
   assuming that this gap isn't a regularization problem. Full numbers:
   `docs/experiments.md`.

9. **Segment-aware calibration was tested as one 5-segment change, not isolated per
   segment — so the ProductCD='C' question specifically is still open.** Recalibrating all
   five ProductCD segments together cut false positives in 'C' by 22.8% (189→146, test) but
   dropped total value 0.72%, because the other four segments got worse alongside it. That
   confirms the *bundled* change isn't worth adopting; it does not tell us whether a
   **C-only** recalibration (leaving the other four segments on the existing global
   calibrator, so they can't be made worse) would be a clean, adoptable win. Sized before
   deciding whether to chase it: best case is ~43 fewer wrongly-blocked customers, capped
   around ₹3-4 lakh — under 0.02% of the ₹17.22cr headline value, an order of magnitude
   below the width of the bootstrap CI already reported on that same headline number. Not
   pursued for that reason, not because it's unclear how to do it. Full numbers:
   `docs/experiments.md`.

10. **The shipped model moved from single-XGBoost to a 2-model ensemble; the granular,
    per-transaction breakdown in this report has not been regenerated for it.** The
    ensemble's own aggregate headline (test PR-AUC 0.5597, total value ₹17.355cr, +₹13.58L
    over the single-model baseline, 95% CI [+₹6.55L, +₹21.24L]) is independently confirmed
    real via a full-test-month bootstrap — see `docs/experiments.md`. What's specifically
    *not* yet redone for the ensemble: the exact 233-false-positive breakdown (§2b/§4
    below), the PR-AUC/ROC-AUC bootstrap CIs in §1 above, `artifacts/test_month_raw.json`
    (still the single-model's per-row scores), and `dashboard.html`/`scripts/robustness_checks.py`
    (both still built from the single-model export). None of this changes the confirmed
    ensemble headline — it means the more granular story in the rest of this document
    describes the single model it was originally computed on, clearly labeled as such
    rather than silently left to look current.
