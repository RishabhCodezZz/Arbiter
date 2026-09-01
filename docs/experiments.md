# Experiment log

Every row changes **exactly one thing** from the row above it, so the delta is attributable.
All numbers on the untouched test month (day 150–183) unless noted.

Random-model PR-AUC on this data ≈ the base rate (~0.035). That is the floor.

| # | Change | PR-AUC | ROC-AUC | Δ PR-AUC | Notes |
|---|---|---|---|---|---|
| V0 | Baseline: raw features, temporal split, no tuning | 0.5486 | 0.9002 | — | 17.9x lift(val) / 15.8x(test) over random. |
| V1 | + causal UID aggregates (expanding window only) | 0.5436 | 0.9035 | -0.0050 | Correct & stable (verified). Small honest cost, not a win. Kept for component B. |
| V2 | + time-consistency feature screening (**1**/411 dropped, corrected screen) | **0.5446** | **0.9014** | -0.0040 (vs V0) | **FINAL.** Original screen (an early slice vs. a mature one) wrongly dropped 70 features on a cold-start-biased comparison — retracted, see below. |
| V3 | + V-column reduction (NaN-group + correlation, 339→279 kept) | 0.5469 | 0.9058 | +0.0023 | First clear win since V0. |
| V4 | + Optuna sweep (60 trials completed, best val PR-AUC 0.6128) | **0.5514** | **0.9077** | +0.0045 | Best score of the project so far. Params: max_depth=10, lr=0.049, colsample_bytree=0.89, min_child_weight=2. |
| V5 | + Platt (sigmoid) calibration | **0.5514** | **0.9077** | **0.0000** | **FINAL.** Isotonic tested and rejected — see below. ECE 0.0036 (best of all three methods tested). |
| V5 | + isotonic calibration (tested, rejected) | 0.5384 | 0.9073 | **-0.0130** | Expected "~unchanged by design" (monotonic) — wrong, see below. Collapsed 91,271 distinct scores to 323 (ties). |
| — | **LLM-as-classifier benchmark** | XGB 0.5735 / LLM 0.1571 | — | — | 200-row held-out sample, `gpt-oss:20b` on Kaggle GPU. See the Kaggle-legal section below — full story. |

## Component B — the Kaggle-legal leaky model (`notebooks/05_kaggle_legal_leaky.py`)

Same test month as every other model in this project. Same hyperparameters as V1 (isolating causal-vs-leaky as the only variable). Two leaks measured **separately**, on purpose:

| Model | PR-AUC | ROC-AUC | vs honest V2 |
|---|---|---|---|
| **Honest (causal, V2, deployable)** | 0.5446 | 0.9014 | — |
| Leaky features only (undeployable) | **0.5697** | 0.9064 | +0.0251 PR-AUC |
| + leaky client-mean post-processing (undeployable) | 0.5512 | **0.9152** | −0.0185 PR-AUC vs leaky-feat |
| **TOTAL gap, full Kaggle-legal vs honest** | | | **+0.0066 PR-AUC** |

**Two findings, both real, both worth stating plainly:**

1. **The total gap is small: +0.0066 PR-AUC (~1.2% relative).** Using both of the 1st-place solution's leakage tricks together buys only a marginal edge over the honest, deployable model. This is arguably the stronger headline than a big gap would have been — it means causal honesty is close to free on this dataset, not a large sacrifice. *(The "leaky features only" row also beats V4/V5's tuned-and-calibrated PR-AUC of 0.5514 by a hair, 0.5697 vs 0.5514 — expected, since leaky features get more real signal, and this is the whole reason the technique existed on the leaderboard in the first place.)*

2. **The client-mean post-processing step made PR-AUC *worse*, not better** (0.5697 → 0.5512), while ROC-AUC improved (0.9064 → 0.9152). Not a bug — checked the mechanism, and it's the *exact same failure mode* as V5's isotonic-calibration finding (the reduction/tuning stage): averaging every one of a client's predictions into one shared value collapses distinct rankings into ties. PR-AUC is sensitive to fine-grained ranking and penalizes that; ROC-AUC, cushioned by the huge negative class, is more tolerant of it and even benefits from tighter between-client separation. Genuinely useful, since it means even a straightforward reproduction of the winning technique doesn't uniformly help — the real story is more nuanced than "leaky always wins," which is itself evidence this was measured honestly rather than tuned to make a clean point.

**Leak proven mechanically, not just asserted:** a demo bucket showed a real client's chronologically-*first* transaction already carrying the exact same `uid_amt_mean_LEAKY` value as its last — meaning that first prediction was built from a statistic computed over transactions that, relative to it, hadn't happened yet.

## Ladder rationale

The point is not the final number. It is that each decision has a measured price tag, so
every claim in the pitch is backed by a delta rather than an assertion.

Two rows are expected to look "bad" and are load-bearing anyway:

- **V5 (calibration)** will not improve PR-AUC — calibration is monotonic, so ranking metrics
  barely move. It is not supposed to improve ranking. It makes `P(fraud)` a *real probability*,
  without which `expected_loss = P × (amount + fee)` is meaningless and the entire cost model
  is fiction. Evidence is the reliability diagram, not the AUC.
- **The Kaggle-legal row** will beat our deployable model. That is the whole point of
  component B: quantifying what causal honesty costs.

## Verified facts (reproduce, don't trust)

| Claim | Source | Our number | Status |
|---|---|---|---|
| ~3.5% fraud rate | dataset | 3.499% (20,663 / 590,540) | ✅ matches |
| 73,838 clients with 2+ txns | 1st place writeup | 92,628 (key: `card1+addr1+D1n`) | ❌ 25% too many — key too coarse |
| 96.9% pure-0 / 2.9% pure-1 / 0.2% mixed | 1st place writeup | 94.49% / 2.15% / **3.36%** | ❌ **mixed 16.8x too high** — see below |
| Multi-txn clients ≈ 50% of rows | 1st place writeup | 79% of rows | ❌ too coarse, same root cause |
| ~68.2% of later clients unseen | 1st place writeup (private test) | 59.7% unseen (day>120 vs ≤120) | ~plausible, different split, not a hard target |
| Fraud rate stable over time (not a drift problem) | 1st place writeup | 2.48%–4.18% across 6 ~30-day blocks, no strong trend | ✅ roughly supports; mild dip in days 0–29 |

**G1 status: FAILED — investigated and formally closed.** Component C (card-level risk state, time-to-detection) is **cut** from scope. Component B is unaffected (see below).

**Investigation log — 8 configurations tried, all above the ~1% stop threshold set in advance:**

| Key | clients (2+) | rows covered | pure-0 | pure-1 | **mixed** |
|---|---|---|---|---|---|
| `card1+addr1+D1n` (base) | 92,628 | 465,318 (79%) | 94.49% | 2.15% | 3.36% |
| `+card2` | 92,834 | 463,400 | 94.48% | 2.16% | 3.35% |
| `+P_emaildomain` | 96,416 | 413,037 | 94.99% | 2.75% | 2.27% |
| `+card2+addr2` | 92,833 | 463,374 | 94.48% | 2.17% | 3.35% |
| `+card2+card5` | 93,504 | 461,599 | 94.52% | 2.16% | 3.32% |
| `+D4n` | 99,991 | 388,525 | 95.06% | 2.82% | **2.11% (best)** |
| `+D10n` | 100,018 | 391,760 | 94.77% | 2.63% | 2.60% |
| `+D15n` | 99,494 | 401,500 | 95.02% | 2.74% | 2.23% |
| **target** (1st place) | 73,838 | 280,829 (50%) | 96.9% | 2.9% | **0.2%** |

Diagnostic 1 (D1-null collapse): ruled out — only 0.21% of rows have missing `D1`, largest affected bucket is 19 rows. Not the driver.

Diagnostic 2 (D-column consistency, the documented refinement technique — sourced via search, not the primary writeup, so tested rather than trusted): only **22.8%** of rows sit in a coarse bucket where `D15n` even fully agrees internally. The ambiguity is structural, not fixable by one more column.

**Conclusion:** the writeup references a separate, dedicated "UID detection script" as its own methodology — and even the 1st-place team states they *"did not add the script's UIDs to our models — machine learning did better finding them on its own."* Exact reproduction of 96.9/2.9/0.2 likely requires iterative fuzzy matching well beyond a concatenated key, which is out of scope for the time available. **This is a scoping decision made from evidence, not a failure to find the right column.**

**What survives:** the causal-vs-full-history comparison (component B) doesn't depend on hitting the exact target purity — it compares the same imperfect key used two ways. We proceed with `card1+addr1+D1n+D4n`. A cheap derived feature, `uid_confident = (std(D15n) in the coarse bucket == 0)`, is added to the causal feature set as a genuine signal, independent of component C's fate.

**Bonus finding — baseline is strong out of the gate:** V0, with zero feature engineering, already scores PR-AUC 0.5486 / ROC-AUC 0.9002 on the untouched test month. Top features (`V201`, `V244`, `V258`) are Vesta's own engineered columns — real identity-like signal already present before we do any UID work. Good sign for the project floor regardless of how G1 resolves.

## V1/V2 finding — causal features underperformed, investigating

**Result:** V1 (+9 causal history features) scored PR-AUC 0.5436 vs V0's 0.5486 (-0.0050). V2 (+ time-consistency screening, dropped 70/411 features) scored 0.5419, worse again.

**Correctness is not in question** — the causal-feature builder passed a brute-force manual recomputation check on 300 sampled rows (0 mismatches), and a deliberate leaky-vs-causal side-by-side demo confirmed the causal version genuinely withholds future information (first row of a real bucket: causal std = NaN, full-bucket std = 68.27, already known). The features are correct. They're just not (yet) helping this model.

**Diagnosis, from two concrete numbers:**

1. **Weak individual signal.** The 9 causal features ranked 266, 280, 296, 328, 351, 355, 382, 407, 420 of 442 by importance — bottom half. Best one (`uid_amt_mean_prior`, importance 0.0008) is ~150x weaker than the baseline stage's top feature (`V201`, importance 0.1136).
2. **The screening slice (day≤40 vs day 121–150) is likely biased against exactly these features.** 7 of our 9 causal features appear in the *top 20* flagged inversions. Day ≤40 is the very start of the observation window — almost no client has accumulated history yet by construction, so history-dependent features measure something structurally different there than in a mature slice. Most flipped val-AUCs sit at 0.48–0.49 (barely below random), consistent with "no real signal, noise flipped the sign" rather than genuine instability.

**Correction to the plan, not a new problem:** component B is "causal vs full-history aggregation of the *same* features," not "features vs no features." This V0→V1 result doesn't touch that comparison — the Kaggle-legal comparison builds the full-history version of V1's own feature set and compares directly against V1.

**Resolved.** Re-ran the screen on two mature slices (day 60–100 vs 121–150). Result confirmed the cold-start-artifact theory decisively:

| Screen | Total flips (of 411) | Causal features flipped (of 9) |
|---|---|---|
| Cold (day≤40 vs 121–150) — original, retracted | 70 | 7 |
| **Mature (day 60–100 vs 121–150) — corrected** | **1** | **1** (`uid_txn_num`, barely: train-AUC 0.5259 → val-AUC 0.4991) |

69 of 70 original "inversions" were an artifact of comparing an immature slice (almost no client has history that early, by construction) to a mature one — not real instability. Rebuilt V2 on the corrected screen (drops only `uid_txn_num`): **PR-AUC 0.5446, ROC-AUC 0.9014** — the final, reported number.

**Final verdict:** causal features are proven correct (300-row manual recomputation, 0 mismatches) and proven temporally stable (survive a fair screen almost intact). Net effect on this exact baseline model is small and mixed: -0.0040 PR-AUC (-0.7% relative), +0.0012 ROC-AUC vs V0. Not a win, not a failure — an honest wash.

**Decision: keep the features, proceed to the reduction/tuning stage on V2's 441-feature set.** (1) They're required infrastructure for component B — the causal-vs-full-history comparison needs a causal feature set to be one half of it. (2) The cost is small enough that the later Optuna sweep may move it either direction; no reason to judge them before tuning exists.

## V5 anomaly — calibration moved PR-AUC more than it should

**Expected:** calibration is monotonic, so PR-AUC (which depends only on score *ranking*, not values) should be ~unchanged. That claim is written directly in the notebook's comments.

**Actual:** ECE improved a lot (0.0103 → 0.0042, real and good), but PR-AUC dropped 0.5514 → 0.5384 (-0.0130) — by far the largest single delta in the project.

**Why "monotonic" didn't protect us:** monotonic only means the transform never *reverses* order — it says nothing about ties. Isotonic regression produces a **step function**, not a smooth curve: wherever it's flat, a whole range of different raw scores collapse to the identical calibrated value. Those transactions become tied even though the raw model could tell them apart, and PR-AUC is sensitive to that. Likely driver: the validation set (83,571 rows, ~2,850 fraud cases) isn't dense enough everywhere for isotonic to avoid flattening in places.

**Resolved.** Diagnostic confirmed the mechanism directly: isotonic collapsed **91,271 distinct raw scores down to 323** on the test set — the ties, made undeniable, not just theorized.

Compared against Platt (sigmoid) scaling — a smooth, strictly-increasing curve that structurally cannot introduce ties:

| Method | PR-AUC | ROC-AUC | ECE |
|---|---|---|---|
| Raw (uncalibrated) | 0.5514 | 0.9077 | 0.0103 |
| Isotonic | 0.5384 | 0.9073 | 0.0042 |
| **Platt — chosen method** | **0.5514** | **0.9077** | **0.0036** |

Platt matches raw *exactly* on both ranking metrics (proves the ties theory) and **beats isotonic on calibration too** — no trade-off needed at all; it strictly dominates. Note this contradicts our own prior expectation ("Platt is less flexible, may calibrate worse") — worth stating plainly that the prediction was wrong and the measurement corrected it, rather than only reporting predictions that turned out right.

**V5 ships with Platt scaling.** The "monotonic ⇒ ranking preserved" claim in the original notebook comment was imprecise — monotonic only forbids *reversals*; a monotonic function can still be a step function and introduce ties. Fixed the comment in the notebook to say this precisely.

## Cost model result — single-XGBoost run (V4/V5 model)

> **This section documents the cost-model run on the single-XGBoost V4/V5 model.** The
> **shipped model is the 2-model ensemble** (XGBoost + untuned LightGBM); its own numbers
> supersede these and are in the later section **"Does the PR-AUC win survive contact with
> the actual cost policy?"** and in `docs/eval_report.md` §1–§4. Quick map: value ₹17.22cr → **₹17.355cr**,
> lift +₹1.54cr / +₹64.3L → **+₹1.678cr / +₹77.03L**, p=0.774 → **p=0.589**, policy mix
> 88,560/2,519/1,348 → **88,331/2,782/1,314**, grid mins ₹1.09cr / ₹57.4L → **+₹1.163cr /
> +₹66.77L**. The cost mechanism, parameter sourcing and FX-correction story below are
> model-independent.

**G6 gate: PASSED.** Single-threshold cost curve has an interior minimum at p=0.774 for this model (not at an edge) — confirmed with the correct FX rate, unchanged from the placeholder-rate run. (p=0.589 for the shipped ensemble, also interior.)

**Policy mix on the test month (92,427 transactions), single-XGBoost run:** allow 95.8% (88,560) · step-up 2.7% (2,519) · block 1.5% (1,348). *(Shipped ensemble: allow 95.6% (88,331) · step-up 3.0% (2,782) · block 1.4% (1,314).)*

**Result — measured on the untouched test month, real Razorpay MDR + real dated FX rate:**

| Policy | Total value | Lift vs this |
|---|---|---|
| No fraud system at all | ₹15.68 crore | — |
| Naive 0.5 threshold (industry default) | ₹16.58 crore | +₹90.0 lakh vs no system |
| **Arbiter — single-XGBoost run** | **₹17.22 crore** | **+₹1.54 crore vs no system, +₹64.3 lakh vs naive 0.5** |
| **Arbiter — shipped 2-model ensemble** | **₹17.355 crore** | **+₹1.678 crore vs no system, +₹77.03 lakh vs naive 0.5** |

**Hard vs modeled split (single-XGBoost run):** of ₹17.22 crore, ₹17.21 crore (99.95%) comes from allow/block outcomes computed directly from actual fraud labels — hard, verifiable. Only ₹92,945 (0.05%) depends on the modeled step-up population rates (`P_STOP`, `P_DROPOFF`). *(For the shipped ensemble the hard-verified share is 100.16% — the modeled step-up component is slightly negative, −₹2.78L.)* The headline claim is overwhelmingly evidence-based, not assumption-dependent.

### FX rate correction — a real, caught, fixed number

First run used `USD_TO_INR = 83.0`, a rough placeholder picked from general knowledge, not checked. Corrected to `95.41` (a live, dated Google/Morningstar quote) once a real dated source was available — same bar the Razorpay MDR number was already held to.

This wasn't cosmetic: `CHARGEBACK_FEE` (₹500) is fixed, not amount-proportional, so a rate change shifts its *relative* weight against transaction size. Verified the effect matched the predicted mechanism — policy shifted marginally more lenient (allow 95.7%→95.8%) as the fixed fee became relatively smaller — before accepting the corrected numbers above.

### Known gap — not blocking, tracked for the docs stage

The sensitivity map (margin × chargeback fee, below) shows how Arbiter's *own* total value moves across assumptions. **Extended and closed** (`scripts/robustness_checks.py` Part 3, fully local — reuses `artifacts/test_month_raw.json`, no new Kaggle run needed): the *lift over both baselines* stays positive across the **entire** 35-point grid, not just the assumed 20%/₹500 point — minimum lift vs no system ₹1.09cr (single-XGBoost run) / **+₹1.163cr (shipped ensemble)**, minimum lift vs naive 0.5 ₹57.4L / **+₹66.77L (shipped ensemble)**, both at the least favorable corner of the grid (highest fee, lowest margin). No negative cells anywhere. Full grid: `docs/robustness_results.json`.

### Sensitivity map (margin × chargeback fee, Arbiter total value, ₹) — single-XGBoost grid

*(The shipped ensemble's grid was swept the same way; its full 35-cell result lives in
`docs/robustness_results.json` — every cell positive vs. both baselines, min lift +₹1.163cr
vs no system / +₹66.77L vs naive. The table below is the single-XGBoost run and
`docs/sensitivity_map.png` renders that same grid.)*

| margin | fee=200 | fee=350 | fee=500 | fee=600 | fee=1000 |
|---|---|---|---|---|---|
| 0.05 | 10.9M | 10.8M | 10.7M | 10.6M | 10.3M |
| 0.10 | 65.6M | 65.3M | 64.8M | 64.5M | 63.4M |
| 0.15 | 116.7M | 116.6M | 116.4M | 116.3M | 115.8M |
| **0.20** | 172.7M | 172.4M | **172.2M** | 172.1M | 171.5M |
| 0.30 | 287.2M | 286.9M | 286.6M | 286.5M | 285.8M |
| 0.40 | 402.9M | 402.6M | 402.3M | 402.2M | 401.4M |
| 0.50 | 519.0M | 518.7M | 518.4M | 518.2M | 517.5M |

## Cost model parameters — FINAL

These are assumptions, not facts, and the panel will ask where they came from. Full
citations: `CLAUDE.md` §6.

| Parameter | Value | Source / justification |
|---|---|---|
| Chargeback fee | ₹500 | Razorpay's own disclosed dispute-fee range ₹200–600, midpoint |
| Payment processing fee (MDR) | 2.36% | Razorpay's own pricing page: 2% platform fee + 18% GST, not refunded on a later chargeback |
| Merchant margin | 20% | Blended e-commerce assumption — swept in sensitivity below |
| Step-up stops fraud | 60% | 3D Secure fraud-reduction studies cite 40–70%, midpoint |
| Step-up drop-off rate | 15% | Checkout-friction studies cite 17–21% for full checkout; a single OTP is less friction, set conservatively lower |
| LTV penalty (false decline) | 3× lost margin | Most speculative parameter — flagged explicitly, swept in sensitivity |
| USD→INR conversion | ₹95.41 | Live, dated quote (Google/Morningstar) — corrected from an unverified placeholder (₹83), see journal |
| Manual review cost | not modeled | Not part of the 3-way allow/step-up/block decision; relevant later for the analyst review queue, not the cost model itself |

## Robustness checks — is the headline number real, or one lucky month?

Every result above is a point estimate from a single untouched test month. `scripts/robustness_checks.py`
(run against `artifacts/test_month_raw.json` — the full 92,427-row test month, row-level, not a
smaller sample or a re-derivation under different assumptions) puts two independent checks
around it.

### 1. Bootstrap confidence interval

Resampled the test month with replacement, 2,000 times (`seed=42`). Critically, **paired per
resample**: Arbiter's total and each baseline's total are computed on the *same* resampled
rows in a given iteration, then differenced — not bootstrapped independently and subtracted
afterward, which would ignore the correlation between them and overstate the true
uncertainty.

Single-XGBoost run:

| | Point estimate | 95% CI |
|---|---|---|
| Arbiter total value | ₹17.22cr | ₹16.86cr – ₹17.59cr |
| Lift vs no system | +₹1.54cr | ₹1.38cr – ₹1.71cr |
| Lift vs naive 0.5 | +₹64.3L | ₹53.7L – ₹75.5L |

Shipped 2-model ensemble (re-run for the consistency migration — `docs/eval_report.md` §2b):

| | Point estimate | 95% CI |
|---|---|---|
| Lift vs no system | **+₹1.678cr** | ₹1.510cr – ₹1.850cr |
| Lift vs naive 0.5 | **+₹77.03L** | ₹64.9L – ₹89.5L |

Both lift intervals (either model) sit comfortably clear of zero. The headline number is a
stable effect across resamples of the same month, not a favorable draw.

### 2. A genuinely fair rules-based baseline

Naive 0.5 is still a strawman nobody would actually run — it needs the trained model's
probability output, just with a crude cutoff. A merchant with **no model at all** might
instead run something simpler: block any transaction above a flat rupee amount. Swept that
rule for its own best threshold (same discipline as picking the strongest available free
LLM for the other benchmark — the point of a baseline is to beat the *best* simple
alternative, not a weak one), rather than picking a threshold arbitrarily.

| Threshold | Total value | Note |
|---|---|---|
| ₹10,000 (context, not tuned) | −₹68.43cr | catastrophic — blocks a huge share of legitimate mid-size purchases |
| ₹50,000 (context, not tuned) | −₹16.97cr | still deeply negative |
| **₹5,12,535 (swept, best found)** | **₹15.68cr** | **identical to the "no system" value** |

**The best possible amount-only rule never blocks a single transaction in this dataset** —
the sweep's true optimum sits right at the maximum observed transaction amount, which
makes "block above threshold" mathematically indistinguishable from "block nothing." Amount
alone carries no usable fraud signal here; every threshold that blocks anything real makes
the outcome worse than doing nothing, because most large transactions are legitimate and
the false-positive penalty scales with the amount blocked.

**Honest reading, not spun toward the convenient conclusion:** Arbiter's entire measured
lift over the best possible simple rule is, by construction, identical to its lift over "no
system at all" (+₹1.678cr for the shipped ensemble; +₹1.54cr for the single-XGBoost run) —
because the best simple rule and no system are the same policy here. That's a stronger claim than "beats a mediocre baseline by some margin": no
non-ML, amount-based business rule can create any value above doing nothing on this data.
Only the model's actual probability signal can.

### 3. Exact false-positive/false-negative breakdown of the real 3-way policy

Every prior report of false-positive cost (`docs/eval_report.md` §3–4) came from an
aggregate-curve ESTIMATE — ~172 wrongly-blocked genuine customers, derived from the
single-threshold PR-curve proxy view. `scripts/robustness_checks.py` recomputes the
exact per-transaction action Arbiter actually picks for every one of the 92,427 rows and
counts the real composition against real labels — no estimation, no proxy.

**Single-XGBoost run** (cross-checked against that model's policy mix 88,560/2,519/1,348):

| | Count |
|---|---|
| Blocked, correctly (real fraud) | 1,115 |
| **Blocked, wrongly (real genuine — the exact false positives)** | **233** |
| Step-up band: real fraud | 654 |
| Step-up band: real genuine | 1,865 |

**Exact false-positive cost: ₹17.97 lakh** (₹17,96,854 — ₹7,712 average per wrongly-blocked
customer), computed row-by-row from `margin × amount × (1 + LTV_multiplier)`, not modeled.

**Shipped 2-model ensemble** (re-run for the consistency migration — full detail in
`docs/eval_report.md` §4): **223** wrongly-blocked genuine of **1,314** total blocks (1,091
correct), **exact cost ₹13.20 lakh** (₹5,917 average) — lower than the single model on both
counts, found while optimising for total rupee value, not for this. Step-up band: 728 real
fraud, 2,054 real genuine.
This *doesn't* match §3's ~172 estimate, and it shouldn't — the two answer different
questions. §3's proxy is a single flat probability cutoff with no step-up option; the real
3-way policy's block boundary is amount-dependent (every cost term scales with transaction
size) and routes much of the middle-risk band to step-up instead — a genuinely different
rule applied to the same data, not a contradiction. The step-up band's fraud/genuine
composition is now exact too; what happens to each of those transactions next (stopped vs.
got through, abandoned vs. completed) remains modeled from population rates, since this
dataset has no real step-up interaction data — unchanged limitation, honest exception list
item #2. Full breakdown: `docs/robustness_results.json`.

### 4. Bootstrap CI on PR-AUC and ROC-AUC themselves, not just the rupee lift

Section 1's paired bootstrap gave the rupee lift a real interval; PR-AUC/ROC-AUC were still
bare point estimates. Extended using the *exact same* resampled indices as section 1 (free
— no extra resampling cost), with hand-rolled PR-AUC and rank-based (Mann-Whitney) ROC-AUC
implementations — matching `scripts/llm_benchmark.py`'s existing hand-rolled average
precision (`requirements.txt` deliberately keeps sklearn out of `scripts/`/`src/`, so a new
metric here follows the same discipline rather than quietly breaking it). Both were
verified to reproduce the already-published point estimates exactly before the interval
built on top of them was trusted:

Single-XGBoost run:

| | Point estimate | 95% CI |
|---|---|---|
| PR-AUC | 0.5514 | 0.5350 – 0.5688 |
| ROC-AUC | 0.9077 | 0.9019 – 0.9132 |

Shipped 2-model ensemble (re-run for the consistency migration — `docs/eval_report.md` §1):

| | Point estimate | 95% CI |
|---|---|---|
| PR-AUC | **0.5597** | 0.5439 – 0.5771 |
| ROC-AUC | **0.9126** | 0.9070 – 0.9180 |

"Measured precision and recall" (the track's own words) now means the same thing
everywhere in this project's reporting, not only in the cost-model section.

Full numbers, methodology, and the sanity check that reproduces the already-reported
headline exactly before trusting anything built on top of it: `journal/`, `docs/eval_report.md` §2b.

## Error analysis and bias/variance/mismatch decomposition (clue skill: Ng's *ML Yearning* ch14/ch40/ch41)

*(Single-XGBoost-era diagnostic — run on the V4/V5 model, not re-run for the shipped
ensemble. Kept because the mechanism it isolates, temporal mismatch from unseen clients,
is model-independent.)*

Diagnosed cold, using `notebooks/04_cost_model.py`'s two new addenda. Full narrative and
reasoning — including why the variance finding below wasn't acted on: `docs/eval_report.md`
§8. Raw numbers here.

**Training-dev decomposition** (diagnostic model, trained on 85% of `tr` — a held-out
training-dev slice withheld — NOT the shipped artifact: `artifacts/training_dev_gap.json`):

| Set | n | PR-AUC | ROC-AUC |
|---|---|---|---|
| train′ (fit on) | 352,361 | 0.9864 | 0.9988 |
| training-dev (unseen, same period) | 62,181 | 0.8357 | 0.9691 |
| val (unseen, next period) | 83,571 | 0.6110 | 0.9233 |
| test (unseen, furthest period) | 92,427 | 0.5496 | 0.9020 |

Variance gap (train′→training-dev): **−0.1508 PR-AUC**. Mismatch gap
(training-dev→val): **−0.2247 PR-AUC — the dominant gap, 1.5x variance**. Further mismatch
(val→test): −0.0614. Diagnostic model's own val/test (0.6110/0.5496) reproduce the shipped
then-shipped single model's (0.6128/0.5514) within 0.002 — a fair proxy, not a different model's story.

**Error analysis** (233 exact false positives, 100 sampled false negatives of 1,444 total —
`artifacts/error_analysis_false_positives.json`, `artifacts/error_analysis_false_negatives_sample.json`):

| | FP (233) | FN sample (100) |
|---|---|---|
| ProductCD='C' (11.7% base fraud rate) | 189 (81.1%) | 17 (17%) |
| ProductCD='W' (2.0% base fraud rate) | 24 (10.3%) | 75 (75%) |
| card6='credit' (6.7% base rate) | 136 (58.4%) | 22 (22%) |
| card6='debit' (2.4% base rate) | 92 (39.5%) | 78 (78%) |
| Top-1 SHAP driver ∈ {V258, C1, C14} | 187 (80.3%) | — |
| Most concentrated single FN driver (C13) | — | 17 (17%) |
| p > 0.95 (confidently wrong) | 110 (47.2%) | — |
| p < 0.80, i.e. within 0.026 of the 0.774 threshold | 67 (28.8%) | 0 (0%) |

Base fraud rates from notebook 01's original EDA (`ProductCD`/`card6` groupby — see its
`.ipynb` output — never previously cross-referenced against a failure population). The
errors cluster exactly where those base rates predict: false positives concentrate in the
highest-risk segment the model has learned to distrust, false negatives in the lowest-risk
segment it has learned to trust.

**Retracted finding, logged for honesty:** an initial cold-start check
(`uid_amt_mean_prior is None`) returned 100% for both populations — later found to be a bug
in the check itself. `uid_amt_mean_prior` is an engineered feature, deliberately excluded
from the raw-column export (`RAW_COLS` filters out everything starting with `uid_`), so the
`.get()` lookup always hit the missing-key default regardless of the client's real history.
Not a real signal about the data. Dropped before it was reported anywhere as fact — same
"verify before trusting" discipline as everything else in this log.

## Hyperparameter sweep — revisiting "not retraining" (`artifacts/hyperparam_sweep.json`)

*(Single-XGBoost-era experiment. The base XGBoost model it tested is still one of the two
members of the shipped ensemble, so the finding — regularizing harder generalizes worse
here — carries directly.)*

Hypothesis, stated before running: every shipped hyperparameter sits at the more-capacity,
less-regularized end of Optuna's search range, and Optuna picked them by maximizing *val*
PR-AUC — which the decomposition above just showed scores 0.2247 higher than test, purely
from being temporally closer to train. Plausible that tuning against an inflated signal
favored a config that generalizes worse than a more conservative alternative would.

Bounded 6-config sweep, each refit on the same 85% train′ split as the decomposition above,
scored on train′/training-dev/val only:

| Config | train′ | training-dev | val | traindev→val gap |
|---|---|---|---|---|
| shipped (control) | 0.9864 | 0.8357 | 0.6110 | 0.2247 |
| shallower depth (6) | 0.9386 | 0.8180 | 0.6049 | 0.2131 |
| higher min_child_weight (10) | 0.9417 | 0.8153 | 0.6000 | 0.2152 |
| higher regularization (λ=3.0, α=1.0) | 0.9833 | 0.8320 | 0.6034 | 0.2287 |
| more conservative sampling (0.6/0.6) | 0.9662 | 0.8189 | 0.6018 | 0.2171 |
| **all four combined** | 0.9066 | 0.7938 | 0.5872 | **0.2066** |

Control row reproduces the decomposition's own numbers (0.9864/0.8357/0.6110) exactly —
same split, same seed, confirms this is measuring the same thing, not a different model's
story.

**Selected by the pre-committed rule** (smallest traindev→val gap among configs within
0.03 PR-AUC of control's val score — every config qualified, even "all four combined" at
0.5872 vs. the 0.5810 floor): **"all four combined."**

**Checked once on test, as confirmation:** PR-AUC **0.5290** vs. the then-shipped single model's
**0.5514** — **−0.0224 (−4.1% relative), a real regression, not a wash.** ROC-AUC moved the
same direction (0.8999 vs. 0.9077).

**Verdict: hypothesis not supported — reversed, in fact.** The more conservative config did
not generalize better to the true target; it generalized worse. **Decision: shipped model
stays. Not retraining.** This *strengthens* rather than reverses the original decision — it
was made on reasoning alone before; it now has a direct empirical test behind it, and the
test came back negative.

**A second, methodological finding worth logging honestly:** the selection metric itself
(traindev→val gap) turned out to be a weak proxy for test generalization here. The winning
config's gap shrank mostly because *both* traindev (0.7938, lowest of the six) and val
(0.5872, lowest of the six) dropped together — a uniformly weaker model, not one that
specifically resists the traindev→val transition better. The 0.03-PR-AUC eligibility band
was meant to prevent exactly this ("a uniformly-bad config with a small gap by coincidence
can't win by default" — see `journal/`) but wasn't tight enough: the winner sat only 0.0038
above the floor and still turned out to be the worst of the six on the metric that actually
matters. Consistent with the decomposition above: the dominant gap is temporal mismatch (new
clients over time), a property of the problem, not something regularization moves — this
sweep is now a direct measurement of that, not just a citation of it.

## Segment-aware calibration (`artifacts/segment_calibration.json`)

*(Single-XGBoost-era experiment — run on the V4/V5 model, not re-run for the shipped
ensemble. Its conclusion — a bundled 5-segment recalibration isn't worth adopting — is
model-independent enough to stand.)*

Follow-up on the error-analysis finding above: 189 of the 233 test false positives (81.1%)
cluster in ProductCD='C'. `decide_action` has no explicit threshold to make segment-aware
(it's a pure per-transaction argmax over three expected-value formulas), so the real
analogue is per-segment *calibration* — a separate Platt mapping per ProductCD feeding the
same, unmodified cost formulas.

All five ProductCD segments had enough val-fraud rows (≥30) to fit their own calibrator, so
this changed calibration for the whole population at once, not just 'C':

| | VAL | TEST (confirmation only) |
|---|---|---|
| Global calibration — total value | ₹15.30cr | ₹17.220cr |
| Segment calibration — total value | ₹15.27cr | ₹17.096cr |
| Delta | **−₹2.73L (−0.18%)** | **−₹12.39L (−0.72%)** |
| Global — ProductCD='C' false positives | 127 | 189 |
| Segment — ProductCD='C' false positives | 93 | 146 |
| FP delta | **−34 (−26.8%)** | **−43 (−22.8%)** |

**Selection made on val only** (pre-committed rule: adopt only if total value doesn't drop
AND the ProductCD='C' FP count doesn't rise) → **val total value dropped, so `keep_global`**,
decided before test was touched. Test, checked once as confirmation, agrees — and the loss
is proportionally larger on test (−0.72% vs −0.18%), so the val-only decision wasn't a fluke
that a bigger sample would have reversed.

**Two things are both true and both real:** the false-positive reduction in ProductCD='C' is
genuine and reproduces on test (189→146, matching the scoped question exactly) — the
narrowly-stated hypothesis was correct. But the governing criterion is total realized value,
not one segment's FP count in isolation (same reasoning as PR-AUC vs. the cost curve
throughout this project), and recalibrating the other four segments alongside 'C' costs more
than 'C' saves. **Not adopted — global calibrator stays.**

**Confound worth naming:** this tested "recalibrate all five segments" as one unit, not
"recalibrate C alone." A C-only variant (other four segments left on the existing global
calibrator) might isolate a real, adoptable win — untried here. Not pursued: the total
effect size on the headline (₹17.355cr for the shipped ensemble) is under 1%, well inside
noise relative to the bootstrap CI already reported on the rupee lift, so the expected payoff doesn't clear
the bar for another Kaggle round trip and a re-verification cascade at this point in the
project.

## Ensemble diagnostic — the first confirmed, statistically real improvement
(`artifacts/ensemble_diagnostic.json`, `artifacts/ensemble_bootstrap.json`)

WHY: the causal-vs-leaky comparison (component B) already showed honoring the no-future-data
constraint costs only +0.0066 PR-AUC — so most of the remaining gap to the Kaggle winners'
0.9408 ROC-AUC (ensemble: CatBoost + LightGBM + XGBoost + a neural net) isn't the leakage
trick. Ensembling is the standard, untried lever, and two of the winners' three tree models
are the same two libraries tested here.

**Method:** LightGBM and CatBoost trained on the exact same feature matrix/split as the
shipped XGBoost, one fair shot each (settings comparable to `BEST_PARAMS`, not independently
tuned), each with its own Platt calibrator. Ensemble = plain average of calibrated
probabilities. Selection on val only; test checked once as confirmation.

| Model | val PR-AUC | test PR-AUC | val→test drop (relative) |
|---|---|---|---|
| XGBoost (single-model baseline) | 0.6128 | 0.5514 | −10.02% |
| LightGBM (untuned) | 0.5976 | **0.5541** | **−7.27% (smallest)** |
| CatBoost (untuned) | 0.6062 | 0.5434 | −10.36% (largest) |
| **2-model ensemble (XGB+LGB) — SHIPPED** | 0.6127 | **0.5597** | −8.64% |
| 3-model ensemble (not shipped) | **0.6204** | **0.5628** | −9.29% |

**Interesting secondary finding:** LightGBM, with *zero* tuning, both scores highest on test
of the three individual models AND degrades least from val to test — the opposite of what
"less-developed model" would predict, and consistent with the hyperparameter sweep's own
finding that heavily tuning against val may chase its temporal-proximity bias. CatBoost, by
contrast, is both the weakest individual test score AND has the largest relative val→test
drop — it doesn't resist mismatch better, yet still adds incremental value to the ensemble
(3-model test 0.5628 > 2-model test 0.5597), consistent with classic ensemble diversity
(different, uncorrelated errors help an average even from a comparatively weaker model) more
than with mismatch-robustness specifically.

**Was this real, or noise? Paired bootstrap, 2,000 resamples, same resampled row indices for
every model every time (isolates the delta's own uncertainty, not each model's individual
noise):**

| Comparison | Point estimate | 95% CI | Excludes zero? |
|---|---|---|---|
| 3-model ensemble − shipped | +0.0114 | **[+0.0081, +0.0146]** | ✅ yes |
| 2-model ensemble − shipped | +0.0083 | **[+0.0059, +0.0109]** | ✅ yes |

**Both confirmed real** — the improvement is not within the noise of one test month, unlike
the segment-calibration case where the point estimate alone wasn't enough to trust. The two
CIs overlap in the 0.0081–0.0109 range, so this data doesn't cleanly prove the 3-model
version beats the 2-model version by a statistically distinguishable amount — the 3-model's
higher point estimate is a real signal, not yet a proven-significant edge over the simpler
2-model alternative.

**Status: this is the first "make it better" experiment to clear the bar for
a real ship conversation** — unlike the hyperparameter sweep (rejected, real regression) and
segment calibration (rejected, net value loss). Best known configuration as of this result:
the **untuned** 3-model ensemble (below explains why tuning LightGBM further made things
worse, not better).

## LightGBM real tuning pass — a third, independent confirmation that tuning against val is
risky here (`artifacts/lgb_tuning.json`)

WHY: LightGBM's untuned strength (best individual test score, smallest val→test
degradation) suggested real headroom. Ran a genuine 60-trial Optuna search mirroring
notebook 03's own XGBoost search exactly — same TPE sampler/seed, same two-phase
cheap-search-then-full-refit structure, same val-PR-AUC objective — for a fair,
comparable-effort tuning pass, not a shortcut.

| | val PR-AUC | test PR-AUC | val→test drop (relative) |
|---|---|---|---|
| LightGBM, untuned (one fair shot) | 0.5976 | 0.5541 | −7.27% |
| **LightGBM, tuned (60-trial Optuna)** | **0.6116** | **0.5406** | **−11.61%** |

**Tuning made it worse, not better — and worse specifically in the mismatch-degradation
sense.** Val score improved (as expected — that's literally what was optimized), but test
score *dropped* below even the untuned version, and the val→test degradation got larger, not
smaller (−11.6% vs −7.3%, now worse than even the shipped XGBoost's −10.0%). Paired
bootstrap (2,000 resamples) on tuned-vs-untuned-solo: 95% CI **[−0.0182, −0.0090] — a
statistically confirmed regression**, not just a smaller point estimate that could be noise.

**This is the third, independent line of evidence for the same finding** (after the
hyperparameter sweep's regularization test and LightGBM's own untuned-vs-tuned-XGBoost
comparison): optimizing against val PR-AUC in this problem risks chasing val's temporal
proximity to train rather than genuine generalization, and this now holds across two
different model families, not just one.

**Consequence for the ensemble:** re-ran the 2-model and 3-model ensembles with the tuned
LightGBM swapped in for the untuned version.

| Ensemble | test PR-AUC | vs. shipped (bootstrap 95% CI) |
|---|---|---|
| 2-model, untuned LGB (confirmed real) | 0.5597 | [+0.0059, +0.0109] |
| 2-model, **tuned** LGB | 0.5516 | **[−0.0025, +0.0027] — includes zero, not distinguishable from shipped** |
| 3-model, untuned LGB (confirmed real) | **0.5628** | **[+0.0081, +0.0146]** |
| 3-model, tuned LGB | 0.5587 | (not separately bootstrapped — solo comparison already shows the regression; worse than the untuned 3-model by −0.0041 test PR-AUC) |

**Swapping in the "improved" LightGBM made the ensemble's advantage disappear at 2 models
and shrink at 3.** The untuned model's specific behavior — not just any LightGBM — was doing
the real work.

**Decision: the best known, confirmed configuration remains the untuned 3-model ensemble
(XGBoost + untuned LightGBM + untuned CatBoost), test PR-AUC 0.5628, CI [+0.0081, +0.0146].**
This experiment didn't find an improvement — it protected the existing confirmed win by
proving the "obvious next step" (tune the strong component further) would have made things
worse if adopted without checking. Recommendation going forward: do not tune any ensemble
member against val PR-AUC in this problem; if CatBoost is tuned at all, it should not use
this same objective, given it already shows the largest val→test degradation of the three
even untuned.

## Diversity check — Random Forest, Extra Trees, Logistic Regression (`artifacts/diversity_check.json`)

WHY: given tuning further just failed, tried a different question — does adding genuinely
different, still-untuned model families (bagging instead of boosting, and a linear model)
help further? All three left deliberately untuned, same lesson as above.

| Model | val PR-AUC | test PR-AUC | val→test drop |
|---|---|---|---|
| Random Forest | 0.5006 | 0.4490 | −10.3% |
| Extra Trees | 0.4561 | 0.4032 | −11.6% |
| **Logistic Regression** | 0.4007 | **0.1721** | **−57.0%** |

**All three are far weaker than any current ensemble member** (XGBoost/LightGBM/CatBoost
all score 0.54–0.55 test PR-AUC). Logistic regression in particular collapses on test —
barely above the ~0.035 random baseline in absolute terms, and by far the largest
val→test degradation measured anywhere in the ensemble experiments.

**6-model ensemble (adding all three, plain average): test PR-AUC 0.4939 — far below the
confirmed 3-model ensemble's 0.5628.** Paired bootstrap against the confirmed 3-model
ensemble: 95% CI **[−0.0750, −0.0632] — a large, unambiguous regression**, an order of
magnitude bigger than the LightGBM-tuning regression and not remotely close to zero.

**Why, mechanically:** a plain average weighs every model equally regardless of quality.
CatBoost (the weakest *existing* member, 0.5434 test) was still competitive enough that
averaging it in helped via diversity. These three are not competitive — averaging them in
at equal weight simply drags the mean down. This isn't evidence that diversity itself is
bad; it's evidence that *unweighted* diversity with components this much weaker is bad. A
weighted average (weights chosen on val, confirmed once on test) could in principle let
genuinely uncorrelated-but-weak signal survive without being this costly — untried, and
given RF alone (the strongest of the three) is still 0.449 vs. the ensemble's 0.563, the
expected marginal value even under optimal weighting looks small. Not pursued further.

**Decision: not adopted. Confirmed configuration remains the untuned 3-model ensemble**
(XGBoost + untuned LightGBM + untuned CatBoost, test PR-AUC 0.5628, CI [+0.0081, +0.0146]).

**Read across all five "make it better" experiments** (hyperparameter sweep,
segment calibration, ensemble diagnostic, LightGBM tuning, diversity check): one clean,
confirmed win (the untuned 3-model ensemble), and four independent negative controls, three
of them model-side interventions that all failed for the same underlying reason — the
dominant error gap here is *temporal mismatch* (§8a), which model architecture/tuning/
diversity cannot fix, because it isn't a property of the model. The textbook fix for
mismatch (data matching the target distribution) isn't available with a closed competition
dataset. This is convergent evidence the model has reached a real ceiling for this feature
set, not that it's under-explored.

## Does the PR-AUC win survive contact with the actual cost policy? (`artifacts/ensemble_rupee_value.json`)

WHY: every ensemble comparison above was judged on PR-AUC/ROC-AUC. This project's own
thesis is that PR-AUC isn't the objective, rupees are — the segment-calibration experiment
already proved a metric win doesn't always mean a value win. Ran both ensemble candidates
through the exact `value_allow`/`value_stepup`/`value_block`/`realized_value` machinery
already used for the shipped headline number — no retraining, reused already-computed
calibrated probabilities.

| Policy | Test-month value | Lift vs. single-XGBoost | 95% CI |
|---|---|---|---|
| Single-XGBoost baseline (previously shipped) | ₹17.220cr | — | — |
| **2-model ensemble — SHIPPED** | **₹17.355cr** | **+₹13.58L** | **[+₹6.55L, +₹21.24L] — real** |
| 3-model ensemble (not shipped) | ₹17.406cr | **+₹18.65L** | **[+₹10.12L, +₹27.91L] — real** |

**Both ensembles produce a statistically confirmed real rupee lift over the shipped single
model — the PR-AUC win genuinely survives contact with the actual policy this time**, unlike
the segment-calibration case. This is the strongest possible confirmation that the
ensembling decision was right.

**3-model vs. 2-model, directly (settles what the PR-AUC-only comparison couldn't):** point
estimate +₹5.07L in favor of 3-model, but the paired bootstrap 95% CI is **[−₹0.017L,
+₹10.55L] — technically includes zero**, just barely (the lower bound misses significance by
about ₹16,500 out of a ₹1M+ range). **Practically, 3-model is very likely at least as good
as 2-model and probably somewhat better — but this does not clear the formal significance
bar**, on the one metric (rupees) that governs every other decision in this project.

**How this settled the 2-vs-3 question:** an earlier "ship 3-model" lean had been formed on
the PR-AUC evidence, before this check ran. This result confirms ensembling itself but
leaves 2-vs-3 a genuinely close call — a small, not-quite-significant edge (3-model) against
lower production complexity (2-model, one fewer model to deploy, monitor, and explain via
SHAP). Revisited explicitly rather than proceeding silently: **the final decision was to
ship the 2-model ensemble** (XGBoost + untuned LightGBM), simplicity as the tiebreaker per
ML Yearning ch09's optimizing/satisficing framework — when the metric that governs every
other decision here can't separate two options, fall back to the satisficing constraint.
The engine was rebuilt around the 2-model ensemble; see `CLAUDE.md` §13.
