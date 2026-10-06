# Experiment log

Historical research record and parameter provenance. [eval_report.md](eval_report.md) is authoritative for current ensemble results. The current model combines **Optuna-tuned XGBoost + untuned LightGBM** after independent Platt calibration; the whole ensemble is not untuned.

Unless stated otherwise, test scores use day > 150 (92,427 rows). This holdout was reused across follow-up comparisons and adaptive model-selection decisions. Validation-first selection in individual experiments does not make the accumulated test evaluation untouched or independent. The record does not establish training directly on test labels. Fresh untouched temporal data is needed for a final independent estimate.

## Historical single-XGBoost ladder

Rows identify stage changes, including alternate calibration choices; they are specific to these experiments. Random-model PR-AUC is approximately the fraud base rate (~0.035).

| # | Change | PR-AUC | ROC-AUC | Δ PR-AUC | Notes |
|---|---|---|---|---|---|
| V0 | Baseline: raw features, temporal split, no tuning | 0.5486 | 0.9002 | — | 17.9x lift(val) / 15.8x(test) over random. |
| V1 | + causal UID aggregates (expanding window only) | 0.5436 | 0.9035 | -0.0050 | Correct & stable (verified). Small honest cost, not a win. Kept for component B. |
| V2 | + time-consistency feature screening (**1**/411 dropped, corrected screen) | **0.5446** | **0.9014** | -0.0040 (vs V0) | **Final at this stage.** Original screen (an early slice vs. a mature one) wrongly dropped 70 features on a cold-start-biased comparison: retracted, see below. |
| V3 | + V-column reduction (NaN-group + correlation, 339→279 kept) | 0.5469 | 0.9058 | +0.0023 | First clear win since V0. |
| V4 | + Optuna sweep (60 trials completed, best val PR-AUC 0.6128) | **0.5514** | **0.9077** | +0.0045 | Highest measured score at this stage. Params: max_depth=10, lr=0.049, colsample_bytree=0.89, min_child_weight=2. |
| V5 | + Platt (sigmoid) calibration | **0.5514** | **0.9077** | **0.0000** | **Final at this stage.** Isotonic tested and rejected: see below. ECE 0.0036 (best of all three methods tested). |
| V5 | + isotonic calibration (tested, rejected) | 0.5384 | 0.9073 | **-0.0130** | Expected "~unchanged by design" (monotonic): wrong, see below. Collapsed 91,271 distinct scores to 323 (ties). |
| — | **LLM-as-classifier benchmark** | XGB 0.5735 / LLM 0.1571 | — | — | 200-row held-out sample, `gpt-oss:20b` on Kaggle GPU. Historical single XGBoost; 10 fraud labels, two LLM failures; see eval report §7. |

## Component B: the Kaggle-legal leaky model

Historical V2 comparison in [notebook 05](../notebooks/05_kaggle_legal_leaky.py), with the same split and hyperparameters for causal and full-history features:

| Model | PR-AUC | ROC-AUC | vs honest V2 |
|---|---|---|---|
| **Honest (causal, V2, deployable)** | 0.5446 | 0.9014 | — |
| Leaky features only (undeployable) | **0.5697** | 0.9064 | +0.0251 PR-AUC |
| + leaky client-mean post-processing (undeployable) | 0.5512 | **0.9152** | −0.0185 PR-AUC vs leaky-feat |
| **TOTAL gap, full Kaggle-legal vs honest** | | | **+0.0066 PR-AUC** |

Features alone add **0.0251** PR-AUC (0.5697−0.5446); after client-mean averaging the gap is **0.0066** (0.5512−0.5446, ~1.2% relative). Averaging decreases PR-AUC by 0.0185 while increasing ROC-AUC. These concern this model/features/split, not universal causality cost or exact reproduction of the winning model. A first-transaction demo carrying its full-bucket mean establishes future-information leakage mechanically.

## Verified facts (reproduce, don't trust)

Historical comparisons to the first-place writeup are retained with their original attribution; its private-test split is different from this temporal split.

| Claim | Source | Our number | Status |
|---|---|---|---|
| ~3.5% fraud rate | dataset | 3.499% (20,663 / 590,540) | ✅ matches |
| 73,838 clients with 2+ txns | 1st place writeup | 92,628 (key: `card1+addr1+D1n`) | ❌ 25% too many: key too coarse |
| 96.9% pure-0 / 2.9% pure-1 / 0.2% mixed | 1st place writeup | 94.49% / 2.15% / **3.36%** | ❌ **mixed 16.8x too high**: see below |
| Multi-txn clients ≈ 50% of rows | 1st place writeup | 79% of rows | ❌ too coarse, same root cause |
| ~68.2% of later clients unseen | 1st place writeup (private test) | 59.7% unseen (day>120 vs ≤120) | ~plausible, different split, not a hard target |
| Fraud rate across time blocks | 1st place writeup | 2.48%–4.18% across 6 ~30-day blocks, no strong trend | Descriptive rates; mild dip in days 0–29; does not rule out distribution drift |

G1 failed; Component C card-level state/time-to-detection was scoped out. Eight key configurations were tested:

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

D1-null collapse was ruled out (0.21% missing, largest affected bucket 19 rows). Only 22.8% of rows had internally agreeing D15n in a coarse bucket. Best mixed share remained 2.11% versus 0.2% target; exact reconstruction via the separate matching methodology was not achieved. Component B compares the same imperfect key two ways and does not require target purity. The retained key is `card1+addr1+D1n+D4n`, with `uid_confident` as an additional signal. Investigation history: [build journal](../journal/build-log.md).

## V1/V2 causal-feature screening

The causal builder matched brute-force recomputation on 300 sampled rows (zero mismatches). Nine features ranked 266, 280, 296, 328, 351, 355, 382, 407, 420 of 442; `uid_amt_mean_prior` importance 0.0008 versus V201 0.1136. The original cold-versus-mature screen was retracted:

| Screen | Total flips (of 411) | Causal features flipped (of 9) |
|---|---|---|
| Cold (day≤40 vs 121–150): original, retracted | 70 | 7 |
| **Mature (day 60–100 vs 121–150): corrected** | **1** | **1** (`uid_txn_num`, barely: train-AUC 0.5259 → val-AUC 0.4991) |

The corrected V2 score is 0.5446 / 0.9014: net −0.0040 PR-AUC and +0.0012 ROC-AUC versus V0. Features were retained for Component B and later tuning despite the PR-AUC decrease. Details of the corrected screen are in the journal.

## V5 calibration study: historical single XGBoost

Isotonic collapsed 91,271 distinct raw scores to 323 ties. A nondecreasing transform can preserve order while introducing ties; this study chose Platt:

| Method | PR-AUC | ROC-AUC | ECE |
|---|---|---|---|
| Raw (uncalibrated) | 0.5514 | 0.9077 | 0.0103 |
| Isotonic | 0.5384 | 0.9073 | 0.0042 |
| **Platt: chosen method** | **0.5514** | **0.9077** | **0.0036** |

Platt preserved ranking and had lower ECE in this study. The current ensemble calibrates each member before averaging; this table and [reliability_diagram.png](reliability_diagram.png) do not measure calibration of the averaged ensemble.

## Cost model result: single-XGBoost run (V4/V5 model)

All monetary results are retrospective simulated value under the parameters below, not observed realized profit. Observed labels/amounts and deterministic routing are distinct from assumed costs and counterfactual outcomes.

| Policy | Simulated value | Lift vs this |
|---|---|---|
| No fraud system at all | ₹15.68 crore | — |
| Reference 0.5 cutoff | ₹16.58 crore | +₹90.0 lakh vs no system |
| **Arbiter: single-XGBoost run** | **₹17.22 crore** | **+₹1.54 crore vs no system, +₹64.3 lakh vs reference 0.5 0.5** |
| **Arbiter: shipped 2-model ensemble** | **₹17.355 crore** | **+₹1.678 crore vs no system, +₹77.03 lakh vs reference 0.5 0.5** |

Historical single policy mix: allow 88,560, step-up 2,519, block 1,348. Current ensemble: 88,331 / 2,782 / 1,314. Two-way presentation curves were optimized on test, with interior minima p = 0.774 (single) and p = 0.589 (ensemble); neither is the amount-dependent three-way operating point.

Historical single-model allow/block component was about ₹17.21cr and modeled step-up contribution ₹92,945. Current ensemble observed-label allow/block component is **₹173.83M**, and step-up contributes about **−₹0.278M**. Both depend on assumed economics; using labels does not make them realized merchant cash outcomes.

The FX scenario changed from 83.0 to 95.41 in the recorded work. The fixed rupee fee then had less weight relative to converted amounts and the policy shifted slightly toward allow. The earlier Google/Morningstar quote's date and supporting historical snapshot could not be verified; 95.41 is retained as a fixed scenario value, not today's live rate.

### Sensitivity map: historical single-XGBoost simulated value, ₹

Only margin and chargeback fee were swept. [sensitivity_map.png](sensitivity_map.png) shows this historical grid; the current ensemble's 35 cells are in [robustness_results.json](robustness_results.json).

| margin | fee=200 | fee=350 | fee=500 | fee=600 | fee=1000 |
|---|---|---|---|---|---|
| 0.05 | 10.9M | 10.8M | 10.7M | 10.6M | 10.3M |
| 0.10 | 65.6M | 65.3M | 64.8M | 64.5M | 63.4M |
| 0.15 | 116.7M | 116.6M | 116.4M | 116.3M | 115.8M |
| **0.20** | 172.7M | 172.4M | **172.2M** | 172.1M | 171.5M |
| 0.30 | 287.2M | 286.9M | 286.6M | 286.5M | 285.8M |
| 0.40 | 402.9M | 402.6M | 402.3M | 402.2M | 401.4M |
| 0.50 | 519.0M | 518.7M | 518.4M | 518.2M | 517.5M |

Current ensemble minimum lift is +₹1.163cr versus no system and +₹66.77L versus reference 0.5, with no negative cells; historical single-model minima were ₹1.09cr / ₹57.4L. The minima need not occur in the same grid cell. LTV, MDR, step-up stopping/drop-off and FX remained fixed; their sensitivity is future validation.

## Cost model parameters and provenance

Values are retained unchanged from [policy.py](../src/policy.py). Access/review date is **2026-10-06**; it is not a publication date or experiment run date. No merchant contract, actual unit margin, customer LTV or step-up interaction data is available. The current official pricing source supports an illustrative scenario, not the exact costs of an unnamed historical merchant.

| Parameter | Retained value | Source / assumption and verification status | Publication date / access date | Sensitivity coverage |
|---|---|---|---|---|
| Chargeback fee | ₹500 | Scenario assumption; earlier unnamed ₹200–600 dispute-fee provenance unverified. ₹500 is not that range's arithmetic midpoint (₹400). | Historical date unknown; reviewed 2026-10-06 | ₹200, ₹350, ₹500, ₹600, ₹1,000 |
| Payment processing fee (MDR in code) | 2.36% = 2% × 1.18 | Illustrative gateway-fee assumption. [Razorpay pricing](https://razorpay.com/pricing/) FAQ describes 2% + GST; [official pricing explanation](https://razorpay.com/blog/razorpay-payment-gateway-pricing-explained/) describes 18% GST on platform fees. Current offerings include exceptions/promotions; this does not establish historical merchant pricing. Retaining the fee after a chargeback is a scenario assumption here, not verified by these citations. | Pricing page undated; explanation dated 2026-02-13; accessed 2026-10-06 | Fixed |
| Merchant margin | 20% | Blended e-commerce scenario assumption; no observed merchant margin. | No publication; reviewed 2026-10-06 | 5%, 10%, 15%, 20%, 30%, 40%, 50% |
| Step-up stops fraud (`P_STOP`) | 60% | Scenario assumption; earlier unnamed 3DS 40–70% provenance unverified; not measured here. | Historical date unknown; reviewed 2026-10-06 | Fixed |
| Genuine step-up drop-off (`P_DROPOFF`) | 15% | Scenario assumption; earlier unnamed checkout-friction 17–21% sources and extrapolation to OTP unverified; not measured here. | Historical date unknown; reviewed 2026-10-06 | Fixed |
| LTV penalty, false decline | 3× lost margin | Speculative scenario assumption, added to the immediate lost margin; no observed customer lifetime loss. | No publication; reviewed 2026-10-06 | Fixed, not swept |
| USD→INR conversion | ₹95.41 per USD | Fixed illustrative conversion. Historical Google/Morningstar source/date unverified; not a current live quote. Earlier value was ₹83. | Historical date unknown; reviewed 2026-10-06 | Fixed |
| Manual review cost | Not modeled | Outside allow/step-up/block economics; no inferred analyst-cost measurement. | No publication; reviewed 2026-10-06 | Not tested |

## Robustness checks: conditional within-month evidence

[robustness_checks.py](../scripts/robustness_checks.py) resamples rows with replacement 2,000 times (seed 42). Paired deltas use the same row indices for each policy/model. Intervals are conditional on this reused month and fixed parameters, under row-resampling assumptions; they do not account for dependence, adaptive model selection, parameter uncertainty or future temporal shifts. Group/time-block bootstrap and new temporal data are follow-ups.

### 1. Bootstrap confidence interval

Historical single-XGBoost:

| | Point estimate | 95% CI |
|---|---|---|
| Arbiter total value | ₹17.22cr | ₹16.86cr – ₹17.59cr |
| Lift vs no system | +₹1.54cr | ₹1.38cr – ₹1.71cr |
| Lift vs naive 0.5 | +₹64.3L | ₹53.7L – ₹75.5L |

Current ensemble:

| | Point estimate | 95% CI |
|---|---|---|
| Lift vs no system | **+₹1.678cr** | ₹1.510cr – ₹1.850cr |
| Lift vs naive 0.5 | **+₹77.03L** | ₹64.9L – ₹89.5L |

The lift intervals exclude zero under those assumptions. Separate policy CIs or their overlap do not test a paired model delta; use the direct paired exports below.

### 2. Amount-only rules baseline

An amount-only threshold was swept on this test month under the same economics:

| Threshold | Total value | Note |
|---|---|---|
| ₹10,000 (context, not tuned) | −₹68.43cr | catastrophic: blocks a huge share of legitimate mid-size purchases |
| ₹50,000 (context, not tuned) | −₹16.97cr | still deeply negative |
| **₹5,12,535 (swept, best found)** | **₹15.68cr** | **identical to the "no system" value** |

The best swept threshold blocks nothing and equals no system. This does not establish that all non-ML rules lack value, or that amount carries no predictive information in every setting.

### 3. Exact policy composition and calculated false-positive cost

Historical single-XGBoost counts:

| | Count |
|---|---|
| Blocked, correctly (real fraud) | 1,115 |
| **Blocked, wrongly (real genuine: the exact false positives)** | **233** |
| Step-up band: real fraud | 654 |
| Step-up band: real genuine | 1,865 |

The 233 genuine blocks yield calculated cost ₹17.97L (₹17,96,854; ₹7,712 average), using `margin × amount × (1 + LTV_multiplier)`. Counts are exact; costs are assumed. Current ensemble has 1,091 fraud / 223 genuine blocks, and 728 fraud / 2,054 genuine step-ups. Calculated genuine-block cost is ₹13.20L (₹5,917 average). Actual block precision is 83.0%, block fraud recall 34.0%, and block-or-step-up routing covers 56.6% of fraud labels, not measured fraud prevention. The old aggregate proxy estimated ~172 genuine blocks for single XGBoost (~215 for the current ensemble), using a different rule.

### 4. Bootstrap model-quality intervals

Historical single XGBoost:

| | Point estimate | 95% CI |
|---|---|---|
| PR-AUC | 0.5514 | 0.5350 – 0.5688 |
| ROC-AUC | 0.9077 | 0.9019 – 0.9132 |

Current ensemble:

| | Point estimate | 95% CI |
|---|---|---|
| PR-AUC | **0.5597** | 0.5439 – 0.5771 |
| ROC-AUC | **0.9126** | 0.9070 – 0.9180 |

## Error analysis and bias/variance/mismatch decomposition

Historical diagnostic only (`artifacts/training_dev_gap.json`), using 85% train′ and a withheld random 15% training-dev slice:

| Set | n | PR-AUC | ROC-AUC |
|---|---|---|---|
| train′ (fit on) | 352,361 | 0.9864 | 0.9988 |
| training-dev (unseen, same period) | 62,181 | 0.8357 | 0.9691 |
| val (unseen, next period) | 83,571 | 0.6110 | 0.9233 |
| test (unseen, furthest period) | 92,427 | 0.5496 | 0.9020 |

Variance gap is −0.1508 PR-AUC; training-dev→validation mismatch gap is −0.2247; validation→test gap is −0.0614. Mismatch is larger, while substantial variance remains. The decomposition does not establish an exclusive cause or a model-independent mechanism. Its validation/test scores are within 0.002 of the then-baseline's 0.6128 / 0.5514.

Historical error analysis reviewed all 233 genuine blocks and 100 sampled fraud allows from 1,444 (`artifacts/error_analysis_false_positives.json`, `artifacts/error_analysis_false_negatives_sample.json`). Base segment rates come from notebook 01's EDA:

| | FP (233) | FN sample (100) |
|---|---|---|
| ProductCD='C' (11.7% base fraud rate) | 189 (81.1%) | 17 (17%) |
| ProductCD='W' (2.0% base fraud rate) | 24 (10.3%) | 75 (75%) |
| card6='credit' (6.7% base rate) | 136 (58.4%) | 22 (22%) |
| card6='debit' (2.4% base rate) | 92 (39.5%) | 78 (78%) |
| Top-1 SHAP driver ∈ {V258, C1, C14} | 187 (80.3%) | — |
| Most concentrated single FN driver (C13) | — | 17 (17%) |
| p > 0.95 (confidently wrong) | 110 (47.2%) | — |
| Historical proxy context: p < 0.80 (FP); 0.10 < p < 0.774 (FN) | 67 (28.8%) | 0 (0%) |

None of the 100 sampled false negatives scored near the old 0.774 two-way proxy; the table's FP band means below 0.80, whereas the FN column asks about 0.10 to 0.774. Small threshold changes look unlikely to help this sample, but this is not proof that all threshold changes are futile. An initial cold-start check used an absent engineered field and was retracted; its history is retained in the journal.

## Hyperparameter sweep: historical single-XGBoost baseline

`artifacts/hyperparam_sweep.json`: six configurations on the same 85% training split. Here and in subsequent comparison tables, historical 'shipped' means the then-single-XGBoost baseline, not today's ensemble.

| Config | train′ | training-dev | val | traindev→val gap |
|---|---|---|---|---|
| then-single-XGBoost baseline (control) | 0.9864 | 0.8357 | 0.6110 | 0.2247 |
| shallower depth (6) | 0.9386 | 0.8180 | 0.6049 | 0.2131 |
| higher min_child_weight (10) | 0.9417 | 0.8153 | 0.6000 | 0.2152 |
| higher regularization (λ=3.0, α=1.0) | 0.9833 | 0.8320 | 0.6034 | 0.2287 |
| more conservative sampling (0.6/0.6) | 0.9662 | 0.8189 | 0.6018 | 0.2171 |
| **all four combined** | 0.9066 | 0.7938 | 0.5872 | **0.2066** |

The pre-committed rule selected the smallest training-dev→validation gap among configurations within 0.03 of control validation PR-AUC (floor 0.5810). All qualified; 'all four combined' won the gap criterion but lowered both training-dev and validation scores. Test PR-AUC was 0.5290 versus baseline 0.5514 (−0.0224, −4.1% relative); ROC-AUC 0.8999 versus 0.9077. The candidate was rejected. This six-config outcome does not prove regularization cannot improve performance.

## Segment-aware calibration: historical single XGBoost

`artifacts/segment_calibration.json`: all five ProductCD segments had at least 30 validation fraud rows for their own Platt calibrator; this tested recalibrating all five together.

| | VAL | TEST (reused holdout) |
|---|---|---|
| Global calibration: simulated value | ₹15.30cr | ₹17.220cr |
| Segment calibration: simulated value | ₹15.27cr | ₹17.096cr |
| Delta | **−₹2.73L (−0.18%)** | **−₹12.39L (−0.72%)** |
| Global: ProductCD='C' false positives | 127 | 189 |
| Segment: ProductCD='C' false positives | 93 | 146 |
| FP delta | **−34 (−26.8%)** | **−43 (−22.8%)** |

Validation-first rule: adopt only if total simulated value does not drop and C-segment false positives do not rise. Validation value decreased, so global calibration was retained; later reused-test results showed the same direction. The C-segment count reduction is observed, but the value comparison uses assumptions. A C-only variant remains untested. The historical rough ₹3 to 4 lakh opportunity estimate is about 0.17 to 0.23% of the current ₹17.355cr headline, not a measured saving or significance test.

## Ensemble diagnostic: historical comparisons, current choice identified

`artifacts/ensemble_diagnostic.json`, `artifacts/ensemble_bootstrap.json`: LightGBM and CatBoost used the same features/split as the tuned XGBoost baseline, comparable initial settings and independent Platt calibrators. Plain averages were compared on validation and the reused test month. The final model choice also used test-month simulated value, so it is not wholly validation-only selection.

| Model | val PR-AUC | test PR-AUC | val→test drop (relative) |
|---|---|---|---|
| XGBoost (single-model baseline) | 0.6128 | 0.5514 | −10.02% |
| LightGBM (untuned) | 0.5976 | **0.5541** | **−7.27% (smallest)** |
| CatBoost (untuned) | 0.6062 | 0.5434 | −10.36% (largest) |
| **2-model ensemble (XGB+LGB): SHIPPED** | 0.6127 | **0.5597** | −8.64% |
| 3-model ensemble (not shipped) | **0.6204** | **0.5628** | −9.29% |

LightGBM had the highest individual test PR-AUC and smallest relative drop among these three. CatBoost was weaker individually but added ranking value to the average; this is consistent with complementary errors, not proof of an independent causal mechanism.

Paired row-bootstrap deltas against the then-single-XGBoost baseline (2,000 resamples):

| Comparison | Point estimate | 95% CI | Excludes zero? |
|---|---|---|---|
| 3-model ensemble − then-single-XGBoost baseline | +0.0114 | **[+0.0081, +0.0146]** | ✅ yes |
| 2-model ensemble − then-single-XGBoost baseline | +0.0083 | **[+0.0059, +0.0109]** | ✅ yes |

Both delta intervals exclude zero conditional on this reused month. Their overlap does not test three versus two models; that question requires a direct paired delta. The table alone does not establish its PR-AUC significance. The later direct simulated-value export informs the final two-model choice below.

## LightGBM tuning pass

`artifacts/lgb_tuning.json`: 60-trial Optuna search using validation PR-AUC, same two-phase search/refit structure and TPE seed as the XGBoost search.

| | val PR-AUC | test PR-AUC | val→test drop (relative) |
|---|---|---|---|
| LightGBM, untuned (one fair shot) | 0.5976 | 0.5541 | −7.27% |
| **LightGBM, tuned (60-trial Optuna)** | **0.6116** | **0.5406** | **−11.61%** |

Tuned-minus-untuned solo paired bootstrap CI [−0.0182, −0.0090] excludes zero. Validation ranking improved while test ranking declined in this comparison. This adds evidence that validation gains may fail to transfer here; it is not an independent temporal replication of earlier experiments.

Ensemble substitutions, all delta CIs against the then-single-XGBoost baseline:

| Ensemble | test PR-AUC | vs. then-single-XGBoost baseline (bootstrap 95% CI) |
|---|---|---|
| 2-model, untuned LGB (paired delta excludes zero) | 0.5597 | [+0.0059, +0.0109] |
| 2-model, **tuned** LGB | 0.5516 | **[−0.0025, +0.0027]: includes zero, not distinguishable from single baseline** |
| 3-model, untuned LGB (paired delta excludes zero) | **0.5628** | **[+0.0081, +0.0146]** |
| 3-model, tuned LGB | 0.5587 | (not separately bootstrapped: solo comparison already shows the regression; worse than the untuned 3-model by −0.0041 test PR-AUC) |

The tuned LGB replacement removed the two-model ranking gain and reduced the three-model score; it was not adopted. 'Untuned LGB ensemble' refers to that member, not the Optuna-tuned XGBoost member.

## Diversity check: Random Forest, Extra Trees, Logistic Regression

`artifacts/diversity_check.json`: untuned additional model families on the same features/split.

| Model | val PR-AUC | test PR-AUC | val→test drop |
|---|---|---|---|
| Random Forest | 0.5006 | 0.4490 | −10.3% |
| Extra Trees | 0.4561 | 0.4032 | −11.6% |
| **Logistic Regression** | 0.4007 | **0.1721** | **−57.0%** |

Equal averaging of all six models gave test PR-AUC 0.4939 versus 0.5628 for the three-model candidate; paired delta CI [−0.0750, −0.0632]. It was rejected. This result concerns these weak additions at equal weight, not all weighted ensembles or model diversity. The unsuccessful follow-ups do not prove a feature/model ceiling or that temporal mismatch cannot be mitigated.

## Ensemble value under the cost policy

`artifacts/ensemble_rupee_value.json` applies the unchanged allow/step-up/block formulas to candidate probabilities. Monetary values are retrospective simulations; row-bootstrap intervals remain conditional on the reused month and fixed economics.

| Policy | Simulated test-month value | Lift vs. single-XGBoost | 95% CI |
|---|---|---|---|
| Single-XGBoost baseline (previously shipped) | ₹17.220cr | — | — |
| **2-model ensemble: SHIPPED** | **₹17.355cr** | **+₹13.58L** | **[+₹6.55L, +₹21.24L]: excludes zero** |
| 3-model ensemble (not shipped) | ₹17.406cr | **+₹18.65L** | **[+₹10.12L, +₹27.91L]: excludes zero** |

The two-model-minus-single delta is +₹13.58L, CI [+₹6.55L, +₹21.24L]; three-model-minus-single is +₹18.65L, CI [+₹10.12L, +₹27.91L]. Both exclude zero under the stated assumptions.

The **direct three-model-minus-two-model paired delta** is +₹5.07L, CI **[−₹0.1654L, +₹10.5502L]**. Original export bounds are **−₹16,539.95779 and +₹1,055,020.49632**. The interval includes zero; it does not establish equivalence or a guaranteed advantage for either candidate.

The final choice is the simpler **two-model ensemble (tuned XGBoost + untuned LightGBM)**. Test-month value comparisons were part of that decision, alongside deployment/explanation complexity. Its current metrics and limitations are in [eval_report.md](eval_report.md).

## Historical LLM benchmark and reproduction limits

[llm_benchmark_results.json](llm_benchmark_results.json) retains single-XGBoost AP 0.5735 versus `gpt-oss:20b` AP 0.1571 on a 200-row sample containing 10 fraud labels, with two LLM timeouts. Median latency is 6665.826ms for 198 successful calls. The 3.65× ratio is average precision, not accuracy. Finalization in [llm_benchmark.py](../scripts/llm_benchmark.py) scores XGBoost on all rows and LLM on successes only; it pairs labels in sample order with scores in result insertion order, so historical reproduction also requires aligned ordering. The export does not identify fraud count among successes.

The current script loads the current ensemble through `FraudModel`; rerunning it does not reproduce historical single-XGBoost scores. Later approximate ensemble `score()` timings (65 to 135ms across reported laptop runs) are separate from this benchmark and from full-engine p95/p99 latency. Synchronous flagged-case SHAP/narrative and persistence are included in the full engine path, with a six-second narrative timeout and possible SDK retry overhead. No 200ms end-to-end SLO or universal comparison with all LLMs is established. See [eval_report.md §7](eval_report.md#7-llm-as-classifier-benchmark--the-ai-judgment-evidence).
