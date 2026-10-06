# Evaluation report

This is the authoritative report for the current **2-model ensemble: Optuna-tuned XGBoost + untuned LightGBM**, averaged after independent Platt calibration. Current full-month metrics come from [robustness_results.json](robustness_results.json), computed by [robustness_checks.py](../scripts/robustness_checks.py) from the ensemble export `artifacts/test_month_raw.json`.

The temporal test month (day > 150) contains 92,427 transactions and 3,213 fraud labels (3.48%). It was reused for follow-up diagnostics, model comparisons and selection decisions. Some experiments selected candidates on validation before inspecting test, but the accumulated test feedback informed later work and the final ensemble choice. This is a reused holdout, not a final untouched evaluation. The record does not establish direct fitting to test labels; adaptive follow-ups can still make the reported estimate optimistic. A fresh untouched temporal holdout is needed for an independent final estimate.

All rupee values below are **retrospective simulated value under explicit cost assumptions**, not observed realized profit. Labels, amounts and policy-routing counts are observed or deterministically computed; counterfactual intervention outcomes and merchant economics are assumed. Parameter provenance and sensitivity coverage are in [experiments.md](experiments.md#cost-model-parameters-and-provenance).

## 1. Model quality

| Metric | Value | 95% bootstrap CI | vs. random baseline |
|---|---|---|---|
| PR-AUC | **0.5597** | 0.5439 – 0.5771 | random ≈ base rate (0.0348) → **16.1x** |
| ROC-AUC | **0.9126** | 0.9070 – 0.9180 | random = 0.5 (context only: see below) |

PR-AUC emphasizes precision and recall at this low fraud prevalence; ROC-AUC supplies ranking context. Historical single-XGBoost scores were 0.5514 / 0.9077. The paired ensemble-minus-single-model PR-AUC delta is +0.0083, CI [+0.0059, +0.0109], in `artifacts/ensemble_bootstrap.json`; separate interval overlap is not a significance test. The experiment ladder and calibration study are historical evidence in [experiments.md](experiments.md).

The intervals use 2,000 row resamples, seed 42. They describe conditional within-month uncertainty under row-resampling assumptions. They do not cover repeated model selection, economic parameter uncertainty, dependence between transactions, or future-month stability. Group/time-block bootstrap and fresh temporal data are follow-up validations, not completed work.

## 2. The cost curve and the chosen operating point

The current policy chooses allow, step-up or block per transaction by comparing expected values, with amount-dependent boundaries; it is not one probability cutoff.

| Policy | Simulated value, test month | Lift |
|---|---|---|
| No fraud system | ₹15.68 crore | — |
| Reference 0.5 cutoff (ensemble-fair) | ₹16.585 crore | +₹90.75 lakh vs no system |
| **Arbiter (2-model ensemble)** | **₹17.355 crore** | **+₹1.678 crore vs no system, +₹77.03 lakh vs reference 0.5** |

Policy mix: allow 88,331 (95.6%), step-up 2,782 (3.0%), block 1,314 (1.4%).

The observed-label allow/block component is **₹173.83M (₹17.383 crore)**; the modeled step-up component is **about −₹0.278M (−₹2.78 lakh)**. Both components are conditional on the assumed economics. Even allow/block value applies counterfactual decisions and assumed margin, fees and lost-customer penalties to historical rows; it is not a verified merchant cash outcome. Step-up additionally assumes 60% fraud stopping and 15% genuine abandonment, neither measured in this dataset.

The presentation's two-way allow/block curve was optimized on this test month: its minimum is p = 0.589. This is a **test-optimized presentation proxy**, not the deployed operating point or a validation-selected cutoff. The historical [cost_curve.png](cost_curve.png) shows the single-XGBoost proxy (p = 0.774).

## 2b. Robustness: conditional within-month uncertainty

Paired row bootstrap computes each policy and baseline on the same resampled rows and then differences their simulated values.

| | Point estimate | 95% CI |
|---|---|---|
| Lift vs no system | +₹1.678cr | ₹1.510cr – ₹1.850cr |
| Lift vs reference 0.5 (ensemble-fair) | +₹77.03L | ₹64.9L – ₹89.5L |

These lift intervals exclude zero conditional on the reused month and fixed economics. The direct paired ensemble-minus-single-model value delta is +₹13.58L, CI [+₹6.55L, +₹21.24L], in `artifacts/ensemble_rupee_value.json`. These results do not establish future-month stability.

The amount-only rule sweep found ₹512,535 as its best threshold: it blocks nothing and equals the no-system baseline. Context thresholds ₹10,000 and ₹50,000 have simulated values −₹68.43cr and −₹16.97cr. This establishes the result for this rule family, dataset and economics, not every possible non-ML rule.

Only margin (5%, 10%, 15%, 20%, 30%, 40%, 50%) and chargeback fee (₹200, ₹350, ₹500, ₹600, ₹1,000) were swept: all 35 cells have positive lift over both baselines, minima +₹1.163cr versus no system and +₹66.77L versus reference 0.5. LTV, MDR, fraud stopping, drop-off and FX were fixed. Sensitivity to those dimensions remains future validation.

## 3. Precision / recall at the operating threshold

The current ensemble export gives these exact counts for the three-way policy:

| Action | Fraud labels | Genuine labels | Total |
|---|---|---|---|
| Block | 1,091 | 223 | 1,314 |
| Step-up | 728 | 2,054 | 2,782 |
| Allow | 1,394 | 86,937 | 88,331 |

Block precision is **1,091 / 1,314 = 83.0%** and block fraud recall is **1,091 / 3,213 = 34.0%**. Fraud routed to either block or step-up is **1,819 / 3,213 = 56.6%**. Routing is not measured fraud prevention: the step-up outcomes remain modeled.

Secondary presentation proxy, from the ensemble PR curve (91,701 points), using the nearest point to the test-optimized p = 0.589:

| | Precision | Recall |
|---|---|---|
| **Test-optimized two-way proxy (0.589)** | **84.1%** | **35.3%** |
| At reference 0.5 (for comparison) | 81.9% | 36.6% |

The following approximate confusion counts are derived from the aggregate proxy precision/recall, not a row recount:

| | Count |
|---|---|
| Fraud caught (TP) | ~1,133 |
| Genuine wrongly flagged (FP) | ~215 |
| Total flagged at/above threshold | ~1,348 |

The historical single-model proxy at p = 0.774 gave 86.1% precision / 33.1% recall. Its cutoff and the current proxy should not be substituted for the amount-dependent three-way policy.

## 4. False-positive cost, explicitly

Exact current policy counts against observed labels:

| | Count |
|---|---|
| Blocked, correctly (real fraud) | 1,091 |
| **Blocked, wrongly (real genuine: the exact false positives)** | **223** |
| Total blocked | 1,314 |

The **calculated false-positive cost is ₹13.20 lakh (₹13,19,554)**, averaging ₹5,917 per wrongly blocked row, using `margin × amount × (1 + LTV_multiplier)`. The 223 genuine labels are exact; the cost and customer-loss interpretation are modeled under 20% margin and a 3× lost-margin LTV penalty. Chargeback fees do not enter the blocked-genuine branch. LTV was fixed, not swept.

Historical single-XGBoost: 233 genuine blocks and calculated cost ₹17.97L. The current ensemble has fewer genuine blocks and lower assumed cost on this reused month. The proxy's ~215 and the policy's 223 refer to different decision rules.

## 5. Kaggle-legal vs. causal gap: Component B

Historical V2 feature/split comparison, not current ensemble evaluation or exact winning-model reproduction:

| Model | PR-AUC | ROC-AUC |
|---|---|---|
| Honest (causal, deployable) | 0.5446 | 0.9014 |
| Leaky features only (undeployable) | 0.5697 | 0.9064 |
| Leaky features + leaky post-processing (undeployable) | 0.5512 | 0.9152 |
| **Total gap, full Kaggle-legal vs. honest** | **+0.0066** | +0.0138 |

The feature-only PR-AUC gap is **0.5697 − 0.5446 = 0.0251**; after client-mean averaging it is **0.5512 − 0.5446 = 0.0066** (~1.2% relative). Averaging reduced PR-AUC by 0.0185 while ROC-AUC increased. These two historical gaps apply to this model, feature construction and split; neither is a universal cost of causal correctness.

## 6. Reliability diagram: before and after calibration

Historical single-XGBoost calibration-method study:

| Method | PR-AUC | ROC-AUC | ECE |
|---|---|---|---|
| Raw (uncalibrated) | 0.5514 | 0.9077 | 0.0103 |
| Isotonic (tested, rejected) | 0.5384 | 0.9073 | 0.0042 |
| **Platt (chosen method)** | **0.5514** | **0.9077** | **0.0036** |

Platt preserved ranking and had lower measured ECE than isotonic in this study. Isotonic collapsed 91,271 distinct scores into 323 ties. The [reliability_diagram.png](reliability_diagram.png) and [sensitivity_map.png](sensitivity_map.png) are historical single-model plots. Each current ensemble member uses its own Platt calibrator; these historical ECE values do not measure the averaged ensemble's calibration. Calibration of the current averaged probabilities remains to be checked.

<a id="7-llm-as-classifier-benchmark--the-ai-judgment-evidence"></a>

## 7. Historical LLM classifier benchmark

Historical comparison retained in [llm_benchmark_results.json](llm_benchmark_results.json):

| Model / scope | Average precision (reported as PR-AUC) | Latency evidence |
|---|---|---|
| Single XGBoost, historical 200-row sample | **0.5735** | Historical export has no XGBoost latency field |
| `gpt-oss:20b`, Kaggle GPU, successful calls | 0.1571 | Median **6665.826ms**, 198 successful calls |

The sample contains 200 rows and 10 fraud labels; two LLM calls failed (timeouts). [llm_benchmark.py](../scripts/llm_benchmark.py)'s finalization scores XGBoost on all sampled rows and the LLM only on successful calls, excluding failures from LLM average precision and latency. Its label array follows sample order while scores follow result-dictionary insertion order, so reproducing the LLM score also requires matching those orders. The export alone does not establish the fraud count among the 198 successes. The rounded AP ratio is **3.65×**, not an accuracy ratio, and the failure-filtered populations differ. This is evidence for this model/prompt/sample, not all LLMs.

The current benchmark script constructs `FraudModel`, which loads the current ensemble. A rerun is a new ensemble comparison, not exact reproduction of the historical single-XGBoost result.

Later reported laptop timings of **about 65 to 135ms for `score()`** (often summarized as ~100ms) are separate approximate scoring measurements. They are not full `Engine.decide()` p95/p99, load-tested latency or a guaranteed 200ms SLO. [engine.py](../src/engine.py) synchronously computes SHAP and narrative for flagged cases, then writes audit/history files. [narrative.py](../src/narrative.py) sets a six-second request timeout; possible SDK retry overhead can extend elapsed time. The recorded engine latency is taken before persistence, so it does not cover the full return path. SHAP is not asynchronous. Deployment needs end-to-end measurements under representative traffic.

## 8. Historical error analysis and the val→test gap

These are historical single-XGBoost V4/V5 diagnostics, not rerun for the ensemble. Details and failed follow-ups are in [experiments.md](experiments.md); the historical development narrative is in [the build journal](../journal/build-log.md).

### 8a. Decomposing the gap: variance vs. temporal mismatch

A diagnostic model trained on 85% of the training period withheld a random 15% training-dev slice (`artifacts/training_dev_gap.json`):

| Set | n | PR-AUC | Gap from previous |
|---|---|---|---|
| train′ (fit on) | 352,361 | 0.9864 | — |
| training-dev (unseen, same period) | 62,181 | 0.8357 | **−0.151 (variance)** |
| val (unseen, next period) | 83,571 | 0.6110 | **−0.225 (mismatch)** |
| test (unseen, furthest period) | 92,427 | 0.5496 | −0.061 (further mismatch) |

The mismatch gap (~0.225) is larger than the variance gap (~0.151), but substantial variance remains. This decomposition supports investigating temporal/distribution mismatch; it does not establish unseen clients as the exclusive cause or prove the mechanism is model-independent. The diagnostic val/test scores (0.6110 / 0.5496) are near the then-baseline's 0.6128 / 0.5514.

A six-configuration regularization sweep selected the smallest training-dev→validation gap among candidates within 0.03 of the control's validation PR-AUC. The selected candidate's test score was 0.5290 versus the single baseline's 0.5514. That candidate failed; this does not prove that regularization, features or other architectures cannot improve the problem.

### 8b. Manual error analysis

Historical review: all 233 genuine blocks and 100 sampled false negatives from 1,444 fraud rows allowed without friction.

| | False positives (233) | False negatives sampled (100) |
|---|---|---|
| ProductCD = 'C' (11.7% base fraud rate: highest of 5 codes) | **81.1%** | 17% |
| ProductCD = 'W' (2.0% base fraud rate: lowest) | 10.3% | **75%** |
| card6 = 'credit' (6.7% base rate vs. debit's 2.4%) | **58.4%** | 22% |
| Dominant SHAP driver (V258/C1/C14 combined, for FP) | **80.3%** | max 17% (diffuse: C13) |
| Historical proxy context (p < 0.80 for FP; 0.10 < p < 0.774 for FN) | 28.8% below 0.80 | **0 of 100** |

Errors concentrate in different segments. None of the 100 sampled false negatives was near the old 0.774 proxy cutoff; that limits the expected benefit of a small cutoff adjustment for this sample, not every possible threshold change. Five-segment recalibration reduced C-segment false positives (127→93 validation; 189→146 test) but decreased simulated value (−0.18% validation; −0.72% test), so it was not adopted. C-only recalibration remains untested. The initial cold-start check used an absent engineered field and was retracted; see the journal.

## Honest exception list

Unresolved limitations; completed bug fixes and renumbering history are retained in [the journal](../journal/build-log.md).

1. Identity reconstruction: Component C was scoped out. Eight tested keys achieved best mixed-client share 2.11% versus the cited 0.2% target; exact UID reconstruction remains unresolved.
2. Counterfactual economics: no observed step-up response data; stopping, drop-off, fees, margin, LTV and FX remain scenario inputs. Observed labels do not make simulated monetary value realized profit.
3. Incomplete sensitivity: only margin and chargeback fee were swept. LTV, MDR, step-up stopping/drop-off and FX require follow-up validation.
4. Transfer: this evaluation uses IEEE-CIS e-commerce data with an illustrative INR scenario; an Indian merchant's actual transaction mix, economics and operating policy need their own validation.
5. Fail-closed records: deliberately unavailable-model demonstrations have no calibrated probability. Their step-up default is a rules fallback, not a fabricated score.
6. Explanation coverage and latency: SHAP/narrative cover flagged transactions only, synchronously; full-engine latency tails and the 200ms target are unverified.
7. Stretch scope: Component D dispute-evidence drafting was never started.
8. Residual error: substantial variance and temporal mismatch remain. The limited failed follow-ups do not prove a model ceiling.
9. Segment confound: five-segment recalibration does not isolate C-only recalibration. The earlier rough ₹3 to 4 lakh opportunity estimate is about 0.17 to 0.23% of ₹17.355cr; it is not a measured benefit or a significance conclusion.
10. Historical snapshots: dashboard curves/headline use current ensemble data; its 11-entry queue and 12-record audit snapshots, and the docket walkthrough, are historical single-model examples. Regeneration needs a risk-inclusive full-feature export; the available 25-row random sample all scored allow under the ensemble.
11. Independent evaluation: the reused holdout and adaptive comparisons require new temporal data. Row bootstrap does not remove selection effects or establish future stability. Current ensemble calibration also needs measurement.
12. Audit integrity: plaintext JSONL with unkeyed SHA-256 detects some edits but is forgeable by a writer who recomputes hashes. Keyed signatures, chained records and external append-only storage remain follow-ups.
13. Serving state: in-process locking and atomic history writes do not provide cross-process/host transactionality or event-time ordering; late events can see later history.
14. Reproducibility: trained artifacts require manual Kaggle export ([artifacts/README.md](../artifacts/README.md)); loader fingerprints are available, but a published checksum-backed release and golden batch/online feature vectors remain follow-ups. Historical benchmark reproduction also needs the corresponding single-model artifacts and sample/results order.
