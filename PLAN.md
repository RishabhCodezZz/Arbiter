# PLAN.md — execution plan, gates and contingencies

> **This document vs CLAUDE.md:** CLAUDE.md holds *context and rationale* — why this track, why this dataset, what the rubric says. **PLAN.md is operational.** Every step has a definition of done, a gate, a pass branch and a fail branch. Update the live sections (§0, §1, §9) as things actually happen.

---

## §0 — Status board *(live — update every session)*

| Stage | Objective | Status |
|---|---|---|
| Track + scope locked | notebook 01 written | ✅ done |
| Baseline + UID investigation | EDA + V0 baseline | ✅ **done — G2 passed strong (PR-AUC 0.5486), G1 failed and closed, C cut** |
| Causal features | V1 causal aggregates, V2 time-consistency | ✅ **done — V2 final: PR-AUC 0.5446 / ROC-AUC 0.9014. Features kept, proceeding on 441-feature set.** |
| Reduction + tuning | V3 V-column reduction, V4 Optuna | ✅ **done — V4 0.5514 (best yet), V5 Platt calibration final** |
| Cost model | V5 calibration, cost model, policy bands | ✅ **done — G6 passed, lift ₹1.54cr vs no system / ₹64.3L vs naive 0.5** |
| Engine build | Decision engine + audit trail | ✅ **done — re-export clean, 13/13 with genuinely correct data, version guard tested and fires correctly** |
| Explainability + fallbacks | SHAP + LLM narrative + fallbacks | ✅ **done — SHAP+narrative wired into engine.py, 2 more real bugs caught+fixed, end-to-end verified. LLM-timeout/garbage-response now formal named tests in `tests/test_narrative.py`.** |
| LLM benchmark + Kaggle-legal comparison | The two differentiators | ✅ **done — both evidence experiments complete.** G9: XGB 0.5735 vs `gpt-oss:20b` 0.1571. Component B: honest V2 0.5446 vs full Kaggle-legal 0.5512 — total gap only **+0.0066 PR-AUC**, smaller than expected, and the post-processing sub-step actually *hurt* PR-AUC (same ties mechanism as V5's isotonic finding). See `docs/experiments.md`. |
| Dashboard | Build the control room | ✅ **done, then superseded.** Streamlit build verified working (all 5 tabs, real data, 2 bugs caught and fixed). Later retired `dashboard.py`/Streamlit entirely, rebuilt as `dashboard.html`. See next row. |
| Docs + exception list | Report, docs, README | ✅ Core docs built and complete: README, eval report (incl. exception list), architecture doc, `.gitignore`, `reliability_diagram.png` downloaded and in place. **Plus: Streamlit dashboard retired, rebuilt as [`dashboard.html`](dashboard.html)** — one self-contained file, real captured data (headline numbers, PR/cost curves with a live client-side slider, an 11-transaction review-queue snapshot, a 12-record audit-log snapshot, sensitivity heatmap, reliability diagram), opens directly in a browser, no install, no server. Verified by rendering it directly (not just reading the code) — all panels, the interactive slider, and the review-queue expand/collapse confirmed working with correct real values; a real keyboard-accessibility bug found (review-queue rows were unreachable without a mouse) and fixed. Also built a narrative companion piece, [`docs/docket.html`](docs/docket.html) — a story page for screen-recording the pitch. **Both pages were briefly hosted via Claude's Artifact platform, then moved to real repo files** — the hosted link was an unnecessary external dependency for a submission that's meant to run standalone from a clean clone; confirmed it was never shared with anyone in the meantime. G10 deferred to an independent friend-tester. | ✅ done |
| Robustness checks | Bootstrap CI + fair rules-baseline on the real full test month | ✅ **done — `scripts/robustness_checks.py`.** Lift vs no system: 95% CI ₹1.38cr–₹1.71cr (point +₹1.54cr). Lift vs naive 0.5: ₹53.7L–₹75.5L (+₹64.3L). Both clear of zero — not a lucky month. Best rules-only threshold (₹512,535) never blocks anything in this data, collapsing to the "no system" value — fraud isn't amount-separable here, the full lift is real model signal. `tests/` (pytest, 36/36 passing) also closes the `demo_fallbacks.py` open item formally. **Extended a second time:** (1) the false-positive-cost estimate (~172, aggregate-curve proxy) replaced with an exact per-transaction count from the real 3-way policy — 233 real false positives of 1,348 total blocks, exact cost ₹17.97 lakh; (2) PR-AUC/ROC-AUC given the same 95% bootstrap CI already given to the rupee lift (0.5350–0.5688 / 0.9019–0.9132), hand-rolled and verified against the published point estimates before being trusted. Honest exception list dropped from 8 items to 7 (item 5, now resolved, removed rather than left stale). See CLAUDE.md §13, `docs/eval_report.md` §2b/§4, `journal/`. |
| Error analysis + val/test gap decomposition | Manual error analysis on real FPs/FNs (clue skill ch14) + training-dev gap decomposition (ch40/41) | ✅ **done — real results in.** Gap decomposition: variance (train′→training-dev) **−0.151 PR-AUC**, temporal mismatch (training-dev→val) **−0.225 PR-AUC — the dominant gap**, further mismatch (val→test) −0.061. Confirms the val/test drop is overwhelmingly temporal mismatch, not overfitting — a controlled measurement of the project's own "unseen clients" thesis, not just a citation of it. Error analysis (233 exact FPs, 100 sampled FNs): errors cluster exactly where base fraud rates predict — FP 81.1% in ProductCD='C' (11.7% base rate), FN 75% in ProductCD='W' (2.0% base rate); 0 of 100 false negatives were anywhere near the operating threshold, so threshold-tuning has a verified **zero** ceiling on that population. **Decision: not retraining** — see decision log below. Sensitivity-map lift-over-baselines gap (self-flagged since the cost-model stage) closed in the same pass: fully local, confirmed positive across the entire 35-point grid (min ₹1.09cr vs no system, ₹57.4L vs naive). Full numbers: `docs/eval_report.md` §8, `docs/experiments.md`, `journal/`. |
| Hyperparameter sweep — revisiting "not retraining" | Bounded, evidence-gated check: is the shipped model's low regularization an artifact of tuning against an inflated (temporally-close-to-train) validation signal? | ✅ **done — hypothesis not supported.** Pre-committed rule selected the most conservative config ("all four combined"); checked once on test it scored **0.5290 PR-AUC vs. shipped 0.5514 (−0.0224, a real regression)**. Shipped model unchanged. Reinforces, doesn't reverse, the original not-retraining decision — now backed by a direct negative test, not just reasoning. Full numbers: `docs/experiments.md`, `journal/`. |
| Segment-aware calibration — acting on the error-analysis finding | Does per-ProductCD Platt calibration (vs. one global calibrator) reduce the 233-FP cluster in ProductCD='C' (81.1% of FPs, `eval_report.md` §8b)? Calibration-only, no retraining. | ✅ **done — real partial win, net rejected.** All 5 segments got their own calibrator. ProductCD='C' false positives dropped 26.8% (val, 127→93) and 22.8% (test, 189→146) — the scoped hypothesis was correct. But total realized value dropped too (−0.18% val, −12.39L/−0.72% test) because recalibrating the other 4 segments cost more than 'C' saved. Pre-committed rule (adopt only if value doesn't drop) correctly says `keep_global`, confirmed on test. **Not adopted.** A C-only variant is a possible future test, not pursued — effect size (<1% of headline value) doesn't clear the bar for another round trip right now. Full numbers: `docs/experiments.md`. |
| Notebook 06 — consolidated, refined cost-model notebook | User asked for a cleaner version of `04_cost_model.py`: same pipeline, minus completed one-off diagnostics whose findings are already fully captured elsewhere. | ✅ **done — `notebooks/06_cost_model_refined.py` created.** Kept: core pipeline (data → causal features → V4 model → V5 calibration → cost model → 3-way policy → all still-needed exports) plus the still-open ensemble diagnostic + bootstrap. Dropped: error-analysis export, training-dev-gap decomposition, hyperparameter sweep, segment calibration (all 4 completed, all 4 fully logged in `docs/experiments.md`/`journal/`, nothing further depends on rerunning them) and the now-dead LLM-benchmark stub. 04 is untouched — stays as the full historical record. Diffed the shared sections to confirm byte-identical apart from the intended header (`diff` on lines 1-614), grepped for dangling references to removed variables (none), syntax-checked, `README.md`'s reproduction steps repointed at 06. |
| Ensemble diagnostic — LightGBM + CatBoost averaged with the shipped XGBoost | Does averaging with 2 more model families close any of the gap to the Kaggle-winning ensemble (0.9408 ROC-AUC)? User explicitly willing to ship a real 3-model engine if the number justifies it. | ✅ **confirmed real — the first "make it better" win this session.** 3-model ensemble: test PR-AUC 0.5628 vs shipped 0.5514 (+0.0114). Paired bootstrap (2,000 resamples): 95% CI **[+0.0081, +0.0146] — excludes zero.** 2-model (XGB+LGB only) also real: +0.0083, CI [+0.0059, +0.0109]. Unlike the sweep and segment calibration, this clears the bar the user set for a real ship conversation. Secondary finding: untuned LightGBM alone has both the best individual test score AND the smallest val→test degradation (−7.3% relative vs XGBoost's −10.0%) — a second, independent hint that Optuna's val-tuning may chase val's temporal-proximity bias, consistent with the hyperparameter sweep's own finding. CatBoost is the weakest individually and degrades *most* val→test, yet still adds incremental ensemble value (diversity, not mismatch-robustness). **Decision now live:** 2-model vs 3-model for an eventual ship, and the scope/timing of the engine rebuild (multi-model loading, SHAP across models, re-verifying every downstream artifact). Full numbers: `docs/experiments.md`, `journal/`. |
| LightGBM real tuning pass — following up on its untuned strength | User chose to start here first. A real 60-trial Optuna search (mirrors notebook 03's XGBoost search exactly), not a one-shot guess. | ✅ **done — confirmed regression, third independent line of evidence for the same finding.** Tuned LightGBM: test PR-AUC 0.5406 vs. untuned's 0.5541 — worse, not better, and its val→test degradation got larger (−11.6% vs −7.3%), not smaller. Paired bootstrap: 95% CI [−0.0182, −0.0090] — a statistically confirmed regression. Swapping tuned LightGBM into the ensemble erased the 2-model win entirely (CI now includes zero) and shrank the 3-model win. **Best known configuration is still the UNTUNED 3-model ensemble** (test PR-AUC 0.5628, CI [+0.0081, +0.0146]) — this experiment protected that confirmed win rather than improving on it. Recommendation: don't tune ensemble members against val PR-AUC in this problem; CatBoost especially, given it already has the largest val→test degradation of the three even untuned. The search cells themselves were removed from `notebooks/06_cost_model_refined.py` once this finding was fully captured — same "not required anymore" logic used for 04→06. Full numbers: `docs/experiments.md`, `journal/`. |
| Diversity check — Random Forest, Extra Trees, Logistic Regression added to the confirmed ensemble | User asked to try other model families in parallel rather than tune further, given tuning just failed. All three left deliberately untuned. | ✅ **done — large, unambiguous regression.** All three individually far weaker than any current ensemble member (RF 0.4490, ET 0.4032, LR **0.1721** test PR-AUC — logistic regression's val→test drop alone was 57%, by far the worst of any model tried this session). 6-model ensemble: test PR-AUC 0.4939, vs. the confirmed 3-model ensemble's 0.5628 — bootstrap 95% CI **[−0.0750, −0.0632]**, an order of magnitude larger regression than the LightGBM-tuning result and not close to zero. Mechanism: plain averaging weighs every model equally — components this much weaker simply drag the mean down; not evidence diversity is bad, evidence *unweighted* diversity with components this weak is bad. **Not adopted.** Confirmed configuration is unchanged: the untuned 3-model ensemble. Full numbers: `docs/experiments.md`, `journal/`. |
| Rupee-value check — does the ensemble's PR-AUC edge translate into real value under the actual 3-way policy? | Every ensemble comparison so far judged models on PR-AUC/ROC-AUC, not the rupee value this project's own thesis says is the real objective. Also settles 3-model vs. 2-model directly (only each vs. shipped had been bootstrapped before). | 🟡 **code written and syntax-checked** (`notebooks/06_cost_model_refined.py`, `artifacts/README.md` updated) — **awaiting a Kaggle run.** Reuses the already-computed `ens2_te`/`ens_te` arrays through the exact same `value_allow`/`value_stepup`/`value_block`/`realized_value` functions as the shipped headline number — no retraining. Runs a direct paired bootstrap of 3-model vs. 2-model rupee value, plus each vs. shipped, on the real test month. ✅ **done — ensembling itself confirmed real; 2-vs-3-model reopened as a genuinely close call.** Both ensembles produce a statistically real rupee lift over the shipped single model: 2-model +₹13.58L (CI [+₹6.55L, +₹21.24L]), 3-model +₹18.65L (CI [+₹10.12L, +₹27.91L]) — the PR-AUC win survives contact with the actual policy this time, unlike segment calibration. **3-model vs. 2-model directly: +₹5.07L point estimate, but CI [−₹0.017L, +₹10.55L] technically includes zero** — misses significance by about ₹16.5K out of a ₹1M+ range. Practically 3-model is very likely at least as good, probably a bit better, but doesn't clear the formal bar on the metric that governs every other decision here. The user's "ship 3-model" call was made on PR-AUC evidence before this ran — revisited explicitly rather than silently proceeding, since the rebuild is a large, not-cheap-to-reverse commitment. Full numbers: `docs/experiments.md`. |
| **Engine rebuild — 2-model ensemble shipped** | Final decision: 2-model (XGBoost + untuned LightGBM), simplicity as the tiebreaker (ch09 optimizing/satisficing framework — 2-vs-3 wasn't statistically separable on rupees). | ✅ **done, fully verified against the real downloaded artifact.** `src/model.py` composes both calibrated models (plain average, matching what was validated); `src/explain.py` averages SHAP across both boosters; `src/audit.py`'s `raw_probability` now `{"xgboost":..., "lightgbm":...}`; `src/engine.py`/`MODEL_VERSION` updated. `verify_and_replay()` needed zero changes (policy-layer-only replay, model-agnostic by design — checked before writing anything). LightGBM pin corrected from a caught, self-flagged unverified guess (`4.5.0`) to the real Kaggle-confirmed version (`4.6.0`). **`scripts/demo_engine.py`: all checks pass, including a new one confirming both ensemble members' raw scores are visible in the audit record. `pytest`: 36/36.** Doc sweep (CLAUDE.md, `docs/eval_report.md`, `README.md`) done via clearly-labeled callouts pointing to the confirmed ensemble numbers, rather than overwriting the single-model's own extensively-bootstrapped granular numbers in place — that per-transaction regeneration (`test_month_raw.json`, `dashboard.html`, `scripts/robustness_checks.py`) is explicitly tracked as exception-list item 10, not done. Full checklist: `REBUILD_CHECKLIST.md`. |
| Pitch video | Record and cut | ⬜ *(shelved — user is directing ML work first, will resume video planning on request)* |
| Submit | Final submission | ⬜ |

**Legend:** ⬜ not started · 🟡 in progress · ✅ done · 🔴 blocked · ⚠️ degraded (shipped, but below plan)

---

## §1 — What is currently going wrong *(live)*

### Open blockers

*(none currently — the one open blocker tracked here (B3) is resolved; see the Resolved table below. Caught stale during a full documentation cross-check: this table and the "known unknowns" list below it hadn't been touched since early in the project, well after every item in both had actually been closed out.)*

### Known unknowns (not blockers, but unresolved)

*(none currently — all three items previously tracked here are resolved: notebook 01 runs clean (V0 baseline, UID investigation — see §4), the cost curve has a confirmed interior minimum at p=0.774 (G6, §3), and the LLM-as-classifier benchmark completed (G9, §3/§4). Left visibly empty rather than deleted, so it's clear these were tracked and closed, not simply forgotten about.)*

### Resolved

| # | Blocker | Resolution |
|---|---|---|
| B1 | Kaggle competition rules not accepted | Accepted; also hit a Kaggle notebook-UI path change (`/kaggle/input/competitions/<slug>/` not `/kaggle/input/<slug>/`) — fixed in `01_eda_baseline.py`. Then a full code cell (Step 0 load) was missing from the Kaggle copy — re-pasted, ran clean. |
| B4 | G1 UID reconstruction — mixed% was 3.36% vs 0.2% target | Investigated across 8 key configurations (D1-null check, 5 static-column variants, documented D-column consistency refinement). Best: 2.11% mixed, ~11x target, plateaued. **Formally cut component C** per the pre-committed ~1% stop threshold. UID key survives for component B; `uid_confident` feature survives independently. Full log: `docs/experiments.md`. |
| B5 | V1/V2 (causal features) underperformed V0 (-0.0050 PR-AUC) | Diagnosed and resolved: original time-consistency screen (an early slice vs. a mature one) was cold-start-biased, wrongly dropping 70 features on an unfair comparison. Re-screened on two mature slices: flips dropped from 70 to 1. Corrected V2 = PR-AUC 0.5446 / ROC-AUC 0.9014 (final). Features proven correct + temporally stable; small net cost vs V0 (-0.0040 PR-AUC) accepted — kept for component B, proceeding to the reduction/tuning stage on the 441-feature set. |
| B7 | **XGBoost version mismatch — most severe bug found so far.** Local xgboost (3.0.0, then 3.4.1) didn't match Kaggle's training version (3.2.0). Verified directly: identical input, raw probability 0.169488 (v3.0.0) vs 0.007395 (v3.2.0 and v3.4.1 agree) — a 23x difference, silent, no error. | **Fixed two ways:** `requirements.txt` hard-pins `xgboost==3.2.0`; `src/model.py` now records the training version in the manifest and raises `XGBoostVersionMismatchError` (fails closed, same as a missing model) if the running version doesn't match. **Guard independently tested**: deliberately faked a version mismatch in a manifest copy, confirmed it blocks loading; confirmed the explicit `allow_version_mismatch=True` override also works. Full story: `journal/`. |
| B6/B8 | Sample-transaction export corrupted every categorical feature to -1 | `te`'s categorical columns get overwritten with integer codes early in the notebook (for training); the export pulled from `te` after that mutation. Fixed by overwriting with true values from `narrow_raw` (read fresh, pre-mutation) matched by `TransactionID` — verified the merge logic with a standalone test before trusting it. Test-data bug only, not a `src/` bug. **Re-exported and confirmed clean**: real values now (`ProductCD='W'`, `card4='visa'`, `card6='debit'`). |
| B9 | `policy.py`'s `degraded=True` confidence penalty could not work, for any parameter value | Blended `value_allow`/`value_block` toward `value_stepup` by a constant factor. Proven algebraically AND empirically that this can never move the decision boundary: if `a(p)==s(p)` at the crossover, `a(p)*(1-k)+s(p)*k` also equals `s(p)` there, for any `k`. Not a tuning bug — the mechanism was incapable. Fixed by shrinking the probability toward 0.5 (max uncertainty) before the value formulas see it — verified directly: confident-allow ceiling narrowed p<0.0393→p<0.0150, confident-block floor rose p>0.6885→p>0.6982. |
| B10 | `verify_and_replay()` never passed `degraded` through | A decision made under the widened step-up band would replay against the NORMAL boundaries, producing a false tamper alarm on a legitimate decision. Fixed: `degraded` now stored in every audit record and threaded through replay. Found while wiring `engine.py` to finally set `degraded=True` when `build_degraded()` is used (it never had before). |
| B3 | No LLM provider key | Ollama Cloud's free tier proved unreliable under sustained sequential calls, even with real exponential backoff (repeated "too many concurrent requests" — see journal). Pivoted to running the LLM benchmark entirely on Kaggle's own GPU (`gpt-oss:20b`, local `ollama serve` inside the notebook, no external API, no rate limit). `artifacts/llm_benchmark_kaggle_results.json` downloaded and combined with local XGBoost scoring via `scripts/llm_benchmark.py --kaggle-results`, verified against a synthetic fake results file before trusting the real one. **G9 passed** on the result (see §3/§4). |

---

## §2 — Priorities and slack

A fixed share of total effort is protected as slack — anything that eats into it gets logged in §1 rather than silently absorbed.

| Phase | Covers | Notes |
|---|---|---|
| Modelling (V0→V5) | Baseline through calibration | The ML ladder. Highest uncertainty. |
| Productisation | Engine, audit, SHAP, LLM, fallbacks | |
| Evidence experiments | LLM benchmark + Kaggle-legal comparison | The two differentiators |
| Presentation | Dashboard, docs, video | |
| Slack | — | Protect this |

**Rule:** if any single stage overruns significantly, invoke the cut order (§7) rather than borrowing from slack twice.

---

## §3 — The gates

Points where the project can fork. Each is binary and checkable.

| Gate | Passes if | Fail cost |
|---|---|---|
| **G1 — UID reconstruction** | ~73,838 clients (±2k), pure-1 ≈ 2.9% (±1pp) | ✅ resolved (FAIL) — component C dropped, see §4 |
| **G2 — Baseline sane** | Test PR-AUC > 0.30 | Bug hunt |
| **G3 — Causal features help** | V1 PR-AUC > V0 | Story changes, not fatal |
| **G4 — Optuna in budget** | Completes inside the time-box | ✅ passed — 60/60 trials, PR-AUC 0.5514 |
| **G5 — Calibration improves** | ECE decreases, reliability diagram straightens | ✅ passed — Platt shipped, ECE 0.0036, ranking untouched |
| **G6 — Cost curve has interior min** | Optimum is not at t=0 or t=1 | ✅ passed — minimum at p=0.774, robust to the FX correction |
| **G7 — Engine runs clean** | `decide()` works from a saved artifact, no notebook | ✅ fully passed — 13/13 on genuinely correct re-exported data; version-mismatch guard independently tested and confirmed to fire |
| **G8 — Fallbacks proven** | Each failure path tested by actually breaking it | Non-negotiable, cannot ship without |
| **G9 — LLM benchmark** | Any clear result | ✅ **PASSED** — XGBoost PR-AUC 0.5735 vs `gpt-oss:20b` PR-AUC 0.1571 on the same 200-row sample (~11.5x random vs ~3.1x random). LLM latency median 6.7s vs XGBoost's microseconds. Clean, expected-direction result — the "where we chose not to use one" evidence. |
| **G10 — Clean clone** | Fresh dir + README only → it runs | Non-negotiable, fix immediately |
| **G11 — Video ≤5 min** | Under time, audio audible | Re-cut |

---

## §4 — Build stages

---

### Baseline + UID investigation

**Objective:** verify the claims we're building on, get an honest floor number.

**Steps**
1. Accept competition rules, attach data, enable GPU (T4×2 or P100)
2. Run `notebooks/01_eda_baseline.py`
3. Record every printed number into `docs/experiments.md`
4. Journal anything that broke

**Definition of done:** V0 row in the experiment log is filled, verification table in `docs/experiments.md` has our numbers next to the claimed ones.

#### 🚦 G1 — UID reconstruction — ✅ **RESOLVED (FAIL branch)**

**Pass:** ~73,838 clients with 2+ transactions (±2k); pure-fraud-0 ≈ 96.9%, pure-fraud-1 ≈ 2.9% (±1pp).

**Actual:** base key gave 92,628 clients, mixed 3.36% (16.8x target). Escalated through 8 configurations — D1-null check (ruled out, only 0.21% nulls), 5 static-column variants, the documented `D`-column consistency refinement (`+D4n`, `+D10n`, `+D15n`). Best: **2.11% mixed**, still ~11x target, plateaued — only 22.8% of coarse buckets are even internally `D15n`-consistent, so the ambiguity is structural, not one-column-away. Root cause: the writeup's real method used a separate, undocumented "UID detection script" as its own methodology, not a simple concatenated key.

**Decision: component C (card-level risk state, time-to-detection) is cut**, per the pre-committed stop threshold (~1%). CLAUDE.md §4 updated. UID key (`card1+addr1+D1n+D4n`) survives for component B, which doesn't depend on exact purity. `uid_confident` (std(D15n)==0 in bucket) survives as a cheap feature into the causal-feature stage. Full log: `docs/experiments.md`. This is real journal material — logged.

#### 🚦 G2 — Baseline sane

**Pass:** test PR-AUC > 0.30 (vs random ≈ 0.035).

- **→ PASS:** proceed to the causal-feature stage.
- **→ FAIL:** this is almost certainly a **bug**, not a modelling failure. Run the diagnostic:
  - Rerun with a **random** split instead of temporal.
  - **Random ≫ temporal** → code is fine; temporal generalisation is genuinely hard. Expected. Proceed.
  - **Both bad** → real bug. Check in order: categorical encoding (did `.map()` produce all-NaN?), target accidentally in features, split producing empty classes.

---

### Causal features

**Objective:** the honest version of the winning solution's "magic".

**Steps**
1. **Causal aggregates.** For each transaction, aggregate over that client's **prior rows only** — expanding window. Count, mean/std of amount, time since previous transaction, distinct devices/emails so far, cumulative amount.
   - *Implementation note:* `groupby('uid').expanding()` or a sorted `groupby().shift()`+`cumsum` pattern. **Verify no row sees its own value** — write an assertion for this, don't eyeball it.
2. Handle the **cold-start half** — ~50% of rows are single-transaction clients with no history. Explicit null-history indicator rather than silent NaN.
3. **Time-consistency screening.** Per feature: train a single-feature model on an early slice, predict a late slice. Drop anything with validation AUC < 0.5 — its relationship inverts over time.
4. Record V1 and V2 rows.

**Definition of done:** two measured deltas, leakage assertion passing.

#### 🚦 G3 — Causal features help — ✅ **RESOLVED (FAIL branch, feature-kept decision)**

**Pass:** V1 > V0. **Actual:** V1 PR-AUC 0.5436 vs V0's 0.5486 (-0.0050) — did not pass.

Correctness independently verified (300-row manual recomputation, 0 mismatches) — not a bug. Diagnosed: the initial suspicion (70 features flagged by time-consistency screening, 7/9 causal ones among them) traced to a **cold-start bias in the screen itself** (an early, immature slice compared against a mature one). Re-screened on two mature slices: flips collapsed from 70 to 1. Corrected V2 = **PR-AUC 0.5446, ROC-AUC 0.9014 (final)**.

**Decision:** features are correct and temporally stable; net cost vs V0 is small (-0.0040 PR-AUC) and may move under later tuning. **Kept.** Proceeding to the reduction/tuning stage on the 441-feature V2 set. Full log: `docs/experiments.md`, `journal/`.

**Reframe (also fixed in CLAUDE.md):** component B is causal-vs-full-history aggregation of the *same* features, not features-vs-no-features — untouched by this gate either way. The Kaggle-legal comparison builds the full-history version of V1's features and compares directly against V1.

---

### Reduction + tuning

**Objective:** get the feature count manageable, then tune.

**Steps**
1. Group the 339 V-columns by identical NaN pattern. Per group: PCA, or max uncorrelated subset, or group mean. Pick per group by validation score.
2. **Optuna, stage 1 only.** Objective = PR-AUC on temporal validation. TPE sampler + MedianPruner. Search `max_depth`, `learning_rate`, `subsample`, `colsample_bytree`, `min_child_weight`, `reg_lambda`, `reg_alpha`.
3. Record V3, V4.

#### 🚦 G4 — Optuna in budget — ✅ **PASSED**

**Hard time-box.** Actual: 60/60 trials completed comfortably inside budget. Best val PR-AUC 0.6128; final refit on full train scored **PR-AUC 0.5514, ROC-AUC 0.9077 on test** — best of the project, +0.0045 over V3. Best params: `max_depth=10, learning_rate=0.049, subsample=0.82, colsample_bytree=0.89, min_child_weight=2, reg_lambda=0.78, reg_alpha=0.023`.

#### 🚦 G5 — Calibration improves — ✅ **PASSED (Platt, not isotonic)**

- **→ PASS:** ECE down, reliability diagram closer to diagonal. Ship it.
- **→ FAIL:** isotonic overfits on small calibration sets. Fall back to Platt/sigmoid.

**Actual:** isotonic's ECE improved (0.0103→0.0042) but cost **-0.0130 PR-AUC** — contradicted the "monotonic ⇒ ~unchanged" assumption. Root cause confirmed directly: isotonic collapsed 91,271 distinct test scores to 323 (ties). Fallback triggered as planned: tested Platt, which **matched raw PR-AUC/ROC-AUC exactly** (0.5514/0.9077, zero cost) and **beat isotonic's ECE too** (0.0036 vs 0.0042) — strictly better on every axis, no trade-off needed. **Shipped Platt.** Full log: `docs/experiments.md`, `journal/`.

---

### Cost model

**Objective:** the actual thesis. This is the most important stage.

**Steps**
1. **Cost model** — see §5 for the full spec.
2. **Threshold sweep** → cost curve → minimum → the two boundaries for allow / step-up / block.
3. **Sensitivity analysis** across cost-parameter ranges.
4. ~~Time-to-detection~~ — **cut with component C**, see status board / §1.

#### 🚦 G6 — Cost curve has an interior minimum — ✅ **PASSED**

**This is the risk nobody thinks about.** If the optimum lands at threshold 0 (block everything) or 1 (allow everything), the demo has no story.

- **→ PASS:** the minimum is interior. Report it, with the rupees saved versus both extremes.
- **→ FAIL:** do **not** fudge the parameters to force a nicer curve. Instead **pivot to the sensitivity map** — a 2-D plot of margin × chargeback-fee showing which policy is optimal in each region, with the merchant's actual position marked. This is arguably a *better* deliverable: it shows the answer depends on economics, not on the model, and that is a genuinely sophisticated point.

**Actual:** minimum at p=0.774, well inside [0,1]. Confirmed robust to a real bug-hunt: caught a stale-placeholder FX rate (USD→INR=83, uncorroborated) before the FIRST run even mattered, then corrected it to a live dated quote (95.41) after the first run and re-verified the gate still passes — same result, slightly stronger. Headline: Arbiter values the test month at ₹17.22cr vs ₹15.68cr (no system) and ₹16.58cr (naive 0.5 threshold). **Lift: +₹1.54cr vs no system, +₹64.3L vs naive.** Of Arbiter's total, 99.95% is hard-computed from actual labels (allow/block), only 0.05% depends on modeled step-up assumptions — the headline claim is overwhelmingly evidence-based. Full numbers: `docs/experiments.md`.

**Closed.** Extended via `scripts/robustness_checks.py` (Part 3) rather than left open: the *lift over both baselines* — not just Arbiter's own value — is now confirmed positive across the **entire** 35-point margin×fee grid, not just the assumed 20%/₹500 point. Minimum lift vs no system across the whole grid: ₹1.09cr (still comfortably positive at the least favorable corner). Minimum lift vs naive 0.5: ₹57.4L. Full grid: `docs/robustness_results.json`.

---

### Engine build

**Objective:** stop being a notebook. This is where "would you trust it" is earned.

**Steps — all built:**
1. ✅ `src/` package: `store.py` (online client-history — real-time analog of notebook 02's batch expanding-window aggregates), `features.py`, `model.py`, `policy.py` (identical cost model to notebook 04), `audit.py`, `engine.py`
2. ✅ `decide(transaction) -> Decision` — loads a saved artifact, no training, **CPU only** (`model.py` never requests a GPU device)
3. ✅ Append-only audit log (`src/audit.py`), one record per decision, fields per CLAUDE.md §8
4. ✅ **Idempotency:** same transaction id twice → same decision returned, not re-decided — proven in the smoke test
5. ✅ **Replay test:** `verify_and_replay()` re-derives the decision from a stored record AND detects a tampered one — proven in the smoke test

**New addition, not in the original plan:** an export addendum at the end of `notebooks/04_cost_model.py` — Kaggle holds the only trained model, so it has to be the source of the portable artifact. Downloads to `artifacts/`.

#### 🚦 G7 — Engine runs clean — ✅ **fully passed** (see §3's gate table — 13/13, real artifact, version guard tested)

**Pass:** `decide()` works from a fresh Python process with only the saved artifact.

- **→ FAIL:** slip to the next stage, and pre-emptively cut the dashboard to CLI. The engine matters more than the dashboard — it is what "does it run, is it structured, would you trust it" is actually asking about.

**Actual:** built a synthetic smoke test (tiny fake model trained locally, no GPU, no real data) specifically to catch plumbing bugs before the real Kaggle artifact export — found and fixed 2 real bugs this way: (1) the test's own synthetic repeat-client pair didn't share a UID because random `D1` values lack the real dataset's day-lockstep structure — fixed in the test, not `src/`; (2) `AuditLog.lookup()` didn't coerce `transaction_id` to `str`, so calling it with a raw int (the natural way, e.g. straight from a pandas row) silently returned "not found" instead of the real record — a false negative on the one property idempotency exists to guarantee. **Fixed in `src/audit.py` itself**, protecting every future caller, not just the one that tripped over it. Ran the actual `scripts/demo_engine.py` file (not a copy) against the fixed code: **13/13 checks passed** on synthetic data.

**Then closed for real.** Real artifacts downloaded (381 features, 31 categorical vocabularies, 25 real held-out transactions). Ran `scripts/demo_engine.py` against the actual trained model: **13/13 checks passed again**, this time for real — model loads standalone (CPU, no GPU, no training), causal history genuinely accumulates across two separate calls using a real repeat client found in the test month, idempotency returns byte-identical decisions, replay + tamper detection both work, fail-closed degrades to a logged `step-up` default when the model artifact is missing.

**A 3rd bug, in the export itself, caught the same way as the others — by running it, not reading it:** raw integer columns (`TransactionID`, `card1`, etc.) exported as `3485114.0` instead of `3485114` — `.iterrows()` upcasts every value in a row to one common dtype. Didn't break the demo, but worth fixing. First fix attempt (`.to_dict()` then re-wrap in `pd.Series()`) was ALSO wrong — re-introduces the identical upcast — caught by testing that specific fix directly before trusting it, not by assuming the line changed meant the bug was gone. Second version verified correct with a standalone dtype check.

**Then two SEVERE bugs (B7, B8 above), found while moving dependencies into a proper venv.** A 23x XGBoost-version probability discrepancy, and a corrupted-categoricals export bug underneath it — both fixed, both verified with standalone tests, full story in `journal/`.

**G7 fully re-closed.** Re-exported from Kaggle with both fixes applied, re-downloaded, re-ran `scripts/demo_engine.py`: **13/13 passed**, with the first genuinely correct probability the engine has produced (`p=0.0170`, using real category data instead of `-1` placeholders). Independently tested the version-mismatch guard itself (not just the pin) — deliberately faked a mismatched manifest, confirmed `XGBoostVersionMismatchError` fires, confirmed the explicit override works too. Nothing outstanding from this investigation.

---

### Explainability + fallbacks

**Objective:** explanation layer, and prove the system degrades gracefully.

**Steps**
1. `TreeExplainer` on **flagged transactions only** — not the full stream
2. SHAP top-k → LLM → one-paragraph analyst narrative
3. Deterministic template renderer covering the same content
4. **Break things on purpose and test each path:**

| Break | Expected behaviour |
|---|---|
| LLM key invalid | decision unchanged, template narrative, `fallbacks_triggered` logged |
| LLM times out | same, within a bounded timeout |
| LLM returns garbage | validate the response; reject and template |
| Model artifact deleted | fail **closed** to rules baseline, alert, do not silently allow |
| Required feature missing | impute, flag degraded confidence, widen step-up band |
| Duplicate transaction id | return original decision from audit log |

#### 🚦 G8 — Fallbacks proven

**Non-negotiable.** The rubric names failure recovery as one of four lines, and the form asks about it directly. **We cannot ship without this working and tested.** If this stage overruns, cut from presentation polish, not from here.

**The invariant to demonstrate on video:** kill the LLM entirely → every decision is byte-identical, only the prose changes. That is the proof the LLM never decides.

---

### LLM benchmark + Kaggle-legal comparison

**Objective:** the differentiators.

**Steps**
1. **LLM-as-classifier benchmark.** Sample ~500 transactions. Serialise features to text, ask an LLM to score fraud probability. Compare against XGBoost on PR-AUC, latency per transaction, and cost per 1,000 decisions.
2. **Kaggle-legal model.** Full-history `groupby('uid').agg()` + prediction post-processing (client-mean). Score it on the same test month.
3. Compute and write up the gap.

#### 🚦 G9 — LLM benchmark

**Both outcomes are usable, which is why this is safe to run:**

- **→ LLM loses (expected):** exactly the "where we chose not to use one" evidence. Report accuracy, latency and cost.
- **→ LLM is competitive (surprise):** **report it honestly.** *"I assumed X, tested it, and was partly wrong"* is a genuinely strong panel answer — arguably stronger than being right. Then argue deployment on latency and cost grounds instead, which will still favour GBDT by orders of magnitude at 590k transactions.

**On the Kaggle-legal model:** it is *supposed* to score higher. That is the finding. Do not tune it to lose.

**Actual:** G9 passed — XGBoost PR-AUC 0.5735 vs `gpt-oss:20b` PR-AUC 0.1571 on the same 200-row sample, 3.65x more accurate and ~6 orders of magnitude faster per call, on a real, fairly-chosen model (not a strawman). LLM lost, as expected, so this is the "where you chose not to use one" evidence stated in the plan above, not the surprise branch. The Kaggle-legal model did score higher, as predicted: honest causal 0.5446 PR-AUC vs full Kaggle-legal 0.5512 — gap only +0.0066 PR-AUC (~1.2%), smaller than expected going in. Full numbers: `docs/experiments.md`, `journal/`.

---

### Dashboard

**Streamlit.** PR curve · **cost curve with a live threshold slider showing rupees move** · review queue with SHAP narratives · audit log viewer · the sensitivity map if G6 failed.

The threshold slider is the money shot for the video — it makes the entire thesis visible in five seconds.

**If behind:** CLI + saved matplotlib plots. Acceptable. The dashboard is presentation, not substance.

**Actual:** built as planned in Streamlit, verified live in a real browser (all 5 tabs, real data, 2 real bugs found and fixed by actually clicking through it). Later superseded, not abandoned: retired the Streamlit app entirely and rebuilt as a single self-contained `dashboard.html` — same real data, no local install needed to view it, which matters more for a judge opening this cold than a slightly nicer local dev experience. See §0 and `journal/` for the full retirement story.

---

### Docs + exception list

**Steps**
1. Eval report: every metric from CLAUDE.md §9
2. **Honest exception list** — what the system could not resolve. Cold-start clients. Cases where SHAP gives no clear driver. Anything the fallbacks caught. *The rubric rewards this; do not soften it.*
3. README: problem, approach, results, **attribution to the 1st place solution**, setup
4. Architecture doc + diagram

#### 🚦 G10 — Clean clone test

Fresh directory. Clone. Follow the README **exactly as written**, no memory of what you meant. Does it run?

**Non-negotiable.** *"Does it run"* is rubric line two. A judge who hits an error in setup never sees the rest.

---

### Pitch video

Script and shot structure kept in local notes, not published — see `private/` (gitignored).

---

### Submit

**Submit in the morning, not at the deadline.** Forms break, uploads fail, videos need re-encoding.

See §8.

---

## §5 — Cost model specification

The heart of the project. These are **assumptions, not facts** — every one gets documented with its source, and the panel will ask.

### Outcomes

| Action | Actually fraud | Actually legitimate |
|---|---|---|
| **Allow** | `−(amount + chargeback_fee)` | `+margin × amount` |
| **Step-up** | `−(1−p_challenge_stops_fraud) × (amount + fee)` | `−p_dropoff × margin × amount` |
| **Block** | `0` | `−(margin × amount + ltv_penalty)` |

### Parameters — fill these in and cite them

**Filled in, `notebooks/04_cost_model.py`:**

| Parameter | Symbol | Value | Justification |
|---|---|---|---|
| Chargeback fee | `fee` | ₹500 | Razorpay's own disclosed dispute-fee range is ₹200–600; midpoint |
| **Payment processing fee (MDR)** | `mdr` | **2.36% of amount** | **Razorpay's own pricing page**: 2% platform fee + 18% GST, uniform across all domestic methods. Not refunded on a later chargeback — a real, separate loss added after the initial cost model, found by checking Razorpay's actual site rather than only generic industry numbers. Fixed/disclosed, not swept in sensitivity (unlike margin/fee, it isn't a business assumption). |
| Merchant margin | `margin` | 20% | Blended e-commerce assumption (5–50%+ by vertical) — swept in sensitivity |
| Step-up drop-off | `p_dropoff` | 15% | Checkout-friction studies cite 17–21% for full checkout complexity; a single OTP is less friction, set conservatively lower |
| Step-up stops fraud | `p_stop` | 60% | 3D Secure fraud-reduction studies cite 40–70%; midpoint |
| False-decline LTV penalty | `ltv_penalty` | 3× lost margin | Most speculative — flagged explicitly, swept in sensitivity |
| Manual review cost | `review` | not modeled | Not part of the 3-way allow/step-up/block decision; relevant later for the analyst review queue, not the cost model itself |

### Required deliverable: sensitivity analysis

**Do not report a single optimal threshold as if the parameters were certain.** Sweep `margin` and `fee` across plausible ranges, show how the optimal policy moves, and mark the assumed operating point.

This does two things: it pre-empts *"where did your ₹500 chargeback fee come from?"*, and it is the fallback deliverable if G6 fails.

---

## §6 — Risk register

| # | Risk | Likelihood | Impact | Mitigation |
|---|---|---|---|---|
| R1 | UID reconstruction fails | ~~Low~~ **materialised** | High | ✅ G1 branch executed; C dropped, shipping A+B |
| R2 | Out of time | **High** | High | Cut order §7; protected slack; fixed gates |
| R3 | Cost curve has no interior minimum | Medium | Medium | G6 → sensitivity map pivot |
| R4 | Kaggle GPU quota exhausted | Medium | Low | CPU `hist`, fewer trials, Colab |
| R5 | LLM benchmark contradicts our claim | Medium | Low | Report honestly; pivot to latency/cost argument |
| R6 | Causal features add ~nothing | ~~Medium~~ **materialised** | Medium | ✅ Confirmed small net cost (-0.0040 PR-AUC vs V0); kept anyway for component B. Reported as finding, not hidden. |
| R7 | Clean-clone setup fails for a judge | Medium | **High** | G10 is non-negotiable; test on a truly fresh dir |
| R8 | Video runs over 5 min | Medium | Medium | Script it; record early if possible |
| R9 | Calibration doesn't work | Low | Medium | Platt fallback; ship uncalibrated + disclose |
| R10 | Scope creep into component D | Medium | Medium | D is stretch. Do not start it before docs/exception-list stage is done. |
| R11 | Solo builder gets sick / loses time | Low | **Critical** | Slack + cut order. Submit whatever exists — partial with an honest README beats nothing. |

---

## §7 — Cut order

**Decided early, because late-stage triage is bad triage.**

1. **D — dispute drafting** (stretch; cut without hesitation)
2. **Dashboard polish** → CLI + saved plots
3. **LLM narrative sophistication** → templates; the *design* is the point
4. ~~C — time-to-detection~~ *(already cut, early — see §4)*
5. **B — Kaggle-legal comparison** ← protect this, it is the differentiator

**Never cut:** model ladder · cost curve · three-way policy · audit trail · one working failure path · README · video.

---

## §8 — Submission checklist

**The build**
- [x] Repo built, structured, and pushed — https://github.com/RishabhCodezZz/arbiter *(currently set to **private** during active development, by deliberate choice — flip back to public before the final submission checkbox below, since the form needs a public URL)*
- [ ] Clean-clone test passed (G10) *(deferred to an independent friend-tester)*
- [x] README: problem, approach, results, **attribution**, setup — `README.md`
- [x] Architecture doc + diagram — `docs/architecture.md`
- [x] Eval report with every metric from CLAUDE.md §9 — `docs/eval_report.md`
- [x] **Honest exception list** — 9 items, in `docs/eval_report.md` (dropped to 7 when the false-positive-cost estimate was closed with an exact number, up to 8 with the error-analysis pass's model-variance finding, up to 9 with the segment-calibration confound found this session; see §0's status board)
- [x] Audit log demonstrable and replayable — `scripts/demo_engine.py` + live in `dashboard.html`
- [x] Fallback paths tested and documented — table in `docs/architecture.md`, each one broken-and-verified in `journal/`
- [x] Notebooks committed, cleaned — all 5, pushed
- [x] No secrets, no keys, no data files in the repo — `.gitignore` added and verified

**The form**
- [ ] Track: **02 — AI Risk Manager**
- [ ] Project name
- [ ] What it solves
- [ ] GitHub URL — verified public in an incognito window
- [ ] Pitch video — unlisted link, verified playable while signed out
- [ ] **"What broke, and how you got out"** ← drafted from `journal/`, written last, read first
- [ ] Standard personal/admin fields (name, college, resume, availability, etc. — not tracked here)
- [ ] Final confirmation checkbox — **irreversible, re-read everything first**

---

## §9 — Decision log *(append-only)*

| Decision | Why |
|---|---|
| Track 02, not 03 | 03 better in the abstract; 02 better for an ML-first solo builder, and least crowded |
| Cost-optimal decision system, not a classifier | Differentiation; matches the track bar's "false-positive cost" wording |
| IEEE-CIS over ULB | ULB's PCA features are unexplainable → no SHAP narrative, no LLM justification |
| Killed idea F (drift monitor) | Deotte: it's unseen clients, not time drift. Premise was wrong. |
| Committed A+B+C, D stretch | Composes into one coherent product |
| No `scale_pos_weight` | Distorts probabilities; fights calibration; cost model needs real probabilities |
| Two-stage optimisation | One model, cost as a knob. Fee changes → re-sweep, no retrain. |
| V0 baseline confirmed strong: PR-AUC 0.5486, ROC-AUC 0.9002 on untouched test month | 15.8x random, zero feature engineering — solid floor for the ladder |
| **Component C cut** — UID reconstruction plateaued at 2.11% mixed vs 0.2% target across 8 tested keys | Pre-committed ~1% stop threshold hit; real method needs an undocumented matching script, out of scope. B unaffected; `uid_confident` feature salvaged. |
| Causal features (V1/V2) kept despite a small net cost (-0.0040 PR-AUC vs V0) | Proven correct + temporally stable; a bad screen (cold-start bias) initially made them look worse than they are, corrected before accepting the number; required for component B regardless of standalone effect |
| Shipped Platt scaling, not isotonic, for calibration (V5) | Isotonic's flexibility introduced ties (91,271→323 distinct scores) costing -0.0130 PR-AUC; Platt matched raw ranking exactly AND had better ECE — strictly dominant, no trade-off needed |
| Added Razorpay's real 2.36% MDR to the cost model; corrected placeholder FX rate (83→95.41) | Both were "check it, don't guess it" corrections — MDR closes a real gap the chargeback-fee number alone missed (permanently lost on a reversed sale); FX was an unverified placeholder swapped for a live dated quote once available. Neither was required to get a working model, both were checked because the whole project's credibility rests on sourced numbers. |
| LLM provider: Ollama Cloud (free tier), `nemotron-3-ultra` for the benchmark specifically | Compared real Intelligence-Index benchmark data across the free models available rather than guessing — Ultra scored highest (47.7) and, unusually, wasn't slow despite being the largest. Deliberately picked the *strongest* free option, not the cheapest, because the benchmark's whole credibility depends on giving the LLM a genuinely fair shot — a rigged comparison against a weak model wouldn't be an honest "where we chose not to use AI" finding. |
| 500-row LLM-benchmark sample export added to `notebooks/04_cost_model.py`, reusing the existing (already-bug-fixed) export logic rather than a new script | The only local sample was 25 rows (from the engine-build stage), which carries under 1 expected fraud case at the dataset's ~3.5% base rate — not enough to compute a PR-AUC anyone should trust. Plain random 500-row sample, deliberately not stratified to hit an exact fraud count — stratifying the benchmark's own test set would be the same dishonesty as tuning the leaky Component B model to lose. |
| Extended `scripts/robustness_checks.py` to compute the false-positive breakdown exactly, and to bootstrap PR-AUC/ROC-AUC | The track card names "false-positive cost" verbatim as part of "the bar" — the prior ~172 was already an honest, stated estimate (so not a hard rubric gap), but the raw data (`artifacts/test_month_raw.json`) and the tested code to compute it exactly (`realized_value()`) already existed, unused for this purpose. PR-AUC/ROC-AUC previously had no CI while the rupee lift did — inconsistent rigor across the same report. Both hand-rolled (not sklearn) to match `requirements.txt`'s stated invariant, and both verified to reproduce the published point estimates before being trusted. Honest exception list item 5 (the estimate) is now resolved and removed, not left stale — 8 items → 7. |
| Found real model variance (−0.151 PR-AUC, train′→training-dev) — chose NOT to retrain/retune the shipped model | The dominant gap is temporal mismatch (−0.225), 1.5x larger — per the diagnostic framework's own rule, work should follow the largest gap, not just any measurable one. Mismatch here is a property of the problem (new clients over time), not fixable by regularization; the project's existing responses (causal features, temporal split, bootstrap CI) are already the right mitigation. The 60-trial Optuna search already searched the regularization space against a real validation objective — re-opening tuning this late cascades into re-verifying every downstream artifact (cost model, dashboard, docket, audit log, 36 tests) for an uncertain payoff against the *smaller* of two measured gaps, with only the video and G10 left as real gates. Reported as honest, quantified evidence instead — a new honest-exception-list item — rather than acted on reflexively. |
| **Revisited the above at explicit request** — ran the bounded 6-config hyperparameter sweep, tested once on test | **Hypothesis not supported, reversed in fact.** Pre-committed rule (smallest traindev→val gap, val within 0.03 of control) selected "all four combined" — the most conservative config — over the shipped one. Checked once on test: PR-AUC **0.5290 vs. shipped 0.5514 (−0.0224, a real regression)**. The more conservative config generalizes *worse*, not better. **Decision unchanged: shipped model stays.** This strengthens rather than reverses the original decision — reasoning alone before, a direct negative empirical test now. Secondary finding: the gap metric itself was a weak proxy here — the winner's smaller gap came from both traindev and val dropping together (a uniformly weaker model), not from resisting the transition better; the 0.03 eligibility band wasn't tight enough to catch it before the test check did. Full numbers: `docs/experiments.md`, `journal/`. |
