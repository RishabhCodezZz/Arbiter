# CLAUDE.md — Razorpay AI Buildathon 2026

> Living document. Updated every time a decision is made, reversed, or a fact is learned. v0.6.
>
> **Companion documents:**
> - **[docs/experiments.md](docs/experiments.md)** — the measured ladder, V0 → V5.
> - **`journal/`** — failure log, written as things happened. Feeds the form field they read first.
>
> This file is *context and rationale*: why this track, why this data, what the rubric wants.

---

## 1. The situation

**Razorpay AI Buildathon 2026** — a student-only hiring funnel, not a prize hackathon.

| | |
|---|---|
| **Builder** | Solo build. |
| **Process** | No resume screen, no aptitude test, no GD. Shortlist goes straight to a panel that interrogates the build. |

The application **is** the submission — the form asks for the repo, the video and the writeup. There is no separate later build deadline.

### What the form asks

Besides standard personal/admin fields (not tracked here), the build side asks: track · project name · what it solves · **public GitHub repo URL** · **5-min pitch video** (unlisted ok) · **"what broke, and how you got out"**.

> Razorpay's own note on the form: *"The last one is the one we read first."*

---

## 2. The rubric (verbatim from the site)

| Line | What they say |
|---|---|
| **Problem taste** | did you pick something that actually matters |
| **Build quality** | does it run, is it structured, would you trust it |
| **AI judgment** | the right tool in the right place, **and where you chose not to use one** |
| **Failure recovery** | what broke, and what you did about it |

### Non-negotiables

1. **It must actually run.** A recorded mock is a fail.
2. **Measured outcomes on a batch.** Never one cherry-picked success.
3. **An audit trail** for every consequential decision.
4. **At least one failure handled gracefully**, by design, not by patch.
5. **Honest exception list** — publish what the system could *not* resolve.
6. **A defensible "where we chose not to use AI."** Most submissions will have no answer here.

---

## 3. Track: **02 — AI Risk Manager**

> *"Build a working detector, verifier or auto-responder for one class of loss, with measured precision and recall on a held-out test set."*
> *The bar: "Honest metrics including false-positive cost. Strictly defense-only: anything offense-capable is disqualified."*
> *Example directions listed on the card: chargeback evidence responder, return-risk scorer, fraud-spike detector, abuse-ring sentinel.*

**On the four example directions:** Arbiter is deliberately none of them individually — it's the decision layer underneath all four, the piece that turns any of their outputs into a priced allow/step-up/block call. The closest overlap is the evidence responder, which is Component D, cut early as a stretch goal (see §4) so the decision engine, audit trail, and causal-honesty evidence stayed the priority.

**Track scores for the record:** 03 Revenue Recovery 9.0 · **02 Risk Manager 8.5** · 01 Agentic Commerce 8.0 (highest relevance, worst crowding) · 04 Finance Controller 7.5 (safest, but AI risks reading as bolt-on) · 05 Open 6.0.

02 chosen because it is the right track *for this builder*: ML is the only self-declared strength, the builder wants the "defend your model" panel conversation, has genuine interest in fraud patterns, and Razorpay signals 02 is under-subscribed — *"surfaces the risk and ML minded builders the others miss."*

---

## 4. What we are building

**One sentence:** a fraud decision system a merchant could actually deploy, evaluated in rupees.

**Scope: A + B, with D as a stretch goal. C cut early — see below.**

| Component | What it is | Rating | Status |
|---|---|---|---|
| **A — Cost-optimal decision engine** | Calibrated probability → cost curve → allow / step-up / block | 9/10 | active |
| **B — Causal-vs-Kaggle leakage study** | Build both models, quantify the gap | 10/10 as a component | active |
| **C — Card-level risk state** | Score cards not transactions; report time-to-detection | 8.5/10 | **cut early — see §5** |
| **D — Dispute evidence drafting** *(stretch)* | Razorpay Disputes API, test mode | 8/10 as a bolt-on | stretch |

**Why C was cut:** required near-perfect client identity resolution (target: 96.9/2.9/0.2 pure-0/pure-1/mixed). Eight key configurations tested, best achieved was 2.11% mixed — ~11x the target, plateaued. Full investigation in §5 and `docs/experiments.md`. A scoping decision made from evidence within a pre-committed time budget, not an unexamined failure — good "what broke, how we got out" material.

**Rejected — and why (worth keeping, it's journal material):**
- **E — Risk policy simulator** (7/10): great demo, thin on ML depth. We chose 02 to show ML.
- **F — Adversarial drift monitor** (5/10): built on a premise the data contradicts. Deotte: *"this competition actually isn't about time… the clients in the dataset change radically over time."* Killed by evidence before any code was written.

---

## 5. The data: IEEE-CIS Fraud Detection

~590k rows, ~394 features + identity table, ~3.5% fraud. 1.35 GB.

**Why a public dataset at all:** the track bar demands "measured precision and recall on a held-out test set", which requires trustworthy labels. Self-generated synthetic fraud means writing both the exam and the answer key — a risk panel spots it instantly and it invalidates every downstream number.

**Why IEEE-CIS over ULB `creditcard.csv`:** ULB's features are anonymised PCA components (`V1`–`V28`) — unexplainable, so no SHAP narrative and no justification for the LLM layer. IEEE-CIS has real semantics (card, email domain, device, addr), so explanations are true statements about the world.

**Stated limitation (say it before they ask):** IEEE-CIS is US card-not-present e-commerce fraud. Razorpay's real mix is UPI-heavy and Indian. The *method* transfers; the specific features would change.

### The label is card-level, not transaction-level — this drives everything

Competition host, verbatim: *"define reported **chargeback on the card** as fraud transaction (isFraud=1) and transactions **posterior to it** with either user account, email address or billing address directly linked… as fraud too."*

Consequences:
- **`isFraud` is literally a reported chargeback.** Our cost model measures the exact thing the labels encode. This is a genuine alignment, not a stretch.
- Of 73,838 clients with 2+ transactions: **96.9% all-fraud-0, 2.9% all-fraud-1, 0.2% mixed.** → predicting **compromised cards**, not anomalous transactions — this was the premise for component C.
- **We verified this ourselves in EDA, and could not reproduce it.** Best key (`card1+addr1+D1n+D4n`) plateaued at **2.11% mixed**, ~11x the target, across 8 tested configurations including the documented `D`-column consistency refinement. Only 22.8% of coarse buckets are even internally consistent on `D15n` — the ambiguity is structural. The writeup references a separate "UID detection script" as its own methodology, and even the 1st-place team didn't feed its raw output directly into their models. **Component C cut as a result — see §4.** Full log in `docs/experiments.md`.
- The UID key survives for **component B** (§ below) — B compares the same imperfect key used two ways (full-history vs. causal), so it doesn't depend on hitting exact purity. A cheap derived feature, `uid_ambiguity_std_prior` (causal version of `uid_confident`), was built alongside the causal feature set — see below.

**Important framing, fixed early:** component B is "causal aggregation vs. full-history aggregation of the *same* feature set" — **not** "features vs. no features." The V0→V1 comparison (does adding history features help at all) is a different, narrower question from B's (does computing them causally vs. leakily change the score). B is unaffected by whichever way V0-vs-V1 goes; the Kaggle-legal comparison builds the full-history version of V1's own features and compares directly against V1.

### Causality: the winning solution is not deployable

The 1st place approach post-processed by *"taking all predictions from a single client and replacing them with that client's average prediction"*, and built features via `groupby('uid').agg('mean')` over the full dataset. Both use **future transactions to score a past one**. Legal on Kaggle; impossible in a gateway with a 200ms decision window.

| Technique | Causal? | Verdict |
|---|---|---|
| UID from `card1 + addr1 + D1n` | ✅ `D1` = days since card began, known at txn time | **Keep** |
| `groupby('uid').agg('mean')` over full data | ❌ includes future rows | **Reimplement as expanding/backward-only** |
| Post-process: replace prediction with client mean | ❌ pure future leakage | **Drop, and say why** |
| V-column reduction by NaN-group + PCA | ✅ time-independent | **Keep** |
| "Time consistency" feature screening | ✅ train early → test late | **Keep — excellent idea** |

**Component B = build both, report both, explain the delta.**

**Built and measured (`notebooks/05_kaggle_legal_leaky.py`).** Honest causal V2: PR-AUC 0.5446. Leaky features alone: 0.5697 (+0.0251). Leaky features + the client-mean post-processing trick on top: 0.5512 — **total gap only +0.0066 PR-AUC (~1.2% relative)**. Two findings, not one: (1) causal honesty costs almost nothing here — a stronger, more reassuring result than a large gap would have been; (2) the post-processing step actually *hurt* PR-AUC relative to leaky-features-alone, while helping ROC-AUC — same tie-collapse mechanism as the isotonic-calibration finding in §6 (averaging every one of a client's predictions into one shared value destroys within-client ranking, which PR-AUC penalizes and ROC-AUC tolerates). Full table: `docs/experiments.md`.

### Attribution — what we take from the 1st place solution

**We borrow heavily and the README will say so explicitly.** Citing them is not a weakness; pretending we derived this independently would be dishonest and trivially caught.

**Taken:** UID construction (`card1 + addr1 + D1n`) · label semantics (card-level chargeback) · group aggregates rather than raw UID as a feature · time-consistency screening · V-column reduction by NaN structure · multiple validation splits · the "unseen clients, not time drift" diagnosis.

**Diverged, deliberately:**

| | They did | We do | Why |
|---|---|---|---|
| Causality | full-history aggregates | expanding window, past only | 200ms, no future |
| Post-processing | replace prediction with client mean | don't | pure leakage |
| Tuning metric | ROC-AUC | PR-AUC | imbalance flatters ROC |
| Calibration | none needed | Platt scaling + reliability diagram (isotonic tested, rejected — see §5) | AUC is rank-only; rupees need real probabilities |
| Objective | maximise AUC | minimise expected rupee loss | no merchant has an AUC line in their P&L |
| Output | a score | allow / step-up / block | a score is not a decision |
| Model | CAT+LGB+XGB stack + NN | 2-model ensemble (XGBoost + LightGBM, both untuned, simple-averaged) | still simpler than a 4-way stack — CatBoost and further tuning were both tried and empirically rejected (see below); a stacking meta-learner and a 4th neural-net member were never attempted |
| Explainability | not required | SHAP → narrative | competitions don't need it, products do |
| Extra metric | — | time-to-detection | nobody reports this |

### On "is ours better" — no, and don't claim it

Their model beats ours on the Kaggle task. Two Grandmasters, a three-model ensemble, months of iteration; we will not touch 0.9408 as a solo build and chasing it is the wrong goal.

**Different game.** They optimised a leaderboard where future data exists and a score is the deliverable. We optimise a merchant's P&L under latency and causality constraints where a *decision* is the deliverable.

*"I beat the Kaggle winners"* is not believable. *"I understood why the winning solution can't ship, and I measured what honesty costs"* is believable, verifiable, and a **judgment** claim rather than a skill claim. Judgment is what's being hired.

### Splitting — a practical landmine

`test_transaction.csv` **has no labels** (hence `sample_submission.csv`). We cannot evaluate on it. The held-out test set must be carved out of `train_transaction.csv` temporally:

```
Day 0 ──────── 120 ──── 150 ──── 183
   TRAIN      │  VAL   │  TEST
              └ calibration + threshold selection
```

Final month is touched **once**, at the end, for the reported numbers.

---

## 6. Architecture

### The core reframe

Precision/recall/F1 are not the objective. **Expected rupee value** is — computed per transaction, not from one global threshold (see below for why that distinction matters).

> **Update — the shipped model is now a 2-model ensemble.** Everything in this section
> describes the single-XGBoost model's own cost-model run, still true as a description of
> that stage and kept as-is rather than rewritten. The engine now ships XGBoost + untuned
> LightGBM, simple-averaged — a confirmed, bootstrapped real improvement, and the full
> granular breakdown (not just the aggregate) is now regenerated and verified for it, not
> just tracked as pending:
>
> | | Single-model (below, historical) | 2-model ensemble (shipped) |
> |---|---|---|
> | Total test-month value | ₹17.22cr | **₹17.355cr** |
> | Lift vs no system | +₹1.54cr | **+₹1.678cr**, CI [+₹1.510cr, +₹1.850cr] |
> | Lift vs naive 0.5 | +₹64.3L | **+₹77.03L**, CI [+₹64.9L, +₹89.5L] |
> | PR-AUC / ROC-AUC | 0.5514 / 0.9077 | **0.5597 / 0.9126** |
> | Exact false positives | 233, ₹17.97L | **223, ₹13.20L** — lower on both counts |
> | Hard-verified share | 99.95% | **100.16%** (modeled step-up component is slightly negative) |
>
> Full detail, every number cross-verified two independent ways (Kaggle export vs. a fresh
> local `scripts/robustness_checks.py` run): `docs/eval_report.md` §1–§4,
> `docs/experiments.md`'s ensemble sections. `dashboard.html`'s curves and headline are now
> the ensemble's real data too; only its review-queue / audit-log *snapshots* remain
> single-model — exception-list item 10, `docs/eval_report.md`.

**Built and run, `notebooks/04_cost_model.py`. G6 gate PASSED. The numbers immediately below are the single-XGBoost cost-model run** — the shipped headline is the 2-model ensemble (see the callout above and `docs/eval_report.md` §1–§4); the reasoning, cost mechanism and parameter sourcing in the rest of this section are model-independent and unchanged.

**Result (single-XGBoost run), on the untouched test month (92,427 transactions), with Razorpay's real MDR and a real dated FX rate:**

| Policy | Total value (single XGBoost) | Total value (shipped 2-model ensemble) |
|---|---|---|
| No fraud system | ₹15.68 crore | ₹15.68 crore |
| Naive 0.5 threshold | ₹16.58 crore | ₹16.585 crore |
| **Arbiter** | ₹17.22 crore | **₹17.355 crore** |

**Lift (single-XGBoost run): +₹1.54 crore vs no system, +₹64.3 lakh vs the industry-default naive threshold** (shipped ensemble: **+₹1.678 crore / +₹77.03 lakh**). Policy mix (single model): 95.8% allow, 2.7% step-up, 1.5% block. Of that run's total value, **99.95% comes from allow/block outcomes computed from actual fraud labels** — hard, verifiable — only 0.05% depends on the modeled step-up population rates (for the shipped ensemble the hard-verified share is 100.16%, the modeled step-up component slightly negative). The headline claim is overwhelmingly evidence-based, not assumption-dependent. Full numbers, sensitivity table: `docs/experiments.md`.

**False-positive cost, exactly — not estimated.** For the single-XGBoost run: of the 1,348 blocks, 233 are real genuine customers wrongly blocked (1,115 are correctly-blocked real fraud) — an exact per-transaction count from `artifacts/test_month_raw.json`, not the aggregate-curve estimate (~172) reported earlier in the project. Exact cost: **₹17.97 lakh** (₹7,712 average per wrongly-blocked customer). **For the shipped 2-model ensemble the exact count is lower on both axes: 223 wrongly-blocked of 1,314 total blocks, ₹13.20 lakh** (`docs/eval_report.md` §4). Both PR-AUC and ROC-AUC also carry a 95% bootstrap CI — single model 0.5514 (0.5350–0.5688) / 0.9077 (0.9019–0.9132); shipped ensemble **0.5597 (0.5439–0.5771) / 0.9126 (0.9070–0.9180)** — the same rigor already given to the rupee lift, extended to the model-quality metrics themselves. Full numbers: `docs/robustness_results.json`, `docs/eval_report.md` §1/§4.

**A second "check it, don't guess it" catch, in the same pass as the MDR one:** first run used an unverified placeholder FX rate (₹83/$1). Corrected to a live dated quote (₹95.41/$1) once available, and verified the correction moved the policy in the theoretically-predicted direction (fixed ₹500 fee became relatively smaller → slightly more lenient) before accepting the new numbers — not just re-run and hoped for the best.

**Parameters — sourced, not invented** (full table + citations below):

| Parameter | Value | Source |
|---|---|---|
| Chargeback fee | ₹500 | Razorpay's own disclosed dispute-fee range ₹200–600, midpoint |
| **Payment processing fee (MDR)** | **2.36%** | **Razorpay's own pricing page** — 2% platform fee + 18% GST, uniform across all domestic methods, not refunded on a later chargeback. Found by checking the actual host platform's site, added after the initial model — a real, separate loss the chargeback-fee number alone doesn't capture. |
| Merchant margin | 20% | Blended e-commerce assumption — swept in sensitivity |
| Step-up stops fraud | 60% | 3D Secure studies cite 40–70%, midpoint |
| Genuine customer drop-off at step-up | 15% | Checkout-friction studies cite 17–21% for full checkout; a single OTP is less friction, set lower |
| LTV penalty, wrongly blocked customer | 3× lost margin | Most speculative — flagged explicitly, swept in sensitivity |

**A bug caught before it ran, not after:** IEEE-CIS is a US dataset — `TransactionAmt` is dollar-scale. Feeding it directly against an Indian ₹500 chargeback fee would make the fee absurdly large relative to typical transaction size and likely produce a degenerate policy (block everything) — plausibly the exact mechanism the pre-written G6 gate ("does the cost curve have an interior minimum") was worried about. Fixed with an illustrative ₹83/$1 conversion before any code ran, stated as a modeling simplification, not hidden.

**Why per-transaction, not a single global threshold:** every cost term scales with `amount`, so the cost-optimal action genuinely depends on transaction size, not just fraud probability. The notebook computes the expected value of **all three actions for every transaction** and picks the best one — the correct decision given the cost model, not an approximation of it. A simpler single-threshold (allow/block only) sweep is also built, purely as the intuitive cost-curve visual for the pitch video, and doubles as the G6 gate check (does the minimum land inside the range, not at an edge — checked explicitly, not assumed).

**MDR applies only where a transaction actually processes:** every allow, and a step-up that completes (genuine finishes, or fraud gets through anyway) — never a block, never a step-up abandonment, because nothing was processed.

**Honesty split, built in on purpose:** allow/block outcomes are computed from **actual test-month labels** — hard, verifiable numbers. Step-up outcomes cannot be computed that way — there's no real step-up interaction data in this dataset (whether a specific fraudster was actually stopped, whether a specific genuine customer actually abandoned) — so those use the cited **population rates** as a modeled expectation. The notebook reports these two pieces separately rather than blending them into one falsely-precise number. This distinction belongs in the honest exception list.

This produces **three actions, not two**: **allow / step-up auth / block**. The uncertain middle band gets friction instead of a decline — converts traffic you'd otherwise destroy, and shifts liability.

**Required, not optional:** a sensitivity map across margin × chargeback fee (both swept, MDR held fixed since it's disclosed fact, not assumption). Never report one number as if the parameters were certain — pre-empts "where did your ₹500 come from."

### Where the LLM goes — and where it deliberately does not

**Not for scoring.** 590k rows of tabular features is gradient-boosting territory: faster, cheaper, more accurate. **Proven, not just claimed — `scripts/llm_benchmark.py`.** On a 200-row held-out sample: XGBoost PR-AUC 0.5735 vs `gpt-oss:20b` (a real, capable model given a fair shot, not a strawman) PR-AUC 0.1571 — **3.65x more accurate**, and ~6 orders of magnitude faster per call (microseconds vs a 6.7s median LLM latency, running on Kaggle's own GPU after the free Ollama Cloud tier proved unreliable under sustained load — see `journal/`). That benchmark is the answer to *"where you chose not to use one."*

**Yes for language:**
1. SHAP contributions → analyst-readable case narrative
2. Dispute evidence drafting *(stretch)*
3. Natural-language querying of the review queue

**Design invariant:** *the LLM never decides, it only renders.* Delete it entirely and every decision is byte-identical; only the prose degrades to a deterministic template. This is the graceful-failure story **and** the AI-judgment story, and it is a property of the design rather than a patch.

### Why SHAP specifically

1. **It is what makes the LLM safe.** Without it the model would have to invent reasons — confident, plausible, fictional. With it, the LLM is doing pure translation from structured attributions it was handed.
2. **The review queue needs "why" in three seconds.** `feature_importances_` is global; SHAP is per-prediction.
3. **It's a leakage detector in development** — one feature dominating every prediction means something is wrong.

*Engineering call:* compute SHAP **only for flagged transactions**, not the full stream. Nobody reads 570,000 explanations.

### Two-stage optimisation — deliberately separated

| Stage | What varies | Objective |
|---|---|---|
| **1 — Model** (Optuna, TPE + pruning) | hyperparameters | **PR-AUC** on temporal validation |
| **2 — Policy** (threshold sweep) | thresholds | **minimum expected rupee loss** |

**Why separate:** cost parameters (margin, chargeback fee, review cost) differ per merchant and change over time. Baking one cost assumption into the model gives a model that works for one merchant on one day. Separated → **one model, cost as a knob**; a fee change re-sweeps in seconds with no retraining.

**Why PR-AUC not ROC-AUC in stage 1:** at 3.5% positives ROC-AUC is flattered by the huge negative class. Report both; tune on PR-AUC; explain the divergence from Kaggle's ROC-AUC scoring.

### Calibration — essential, usually skipped

XGBoost output is **not a probability**. But `expected_loss = P(fraud) × (amount + fee)` requires a real one, or every rupee figure is wrong and the thesis collapses.

**Resolved — Platt, not isotonic.** "Monotonic ⇒ ranking metrics unchanged" is imprecise: monotonic only forbids reversing order, not tying it, and isotonic is a step function — it collapsed 91,271 distinct test scores to 323, costing -0.0130 PR-AUC. Platt (smooth, strictly increasing, cannot tie) matched raw PR-AUC/ROC-AUC exactly *and* beat isotonic's ECE (0.0036 vs 0.0042) — strictly dominant, no trade-off. **Numbers computed on Kaggle; `docs/reliability_diagram.png` is downloaded and committed.** (Caught once already, mid-project: the plot was claimed "shipped" here before it had actually been downloaded — checked the file exists before trusting that claim, found it didn't, fixed by re-running the plotting code already complete and correct in `notebooks/03_reduce_tune_calibrate.py` and downloading for real. See `journal/`.)

---

## 7. Where the work happens

```
Kaggle notebook  →  research: EDA, features, Optuna, training   [GPU]
       ↓  export artifacts (model, calibrator, feature spec, thresholds)
GitHub repo      →  product: pipeline, decision engine, audit log,
                    API, dashboard, tests                        [CPU-only]
notebooks/       →  the research record, cleaned, committed
```

**Kaggle for ML:** data pre-mounted (no 1.35 GB download), free GPU (~30 hrs/week, sessions ~9–12h), ~30 GB RAM. GPU matters mainly for the Optuna sweep — CPU `hist` is minutes per fold, which × 6 folds × 50 trials adds up fast.

**Repo for the product:** a notebook cannot demonstrate graceful degradation, an audit trail, or "would you trust it". The cloned repo must run **without a GPU** — it loads a trained artifact and serves decisions.

*"Research in notebooks, product in modules"* is itself a build-quality signal.

Use `device="cuda"` + `tree_method="hist"` (not the deprecated `gpu_hist`).

**Built exactly as diagrammed** — `src/{store,features,model,policy,audit,engine}.py`, CPU-only. Verified four times, each round surfacing something real: a synthetic smoke test before the real artifact existed (2 bugs), the real trained model (mechanics: 13/13), a proper `requirements.txt`-pinned venv (2 severe bugs — a 23x XGBoost-version probability gap, and a corrupted-categorical export underneath it), then a clean re-export closing both: **13/13 on genuinely correct data**, plus the version guard itself independently tested. (Engine later rebuilt for the 2-model ensemble — now `ALL CHECKS PASSED` / 98 pytest, both version guards tested; see §13.) Fully proven, nothing outstanding.

---

## 8. Audit record — what every decision writes

Model version · feature vector hash · raw score · **calibrated probability** · threshold in force · cost parameters in force · action (allow/step-up/block) · SHAP top-k contributors · LLM prompt + response id (nullable) · latency · **fallbacks triggered** · timestamp.

Must be **replayable**: given the record, re-derive the decision.

---

## 9. Metrics we will report

### Why PR-AUC over ROC-AUC (and why we still report both)

Both are **threshold-free** summaries: sweep every threshold, plot a curve, take the area. Neither tells you where to actually cut — that is what the cost curve is for. These judge the *model*; the cost curve judges the *decision*.

```
ROC:  TPR = TP/(TP+FN)   vs   FPR = FP/(FP+TN)
PR:   Precision = TP/(TP+FP)  vs  Recall = TP/(TP+FN)
```

**ROC-AUC is flattered by imbalance.** Its `FPR` denominator contains `TN`, which is enormous (~570k legit vs ~20k fraud). Worked example — 100k txns, 3.5k fraud, model flags 10k and catches 2.8k:

- Recall 80%, **FPR 7.5%** → *"we catch 80% of fraud at a 7.5% false positive rate"* — sounds excellent
- **Precision 28%** → *"72% of the customers we block are innocent"* — sounds alarming

Same model. The second is what the merchant experiences. **Precision contains no `TN`** — no cushion — so PR-AUC reports it bluntly.

**Baselines differ and this trips everyone up:** a random model scores ROC-AUC **0.5** always, but PR-AUC **= the base rate (~0.035)**. So PR-AUC 0.50 is ~14× random here, not "a coin flip". The two numbers are not on the same scale.

**We report both** because Kaggle scored this competition on ROC-AUC (winners: 0.9408 private LB) so it gives context, and because reporting only the flattering metric is exactly what the rubric penalises.

### The list

- PR-AUC and ROC-AUC on the untouched final month
- **Cost curve** and the chosen operating point, in rupees
- Precision/recall **at the operating threshold** (not at 0.5)
- **False-positive cost** explicitly — the bar names it
- ~~Time-to-detection (component C)~~ — cut, see §4
- **Kaggle-legal vs causal AUC gap** (component B)
- **Reliability diagram**, before and after calibration
- **Honest exception list** — what the system could not resolve

---

## 10. Questions the panel will ask, and our answers

**Q: Why this dataset? Razorpay didn't specify one.**
The bar requires measured precision/recall on a held-out test set → requires trustworthy labels. Three options: real merchant data (impossible, PII), self-generated synthetic (writing the exam *and* the answer key — invalidates every number), or a real public labelled set. Only the third survives. Among those, IEEE-CIS over ULB because ULB's PCA features are unexplainable and would make the SHAP/LLM layer impossible.

**Q: Isn't this just a Kaggle notebook?**
No — the Kaggle objective is ROC-AUC on a static file with future information available. Ours is minimum rupee loss under a causal constraint. We built both and measured the gap.

**Q: Why not 0.5 as the threshold?**
Because nobody chose 0.5; it's a default. There isn't even a single threshold — every cost term scales with transaction amount, so the cost-optimal action genuinely depends on amount, not just probability. We compute the expected value of allow/step-up/block per transaction and pick the best one; the simple single-threshold view we do show (for the intuitive cost-curve visual) comes from the minimum of that curve, and moves when the merchant's economics move.

**Q: How did you price the outcomes — where did ₹500 and 2.36% come from?**
₹500 is Razorpay's own disclosed chargeback dispute-fee range (₹200–600), midpoint. 2.36% is Razorpay's own disclosed processing fee (2% + 18% GST) — found by checking their actual pricing page, not assumed, and modeled correctly as non-refundable on a later chargeback. Margin and the LTV penalty are the genuinely uncertain ones; both are swept in a sensitivity analysis rather than reported as if certain.

**Q: Why isn't the LLM doing the scoring?**
Benchmarked it; it loses to GBDT on accuracy, latency and cost. Evidence is in the repo. The LLM renders explanations, it never decides.

**Q: How do you know your probabilities are real?**
Calibrated with Platt scaling — tested isotonic first, rejected it because its extra flexibility introduced ties (91,271 distinct scores collapsed to 323) that cost real ranking accuracy for no net calibration benefit. Platt matched raw ranking exactly and calibrated better. Reliability diagram before/after is in the report.

**Q: What breaks in production?**
See §11 and `journal/`.

**Q: Your model shows real variance (a 15-point PR-AUC gap between training and a held-out training-dev slice) — why didn't you fix it?**
Because it isn't the dominant problem, and when we actually tested a fix it made things worse. A training-dev decomposition put a number on both candidate explanations for the model's known val/test gap: variance is 0.151 PR-AUC, temporal mismatch is 0.225 — 1.5x larger. The largest gap should drive the response, and mismatch here is a property of the problem (new clients appearing over time), not something more regularization fixes. We didn't just assume that — we ran a bounded 6-config hyperparameter sweep (shipped config as control plus 4 more-conservative variants and a combined one), selected the smallest traindev→val gap on val only, then checked once on test: the selected (most conservative) config scored 0.5290 PR-AUC vs. the then-shipped single model's 0.5514 — a real regression, not an improvement. Regularizing harder generalizes worse here, not better. Model unchanged (and the later ensemble rebuild kept that same base XGBoost).

**Q: You said "unseen clients, not time drift" — how do you know that's actually what's happening here, not just an assumption borrowed from the Kaggle writeup?**
We measured it, not just cited it. A held-out training-dev slice (same period as training, never trained on) already shows most of the degradation the model will ever show from generalizing within one time window — 0.151 PR-AUC. Moving from that slice into the next chronological period (val) costs another 0.225 — 50% more than the entire within-period variance. That's a controlled comparison, not an assumption: if it were really about the model not fitting training data well, that would show up already in the training-dev number, not specifically at the boundary into a new time period.

**Q: Didn't you say you'd predict compromised cards, not transactions?**
That was the plan (component C). We tried to reproduce the 1st place team's client-identity numbers ourselves, couldn't get within 10x of their purity across 8 tested key configurations, traced it to their reliance on an undocumented dedicated matching script, and cut the component rather than ship a metric (time-to-detection) built on identity resolution we couldn't trust. Full log is in the repo. We kept the transaction-level system, which doesn't depend on this.

**Q: You said latency/deployability was why you shipped one model — why does it now ship two?**
Because we tested it, the same way we test everything else here. LightGBM, trained untuned on the exact same features, turned out to be genuinely better at resisting the temporal-mismatch degradation that dominates this problem — smallest val→test drop of any model tried. Averaging it with the shipped XGBoost produced a statistically confirmed real rupee lift (bootstrap 95% CI [+₹6.55L, +₹21.24L] on the test month), not just a better-looking metric — we specifically checked it survives contact with the real cost policy, because a related earlier experiment (segment calibration) proved a metric win doesn't always mean a value win. We also tried a 3rd model (CatBoost) and further tuning of LightGBM itself — both tested, both not adopted (CatBoost's edge over 2 models wasn't statistically distinguishable from zero; tuning LightGBM made it *worse*, confirming the same val-mismatch risk a third time on a second model family). Two models, not one, and not three — each a tested decision, not a default.

**Q: This is US card data — Razorpay is UPI-heavy.**
Correct, and stated up front. The method is rail-agnostic; the features would change.

---

## 11. Failure paths (designed, not patched)

| Failure | Behaviour |
|---|---|
| LLM API timeout / error / rate limit | Decision unaffected; narrative falls back to deterministic template; `fallbacks_triggered` logged |
| Model artifact missing / corrupt | Fail closed to a conservative rules baseline; alert; do not silently allow |
| Feature unavailable at scoring time | Impute + flag degraded confidence; widen the step-up band rather than guess |
| SHAP too slow inline | Async — decision returns immediately, explanation backfills into the queue |
| Duplicate/replayed transaction id | Idempotency key; return the original decision, do not re-decide |

---

## 11b. Contingencies — decided while calm

Every one of these is **material for the field they read first.** A build with no failures was either unambitious or is being reported dishonestly. The plan is not to avoid them; it is to have a response ready and log it properly.

| If this breaks | What it means | Response |
|---|---|---|
| **UID doesn't reproduce 96.9/2.9/0.2** | identity reconstruction is wrong | Debug `D1` semantics; try adding `P_emaildomain` to the key. If unresolved after a bounded budget, **drop component C**, ship A+B. Still complete. |
| OOM on load | predictable, low severity | column subsets, drop V-columns before merge, chunk it |
| **V0 PR-AUC terrible (<0.2)** | almost certainly a *bug*, not a modelling failure | Diagnostic: rerun with a **random** split. Random ≫ temporal → code is fine, temporal is just hard (expected). Both bad → real bug; check categorical encoding first. |
| Causal features (V1) don't help | **a finding, not a failure** | Report the delta honestly even if +0.01. Gap analysis still works. |
| Optuna too slow | time risk | Hard time-box. Subsample for the search, refit on full. Aggressive pruning. |
| ~~Calibration makes it worse~~ | ✅ **materialised exactly as planned** — isotonic cost -0.0130 PR-AUC via tie-collapse | Fell back to Platt as planned; it won on every axis (ranking untouched, ECE 0.0036 < isotonic's 0.0042). Both reported in `docs/experiments.md`. |
| No LLM API access | threatens the AI-judgment story | Design already survives it — the LLM never decides. Ship template narratives; the LLM-as-classifier benchmark needs only a few hundred calls. |
| GPU quota exhausted | slower, not fatal | CPU `hist`; cut trials; Colab backup. |

### Cut order — fixed early, because late-stage triage is bad triage

1. **D — dispute drafting** (stretch; cut without hesitation)
2. **Dashboard polish** → degrade to CLI + saved plots
3. **LLM narrative sophistication** → templates are fine, the *design* is the point
4. ~~C — time-to-detection~~ *(already cut — see §4)*
5. **B — Kaggle-legal comparison** ← protect this, it is the differentiator

**Never cut:** the model ladder · the cost curve · the three-way policy · the audit trail · one working failure path · the README · the video.

---

## 12. The engineering journal is a first-class deliverable

Because *"the last one is the one we read first."*

`journal/` holds entries written as things happened, in order. Every genuine failure gets written down **when it happens** — never reconstructed at the end. Reconstructed failure stories read as fiction, and this panel reads that field first.

Each entry: what broke · how it surfaced · the first (wrong) hypothesis · what it actually was · what changed as a result.

Watch for: temporal split collapsing the score vs. the random-split number · calibration wrecked by imbalance · SHAP too slow inline · LLM latency/timeouts · memory blowing up on 590k × 394 · UID reconstruction not reproducing the 96.9/2.9/0.2 split.

---

## 13. How the build unfolded

The measured ladder ran V0 → V5: a plain XGBoost baseline (PR-AUC 0.5486, ~15.8x random),
causal client-history features (small honest cost, kept for component B), V-column reduction
(+0.0023), a 60-trial Optuna search (V4, PR-AUC 0.5514), and Platt calibration (isotonic
tested and rejected for tie-collapse; ranking untouched, ECE 0.0036 vs 0.0103 raw). The cost
model turned that probability into a priced three-way decision and passed the G6 interior-
minimum gate. The engine (`src/`) was then built as a CPU-only package that only ever *loads*
an artifact — which is where the project's most severe bug surfaced: a silent 23x probability
difference from an XGBoost version mismatch, fixed with a hard pin plus a runtime guard.
SHAP + LLM-narrative + every fallback path were wired in and each broken on purpose to prove
recovery. Two evidence experiments followed: the LLM-as-classifier benchmark (XGBoost 0.5735
vs `gpt-oss:20b` 0.1571 PR-AUC) and the Kaggle-legal leakage study (causal 0.5446 vs full
leaky 0.5512 → gap only +0.0066). Robustness checks put a bootstrap CI on the rupee lift and
an exact per-transaction false-positive count on the policy. Error analysis and a training-
dev decomposition attributed the val→test gap to temporal mismatch, not overfitting. Six
follow-up experiments then tested whether the shipped model could be beaten — five negative
(harder regularization, per-segment calibration, LightGBM tuning, weak-model diversity, 3-vs-
2-model rupee value), one positive: averaging in an untuned LightGBM, which shipped as the
2-model ensemble after confirming the gain survives the real rupee policy. Finally, two
rounds of automated AI code review over the repo found three real defects (a train/serve
`ddof` skew, a malformed-artifact crash path, a NaN calibrator that could produce a silent
`allow`), all fixed with regression tests, plus three production-grade gaps disclosed in
the exception list rather than closed. Every number above is in `docs/experiments.md`; every failure and
fix, written as it happened, is in `journal/`.

---

## 14. Reference

- Buildathon: https://razorpay.com/buildathon/
- IEEE-CIS 1st place, Part 1: https://www.kaggle.com/c/ieee-fraud-detection/discussion/111284
- IEEE-CIS 1st place, Part 2 (technical): https://www.kaggle.com/c/ieee-fraud-detection/discussion/111321
- UID detection script: https://www.kaggle.com/kyakovlev/ieee-uid-detection-v6
- XGB "magic" 0.9600: https://www.kaggle.com/cdeotte/xgb-fraud-with-magic-0-9600
- EDA, first 150 cols: https://www.kaggle.com/alijs1/ieee-transaction-columns-reference
- EDA, V and ID cols: https://www.kaggle.com/cdeotte/eda-for-columns-v-and-id
- Razorpay Disputes API: https://razorpay.com/docs/api/disputes/
- Razorpay official MCP server: https://github.com/razorpay/razorpay-mcp-server

---

## 15. Working agreements

- Update this file whenever a decision changes. It is the source of truth.
- Log failures to `journal/` **as they happen**.
- Report honestly: if a number is bad, the number goes in the report. The rubric rewards it and the panel will find it anyway.
- Prefer deterministic where determinism belongs. Every LLM call must survive being asked "why not a rule?"
- **Verify claims from writeups ourselves.** Trust nothing we haven't reproduced.
