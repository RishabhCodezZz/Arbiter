# artifacts/

Not committed to the repo (model weights + real transaction samples don't belong in git
history). Populated by running the addenda in `notebooks/06_cost_model_refined.py` in your
Kaggle session, then downloading these files from Kaggle's Output panel:

**The shipped model is a 2-model ensemble** (XGBoost + LightGBM, both trained untuned on the
same feature set, simple-averaged after independent Platt calibration) — a confirmed,
statistically real rupee-value improvement over the single-XGBoost baseline (95% CI
[+₹6.55L, +₹21.24L] on the test month; see `docs/experiments.md`'s ensemble sections for the
full evidence trail, including a 3rd model (CatBoost) and further per-model tuning, both
tried and NOT adopted). Both `model.json`+`model_lgb.txt` and both calibrators are required —
`src/model.py` fails closed (same as a fully-missing model) if either is absent.

- `model.json` — the trained XGBoost model (V4, Optuna-tuned), native format, CPU-loadable
- `calibrator.json` — Platt calibration coefficients (`{"coef": ..., "intercept": ...}`)
- `model_lgb.txt` — the trained LightGBM model (untuned — see `docs/experiments.md` for why
  tuning it was tried and reversed), LightGBM's own native text format, CPU-loadable
- `calibrator_lgb.json` — LightGBM's own independently-fit Platt calibration coefficients
- `feature_manifest.json` — the exact feature list, categorical vocabularies, kept V-columns,
  and both `xgboost_version`/`lightgbm_version` (the runtime version-mismatch guards in
  `src/model.py` check against these, not against `requirements.txt`'s pins)
- `sample_transactions.json` — 25 real held-out test-month transactions, for `scripts/demo_engine.py`
- `llm_benchmark_sample.json` — 500 real held-out test-month transactions, for `scripts/llm_benchmark.py`
- `llm_benchmark_kaggle_results.json` — LLM-side scores from Ollama running on Kaggle's GPU
- `dashboard_data.json` — PR/ROC/cost curves + sensitivity grid from the full test month. Not
  consumed at runtime by anything (the data is already baked into `dashboard.html`) — kept
  as the source-of-truth export that page's numbers are traceable back to.
- `test_month_raw.json` — raw (probability, label, amount) for all 92,427 test-month rows,
  row-level not aggregated. Only needed for local bootstrap-resampling / rules-baseline work
  against the exact headline population; nothing else reads it. Self-checked at export time
  (the notebook cell that writes it recomputes `arbiter_value` from the raw arrays alone and
  asserts it matches the headline number before printing success).
- `error_analysis_false_positives.json` — all 233 real, exact false-positive transactions
  (raw features, calibrated probability, amount, top-5 SHAP contributions each) — for the
  manual error-analysis pass (clue skill ch14).
- `error_analysis_false_negatives_sample.json` — 100 real transactions (of 1,444) where
  fraud was allowed straight through with zero friction — same fields as above.
- `training_dev_gap.json` — the train'/training-dev/val/test PR-AUC and ROC-AUC comparison
  used to decompose the val→test gap (clue skill ch40/41). Comes from a diagnostic-only
  model refit on 85% of the training period, not the shipped artifact.
- `hyperparam_sweep.json` — a bounded 6-config sweep testing whether the shipped model's
  low regularization is an artifact of tuning against an inflated validation signal.
  Selection uses train'/training-dev/val only (never test); test is checked once afterward,
  as a confirmation. Diagnostic only — adopting the result means a full refit on 100% of
  `tr` before it could ever replace the shipped `model.json`.
- `segment_calibration.json` — tests whether a per-ProductCD Platt calibrator (vs. the one
  global calibrator) reduces the false-positive cluster in ProductCD='C' (81.1% of the 233
  real false positives, per `docs/eval_report.md` §8b), using the shipped model's own raw
  scores — no retraining, calibration-only. Scoped narrowly to that precision question; the
  false-negative cluster in ProductCD='W' has an already-proven zero ceiling for any
  threshold/calibration fix and isn't what this tests. Selection on val only, test checked
  once as confirmation.
- `ensemble_diagnostic.json` — tests whether averaging the shipped XGBoost with a
  LightGBM and a CatBoost model (same features, same split, one fair shot each — not
  independently tuned) closes any of the gap to the Kaggle-winning ensemble. Selection on
  val only (does the 3-model average beat XGBoost-alone), test checked once as
  confirmation. Diagnostic only — if it looks promising, a real per-model tuning pass and
  a full engine rebuild (multi-model loading, SHAP across 3 models, re-verifying every
  downstream artifact) would be the next step before shipping, not done here.
- `ensemble_bootstrap.json` — a paired bootstrap on the test PR-AUC delta (ensemble vs.
  shipped), checking whether the point-estimate improvement is statistically real or
  within the noise of a single test month. Also scores a leaner 2-model (XGBoost+LightGBM,
  dropping CatBoost) ensemble as a simplicity check. **Result: both confirmed real** (95%
  CI excludes zero for both the 2- and 3-model ensembles) — the first "make it better"
  experiment this session to actually clear the bar for shipping.
- `lgb_tuning.json` — no longer produced by `06_cost_model_refined.py` (the search itself
  was removed once its result was fully captured — see below); the file that was
  downloaded from the one Kaggle run that produced it is still referenced from
  `docs/experiments.md`/`journal/` and doesn't need to exist locally for anything to work.
  **Result: tuning LightGBM against val made it WORSE** — test PR-AUC dropped from 0.5541
  (untuned) to 0.5406 (tuned), confirmed by a paired bootstrap (95% CI excludes zero,
  entirely negative). Swapping the tuned version into the ensemble erased its confirmed
  win. Decision: don't tune ensemble members against val PR-AUC in this problem — acted on
  by leaving every model in `diversity_check.json` below untuned.
- `diversity_check.json` — adds three more untuned model families (Random Forest, Extra
  Trees, Logistic Regression) to the confirmed 3-model ensemble, testing whether more
  diversity helps further now that tuning has been ruled out. Selection is a paired
  bootstrap of the resulting 6-model ensemble against the CONFIRMED 3-model ensemble
  (not the shipped model directly) — the real question this answers.
- `ensemble_rupee_value.json` — runs the 2-model and 3-model ensembles through the REAL
  3-way cost policy (identical `value_allow`/`value_stepup`/`value_block`/`realized_value`
  machinery as the shipped headline number), since every prior ensemble comparison judged
  on PR-AUC/ROC-AUC, not the rupee value this project actually optimizes for. Also runs a
  direct paired bootstrap of 3-model vs. 2-model rupee value — the one comparison the
  PR-AUC-only analysis couldn't settle.
- `ensemble_dashboard_headline.json` — readable summary of the two files below, since
  parsing full curve arrays for a quick number check is annoying.
- `dashboard_data.json` / `test_month_raw.json` — **as of the ensemble-rebuild addendum,
  these are OVERWRITTEN with the shipped 2-model ensemble's real data** (PR/ROC/cost
  curves, policy mix, sensitivity map, and full row-level calibrated probabilities), not
  the single-model numbers they originally held. Closes exception-list item 10
  (`docs/eval_report.md`) — `dashboard.html` and `scripts/robustness_checks.py`'s output
  need to be regenerated from the new copies of these two files to actually reflect this.

Once `model.json`, `calibrator.json`, `model_lgb.txt`, `calibrator_lgb.json`,
`feature_manifest.json`, and `sample_transactions.json` are all here, run from the repo root:

```
python scripts/demo_engine.py
```

The dashboard needs none of this downloaded locally — `../dashboard.html` is one
self-contained file with real data already captured into it. Open it directly in a
browser, nothing to install. (An earlier version ran locally via `streamlit run
dashboard.py`; later retired in favor of this.)
