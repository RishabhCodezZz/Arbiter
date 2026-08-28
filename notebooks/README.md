# notebooks/ — the Kaggle research record

Each stage is a **jupytext pair** sharing one stem:

| File | Role |
|---|---|
| `NN_name.py` | the readable, diff-able source — this is what gets reviewed |
| `NN_name.ipynb` | the same code, executed on Kaggle, outputs committed as proof the numbers in [`docs/experiments.md`](../docs/experiments.md) are real and reproducible |

All notebooks are **self-contained** — each reloads its own state from the competition
data rather than assuming a prior kernel survived. Run on Kaggle (competition data
attached, GPU on). The `src/` package never runs any of this — it only loads the exported
artifact.

## The ladder

| # | Notebook | What it adds | Headline number | Keep because |
|---|---|---|---|---|
| 01 | `01_eda_baseline` | V0 floor — plain XGBoost, no history features, no tuning. EDA + the UID-identity investigation. | PR-AUC 0.5486 / ROC-AUC 0.9002 | Bottom rung of the measured ladder; the only place the "could not reproduce 96.9/2.9/0.2 → Component C cut" investigation lives. |
| 02 | `02_causal_features` | V1/V2 — causal (expanding-window, backward-only) client-history aggregates + time-consistency screening. | V2 PR-AUC 0.5446 | Builds the feature set every downstream notebook and Component B depend on. |
| 03 | `03_reduce_tune_calibrate` | V3 (V-column reduction) + V4 (60-trial Optuna) + V5 (calibration: isotonic rejected, Platt shipped). | V4 PR-AUC 0.5514 / ROC-AUC 0.9077 | Produces the winning hyperparameters `04`/`06` hard-code, and the reliability-diagram. |
| 04 | `04_cost_model` | **History only.** The full record of the cost-model stage — 3-way policy, G6 gate, artifact exports — **plus** four one-off diagnostics (error-analysis export, training-dev/bias-variance-mismatch decomposition, bounded hyperparameter sweep, per-segment calibration). | +₹1.54cr vs no system | Superseded by `06` for the pipeline, but it is the only place the four diagnostics' code lives, and they are cited throughout `docs/experiments.md`. Not the file to re-run. |
| 05 | `05_kaggle_legal_leaky` | Component B, second half — the deliberately leaky full-history twin of `02`'s causal features. Same names, same stats, the only variable is whether the window sees the future. | causal 0.5446 vs leaky 0.5512 → gap +0.0066 | The differentiator: measures what causal honesty costs. |
| 06 | `06_cost_model_refined` | **The current entry point.** `04` minus the four completed diagnostics, plus the still-active XGBoost + LightGBM (+ CatBoost) ensemble diagnostic and its bootstrap. Produces every artifact `src/` still needs. | 2-model ensemble +₹13.58L over single-model, 95% CI [+₹6.55L, +₹21.24L] | Run this one to regenerate artifacts. |

## To reproduce the artifacts

Run **`06_cost_model_refined`** on Kaggle and download the outputs listed in
[`artifacts/README.md`](../artifacts/README.md). `04`'s executed pipeline sections are
byte-identical to `06`'s apart from the removed diagnostics (diffed at consolidation
time), so `04_cost_model.ipynb` remains a valid proof-of-run for the shared pipeline.

## Known cosmetic debt

`reduce_mem()` is copy-pasted into every notebook rather than imported — a deliberate
trade for self-containment (a Kaggle notebook can't `import` from a sibling file without
setup). Same function, same behaviour, in all six.
