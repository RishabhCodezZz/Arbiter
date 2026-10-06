# Notebook index

Research runs on Kaggle with IEEE-CIS competition data attached and GPU enabled.
Each notebook reloads its own data/state; the local `src/` engine only loads exported
artifacts. [Notebook 06](06_cost_model_refined.py) is the current regeneration entry.

## Paired source and outputs

Each stage keeps a jupytext pair with the same stem:

| File | Role |
|---|---|
| `NN_name.py` | Readable source for review and execution |
| `NN_name.ipynb` | Executed notebook retaining the code and outputs from a recorded run |

Executed outputs are historical evidence for the code recorded in that notebook. They
do not automatically prove an unchanged result for the current Python source: source
and output pairs may reflect different revisions. Reproduce a claim by executing its
corresponding source with the required data/environment and inspecting fresh outputs.
Keep both files when archiving research. [Experiments](../docs/experiments.md) identifies
historical stages; the [evaluation report](../docs/eval_report.md) owns current results.

## Stages

| Stage | Source / recorded run | Purpose and status |
|---|---|---|
| 01 | [01_eda_baseline.py](01_eda_baseline.py) / [outputs](01_eda_baseline.ipynb) | Historical baseline, EDA and UID investigation; card-level state was cut after identity reconstruction fell short |
| 02 | [02_causal_features.py](02_causal_features.py) / [outputs](02_causal_features.ipynb) | Expanding history aggregates and time-consistency screening; feature basis for later stages |
| 03 | [03_reduce_tune_calibrate.py](03_reduce_tune_calibrate.py) / [outputs](03_reduce_tune_calibrate.ipynb) | V-column reduction, XGBoost tuning and historical single-model calibration comparisons |
| 04 | [04_cost_model.py](04_cost_model.py) / [outputs](04_cost_model.ipynb) | Archival research evidence: historical single-XGBoost cost pipeline and the retained error analysis, training-dev decomposition, bounded hyperparameter sweep and segment calibration source |
| 05 | [05_kaggle_legal_leaky.py](05_kaggle_legal_leaky.py) / [outputs](05_kaggle_legal_leaky.ipynb) | Historical full-history feature/post-processing comparison against causal stage 02; results apply to that comparison |
| 06 | [06_cost_model_refined.py](06_cost_model_refined.py) / [outputs](06_cost_model_refined.ipynb) | Current artifact regeneration entry, including model exports, ensemble comparisons and final two-model dashboard/raw re-export |

Notebook 04 remains at its existing path because it contains unique diagnostic code
and evidence cited by the journal and experiments. Execute the relevant 04 diagnostics
when reproducing those historical analyses; use 06 for current engine artifacts.
The historical LightGBM tuning search and LLM benchmark export cells were removed from
current sources, so retain their downloaded outputs and recorded runs.

## Regeneration

Run notebook 06 through the final ensemble re-export and download the files listed in
the [artifact guide](../artifacts/README.md). Its earlier cells export single-XGBoost
curve/raw data, which the final addendum overwrites with ensemble probabilities. Each
model has an independent Platt calibrator; the engine uses their simple average.

Successful artifact regeneration checks that the current pipeline executes. It does
not create an independent evaluation: the test month has been reused for follow-ups
and final selection. Notebook 04's outputs cannot stand in for a fresh run of current
06, even where pipeline sections were shared at consolidation time.

`reduce_mem()` is copied into the notebooks to keep them self-contained on Kaggle.
Caches are regenerable; source/output pairs and result exports should be retained.
See [retention guidance](../artifacts/README.md#retention) before resetting runtime state.
