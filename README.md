# Arbiter

Arbiter is a solo fraud decision project for card-not-present payments. It uses a
calibrated fraud probability and an illustrative INR cost model to choose allow,
step-up verification or block for each transaction. The local Python engine loads
trained models, records decisions and explains flagged cases.

## Current snapshot

The engine uses Optuna-tuned XGBoost and untuned LightGBM on the same features,
with independent Platt calibration followed by a simple average.

| Measure | Current ensemble |
|---|---|
| Evaluation rows | 92,427 transactions; 3,213 fraud labels |
| PR-AUC / ROC-AUC | 0.5597 / 0.9126 |
| Simulated policy value | ₹17.355 crore |
| Simulated lift | +₹1.678 crore vs no system; +₹77.03 lakh vs reference 0.5 cutoff |
| Genuine transactions routed to block | 223; calculated penalty ₹13.20 lakh |

These are retrospective results on a reused temporal holdout. Follow-up comparisons
and the final ensemble choice used this month's results. Rupee values apply assumed
fees, margin, customer-loss penalties and intervention outcomes to historical rows;
they are not observed merchant profit. Bootstrap intervals describe uncertainty
within this month under fixed assumptions. See the [evaluation report](docs/eval_report.md)
for current results, intervals and limitations, and [experiments](docs/experiments.md)
for provenance and historical comparisons.

## Quickstart

Clone the repository and enter its root:

```text
git clone https://github.com/RishabhCodezZz/arbiter.git
cd arbiter
python -m venv .venv
```

Install dependencies using the new environment's interpreter.

Windows (PowerShell):

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

macOS/Linux:

```bash
.venv/bin/python -m pip install -r requirements.txt
```

Generate and download the models with [notebook 06](notebooks/06_cost_model_refined.py)
on Kaggle, with competition data attached and GPU enabled. Complete the final
ensemble exports, since earlier cells write single-model evaluation files that the
last addendum overwrites. Put these files in `artifacts/`:

```text
model.json             calibrator.json
model_lgb.txt          calibrator_lgb.json
feature_manifest.json  sample_transactions.json
```

The first five files are required for ensemble scoring; the sample is for the demo.
[Artifact instructions](artifacts/README.md) distinguish runtime files from optional
exports. Keep the training library versions recorded in the manifest; the loader
fails closed if required files or versions are unsuitable.

Run the demo and tests from the repository root.

Windows:

```powershell
.\.venv\Scripts\python.exe scripts\demo_engine.py
.\.venv\Scripts\python.exe -m pytest tests/
```

macOS/Linux:

```bash
.venv/bin/python scripts/demo_engine.py
.venv/bin/python -m pytest tests/
```

The demo should print `ALL CHECKS PASSED` after checking idempotency, policy replay,
tamper detection and fail-closed behavior. Some tests need the downloaded artifacts.
Local inference uses CPU. Narrative generation falls back to a template without
`ANTHROPIC_API_KEY`; configured LLM calls run synchronously for flagged cases.

## Offline dashboard

Open [dashboard.html](dashboard.html) directly in a browser. It embeds its data,
styles and scripts and needs no server or Python environment. Headline, ranking
curves and sensitivity data use the current ensemble. The two-way cost-curve minimum
is a test-optimized presentation proxy, separate from the three-way runtime policy.
Queue/audit snapshots and the calibration panel contain historical single-model
evidence. The [docket](docs/docket.html) is a historical walkthrough.

## Limitations

- The dataset is IEEE-CIS e-commerce data; applying an INR scenario does not validate
  Indian merchant economics or an Indian payment mix.
- Step-up stopping and abandonment are assumed. Sensitivity currently covers margin
  and chargeback fee; the remaining economic inputs need validation.
- Fresh temporal evaluation and current ensemble calibration measurement remain open.
- The engine has local file persistence, no event-time ordering guarantee and no
  verified end-to-end latency SLO. Audit hashes are unkeyed and forgeable by a writer
  who also recomputes them. See [architecture](docs/architecture.md).
- The [historical LLM benchmark](docs/experiments.md#historical-llm-benchmark-and-reproduction-limits)
  used single XGBoost and had ordering/timeout limitations. The current benchmark
  script loads the ensemble, so rerunning it does not reproduce that comparison.

## Attribution

The [IEEE-CIS first-place solution](https://www.kaggle.com/c/ieee-fraud-detection/discussion/111284)
by Chris Deotte and team, including its
[technical writeup](https://www.kaggle.com/c/ieee-fraud-detection/discussion/111321),
informed UID construction, group aggregates, time-consistency screening and V-column
reduction. Arbiter uses expanding history features and omits future-dependent client
post-processing. The leakage comparison is documented in [experiments](docs/experiments.md);
it does not establish superiority to the competition solution.

## Document map

| Document | Role |
|---|---|
| [Evaluation report](docs/eval_report.md) | Current results and unresolved limitations |
| [Experiments](docs/experiments.md) | Parameter sources, selection history and evidence |
| [Architecture](docs/architecture.md) | Current engine flow and serving constraints |
| [CLAUDE.md](CLAUDE.md) | Project scope and working agreements |
| [Artifacts](artifacts/README.md) / [notebooks](notebooks/README.md) | Export requirements and regeneration |
| [Build journal](journal/build-log.md) | Sequential historical record of decisions and fixes |
