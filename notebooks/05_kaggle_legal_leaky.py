# %% [markdown]
# # 05 — The "Kaggle-legal" leaky model (Component B, second half)
#
# Component B's whole point (see CLAUDE.md sec 5, journal): the 1st-place
# solution's "magic" uses each client's FUTURE transactions to score a PAST one —
# `groupby('uid').agg('mean')` over the full dataset, plus a post-processing step that
# replaces every prediction with that client's average prediction. Both are legal on a
# static Kaggle leaderboard. Neither is possible in a gateway with a 200ms decision window
# and no knowledge of what a card will do next week.
#
# notebooks/02 already built and shipped the HONEST version (V1/V2: expanding-window,
# shift(1), never sees its own future). This notebook builds the DELIBERATELY LEAKY twin —
# same feature *names*, same aggregate *statistics*, the ONLY thing that changes is whether
# the aggregation window includes the future. That's the whole comparison: not "different
# features", one specific methodological choice, isolated.
#
# EXPECTED RESULT, stated up front so nobody mistakes this for a failure later:
# the leaky model is SUPPOSED to score higher. That's the finding, not a bug. We do
# not tune this model to lose, and we do not tune it to win either — same hyperparameters as
# V1, so the ONLY variable being measured is causal-vs-leaky, nothing else.
#
# SELF-CONTAINED, same convention as notebook 02 — reloads data fresh.

# %%
import gc
import warnings

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import average_precision_score, roc_auc_score

warnings.filterwarnings("ignore")
np.seterr(invalid="ignore", divide="ignore")
pd.set_option("display.max_columns", 100)

DATA = "/kaggle/input/competitions/ieee-fraud-detection"
print("xgboost:", xgb.__version__)

# %% [markdown]
# ## Rebuild state (identical to notebook 02 through the uid/split setup)

# %%
def reduce_mem(df, verbose=True):
    start = df.memory_usage(deep=True).sum() / 1024**2
    for col in df.columns:
        t = df[col].dtype
        if t == object or str(t) == "category":
            continue
        c_min, c_max = df[col].min(), df[col].max()
        if str(t).startswith("int"):
            if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                df[col] = df[col].astype(np.int8)
            elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                df[col] = df[col].astype(np.int16)
            elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                df[col] = df[col].astype(np.int32)
        else:
            df[col] = df[col].astype(np.float32)
    end = df.memory_usage(deep=True).sum() / 1024**2
    if verbose:
        print(f"  memory {start:.0f}MB -> {end:.0f}MB")
    return df


print("loading...")
txn = reduce_mem(pd.read_csv(f"{DATA}/train_transaction.csv"))
idt = reduce_mem(pd.read_csv(f"{DATA}/train_identity.csv"))
df = txn.merge(idt, how="left", on="TransactionID")
del txn, idt
gc.collect()

df["day"] = (df["TransactionDT"] // (24 * 60 * 60)).astype(np.int16)
df["hour"] = ((df["TransactionDT"] // 3600) % 24).astype(np.int8)
df["dow"] = (df["day"] % 7).astype(np.int8)

for col in ["D1", "D4", "D10", "D15"]:
    df[f"{col}n"] = df["day"] - df[col]

df["uid"] = (
    df["card1"].astype(str) + "_" + df["addr1"].astype(str) + "_"
    + df["D1n"].astype(str) + "_" + df["D4n"].astype(str)
)
df["coarse_uid"] = df["card1"].astype(str) + "_" + df["addr1"].astype(str) + "_" + df["D1n"].astype(str)

TRAIN_END, VAL_END = 120, 150
print(f"shape: {df.shape}, day range {df['day'].min()}-{df['day'].max()}")

# %% [markdown]
# ## LEAKY full-history aggregates — the same statistics as V1's causal features,
# computed over the WHOLE group instead of expanding-window-then-shift(1)
#
# Direct analogs of every V1 `uid_*` feature that has one. `uid_time_since_prev` has no
# natural full-history twin (it's inherently about sequence, not a full-history aggregate
# statistic like Deotte's actual `groupby('uid').agg('mean')` trick) — skipped rather than
# invented, so this stays a faithful reproduction of the documented technique, not a
# different thing wearing its name.
#
# NOTE — no `.shift(1)` anywhere below. That absence IS the leak. Every row, including a
# client's very first transaction chronologically, sees a statistic computed from ALL of
# that client's rows, including ones that haven't happened yet relative to it.

# %%
df["uid_txn_count_LEAKY"] = df.groupby("uid")["TransactionID"].transform("count")
df["uid_amt_mean_LEAKY"] = df.groupby("uid")["TransactionAmt"].transform("mean")
df["uid_amt_std_LEAKY"] = df.groupby("uid")["TransactionAmt"].transform("std")

for col, out in [("DeviceInfo", "uid_device_used_ever_LEAKY"),
                  ("P_emaildomain", "uid_email_used_ever_LEAKY")]:
    if col in df.columns:
        # "has this uid EVER used this device/email" — dropna=False for the same reason as
        # notebook 02 (most rows have missing device; default groupby silently drops NaN
        # keys, which would produce NaN instead of a real 0 for the majority of rows)
        grp_size = df.groupby(["uid", col], dropna=False)[col].transform("size")
        df[out] = (grp_size > 0).astype(np.int8)

# the near-miss feature from notebook 02, RESERVED for exactly this notebook — rebuilt here
# since this notebook is self-contained and can't assume 02's kernel state survived
df["uid_ambiguity_std_full_LEAKY"] = df.groupby("coarse_uid")["D15n"].transform("std")

leaky_cols = [c for c in df.columns if c.endswith("_LEAKY")]
print("leaky features built:", leaky_cols)

# %% [markdown]
# ## Prove the leak is real, on the same demo bucket notebook 02 used — not asserted
#
# Same bucket, same point notebook 02 already made for one feature (D15n ambiguity) — here
# for `uid_amt_mean_LEAKY` too, since that's the one that actually feeds the model as a raw
# amount statistic, closer to Deotte's literal `agg('mean')` trick than the ambiguity
# feature is.

# %%
big_bucket_uid = df.groupby("uid").size().sort_values(ascending=False).index[3]
demo = df.loc[df["uid"] == big_bucket_uid,
              ["TransactionDT", "TransactionAmt", "uid_amt_mean_LEAKY"]].sort_values("TransactionDT")
print(demo.to_string(index=False))
print(f"\n>>> uid_amt_mean_LEAKY is IDENTICAL on every row for this client ({demo['uid_amt_mean_LEAKY'].nunique()} "
      f"unique value across {len(demo)} rows) — including the first row, which has already 'seen' the mean of "
      f"every future transaction this client will ever make. That's the leak, made visible, not theoretical.")

# %% [markdown]
# ## Temporal split — IDENTICAL boundaries to every other model in this project
#
# Same test month, never touched until this exact score. The leak is in the FEATURES
# (computed over the full timeline before splitting), not in the split itself — this stays
# faithful to how the 1st-place solution actually worked: full-history aggregates built
# once, then a normal train/val/test split applied on top.

# %%
tr = df[df["day"] <= TRAIN_END]
va = df[(df["day"] > TRAIN_END) & (df["day"] <= VAL_END)]
te = df[df["day"] > VAL_END]
for name, part in [("train", tr), ("val", va), ("test", te)]:
    print(f"{name:5s}  {len(part):7,} rows  fraud {part['isFraud'].mean():.3%}")

# %% [markdown]
# ## Feature prep — same DROP list as V1, leaky uid_* features instead of causal ones
#
# Deliberately NOT re-running the time-consistency screen (notebook 02's V2 step): that
# screen exists to catch a specific pathology in CAUSAL features — a cold-start slice looking
# structurally different from a mature one. A full-history leaky feature has no cold start by
# construction (even day-1 rows see the client's whole future), so the same screen isn't
# obviously meaningful here, and re-deriving a parallel screening methodology is out of scope
# for what this comparison needs to prove. Same hyperparameters as V1 for the same reason:
# isolate ONE variable (causal vs leaky), not two (causal-vs-leaky AND re-tuned).

# %%
DROP = ["TransactionID", "TransactionDT", "isFraud", "day",
        "D1n", "D4n", "D10n", "D15n", "uid", "coarse_uid"]
features_leaky = [c for c in df.columns if c not in DROP]
cat_cols = [c for c in features_leaky if df[c].dtype == object]
print(f"{len(features_leaky)} features ({len(cat_cols)} categorical, {len(leaky_cols)} leaky uid features)")

for c in cat_cols:
    mapping = {v: i for i, v in enumerate(tr[c].dropna().unique())}
    for part in (tr, va, te):
        part[c] = part[c].map(mapping).fillna(-1).astype(np.int32)

X_tr, y_tr = tr[features_leaky], tr["isFraud"]
X_va, y_va = va[features_leaky], va["isFraud"]
X_te, y_te = te[features_leaky], te["isFraud"]

# %% [markdown]
# ## Train — same hyperparameters as V1 (notebook 02), on purpose

# %%
def make_model():
    return xgb.XGBClassifier(
        n_estimators=2000, learning_rate=0.05, max_depth=8,
        subsample=0.8, colsample_bytree=0.5, min_child_weight=4,
        reg_lambda=1.0, tree_method="hist", device="cuda",
        eval_metric="aucpr", early_stopping_rounds=100, random_state=42,
    )


def report(name, y_true, y_prob):
    pr = average_precision_score(y_true, y_prob)
    roc = roc_auc_score(y_true, y_prob)
    print(f"{name:6s}  PR-AUC {pr:.4f}   ROC-AUC {roc:.4f}   lift {pr/y_true.mean():.1f}x")
    return pr, roc


model_leaky = make_model()
model_leaky.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], verbose=100)

print("\n" + "=" * 62)
print("LEAKY features only (no post-processing yet)")
print("=" * 62)
report("val", y_va, model_leaky.predict_proba(X_va)[:, 1])
pr_feat, roc_feat = report("test", y_te, model_leaky.predict_proba(X_te)[:, 1])

# %% [markdown]
# ## The SECOND leak — post-processing: replace each prediction with the client's average
#
# The 1st-place writeup's own words: "taking all predictions from a single client and
# replacing them with that client's average prediction." This is separate from and IN
# ADDITION TO the leaky features above — it uses the TEST SET'S OWN future predictions
# (including ones after the row being scored) to overwrite a past prediction. Doubly
# impossible in production: not just future transaction DATA, future MODEL OUTPUT.

# %%
test_probs_raw = model_leaky.predict_proba(X_te)[:, 1]
test_probs_series = pd.Series(test_probs_raw, index=te.index)
test_uid = te["uid"]

test_probs_postprocessed = test_probs_series.groupby(test_uid).transform("mean")

print("=" * 62)
print("LEAKY features + LEAKY post-processing (client-mean override)")
print("=" * 62)
pr_full, roc_full = report("test", y_te, test_probs_postprocessed)

# %% [markdown]
# ## The gap — Component B's actual deliverable

# %%
# Causal numbers from notebook 02 / docs/experiments.md — pasted, not recomputed, since this
# notebook is self-contained and doesn't share kernel state with 02. If 02's numbers ever
# change, update these two lines to match before trusting this comparison table.
PR_CAUSAL_V2 = 0.5446
ROC_CAUSAL_V2 = 0.9014

print(f"""
{'='*70}
COMPONENT B — the full comparison
{'='*70}
{'Model':<45} {'PR-AUC':>10} {'ROC-AUC':>10}
{'-'*70}
{'Honest (causal, V2, deployable)':<45} {PR_CAUSAL_V2:>10.4f} {ROC_CAUSAL_V2:>10.4f}
{'Leaky features only (undeployable)':<45} {pr_feat:>10.4f} {roc_feat:>10.4f}
{'Leaky features + leaky post-processing':<45} {pr_full:>10.4f} {roc_full:>10.4f}
{'-'*70}
Gap from leaky FEATURES alone:            {pr_feat - PR_CAUSAL_V2:+.4f} PR-AUC
Gap from leaky POST-PROCESSING on top:     {pr_full - pr_feat:+.4f} PR-AUC
TOTAL gap (full Kaggle-legal vs honest):   {pr_full - PR_CAUSAL_V2:+.4f} PR-AUC
{'='*70}

--- paste into docs/experiments.md ---
| B-leaky-feat | LEAKY full-history aggregates (undeployable) | {pr_feat:.4f} | {roc_feat:.4f} | {pr_feat-PR_CAUSAL_V2:+.4f} vs honest V2 |
| B-leaky-full | + LEAKY client-mean post-processing (undeployable) | {pr_full:.4f} | {roc_full:.4f} | {pr_full-pr_feat:+.4f} vs leaky-feat |
""")

# %% [markdown]
# ## Download
#
# No artifact export needed here (this model is never deployed, by design — that's the
# whole point). Just copy the printed table above into docs/experiments.md and the journal,
# with today's date.
