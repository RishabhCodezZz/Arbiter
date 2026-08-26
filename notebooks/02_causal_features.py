# %% [markdown]
# # 02 — Causal Aggregates (V1) + Time-Consistency Screening (V2)
#
# Goal: the "honest version of the magic" — client-history features that only ever
# look backward in time, the way a real gateway would have to. Then screen out any feature
# whose relationship with fraud inverts over time (Deotte's trick).
#
# This notebook is SELF-CONTAINED (reloads data fresh — don't assume a prior kernel
# state survived). See notebook 01 for the full EDA and the reasoning behind each setup
# decision; this repeats only what's needed to rebuild df/uid/split, without re-deriving
# decisions already made and logged in CLAUDE.md / docs/experiments.md.
#
# UID KEY USED HERE: card1+addr1+D1n+D4n. NOT the same as the 1st place team's — ours
# measured 2.11% cross-client mixing in the baseline/UID stage (target was 0.2%; component C was cut as a
# result). We use it anyway for AGGREGATION here because this comparison (causal vs
# full-history) doesn't depend on exact identity purity — see CLAUDE.md sec 4/5.

# %%
import gc
import warnings

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import average_precision_score, roc_auc_score

warnings.filterwarnings("ignore")
np.seterr(invalid="ignore", divide="ignore")  # 0/0 and x/0 -> NaN is INTENTIONAL below
                                                # (it means "no prior history yet"), not a bug
pd.set_option("display.max_columns", 100)

DATA = "/kaggle/input/competitions/ieee-fraud-detection"
print("xgboost:", xgb.__version__)

# %% [markdown]
# ## Rebuild state from the baseline/UID stage (condensed — see notebook 01 for the why)

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

# D1n/D4n/D10n/D15n: "day this reference event happened" (day - D_column). D1n is used for
# the coarse identity bucket; D4n is what pushed our best key to 2.11% mixed earlier;
# D10n/D15n are only needed below for the ambiguity diagnostic.
for col in ["D1", "D4", "D10", "D15"]:
    df[f"{col}n"] = df["day"] - df[col]

df["uid"] = (
    df["card1"].astype(str) + "_" + df["addr1"].astype(str) + "_"
    + df["D1n"].astype(str) + "_" + df["D4n"].astype(str)
)
# coarse_uid: the ORIGINAL 3-column key from the writeup, kept separate on purpose —
# it's what the ambiguity feature below measures the consistency OF.
df["coarse_uid"] = df["card1"].astype(str) + "_" + df["addr1"].astype(str) + "_" + df["D1n"].astype(str)

TRAIN_END, VAL_END = 120, 150
print(f"shape: {df.shape}, day range {df['day'].min()}-{df['day'].max()}")

# %% [markdown]
# ## Causal aggregate features — THE core of this notebook
#
# THE RULE: every feature below may use a row's OWN transaction data (always legitimately
# known at decision time) and any PRIOR transaction of the same client. It may NEVER use
# anything from that client's FUTURE transactions. This is what "causal" means here, and
# it's the entire reason component B (Kaggle-legal vs causal) is worth doing at all.
#
# IMPLEMENTATION PATTERN: compute a per-uid running total *including* the current row
# (fast, vectorized via cumsum/cumcount), then `.shift(1)` it within the same group. Shift
# inserts NaN at each group's first row and, for every later row, gives the running total
# through the PREVIOUS row only — excluding the current one. That's not a leakage patch
# bolted on afterward; it's the literal definition of "prior history."
#
# WHY COMPUTE ON THE FULL CHRONOLOGICAL TIMELINE, NOT PER-SPLIT: a real client's history
# doesn't reset at our train/val/test boundary — a day-121 transaction should still benefit
# from that same client's day-90 history. This stays 100% causal regardless, because each
# row only ever looks at ITS OWN past, never anything after itself, no matter which split
# it or its history lands in.
#
# We use pandas' automatic INDEX ALIGNMENT throughout: compute each feature in whatever
# sort order that feature needs (time-within-uid), then assign straight back into `df`.
# Pandas matches by index, not row position, so nothing needs to share one global order.

# %%
order = df.sort_values(["uid", "TransactionDT", "TransactionID"]).index
uid_ord = df.loc[order, "uid"]

# number of PRIOR transactions by this uid (0 = this client's first-ever transaction)
df["uid_txn_num"] = df.loc[order].groupby("uid", sort=False).cumcount()

# running mean/std of amount, PRIOR transactions only, via the cumsum/cumsumsq trick
# (much faster than pandas .expanding() across ~218k groups)
amt = df.loc[order, "TransactionAmt"].astype(np.float64)
cum_sum_incl = amt.groupby(uid_ord, sort=False).cumsum()
cum_sumsq_incl = (amt ** 2).groupby(uid_ord, sort=False).cumsum()

df["uid_amt_sum_prior"] = cum_sum_incl.groupby(uid_ord, sort=False).shift(1)
df["uid_amt_sumsq_prior"] = cum_sumsq_incl.groupby(uid_ord, sort=False).shift(1)

n_prior = df["uid_txn_num"].astype(np.float64)
df["uid_amt_mean_prior"] = df["uid_amt_sum_prior"] / n_prior
var_prior = (df["uid_amt_sumsq_prior"] / n_prior) - df["uid_amt_mean_prior"] ** 2
df["uid_amt_std_prior"] = np.sqrt(var_prior.clip(lower=0))  # clip: guards float error, not a real negative variance

# seconds since this uid's own previous transaction. NaN = this client's first transaction
# (a real, informative cold-start signal — not something to fill in).
df["uid_time_since_prev"] = df.loc[order].groupby("uid", sort=False)["TransactionDT"].diff()

# "has this uid used this exact device / email before?" — causal repeat-flags.
# dropna=False is deliberate: most rows have a MISSING device (only 24.4% identity
# coverage from notebook 01), and pandas groupby silently DROPS NaN keys by default,
# which would silently produce NaN/-1 for most of the dataset instead of a real 0/1.
for col, out in [("DeviceInfo", "uid_device_seen_before"), ("P_emaildomain", "uid_email_seen_before")]:
    if col in df.columns:
        seen = df.loc[order].groupby(["uid", col], sort=False, dropna=False).cumcount()
        df[out] = (seen > 0).astype(np.int8)

print("causal features built:")
print([c for c in df.columns if c.startswith("uid_")])

# %% [markdown]
# ## Correctness check — PROVE it, don't eyeball it
#
# WHY: a silent off-by-one here (using .expanding() instead of .expanding().shift(1), or
# grouping/sorting mismatched) would let the current row leak into its own "prior" stats.
# That's exactly the kind of bug that produces a suspiciously great score and an indefensible
# number in the panel room. Brute-force recompute a sample by hand and compare.

# %%
rng = np.random.default_rng(42)
sample_idx = rng.choice(df.index, size=300, replace=False)

mismatches = []
for idx in sample_idx:
    row = df.loc[idx]
    prior = df[(df["uid"] == row["uid"]) & (df["TransactionDT"] < row["TransactionDT"])]
    exp_n = len(prior)
    exp_mean = prior["TransactionAmt"].mean() if exp_n > 0 else np.nan

    n_ok = exp_n == row["uid_txn_num"]
    mean_ok = (pd.isna(exp_mean) and pd.isna(row["uid_amt_mean_prior"])) or \
              np.isclose(exp_mean, row["uid_amt_mean_prior"], equal_nan=True)
    if not (n_ok and mean_ok):
        mismatches.append((idx, exp_n, row["uid_txn_num"], exp_mean, row["uid_amt_mean_prior"]))

print(f"correctness check: {len(mismatches)} mismatches out of {len(sample_idx)} sampled rows")
if mismatches:
    print("(a handful can be legitimate ties: two transactions sharing the exact same")
    print(" TransactionDT for one uid, broken by TransactionID in our sort but treated as")
    print(" simultaneous by this row-by-row check. Large counts mean a real bug — stop.)")
    print(pd.DataFrame(mismatches[:10],
          columns=["idx", "expected_n", "actual_n", "expected_mean", "actual_mean"]))
assert len(mismatches) < 5, "STOP: causal features don't match manual recomputation. Do not trust V1 until fixed."
print(">>> PASSED — causal features verified against manual recomputation, not just asserted.")

# %% [markdown]
# ## The near-miss: a feature we almost shipped leaky
#
# The natural next feature is "how confident are we that this coarse identity bucket
# (card1+addr1+D1n) is really one client?" — from the baseline/UID stage, using std(D15n) across the WHOLE
# bucket. But computing that std across the whole bucket uses every transaction in it,
# INCLUDING ones that haven't happened yet relative to the row being scored. That's future
# information leaking into a "confidence" feature — precisely the bug this whole project
# is about catching in OTHER people's solutions. We almost put it in our own.
#
# Fix: same shift-after-expanding pattern as above, computed causally per coarse_uid.
# Below, we build BOTH versions side by side and prove the difference is real, not
# theoretical — then only the causal one goes into V1. The leaky one is kept, clearly
# labeled, for the later Kaggle-legal comparison model (component B) — that model is
# SUPPOSED to use future information, that's the entire point of measuring the gap.

# %%
order2 = df.sort_values(["coarse_uid", "TransactionDT", "TransactionID"]).index
coarse_ord = df.loc[order2, "coarse_uid"]

d15n_std_incl = (
    df.loc[order2, "D15n"].groupby(coarse_ord, sort=False)
    .transform(lambda s: s.expanding().std())
)
df["uid_ambiguity_std_prior"] = d15n_std_incl.groupby(coarse_ord, sort=False).shift(1)  # CAUSAL — use this

# LEAKY on purpose. Reserved for the later Kaggle-legal comparison model. NOT added to
# the V1/V2 feature lists below.
df["_LEAKY_coarse_d15n_std_full_bucket_RESERVED_FOR_DAY8_ONLY"] = (
    df.groupby("coarse_uid")["D15n"].transform("std")
)

# demonstrate the gap concretely on one real, sizeable bucket
big_bucket = df.groupby("coarse_uid").size().sort_values(ascending=False).index[5]
demo = df.loc[df["coarse_uid"] == big_bucket,
              ["TransactionDT", "D15n", "uid_ambiguity_std_prior",
               "_LEAKY_coarse_d15n_std_full_bucket_RESERVED_FOR_DAY8_ONLY"]].sort_values("TransactionDT")
demo.columns = ["TransactionDT", "D15n", "causal_std_so_far (ours)", "full_bucket_std (LEAKY)"]
print(demo.to_string(index=False))
print("\n>>> Watch the first row: causal_std_so_far is NaN (no history yet) while")
print(">>> full_bucket_std already has a concrete answer, computed from transactions that")
print(">>> haven't happened yet relative to that first row. That's the leak, made visible.")

# %% [markdown]
# ## Temporal split (same boundaries as the baseline stage — test stays untouched)

# %%
tr = df[df["day"] <= TRAIN_END]
va = df[(df["day"] > TRAIN_END) & (df["day"] <= VAL_END)]
te = df[df["day"] > VAL_END]
for name, part in [("train", tr), ("val", va), ("test", te)]:
    print(f"{name:5s}  {len(part):7,} rows  fraud {part['isFraud'].mean():.3%}")

# %% [markdown]
# ## Feature prep for V1
#
# NEW to the DROP list vs notebook 01: D1n, D4n, D10n, D15n — the raw identity-pointer
# columns. Deotte is explicit about this ("Preventing Overfitting"): don't feed columns
# that directly reveal client identity as raw features, because the model will memorise
# SEEN clients instead of learning generalisable patterns — and ~60% of later clients are
# unseen (measured in notebook 01). We keep card1/addr1/card2 etc as regular features
# (Deotte does too, throughout) — only the derived identity POINTERS are excluded.

# %%
DROP = ["TransactionID", "TransactionDT", "isFraud", "day",
        "D1n", "D4n", "D10n", "D15n", "uid", "coarse_uid",
        "_LEAKY_coarse_d15n_std_full_bucket_RESERVED_FOR_DAY8_ONLY"]
features_v1 = [c for c in df.columns if c not in DROP]
cat_cols = [c for c in features_v1 if df[c].dtype == object]
print(f"{len(features_v1)} features for V1 ({len(cat_cols)} categorical, "
      f"{len([c for c in features_v1 if c.startswith('uid_')])} new causal features)")

for c in cat_cols:
    mapping = {v: i for i, v in enumerate(tr[c].dropna().unique())}
    for part in (tr, va, te):
        part[c] = part[c].map(mapping).fillna(-1).astype(np.int32)

X_tr, y_tr = tr[features_v1], tr["isFraud"]
X_va, y_va = va[features_v1], va["isFraud"]
X_te, y_te = te[features_v1], te["isFraud"]

# %% [markdown]
# ## V1 model
#
# IDENTICAL hyperparameters to V0 (notebook 01), on purpose — no tuning yet (that's later).
# Isolating ONE change (the causal features) means the score delta is attributable to
# exactly that change, not tangled up with a simultaneous hyperparameter change. This is
# the whole point of the ladder in docs/experiments.md.

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


model_v1 = make_model()
model_v1.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], verbose=100)

print("\n" + "=" * 62)
print("V1 — raw features + causal client-history aggregates")
print("=" * 62)
report("val", y_va, model_v1.predict_proba(X_va)[:, 1])
pr_v1, roc_v1 = report("test", y_te, model_v1.predict_proba(X_te)[:, 1])
print(f"\nV0 test was PR-AUC 0.5486 / ROC-AUC 0.9002 — delta: "
      f"{pr_v1-0.5486:+.4f} PR-AUC / {roc_v1-0.9002:+.4f} ROC-AUC")

# %% [markdown]
# ## Do the new features matter? (quick check before screening)

# %%
imp = pd.Series(model_v1.feature_importances_, index=features_v1).sort_values(ascending=False)
uid_rank = [i for i, f in enumerate(imp.index) if f.startswith("uid_")]
print("causal feature ranks (0 = most important):", uid_rank[:10])
print("\ntop 5 causal features by importance:")
print(imp[imp.index.str.startswith("uid_")].head(5).to_string())

# %% [markdown]
# ## Time-consistency screening (V2)
#
# Deotte's method: does a feature's relationship with fraud hold up over time, or does it
# INVERT — find patterns in the present that don't exist in the future? He trained a tiny
# single-feature model on month 1 and tested on month 6; ~5% of columns failed, including
# the entire V322-V339 block.
#
# We approximate the same check faster: for each numeric feature, treat its raw value as a
# ranking score and compute ROC-AUC against the target on an EARLY slice of train, then
# again on VAL. A real, stable feature keeps the same side of 0.5 in both. A feature whose
# sign flips found a pattern that doesn't generalise forward — drop it.
#
# WHY day<=40 vs VAL (121-150), not train vs TEST: keeps TEST completely untouched, exactly
# as committed in notebook 01 — it's read once, at the very end, for the numbers we report.
# Categorical columns are skipped in this pass (lower risk for this specific inversion
# pathology, and 31 encoded categoricals vs ~400 numeric columns — scoped for time).

# %%
SCREEN_TRAIN_END = 40
screen_tr = df[df["day"] <= SCREEN_TRAIN_END]
screen_va = df[(df["day"] > TRAIN_END) & (df["day"] <= VAL_END)]

numeric_features = [c for c in features_v1 if c not in cat_cols]

flips = []
for col in numeric_features:
    tv = screen_tr[col].fillna(-999)
    vv = screen_va[col].fillna(-999)
    if tv.nunique() < 2 or vv.nunique() < 2:
        continue
    try:
        tr_auc = roc_auc_score(screen_tr["isFraud"], tv)
        va_auc = roc_auc_score(screen_va["isFraud"], vv)
    except ValueError:
        continue
    has_signal = abs(tr_auc - 0.5) > 0.02
    inverted = (tr_auc - 0.5) * (va_auc - 0.5) < 0
    if has_signal and inverted:
        flips.append((col, round(tr_auc, 4), round(va_auc, 4)))

flips_df = pd.DataFrame(flips, columns=["feature", "train_auc", "val_auc"]) \
             .sort_values("train_auc", ascending=False) if flips else pd.DataFrame(columns=["feature","train_auc","val_auc"])
print(f"{len(flips_df)} of {len(numeric_features)} numeric features show a time-consistency "
      f"sign flip (Deotte found ~5% of ~400 columns; scale check, not a hard target):")
print(flips_df.head(20).to_string(index=False))

features_v2 = [f for f in features_v1 if f not in set(flips_df["feature"])]
print(f"\nV2 feature count: {len(features_v2)} (dropped {len(features_v1)-len(features_v2)})")

# %% [markdown]
# ## V2 model — same features minus the ones that inverted

# %%
X_tr2, X_va2, X_te2 = tr[features_v2], va[features_v2], te[features_v2]

model_v2 = make_model()
model_v2.fit(X_tr2, y_tr, eval_set=[(X_va2, y_va)], verbose=100)

print("\n" + "=" * 62)
print("V2 — V1 minus time-inconsistent features")
print("=" * 62)
report("val", y_va, model_v2.predict_proba(X_va2)[:, 1])
pr_v2, roc_v2 = report("test", y_te, model_v2.predict_proba(X_te2)[:, 1])
print(f"\nV1 test was PR-AUC {pr_v1:.4f} / ROC-AUC {roc_v1:.4f} — delta: "
      f"{pr_v2-pr_v1:+.4f} PR-AUC / {roc_v2-roc_v1:+.4f} ROC-AUC")

# %% [markdown]
# ## Record this run

# %%
print(f"""
--- paste into docs/experiments.md ---
| V1 | + causal UID aggregates (expanding window) | {pr_v1:.4f} | {roc_v1:.4f} | {pr_v1-0.5486:+.4f} |
| V2 | + time-consistency screening ({len(flips_df)} features dropped) | {pr_v2:.4f} | {roc_v2:.4f} | {pr_v2-pr_v1:+.4f} |

correctness check mismatches: {len(mismatches)} / {len(sample_idx)}
causal features built: {[c for c in df.columns if c.startswith('uid_')]}
top causal feature by importance: {imp[imp.index.str.startswith('uid_')].index[0] if any(imp.index.str.startswith('uid_')) else 'none'}
time-consistency flips: {len(flips_df)} of {len(numeric_features)}
""")
