# %% [markdown]
# # 01 — EDA + Baseline
#
# Goal: verify the claims we're building on, then get an honest number on the board.
#
# This notebook is deliberately PLAIN. No UID aggregates, no feature selection, no tuning.
# It establishes the floor. Every later notebook adds exactly one thing and measures what
# that thing was worth. That delta is the experiment log.
#
# Kaggle settings: GPU T4 x2 or P100, competition data attached.

# %%
import gc
import warnings

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import average_precision_score, roc_auc_score

warnings.filterwarnings("ignore")
pd.set_option("display.max_columns", 100)

DATA = "/kaggle/input/competitions/ieee-fraud-detection"
# NOTE: Kaggle's current notebook UI nests competition data one level deeper than it
# used to (/kaggle/input/competitions/<slug>/ instead of /kaggle/input/<slug>/).
# Verified against this session. If this breaks again on a future session,
# run: os.listdir("/kaggle/input") then os.listdir("/kaggle/input/competitions")
# to find the current path, rather than guessing.

print("xgboost:", xgb.__version__)

# %% [markdown]
# ## Step 0 — Load with memory reduction
#
# WHY THIS IS STEP ZERO: 590k x 394 in float64 is ~1.9GB before we merge identity or
# create a single feature. Every groupby makes copies. Kaggle gives ~30GB and you will
# eat it faster than you expect. Downcasting typically halves memory.
#
# If this step OOMs, that is journal entry #1. It is not a hypothetical.


def reduce_mem(df, verbose=True):
    """Downcast numeric columns to the smallest dtype that holds their actual range."""
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
            # float32 is plenty. float16 loses too much precision for D/V columns.
            df[col] = df[col].astype(np.float32)
    end = df.memory_usage(deep=True).sum() / 1024**2
    if verbose:
        print(f"  memory {start:.0f}MB -> {end:.0f}MB  ({100*(start-end)/start:.0f}% saved)")
    return df


print("loading transaction...")
txn = pd.read_csv(f"{DATA}/train_transaction.csv")
txn = reduce_mem(txn)

print("loading identity...")
idt = pd.read_csv(f"{DATA}/train_identity.csv")
idt = reduce_mem(idt)

# left join: not every transaction has identity data. That missingness is itself signal.
df = txn.merge(idt, how="left", on="TransactionID")
del txn, idt
gc.collect()

print(f"\nshape: {df.shape}")
print(f"identity coverage: {df['id_01'].notna().mean():.1%} of transactions")

# %% [markdown]
# ## Step 2 — The target
#
# WHY: we need the base rate for two reasons. (1) It tells us what a random model scores
# on PR-AUC (answer: the base rate itself), which is the only honest point of comparison.
# (2) The cost model needs it.
#
# The fraud-rate-over-time plot tests Deotte's claim that this problem is NOT about time.
# We killed idea F (drift monitor) on the strength of that claim. Verify it, don't assume it.

# %%
BASE_RATE = df["isFraud"].mean()
print(f"fraud rate: {BASE_RATE:.4%}  ({df['isFraud'].sum():,} of {len(df):,})")
print(f"\n>>> A random model scores PR-AUC = {BASE_RATE:.4f}. That is the floor.")
print(">>> A model that always predicts 'not fraud' is "
      f"{1-BASE_RATE:.2%} accurate and completely useless.")

# %% [markdown]
# ## Step 3 — Time structure
#
# WHY: everything downstream (temporal split, expanding-window features, time-consistency
# screening) depends on the day axis being correct. Get it wrong here and every later
# number is silently wrong.

# %%
df["day"] = (df["TransactionDT"] // (24 * 60 * 60)).astype(np.int16)

# These two DO generalise forward in time, unlike raw day/TransactionDT.
df["hour"] = ((df["TransactionDT"] // 3600) % 24).astype(np.int8)
df["dow"] = (df["day"] % 7).astype(np.int8)

print(f"day range: {df['day'].min()} to {df['day'].max()}")

daily = df.groupby("day")["isFraud"].agg(["count", "mean"])
print("\nfraud rate by month-ish block:")
for lo in range(0, 184, 30):
    blk = daily.loc[lo:lo + 29]
    if len(blk):
        rate = (blk["count"] * blk["mean"]).sum() / blk["count"].sum()
        print(f"  day {lo:3d}-{lo+29:3d}: {blk['count'].sum():7,} txns, fraud {rate:.3%}")

print("\n>>> If that rate is roughly stable, Deotte is right that this isn't a time-drift")
print(">>> problem, and killing idea F was correct. If it swings, revisit.")

# %% [markdown]
# ## Step 4 — Reconstruct UID  *** THE CRITICAL STEP ***
#
# WHY THIS MATTERS MORE THAN ANYTHING ELSE HERE:
#
# The competition host defined labels at the CARD level, not the transaction level:
#   "define reported chargeback on the card as fraud transaction (isFraud=1) and
#    transactions posterior to it ... as fraud too."
#
# Deotte reports that of 73,838 clients with 2+ transactions:
#   96.9% are always isFraud=0, 2.9% always isFraud=1, 0.2% mixed.
#
# Our component C (card-level risk state) and much of component B depend on being able to
# reconstruct client identity. If our numbers come out meaningfully different, our UID
# definition is WRONG and we need to know NOW, not later with a pipeline on top of it.
#
# D1 = "days since this card began". So day - D1 = the day the card started.
# card1 + addr1 + that start-day identifies a client.
#
# NOTE: D1n is causally legal. D1 is known at transaction time. Nothing here peeks forward.

# %%
df["D1n"] = df["day"] - df["D1"]
df["uid"] = (
    df["card1"].astype(str) + "_"
    + df["addr1"].astype(str) + "_"
    + df["D1n"].astype(str)
)

g = df.groupby("uid")["isFraud"].agg(["count", "mean"])
multi = g[g["count"] >= 2]

pure0 = (multi["mean"] == 0).mean()
pure1 = (multi["mean"] == 1).mean()
mixed = 1 - pure0 - pure1

print(f"total clients (UIDs):        {len(g):,}")
print(f"clients with 2+ txns:        {len(multi):,}   (target: 73,838)")
print(f"  rows they account for:     {multi['count'].sum():,}   (target: 280,829 = 50%)")
print()
print(f"  all isFraud=0:  {pure0:6.2%}   (target: 96.9%)")
print(f"  all isFraud=1:  {pure1:6.2%}   (target:  2.9%)")
print(f"  mixed:          {mixed:6.2%}   (target:  0.2%)")

ok = abs(len(multi) - 73838) < 2000 and abs(pure1 - 0.029) < 0.01
print(f"\n>>> RECONSTRUCTION {'VERIFIED' if ok else 'FAILED — STOP AND INVESTIGATE'}")

# %% [markdown]
# ## Step 5 — The cold-start finding
#
# WHY: if 73,838 multi-transaction clients account for only ~50% of rows, the other ~50%
# are SINGLE-transaction clients. For those, card-history features are worthless — there
# is no history.
#
# That means roughly half our traffic has a cold-start problem. This is not a flaw in the
# plan, it is a FINDING, and it belongs in the honest exception list. It also tells us the
# system needs two paths: history-rich and cold-start.

# %%
single_rows = len(df) - multi["count"].sum()
print(f"single-transaction clients account for {single_rows:,} rows "
      f"({single_rows/len(df):.1%} of traffic)")

single_uids = g[g["count"] == 1]
print(f"their fraud rate:     {single_uids['mean'].mean():.3%}")
print(f"multi-txn fraud rate: {(multi['count']*multi['mean']).sum()/multi['count'].sum():.3%}")
print("\n>>> If these differ a lot, cold-start traffic needs its own policy band.")

# %% [markdown]
# ## Step 6 — Client overlap across OUR split
#
# WHY: Deotte found 68.2% of private-test clients were unseen in train. We need OUR number
# for OUR split, because it sets the ceiling on how much card-history features can help.
# If most clients are unseen, aggregate group features matter far more than per-client lookups.

# %%
TRAIN_END, VAL_END = 120, 150

uid_train = set(df.loc[df["day"] <= TRAIN_END, "uid"])
later = df.loc[df["day"] > TRAIN_END]
seen = later["uid"].isin(uid_train).mean()

print(f"clients in day>{TRAIN_END} that were ALSO in day<={TRAIN_END}: {seen:.1%}")
print(f"unseen: {1-seen:.1%}   (Deotte's private-test figure was 68.2% unseen)")

# %% [markdown]
# ## Step 7 — Amount distribution
#
# WHY THIS IS NOT OPTIONAL: our cost model is
#     expected_loss = P(fraud) x (amount + chargeback_fee)
# The amount distribution is a DIRECT INPUT to the objective function. If fraud concentrates
# at high amounts, the cost-optimal threshold shifts substantially versus a flat assumption.
# This step is what makes the cost curve real rather than decorative.

# %%
amt = df.groupby("isFraud")["TransactionAmt"].describe()[
    ["count", "mean", "50%", "75%", "max"]
]
print(amt.round(2))

fraud_amt = df.loc[df.isFraud == 1, "TransactionAmt"].sum()
total_amt = df["TransactionAmt"].sum()
print(f"\nfraud is {BASE_RATE:.2%} of transactions but "
      f"{fraud_amt/total_amt:.2%} of rupee volume")
print(">>> That ratio is the single most important number for the cost model.")

# known signal: non-round amounts often indicate foreign currency conversion
df["amt_decimal"] = ((df["TransactionAmt"] - df["TransactionAmt"].round(0)).abs() > 0.01)
print(f"\nnon-round amounts: fraud rate {df.groupby('amt_decimal')['isFraud'].mean().to_dict()}")

# %% [markdown]
# ## Step 8 — Key categoricals (sanity check on reality)
#
# WHY: fraud should skew toward credit over debit, toward disposable email domains, toward
# certain product codes. If it doesn't, something is wrong with our load.
#
# These are also the NARRATABLE features — what the LLM will eventually be describing.

# %%
for col in ["ProductCD", "card4", "card6", "DeviceType"]:
    if col in df.columns:
        s = df.groupby(col)["isFraud"].agg(["count", "mean"]).sort_values("mean", ascending=False)
        print(f"\n{col}:")
        print(s.head(8).to_string())

top_email = df["P_emaildomain"].value_counts().head(10).index
print("\nP_emaildomain (top 10 by volume):")
print(df[df.P_emaildomain.isin(top_email)]
      .groupby("P_emaildomain")["isFraud"].agg(["count", "mean"])
      .sort_values("mean", ascending=False).to_string())

# %% [markdown]
# ## Temporal split
#
# WHY WE CANNOT USE test_transaction.csv: it has no labels (that's what sample_submission.csv
# is for). Kaggle keeps them hidden. Razorpay's required "held-out test set" must therefore be
# carved out of train_transaction.csv ourselves.
#
#   day 0 -------- 120 ---- 150 ---- 183
#      TRAIN       |  VAL   |  TEST
#
# TEST is touched ONCE, at the very end, for the reported numbers. Not for tuning.

# %%
tr = df[df["day"] <= TRAIN_END]
va = df[(df["day"] > TRAIN_END) & (df["day"] <= VAL_END)]
te = df[df["day"] > VAL_END]

for name, part in [("train", tr), ("val", va), ("test", te)]:
    print(f"{name:5s}  {len(part):7,} rows  fraud {part['isFraud'].mean():.3%}  "
          f"days {part['day'].min()}-{part['day'].max()}")

# %% [markdown]
# ## Feature prep — deliberately minimal
#
# DROPPED AND WHY:
#   TransactionID  - an identifier, no signal
#   TransactionDT  - raw timestamp. The model would learn "day > 150 doesn't exist in
#                    training" and generalise terribly forward in time. hour/dow keep the
#                    useful cyclical part.
#   day, D1n       - same reason
#   uid            - Deotte is explicit: do NOT use UID directly. ~68% of later clients are
#                    unseen, so the model would memorise identities that never recur.
#                    We keep the column for ANALYSIS, never as a feature.
#
# CATEGORICALS: label-encoded, FIT ON TRAIN ONLY. Unseen categories -> -1.
# Fitting the encoder on the full dataset would mean knowing which categories will exist in
# the future. That's a mild leak, and given this project is literally about causal honesty,
# we don't take it. Costs nothing.

# %%
DROP = ["TransactionID", "TransactionDT", "isFraud", "day", "D1n", "uid", "amt_decimal"]
features = [c for c in df.columns if c not in DROP]

cat_cols = [c for c in features if df[c].dtype == object]
print(f"{len(features)} features, {len(cat_cols)} categorical")

for c in cat_cols:
    mapping = {v: i for i, v in enumerate(tr[c].dropna().unique())}
    for part in (tr, va, te):
        part[c] = part[c].map(mapping).fillna(-1).astype(np.int32)

X_tr, y_tr = tr[features], tr["isFraud"]
X_va, y_va = va[features], va["isFraud"]
X_te, y_te = te[features], te["isFraud"]

# %% [markdown]
# ## Baseline model
#
# TWO DELIBERATE CHOICES, BOTH WORTH DEFENDING:
#
# 1. eval_metric = "aucpr", NOT "auc".
#    At 3.5% positives, ROC-AUC is flattered by the huge negative class and looks good even
#    for mediocre models. Precision-Recall AUC focuses on the rare positive class. We report
#    both (Kaggle scored on ROC-AUC) but we TUNE on PR-AUC.
#
# 2. NO scale_pos_weight, despite every tutorial recommending it for imbalance.
#    scale_pos_weight deliberately distorts the output distribution to push predictions
#    upward. That directly fights our calibration step later — and our entire cost model
#    requires REAL probabilities, because expected_loss = P(fraud) x amount. If P isn't a
#    genuine probability, every rupee figure we produce is wrong.
#    We handle imbalance in the METRIC (PR-AUC) and fix the probabilities with isotonic
#    calibration afterwards. This is a real fork in the road; note it for the panel.

# %%
model = xgb.XGBClassifier(
    n_estimators=2000,
    learning_rate=0.05,
    max_depth=8,
    subsample=0.8,
    colsample_bytree=0.5,
    min_child_weight=4,
    reg_lambda=1.0,
    tree_method="hist",
    device="cuda",
    eval_metric="aucpr",
    early_stopping_rounds=100,
    random_state=42,
)

model.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], verbose=100)
print(f"\nbest iteration: {model.best_iteration}")

# %% [markdown]
# ## Results

# %%
def report(name, y_true, y_prob):
    pr = average_precision_score(y_true, y_prob)
    roc = roc_auc_score(y_true, y_prob)
    lift = pr / y_true.mean()
    print(f"{name:6s}  PR-AUC {pr:.4f}   ROC-AUC {roc:.4f}   "
          f"lift over random {lift:.1f}x")
    return pr, roc


print("=" * 62)
print("V0 BASELINE — raw features, temporal split, no tuning, no UID work")
print("=" * 62)
p_va = model.predict_proba(X_va)[:, 1]
p_te = model.predict_proba(X_te)[:, 1]
report("val", y_va, p_va)
pr_te, roc_te = report("test", y_te, p_te)
print(f"\nrandom-model PR-AUC would be {y_te.mean():.4f}")

# %% [markdown]
# ## What the top features tell us
#
# WHY LOOK: if one feature dominates everything, we probably have leakage. This is a
# debugging instrument before it is ever a presentation one. (SHAP comes later and does this
# properly per-prediction; gain importance is the cheap global version.)

# %%
imp = (pd.Series(model.feature_importances_, index=features)
       .sort_values(ascending=False).head(25))
print(imp.to_string())

# %% [markdown]
# ## Record this run
#
# Paste the numbers into docs/experiments.md. Every subsequent notebook changes ONE thing
# and we measure the delta. That ladder is the story we tell the panel.

# %%
print(f"""
--- paste into docs/experiments.md ---
| V0 | baseline, raw features | {pr_te:.4f} | {roc_te:.4f} | — |
UID verified: {ok}
unseen clients after day {TRAIN_END}: {1-seen:.1%}
fraud = {BASE_RATE:.2%} of txns, {fraud_amt/total_amt:.2%} of volume
best_iteration: {model.best_iteration}
""")
