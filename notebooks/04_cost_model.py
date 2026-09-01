# %% [markdown]
# # 04 — The Cost Model: allow / step-up / block
#
# This is the actual thesis of the project. Everything through V4/V5 built a good
# RANKING of risk. This notebook turns that ranking into a DECISION, priced in rupees.
#
# Self-contained — rebuilds through V4's best hyperparameters directly (the Optuna search
# itself is not re-run; its winning params are hardcoded below from notebook 03's result).

# %%
import gc
import warnings

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score

warnings.filterwarnings("ignore")
np.seterr(invalid="ignore", divide="ignore")
pd.set_option("display.max_columns", 100)

DATA = "/kaggle/input/competitions/ieee-fraud-detection"

# %% [markdown]
# ## Rebuild through V4/V5 (condensed — see notebooks 01-03 for the reasoning)

# %%
def reduce_mem(df):
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
df["uid"] = (df["card1"].astype(str) + "_" + df["addr1"].astype(str) + "_"
             + df["D1n"].astype(str) + "_" + df["D4n"].astype(str))
df["coarse_uid"] = df["card1"].astype(str) + "_" + df["addr1"].astype(str) + "_" + df["D1n"].astype(str)

TRAIN_END, VAL_END = 120, 150

order = df.sort_values(["uid", "TransactionDT", "TransactionID"]).index
uid_ord = df.loc[order, "uid"]
df["uid_txn_num"] = df.loc[order].groupby("uid", sort=False).cumcount()
amt_f = df.loc[order, "TransactionAmt"].astype(np.float64)
cum_sum_incl = amt_f.groupby(uid_ord, sort=False).cumsum()
cum_sumsq_incl = (amt_f ** 2).groupby(uid_ord, sort=False).cumsum()
df["uid_amt_sum_prior"] = cum_sum_incl.groupby(uid_ord, sort=False).shift(1)
df["uid_amt_sumsq_prior"] = cum_sumsq_incl.groupby(uid_ord, sort=False).shift(1)
n_prior = df["uid_txn_num"].astype(np.float64)
df["uid_amt_mean_prior"] = df["uid_amt_sum_prior"] / n_prior
var_prior = (df["uid_amt_sumsq_prior"] / n_prior) - df["uid_amt_mean_prior"] ** 2
df["uid_amt_std_prior"] = np.sqrt(var_prior.clip(lower=0))
df["uid_time_since_prev"] = df.loc[order].groupby("uid", sort=False)["TransactionDT"].diff()
for col, out in [("DeviceInfo", "uid_device_seen_before"), ("P_emaildomain", "uid_email_seen_before")]:
    if col in df.columns:
        seen = df.loc[order].groupby(["uid", col], sort=False, dropna=False).cumcount()
        df[out] = (seen > 0).astype(np.int8)
order2 = df.sort_values(["coarse_uid", "TransactionDT", "TransactionID"]).index
coarse_ord = df.loc[order2, "coarse_uid"]
d15n_std_incl = (df.loc[order2, "D15n"].groupby(coarse_ord, sort=False)
                 .transform(lambda s: s.expanding().std()))
df["uid_ambiguity_std_prior"] = d15n_std_incl.groupby(coarse_ord, sort=False).shift(1)

tr = df[df["day"] <= TRAIN_END]
va = df[(df["day"] > TRAIN_END) & (df["day"] <= VAL_END)]
te = df[df["day"] > VAL_END]

DROP = ["TransactionID", "TransactionDT", "isFraud", "day",
        "D1n", "D4n", "D10n", "D15n", "uid", "coarse_uid", "uid_txn_num"]
features_v2 = [c for c in df.columns if c not in DROP]
cat_cols = [c for c in features_v2 if df[c].dtype == object]
for c in cat_cols:
    mapping = {v: i for i, v in enumerate(tr[c].dropna().unique())}
    for part in (tr, va, te):
        part[c] = part[c].map(mapping).fillna(-1).astype(np.int32)

# V3: same correlation-based V-column reduction as notebook 03
import re
from collections import defaultdict

v_cols = [c for c in features_v2 if re.match(r"^V\d+$", c)]
nan_pattern = tr[v_cols].isna()
pattern_key = {c: hash(nan_pattern[c].values.tobytes()) for c in v_cols}
groups = defaultdict(list)
for c, k in pattern_key.items():
    groups[k].append(c)


def reduce_group(df_train, cols, threshold=0.95):
    if len(cols) <= 1:
        return list(cols)
    corr = df_train[cols].corr().abs()
    keep = []
    for c in cols:
        if all(corr.loc[c, k] < threshold for k in keep):
            keep.append(c)
    return keep


kept_v_cols = []
for k, cols in groups.items():
    kept_v_cols.extend(reduce_group(tr, cols, threshold=0.95))
features_v3 = [c for c in features_v2 if c not in v_cols] + kept_v_cols

y_tr, y_va, y_te = tr["isFraud"], va["isFraud"], te["isFraud"]
X_tr3, X_va3, X_te3 = tr[features_v3], va[features_v3], te[features_v3]

# V4: best params from notebook 03's 60-trial Optuna search (hardcoded — not re-searching)
BEST_PARAMS = {
    "max_depth": 10, "learning_rate": 0.04920458917496633, "subsample": 0.815514794624691,
    "colsample_bytree": 0.887187653957725, "min_child_weight": 2,
    "reg_lambda": 0.781547599058096, "reg_alpha": 0.023223970495058505,
}
model = xgb.XGBClassifier(
    n_estimators=2000, tree_method="hist", device="cuda",
    eval_metric="aucpr", early_stopping_rounds=100, random_state=42, **BEST_PARAMS,
)
model.fit(X_tr3, y_tr, eval_set=[(X_va3, y_va)], verbose=200)

raw_va = model.predict_proba(X_va3)[:, 1]
raw_te = model.predict_proba(X_te3)[:, 1]

# V5: Platt calibration, fit on VAL, applied to TEST (matches notebook 03's shipped choice)
platt = LogisticRegression()
platt.fit(raw_va.reshape(-1, 1), y_va)
cal_va = platt.predict_proba(raw_va.reshape(-1, 1))[:, 1]
cal_te = platt.predict_proba(raw_te.reshape(-1, 1))[:, 1]

print(f"\nreproduced test PR-AUC: {average_precision_score(y_te, cal_te):.4f} "
      f"(notebook 03 reported 0.5514 — should match closely)")

# %% [markdown]
# ## The currency problem, caught before it caused a silent distortion
#
# IEEE-CIS is a US dataset. `TransactionAmt` is dollar-scale, not real rupees. Feeding it
# directly into a cost model with an Indian ₹500 chargeback fee would make the fee absurdly
# large relative to typical transaction size (median legit transaction ≈ 68 raw units —
# treated as ₹68, a ₹500 fee would dwarf it and make "block everything" trivially optimal).
# That failure mode was anticipated (G6 gate: "does the cost curve even have
# an interior minimum") — catching the actual mechanism here, before running anything,
# beats discovering it empirically as a mystery later.
#
# Fix: convert at an illustrative rate. Only the ORDER OF MAGNITUDE matters for the
# methodology — this is a modeling simplification for a US dataset applied to an Indian
# cost framework, and it's stated plainly, not hidden.

# %%
USD_TO_INR = 95.41  # live, dated quote, Google/Morningstar — was 83.0 (a rough
# placeholder, not checked) in the first run. Corrected once a real dated source was
# available, same standard as the Razorpay MDR number above: cite it, don't guess it.
# Note CHARGEBACK_FEE (fixed Rs) does NOT scale with this rate the way amount-proportional
# terms (margin, MDR, LTV penalty) do — so this correction has a real, if modest, effect on
# the reported policy and lift figures, not just a cosmetic one.

amt_va_inr = va["TransactionAmt"].values.astype(np.float64) * USD_TO_INR
amt_te_inr = te["TransactionAmt"].values.astype(np.float64) * USD_TO_INR

print(f"median legit transaction: Rs {np.median(amt_te_inr[y_te.values == 0]):,.0f}")
print(f"median fraud transaction: Rs {np.median(amt_te_inr[y_te.values == 1]):,.0f}")

# %% [markdown]
# ## Cost model parameters — sourced, not invented
#
# Every number below is a real, cited assumption, not a fit to this data. The panel will
# ask where these came from — here's the answer, in the code, not just in a slide.
#
# | Parameter | Value | Source |
# |---|---|---|
# | Chargeback fee | Rs 500 | Razorpay's own disclosed dispute-fee range is Rs 200-600; midpoint |
# | Payment processing fee (MDR) | 2.36% of amount | **Razorpay's own pricing page**: uniform 2% platform fee on every successful domestic transaction (cards/UPI/netbanking/wallets alike) + 18% GST on that fee = 2.36% effective. Charged at processing time; NOT refunded when a sale later reverses via chargeback — a genuinely separate loss from the chargeback fee itself, not double-counting. |
# | Merchant margin | 20% | Blended e-commerce assumption (5%-50%+ by vertical) — swept below |
# | Step-up stops fraud | 60% | 3D Secure fraud-reduction studies cite 40-70%; midpoint |
# | Genuine customer drop-off at step-up | 15% | Checkout-friction studies cite 17-21% for FULL checkout complexity; a single OTP prompt is less friction than that, so set conservatively lower |
# | LTV penalty, wrongly blocked customer | 3x the lost margin | Most speculative parameter here — flagged explicitly, swept below |
#
# MDR is a fixed, disclosed platform fee, not a business assumption — so unlike margin and
# chargeback fee (which vary by merchant/vertical and get swept in the sensitivity analysis
# below), it isn't swept. It only applies when a transaction actually PROCESSES: paid on
# every allow, paid on a step-up that completes (genuine finishes, or fraud gets through
# anyway) — never paid on a block or a step-up abandonment, because no transaction happened.

# %%
CHARGEBACK_FEE = 500.0
MDR_RATE = 0.02 * 1.18  # 2% platform fee + 18% GST, Razorpay's own disclosed pricing
MARGIN = 0.20
P_STOP = 0.60
P_DROPOFF = 0.15
LTV_MULTIPLIER = 3.0


def value_allow(p, amt):
    """Expected Rs value of allowing a transaction. MDR is paid on EVERY processed
    transaction regardless of outcome — genuine or fraud, Razorpay keeps its fee."""
    return (1 - p) * (MARGIN * amt - MDR_RATE * amt) - p * (amt + CHARGEBACK_FEE + MDR_RATE * amt)


def value_stepup(p, amt):
    """Genuine + completes (1-p_dropoff): margin earned, MDR paid (transaction processed).
    Genuine + abandons (p_dropoff): Rs0 — no transaction, no MDR either.
    Fraud + gets through (1-p_stop): full loss + MDR paid (it did process, then reversed).
    Fraud + stopped (p_stop): Rs0 — step-up prevented processing, no MDR."""
    genuine_value = (1 - p) * (1 - P_DROPOFF) * (MARGIN * amt - MDR_RATE * amt)
    fraud_cost = p * (1 - P_STOP) * (amt + CHARGEBACK_FEE + MDR_RATE * amt)
    return genuine_value - fraud_cost


def value_block(p, amt):
    """Fraud blocked: Rs0 (correct call, no formula needed). Genuine blocked: lose the
    margin AND the LTV penalty — this branch is where a wrong block actually costs."""
    genuine_penalty = MARGIN * amt * (1 + LTV_MULTIPLIER)
    return -(1 - p) * genuine_penalty


# %% [markdown]
# ## The classic 2-way cost curve (allow vs block, single threshold) — for the pitch video
#
# Simpler than the real 3-way policy below, but this is the intuitive visual: sweep every
# threshold, plot total REALIZED value (computed from actual test-month labels, not the
# expected-value formula — this is what actually happened, not a projection), find the
# minimum-loss point.
#
# GATE CHECK (G6): does the minimum land INSIDE the range, not at threshold 0 or 1?
# If it's at an edge, the whole "choose a threshold" framing has no story — pivot to the
# sensitivity map instead. Checked explicitly below, not assumed.

# %%
thresholds = np.linspace(0.001, 0.999, 400)
y_te_arr = y_te.values.astype(int)
curve_values = []

for t in thresholds:
    blocked = cal_te >= t
    val = np.where(
        blocked,
        np.where(y_te_arr == 1, 0.0, -(MARGIN * amt_te_inr * (1 + LTV_MULTIPLIER))),
        np.where(y_te_arr == 1, -(amt_te_inr + CHARGEBACK_FEE + MDR_RATE * amt_te_inr),
                 MARGIN * amt_te_inr - MDR_RATE * amt_te_inr),
    )
    curve_values.append(val.sum())

curve_values = np.array(curve_values)
best_idx = curve_values.argmax()
best_threshold = thresholds[best_idx]

print(f"best single threshold: {best_threshold:.3f}")
print(f"value at best threshold: Rs {curve_values[best_idx]:,.0f}")
print(f"value at threshold=0 (block everything): Rs {curve_values[0]:,.0f}")
print(f"value at threshold=1 (allow everything):  Rs {curve_values[-1]:,.0f}")

is_interior = 0.02 < best_threshold < 0.98
print(f"\n>>> G6 CHECK: minimum is {'INTERIOR — gate PASSED' if is_interior else 'AT AN EDGE — gate FAILED, pivot to sensitivity map'}")

import matplotlib.pyplot as plt

fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(thresholds, curve_values / 1e5, color="C0")
ax.axvline(best_threshold, color="C1", linestyle="--", label=f"optimal threshold = {best_threshold:.3f}")
ax.axvline(0.5, color="gray", linestyle=":", label="naive default (0.5)")
ax.set_xlabel("block threshold (probability)")
ax.set_ylabel("total portfolio value (Rs, lakhs)")
ax.set_title("Cost curve — test month, allow/block only")
ax.legend()
plt.tight_layout()
plt.savefig("cost_curve.png", dpi=120)
plt.show()
print("saved: cost_curve.png")

# %% [markdown]
# ## The real policy: per-transaction optimal action (allow / step-up / block)
#
# WHY NOT A SINGLE GLOBAL THRESHOLD FOR THE SHIPPED POLICY: the cost formulas above all
# scale with `amt` — a high-value transaction has more at stake in both directions than a
# low-value one, so the cost-optimal action genuinely depends on amount, not just
# probability. A single global cutoff (as used for the illustrative curve above) ignores
# this. The actual policy computes the expected value of ALL THREE actions for EVERY
# transaction and picks the best one — the correct decision given this cost model, not an
# approximation of it.

# %%
p = cal_te
amt = amt_te_inr

va_allow = value_allow(p, amt)
va_stepup = value_stepup(p, amt)
va_block = value_block(p, amt)

values = np.vstack([va_allow, va_stepup, va_block]).T
actions = np.argmax(values, axis=1)  # 0=allow, 1=stepup, 2=block
action_names = np.array(["allow", "step-up", "block"])

counts = pd.Series(action_names[actions]).value_counts()
print("policy action distribution on the test month:")
print(counts)
print(f"\nas % of traffic:\n{(counts / len(actions) * 100).round(1)}")

# %% [markdown]
# ## Measured result — Arbiter vs two baselines, on the same batch
#
# IMPORTANT HONESTY NOTE: allow/block outcomes below are computed from ACTUAL test-month
# labels — these are hard, verifiable numbers, not projections. Step-up outcomes CANNOT be
# computed this way — we don't have real step-up interaction data (whether a specific
# fraudster was actually stopped, or a specific genuine customer actually abandoned). Those
# use the cited POPULATION RATES (P_STOP, P_DROPOFF) as a modeled expectation. This
# distinction is stated explicitly here and belongs in the honest exception list, not
# glossed over.

# %%
def realized_value(action, is_fraud, amt, margin=MARGIN, fee=CHARGEBACK_FEE,
                    p_stop=P_STOP, p_dropoff=P_DROPOFF, ltv_mult=LTV_MULTIPLIER,
                    mdr=MDR_RATE):
    """Parameterized so sensitivity sweeps evaluate value under the SAME assumptions used
    to choose the action, not the global defaults. Passing no args reproduces the
    headline result exactly. MDR applies only where a transaction actually processes:
    every allow, and a step-up that completes (genuine finishes OR fraud gets through) —
    never a block, never a step-up abandonment."""
    out = np.zeros_like(amt)
    m = action == 0  # allow — hard, from actual labels
    out[m] = np.where(is_fraud[m] == 1, -(amt[m] + fee + mdr * amt[m]), margin * amt[m] - mdr * amt[m])
    m = action == 1  # step-up — MODELED, using population rates (see note above)
    out[m] = np.where(
        is_fraud[m] == 1,
        -(1 - p_stop) * (amt[m] + fee + mdr * amt[m]),
        (1 - p_dropoff) * (margin * amt[m] - mdr * amt[m]),
    )
    m = action == 2  # block — hard, from actual labels. No MDR: nothing processed.
    out[m] = np.where(is_fraud[m] == 1, 0.0, -(margin * amt[m] * (1 + ltv_mult)))
    return out


arbiter_value = realized_value(actions, y_te_arr, amt).sum()
hard_mask = actions != 1  # allow + block only — fully verifiable portion
arbiter_hard_value = realized_value(actions, y_te_arr, amt)[hard_mask].sum()
arbiter_modeled_value = arbiter_value - arbiter_hard_value

# Baseline 1: no fraud system at all (always allow) — MDR still applies, every transaction processes
baseline_noop = np.where(
    y_te_arr == 1, -(amt + CHARGEBACK_FEE + MDR_RATE * amt), MARGIN * amt - MDR_RATE * amt
).sum()

# Baseline 2: naive industry-default — single 0.5 threshold, allow/block only, no step-up
naive_block = p >= 0.5
baseline_naive = np.where(
    naive_block,
    np.where(y_te_arr == 1, 0.0, -(MARGIN * amt * (1 + LTV_MULTIPLIER))),
    np.where(y_te_arr == 1, -(amt + CHARGEBACK_FEE + MDR_RATE * amt), MARGIN * amt - MDR_RATE * amt),
).sum()

print(f"{'policy':22s} {'total value (Rs)':>18s}")
print(f"{'no fraud system':22s} {baseline_noop:18,.0f}")
print(f"{'naive 0.5 threshold':22s} {baseline_naive:18,.0f}")
print(f"{'Arbiter (this system)':22s} {arbiter_value:18,.0f}")
print(f"\n  of which HARD (allow/block, from real labels):    Rs {arbiter_hard_value:,.0f}")
print(f"  of which MODELED (step-up, from population rates): Rs {arbiter_modeled_value:,.0f}")
print(f"\nlift vs no system:    Rs {arbiter_value - baseline_noop:,.0f}")
print(f"lift vs naive 0.5:    Rs {arbiter_value - baseline_naive:,.0f}")

# %% [markdown]
# ## Illustrative policy bands — "what does this actually mean for one transaction"
#
# The real policy is per-transaction (varies with amount). For the pitch video / README,
# show the effective probability bands at a few representative amounts — this is what a
# merchant or panelist will actually want to see.

# %%
for demo_amt in [500, 2000, 10000, 50000]:
    ps = np.linspace(0.001, 0.999, 500)
    vals = np.vstack([value_allow(ps, demo_amt), value_stepup(ps, demo_amt), value_block(ps, demo_amt)]).T
    acts = np.argmax(vals, axis=1)
    allow_to = ps[acts == 0].max() if (acts == 0).any() else 0
    block_from = ps[acts == 2].min() if (acts == 2).any() else 1
    print(f"Rs {demo_amt:>6,}: allow below p={allow_to:.3f}  |  step-up {allow_to:.3f}-{block_from:.3f}  |  block above p={block_from:.3f}")

# %% [markdown]
# ## Sensitivity analysis — required, not optional
#
# Never report one number as if the parameters were certain. Sweep the two most uncertain
# ones (margin, chargeback fee) and show how total value moves — this pre-empts "where did
# your Rs 500 come from" and is the honest fallback if G6 had failed.

# %%
margins = [0.05, 0.10, 0.15, 0.20, 0.30, 0.40, 0.50]
fees = [200, 350, 500, 600, 1000]

results = []
for m in margins:
    for f in fees:
        va_a = (1 - p) * (m * amt - MDR_RATE * amt) - p * (amt + f + MDR_RATE * amt)
        va_s = (1 - p) * (1 - P_DROPOFF) * (m * amt - MDR_RATE * amt) - p * (1 - P_STOP) * (amt + f + MDR_RATE * amt)
        va_b = -(1 - p) * (m * amt * (1 + LTV_MULTIPLIER))
        v = np.vstack([va_a, va_s, va_b]).T
        act = np.argmax(v, axis=1)
        total = realized_value(act, y_te_arr, amt, margin=m, fee=f).sum()
        results.append({"margin": m, "fee": f, "total_value": total,
                         "pct_blocked": (act == 2).mean() * 100})

sens_df = pd.DataFrame(results)
pivot = sens_df.pivot(index="margin", columns="fee", values="total_value")
print("total portfolio value (Rs) by margin x chargeback fee assumption:")
print(pivot.round(0).to_string())

fig, ax = plt.subplots(figsize=(7, 5))
im = ax.imshow(pivot.values, aspect="auto", cmap="RdYlGn", origin="lower")
ax.set_xticks(range(len(fees))); ax.set_xticklabels(fees)
ax.set_yticks(range(len(margins))); ax.set_yticklabels(margins)
ax.set_xlabel("chargeback fee (Rs)"); ax.set_ylabel("merchant margin")
ax.set_title("Sensitivity: total value across cost assumptions")
ax.scatter([fees.index(500)], [margins.index(0.20)], color="black", marker="*", s=200, label="our assumption")
ax.legend()
plt.colorbar(im, label="total value (Rs)")
plt.tight_layout()
plt.savefig("sensitivity_map.png", dpi=120)
plt.show()
print("saved: sensitivity_map.png")

# %% [markdown]
# ## Record this run

# %%
print(f"""
--- Cost model summary ---
G6 (interior minimum): {'PASSED' if is_interior else 'FAILED'} — optimal single-threshold at {best_threshold:.3f}

Policy mix: {dict(counts)}

Total value (Rs) on test month:
  no fraud system:      {baseline_noop:,.0f}
  naive 0.5 threshold:   {baseline_naive:,.0f}
  Arbiter:               {arbiter_value:,.0f}
  lift vs no system:     {arbiter_value - baseline_noop:,.0f}
  lift vs naive:         {arbiter_value - baseline_naive:,.0f}

Cost parameters: fee=Rs{CHARGEBACK_FEE}, margin={MARGIN}, p_stop={P_STOP}, p_dropoff={P_DROPOFF}, ltv_mult={LTV_MULTIPLIER}
USD->INR conversion: {USD_TO_INR}
""")

# %% [markdown]
# ## Engine-build addendum — export portable artifacts for the local decision engine
#
# WHY THIS EXISTS: CLAUDE.md is explicit that this is where the project STOPS being a
# notebook — the decision engine has to load a saved artifact and run on plain CPU, no GPU,
# no training, from a fresh Python process on the local machine. This cell exports exactly
# what `src/` needs to do that: the trained model, the calibrator, the feature list, the
# categorical vocabularies, and a small sample of REAL held-out transactions to test against.
#
# Run this ONCE, now, in this same session (reuses the already-trained `model` and `platt` —
# no retraining). Then download the output files from Kaggle's Output panel and place them
# in the local repo's `artifacts/` folder.

# %%
import json
import os

os.makedirs("/kaggle/working/artifacts", exist_ok=True)

# 1. The trained model — XGBoost's native format is portable and CPU-loadable regardless
# of what device it was trained on. No GPU needed to load or predict with it later.
model.save_model("/kaggle/working/artifacts/model.json")

# 2. The Platt calibrator — just two numbers. Storing them directly (not pickling the
# sklearn object) avoids any sklearn-version fragility and keeps the calibration
# transparent/auditable: anyone can read model.json and verify the sigmoid by hand.
with open("/kaggle/working/artifacts/calibrator.json", "w") as f:
    json.dump({"coef": float(platt.coef_[0][0]), "intercept": float(platt.intercept_[0])}, f)

print(f"model + calibrator saved. calibrator: coef={platt.coef_[0][0]:.6f}, intercept={platt.intercept_[0]:.6f}")

# %% [markdown]
# ## Recover categorical vocabularies
#
# WHY A FRESH TARGETED RELOAD, NOT REUSING `tr`: the categorical-encoding cell earlier in
# this notebook OVERWROTE tr[c]/va[c]/te[c] in place with integer codes — the original
# string values are gone from memory. Re-running that cell again now would build mappings
# from the ALREADY-ENCODED integers, silently producing garbage. Recovering the true
# string->int vocabulary requires reading the raw strings again — but only the ~31
# categorical columns (usecols), not the full 434-column file, so this is fast, not a full
# reload, and touches nothing already in memory.

# %%
narrow_txn = pd.read_csv(f"{DATA}/train_transaction.csv",
                          usecols=lambda c: c in cat_cols or c in ("TransactionID", "TransactionDT"))
narrow_idt = pd.read_csv(f"{DATA}/train_identity.csv",
                          usecols=lambda c: c in cat_cols or c == "TransactionID")
narrow_raw = narrow_txn.merge(narrow_idt, how="left", on="TransactionID")
narrow_raw["day"] = (narrow_raw["TransactionDT"] // 86400).astype(int)
narrow_train = narrow_raw[narrow_raw["day"] <= TRAIN_END]

cat_mappings = {}
for c in cat_cols:
    cat_mappings[c] = {str(v): i for i, v in enumerate(narrow_train[c].dropna().unique())}

manifest = {
    "features": features_v3,
    "categorical_columns": cat_cols,
    "categorical_mappings": cat_mappings,
    "kept_v_cols": kept_v_cols,
    "train_end_day": int(TRAIN_END),
    # Recorded so src/model.py can detect a version mismatch loudly instead of silently
    # scoring wrong — see the journal entry: xgboost 3.0.0 vs 3.2.0 (this training
    # version) gave a 23x different RAW probability (0.169 vs 0.007) on byte-identical
    # input. That is not floating-point noise, it's a real incompatibility, and it produced
    # no error or warning of any kind on its own.
    "xgboost_version": xgb.__version__,
}
with open("/kaggle/working/artifacts/feature_manifest.json", "w") as f:
    json.dump(manifest, f)

print(f"feature manifest saved: {len(features_v3)} features, {len(cat_cols)} categorical vocabularies")

# %% [markdown]
# ## Export sample transactions — REAL held-out data, for testing the engine's plumbing
#
# RAW columns only (excludes everything WE engineered: day/hour/dow, D1n/D4n/D10n/D15n,
# uid/coarse_uid, every uid_* causal feature). The local engine has to compute all of that
# itself from raw input — exporting it pre-computed would test nothing.
#
# Deliberately includes one client's FIRST TWO transactions in the test month (found by
# uid) — this is what proves the client-history store actually accumulates state between
# calls, not just that decide() runs once.

# %%
ENGINEERED_COLS = {"day", "hour", "dow", "D1n", "D4n", "D10n", "D15n", "uid", "coarse_uid"}
ENGINEERED_COLS |= {c for c in df.columns if c.startswith("uid_")}
RAW_COLS = [c for c in df.columns if c not in ENGINEERED_COLS]

repeat_uid = te.groupby("uid").filter(lambda g: len(g) >= 2)["uid"].iloc[0]
repeat_rows = te[te["uid"] == repeat_uid].sort_values("TransactionDT").head(2)

rng_sample = te.sample(n=23, random_state=42)
sample_df = pd.concat([repeat_rows, rng_sample]).drop_duplicates(subset="TransactionID")

# BUG (caught while chasing an unrelated probability discrepancy between local
# environments — this is the real explanation, not xgboost versions): `te`'s categorical
# columns were already overwritten with integer codes by the encoding loop near the top of
# this notebook (needed for training), the SAME mutation that forced the targeted reload
# for cat_mappings above. This export pulls from `te` too, so it was exporting ALREADY-
# ENCODED integers (e.g. ProductCD=0.0) as if they were the true raw values (e.g. "W") —
# meaning every sample transaction's categoricals were corrupted before src/features.py
# ever saw them, and every one of them fell through to -1 ("unknown category") at serving
# time. Confirmed directly: str(0.0) never matches a manifest key like "W".
#
# NOTE this is a TEST-DATA bug only, not a src/ bug — src/features.py behaved completely
# correctly given what it was handed. Real production transactions from Razorpay's actual
# gateway would never carry this corruption; it's an artifact of this export process alone.
#
# Fix: overwrite the corrupted categorical columns with the TRUE raw values from
# `narrow_raw` (built above for the cat_mappings recovery, and for the same reason: it was
# read fresh from the original CSVs before the encoding loop ever touched it).
true_cats = narrow_raw.set_index("TransactionID")[cat_cols]
sample_df = sample_df.set_index("TransactionID")
sample_df[cat_cols] = true_cats.loc[sample_df.index]
sample_df = sample_df.reset_index()


def row_to_json_safe(row_dict):
    out = {}
    for k, v in row_dict.items():
        if v is None or (isinstance(v, float) and pd.isna(v)):
            out[k] = None
        elif isinstance(v, (np.integer,)):
            out[k] = int(v)
        elif isinstance(v, (np.floating,)):
            out[k] = float(v)
        else:
            out[k] = v
    return out


# BUG (caught running the real export for the first time): .iterrows() returns
# each row as a pandas Series, which can only hold ONE dtype — mixing an int column with
# any float column in the same row silently upcasts the int to float too. Every naturally-
# integer column (TransactionID, card1, C1, addr1, ...) was exported as "3485114.0" instead
# of 3485114. Didn't break anything downstream (str(3485114.0) still works fine as an
# idempotency key, all 11 engine checks passed regardless) but was sloppy — a real
# downstream integration could reasonably expect a clean int.
#
# FIRST FIX ATTEMPT WAS ALSO WRONG: wrapping each to_dict() row back in pd.Series() before
# calling row_to_json_safe re-introduces the exact same upcast, since pd.Series() forces
# one dtype across the dict's values just like iterrows() does — verified this directly
# before trusting the fix. to_dict(orient="records") already returns plain Python dicts
# with each column's own dtype intact; row_to_json_safe now takes that dict straight, no
# Series in between at all.
samples = [row_to_json_safe(r) for r in sample_df[RAW_COLS].to_dict(orient="records")]
with open("/kaggle/working/artifacts/sample_transactions.json", "w") as f:
    json.dump(samples, f)

print(f"{len(samples)} sample transactions exported ({len(RAW_COLS)} raw columns each)")
print(f"repeat-client demo uid: {repeat_uid}, transactions: "
      f"{repeat_rows['TransactionID'].tolist()}")

# %% [markdown]
# ## LLM benchmark — already run, results already shipped, removed from this notebook
#
# The 500-row export + the Ollama-on-Kaggle-GPU benchmark (install, serve, `gpt-oss:20b`,
# 200 timed calls) used to live here. G9 already passed with those results —
# `artifacts/llm_benchmark_sample.json` and `artifacts/llm_benchmark_kaggle_results.json`
# are both already downloaded and committed, and `docs/experiments.md`/`CLAUDE.md` already
# report the final XGBoost-vs-LLM comparison (0.5735 vs 0.1571 PR-AUC). Re-running this
# every session just to skip/comment it out for unrelated addenda (the hyperparameter
# sweep, segment calibration) was pure friction with nothing left to prove — removed rather
# than left to comment out by hand each time. Full history of why Ollama was even here
# (the earlier Ollama Cloud rate-limit saga) is in `journal/`, untouched.

# %% [markdown]
# ## Dashboard addendum — dashboard data export
#
# Streamlit dashboard needs two things that only exist inside THIS notebook session (the
# full 92,427-row test month's real scores): a PR curve and the cost-curve threshold sweep.
# The other two dashboard panels (review queue, audit log) don't need anything new — they
# already work from `src/engine.py` run live on local sample data and `data/audit_log.jsonl`
# respectively. This cell reuses variables already computed above (`cal_te`, `y_te_arr`,
# `thresholds`, `curve_values`, `sens_df`) rather than recomputing anything.

# %%
from sklearn.metrics import precision_recall_curve, roc_curve

precision, recall, pr_thresholds = precision_recall_curve(y_te_arr, cal_te)
fpr, tpr, roc_thresholds = roc_curve(y_te_arr, cal_te)

# precision_recall_curve returns one MORE point than thresholds (the last point is
# precision=1,recall=0 with no corresponding threshold) — pad so arrays zip cleanly for
# the dashboard rather than silently truncating one off the end.
pr_thresholds_padded = np.append(pr_thresholds, 1.0)

dashboard_data = {
    "pr_curve": {
        "precision": precision.tolist(), "recall": recall.tolist(),
        "thresholds": pr_thresholds_padded.tolist(),
    },
    "roc_curve": {
        "fpr": fpr.tolist(), "tpr": tpr.tolist(), "thresholds": roc_thresholds.tolist(),
    },
    "cost_curve": {
        "thresholds": thresholds.tolist(), "values_inr": curve_values.tolist(),
        "best_threshold": float(best_threshold),
    },
    "headline": {
        "no_system_inr": float(baseline_noop), "naive_threshold_inr": float(baseline_naive),
        "arbiter_inr": float(arbiter_value), "arbiter_hard_inr": float(arbiter_hard_value),
        "arbiter_modeled_inr": float(arbiter_modeled_value),
    },
    "policy_mix": {k: int(v) for k, v in counts.items()},
    "policy_bands": [
        {"amount_inr": demo_amt,
         "allow_to": float(ps[acts == 0].max()) if (acts == 0).any() else 0.0,
         "block_from": float(ps[acts == 2].min()) if (acts == 2).any() else 1.0}
        for demo_amt in [500, 2000, 10000, 50000]
        for ps in [np.linspace(0.001, 0.999, 500)]
        for vals in [np.vstack([value_allow(ps, demo_amt), value_stepup(ps, demo_amt),
                                 value_block(ps, demo_amt)]).T]
        for acts in [np.argmax(vals, axis=1)]
    ],
    "sensitivity_map": {
        "margins": margins, "fees": fees,
        "grid_inr": pivot.values.tolist(),  # rows=margins, cols=fees, matches pivot's own axes
        "our_assumption": {"margin": 0.20, "fee": 500},
    },
    "test_month": {"n_transactions": int(len(y_te_arr)), "n_fraud": int(y_te_arr.sum())},
    "model_pr_auc": float(average_precision_score(y_te_arr, cal_te)),
    "model_roc_auc": float(roc_auc_score(y_te_arr, cal_te)),
}

with open("/kaggle/working/artifacts/dashboard_data.json", "w") as f:
    json.dump(dashboard_data, f)
print(f"dashboard_data.json written — PR curve ({len(precision)} pts), "
      f"cost curve ({len(thresholds)} pts), sensitivity grid ({pivot.shape[0]}x{pivot.shape[1]})")

# %% [markdown]
# ## Row-level export addendum — raw per-transaction data for local bootstrap/baseline work
#
# Everything exported so far is either a single artifact (model/calibrator) or an already
# AGGREGATED summary (curves, grids, headline totals) — none of it lets local code resample
# individual transactions with replacement, which a bootstrap confidence interval needs, or
# score a candidate rules-based baseline on the exact same population the headline number
# was computed on. This cell exports the one thing missing: the raw (probability, label,
# amount) triple for every one of the 92,427 test-month transactions, at full row level.
#
# No GPU needed for this cell specifically — `cal_te`, `y_te_arr`, and `amt` already exist
# in memory from the cells above; this only serializes them. Kept as its own file rather
# than folded into dashboard_data.json so the aggregated exports stay lightweight and this
# one is easy to skip if it's ever not needed.

# %%
test_month_raw = {
    "calibrated_probability": cal_te.tolist(),
    "is_fraud": y_te_arr.astype(int).tolist(),
    "amount_inr": amt.tolist(),
}
with open("/kaggle/working/artifacts/test_month_raw.json", "w") as f:
    json.dump(test_month_raw, f)
print(f"test_month_raw.json written — {len(cal_te):,} rows, {int(y_te_arr.sum()):,} fraud "
      f"({y_te_arr.mean():.2%})")

# Sanity check before trusting the file: recompute the exact headline arbiter_value from
# these three raw arrays alone, using the same realized_value() defined above, and confirm
# it matches the number already printed earlier in this notebook to the rupee.
_check_values = np.vstack([
    value_allow(cal_te, amt), value_stepup(cal_te, amt), value_block(cal_te, amt)
]).T
_check_actions = np.argmax(_check_values, axis=1)
_check_total = realized_value(_check_actions, y_te_arr.astype(int), amt).sum()
assert abs(_check_total - arbiter_value) < 1.0, (
    f"Row-level export doesn't reproduce the headline number: {_check_total:,.2f} vs "
    f"{arbiter_value:,.2f} — do not trust this file until this passes."
)
print(f"Sanity check passed: recomputed Rs{_check_total:,.0f} from the raw export alone, "
      f"matches the headline Rs{arbiter_value:,.0f} exactly.")

# %% [markdown]
# ## Error-analysis export addendum — raw feature rows for the REAL false positives/negatives
#
# WHY THIS EXISTS: `docs/robustness_results.json` already has the EXACT false-positive/
# negative transaction OUTCOMES (233 real genuine customers wrongly blocked; separately,
# real fraud that got allowed straight through with zero friction at all). Neither that file
# nor any other local artifact has the RAW FEATURE ROWS for those specific transactions —
# only (probability, label, amount). Per the clue skill's ch14 (error analysis): before
# spending more engineering time on new features or more tuning, manually inspect ~100 real
# misclassified examples and bucket them by failure cause — this addendum is what makes that
# possible, exporting real rows with real SHAP contributions attached, not just bare numbers.
#
# SCOPE, stated explicitly: "false negative" here means fraud ALLOWED straight through with
# NO intervention at all (`action == allow`) — the worst, most complete miss, not the softer
# case of fraud that at least got routed to step-up. That softer group (654 real fraud cases
# in the step-up band, per docs/robustness_results.json) is a separate, less severe category,
# deliberately out of scope for this specific pass — a possible follow-up, not silently
# folded in here.

# %%
get_ipython().system("pip install -q shap")
import shap

fp_mask = (actions == 2) & (y_te_arr == 0)   # real false positives — ALL 233, not sampled
fn_mask = (actions == 0) & (y_te_arr == 1)   # fraud ALLOWED straight through — the worst misses

fp_te_idx = te.index[fp_mask]
fn_te_idx_all = te.index[fn_mask]

print(f"exact false positives (blocked, wrongly): {len(fp_te_idx)}")
print(f"fraud allowed straight through, zero friction: {len(fn_te_idx_all)} "
      f"(sampling 100 of these — see clue skill ch14, ~100 examples is enough for a first pass)")

rng_ea = np.random.default_rng(42)
fn_sample_idx = rng_ea.choice(fn_te_idx_all, size=min(100, len(fn_te_idx_all)), replace=False)

explainer = shap.TreeExplainer(model)


def export_error_rows(row_index, tag):
    # BUG (caught running this for real on Kaggle, not by reading it): the first version
    # overwrote `sub`'s categorical columns with the TRUE raw strings (needed for the export
    # JSON) and then built X_sub straight from that SAME overwritten frame to feed the model
    # and SHAP -- but the model was trained on the INTEGER-ENCODED categoricals (`te`'s own
    # encoding, done early in this notebook), not raw strings. Result: XGBoost/SHAP saw
    # object-dtype columns (ProductCD, card4, ... every categorical) and raised
    # "DataFrame.dtypes for data must be int, float, bool or category." The inverse of a bug
    # already found and fixed once in this same notebook's sample-transaction export --
    # reused that fix's pattern in a spot that needed BOTH representations at once, and only
    # checked the export path, not the scoring path.
    #
    # FIX: keep two separate copies. `sub_encoded` (untouched -- `te`'s own encoding) feeds
    # the model and SHAP. `sub_raw` (a SEPARATE copy, true strings overwritten in) feeds the
    # export JSON only. Never let one dataframe serve both purposes again.
    sub_encoded = te.loc[row_index]
    sub_raw = sub_encoded.copy()

    sub_true_cats = true_cats.loc[sub_raw["TransactionID"].values]
    sub_raw = sub_raw.set_index("TransactionID")
    sub_raw[cat_cols] = sub_true_cats
    sub_raw = sub_raw.reset_index()

    X_sub = sub_encoded[features_v3]   # encoded ints -- matches what the model was trained on
    shap_vals = explainer.shap_values(X_sub)
    arr = np.asarray(shap_vals[-1]) if isinstance(shap_vals, list) else np.asarray(shap_vals)
    if arr.ndim == 3:
        arr = arr[-1]

    raw_probs = model.predict_proba(X_sub)[:, 1]
    cal_probs = platt.predict_proba(raw_probs.reshape(-1, 1))[:, 1]

    # .to_dict(orient="records") ONCE on the whole frame — NOT .iloc[i].to_dict() per row
    # inside the loop below, which would reintroduce the exact int->float upcast bug already
    # found and fixed earlier in THIS notebook's sample-transaction export (see journal:
    # .iterrows()/per-row Series access forces one dtype across a mixed int/float row).
    raw_records = sub_raw[RAW_COLS].to_dict(orient="records")

    out_rows = []
    for i, tid in enumerate(sub_encoded["TransactionID"].values):
        contribs = sorted(zip(features_v3, arr[i]), key=lambda kv: abs(kv[1]), reverse=True)[:5]
        out_rows.append({
            "TransactionID": int(tid),
            "raw_features": row_to_json_safe(raw_records[i]),
            "calibrated_probability": float(cal_probs[i]),
            "amount_inr": float(sub_encoded["TransactionAmt"].iloc[i]) * USD_TO_INR,
            "top5_shap": [{"feature": f, "contribution": float(v)} for f, v in contribs],
        })
    with open(f"/kaggle/working/artifacts/error_analysis_{tag}.json", "w") as f:
        json.dump(out_rows, f)
    print(f"{tag}: exported {len(out_rows)} rows with SHAP")


export_error_rows(fp_te_idx, "false_positives")
export_error_rows(fn_sample_idx, "false_negatives_sample")

# %% [markdown]
# ## Training-dev addendum — decomposing the val→test gap (clue skill ch40/ch41)
#
# WHY: the FINAL V4/V5 refit's own executed output already shows val PR-AUC 0.6128 vs test
# PR-AUC 0.5514 — an 11% relative gap that has never been reported or investigated anywhere
# in this project, despite sitting in plain sight in this notebook's own history. Two
# candidate explanations, genuinely different in what they'd imply: (1) mechanical —
# `early_stopping_rounds` picks the boosting-round count that minimizes VAL loss specifically,
# which optimistically biases the val number a little by construction; (2) real — the test
# window is chronologically farther from train than val is, and this project's own thesis
# (borrowed from Deotte) is "unseen clients, not time drift" — if that effect compounds
# further forward in time, this gap is measuring exactly that.
#
# METHOD (ch40's four-way split): carve a TRAINING-DEV set — a random 15% slice of the
# TRAINING period itself (day <= 120), held out and NEVER trained on — so "unseen, same
# distribution as train" (training-dev) can be compared against "unseen, next period" (val)
# and "unseen, furthest period" (test), isolating generic variance from temporal mismatch.
#
# IMPORTANT — this is a DIAGNOSTIC side-experiment, not a replacement model: `model_diag`
# below is trained on 85% of the training period (a training-dev slice withheld), so its
# exact numbers will differ slightly from the shipped V4/V5 model's. That's expected and
# correct — this run exists only to decompose the gap, not to replace the deployed artifact.

# %%
TRAINDEV_FRAC = 0.15
rng_td = np.random.default_rng(42)
train_idx_all = tr.index.values.copy()
rng_td.shuffle(train_idx_all)
n_traindev = int(len(train_idx_all) * TRAINDEV_FRAC)
traindev_idx = train_idx_all[:n_traindev]
trainprime_idx = train_idx_all[n_traindev:]

tr_prime = tr.loc[trainprime_idx]
traindev = tr.loc[traindev_idx]
print(f"train' (fit on): {len(tr_prime):,} rows   training-dev (held out, unseen): {len(traindev):,} rows")

X_trprime3, y_trprime = tr_prime[features_v3], tr_prime["isFraud"]
X_traindev3, y_traindev = traindev[features_v3], traindev["isFraud"]

model_diag = xgb.XGBClassifier(
    n_estimators=2000, tree_method="hist", device="cuda",
    eval_metric="aucpr", early_stopping_rounds=100, random_state=42, **BEST_PARAMS,
)
model_diag.fit(X_trprime3, y_trprime, eval_set=[(X_va3, y_va)], verbose=200)

raw_trprime = model_diag.predict_proba(X_trprime3)[:, 1]
raw_traindev = model_diag.predict_proba(X_traindev3)[:, 1]
raw_va_diag = model_diag.predict_proba(X_va3)[:, 1]
raw_te_diag = model_diag.predict_proba(X_te3)[:, 1]

# Platt fit on the SAME val set as everywhere else — unchanged, only the underlying model differs.
platt_diag = LogisticRegression()
platt_diag.fit(raw_va_diag.reshape(-1, 1), y_va)


def cal_diag(raw):
    return platt_diag.predict_proba(raw.reshape(-1, 1))[:, 1]


gap_rows = []
for name, yy, raw in [
    ("train_prime_fit_on", y_trprime, raw_trprime),
    ("training_dev_unseen_same_period", y_traindev, raw_traindev),
    ("val_unseen_next_period", y_va, raw_va_diag),
    ("test_unseen_furthest_period", y_te, raw_te_diag),
]:
    c = cal_diag(raw)
    pr = average_precision_score(yy, c)
    roc = roc_auc_score(yy, c)
    gap_rows.append({"set": name, "n": int(len(yy)), "pr_auc": float(pr), "roc_auc": float(roc)})
    print(f"{name:32s}  n={len(yy):>7,}  PR-AUC {pr:.4f}   ROC-AUC {roc:.4f}")

variance_gap = gap_rows[0]["pr_auc"] - gap_rows[1]["pr_auc"]
mismatch_gap_val = gap_rows[1]["pr_auc"] - gap_rows[2]["pr_auc"]
mismatch_gap_test = gap_rows[2]["pr_auc"] - gap_rows[3]["pr_auc"]

print(f"""
gap decomposition (ch41's three-gap diagnosis, PR-AUC):
  variance gap      (train' -> training-dev): {variance_gap:+.4f}
  mismatch gap       (training-dev -> val):    {mismatch_gap_val:+.4f}
  further mismatch   (val -> test):            {mismatch_gap_test:+.4f}

Read: a small variance gap + a large training-dev->val/test gap means the val/test drop is
real temporal mismatch, not overfitting to train. A large variance gap would instead point
at overfitting/regularization as the first lever to pull.
""")

with open("/kaggle/working/artifacts/training_dev_gap.json", "w") as f:
    json.dump({
        "traindev_frac": TRAINDEV_FRAC, "seed": 42,
        "results": gap_rows,
        "variance_gap_pr_auc": variance_gap,
        "mismatch_gap_traindev_to_val_pr_auc": mismatch_gap_val,
        "mismatch_gap_val_to_test_pr_auc": mismatch_gap_test,
        "note": ("Diagnostic side-experiment — model_diag is trained on 85% of the training "
                 "period (training-dev slice withheld), NOT the shipped model. Small "
                 "differences from the official V4/V5 numbers are expected."),
    }, f, indent=2)
print("Saved: training_dev_gap.json")

# %% [markdown]
# ## Bounded hyperparameter sweep — is low regularization an artifact of tuning against
# an inflated validation signal?
#
# WHY: every value in BEST_PARAMS sits at the more-capacity / less-regularized end of its
# Optuna search range — max_depth=10 (the deepest allowed), min_child_weight=2 (near the
# most permissive), reg_lambda=0.78 and reg_alpha=0.023 (both low). Optuna picked these
# because they maximized VAL PR-AUC — but the training-dev decomposition just above shows
# val itself scores 0.225 PR-AUC higher than test, i.e. val is NOT a temporally-neutral
# judge. HYPOTHESIS, not yet proven: tuning against an inflated validation signal may have
# systematically favoured a higher-capacity, more overfit configuration than what actually
# generalises best to the true target distribution (test).
#
# METHOD: a SMALL, bounded sweep (6 configs, not a new 60-trial search) of deliberately more
# conservative alternatives to BEST_PARAMS, each refit on train' (the SAME 85% split already
# built above) and scored on train'/training-dev/val — NEVER on test at this stage.
# Selection criterion is deliberately NOT "highest val score" (the thing under suspicion) —
# it's the SMALLEST training-dev-to-val gap, among configs whose val score isn't
# catastrophically worse than the control (within 0.03 PR-AUC), so a uniformly-bad config
# with a small gap by coincidence can't win by default.
#
# TEST IS NOT TOUCHED IN THIS CELL — per the project's own "test touched once" rule, the
# selected config gets checked against test exactly once, in the NEXT cell, as a
# confirmation of the choice, never as the thing that makes the choice.

# %%
CANDIDATE_PARAMS = {
    "shipped (BEST_PARAMS, control)": dict(BEST_PARAMS),
    "shallower depth": {**BEST_PARAMS, "max_depth": 6},
    "higher min_child_weight": {**BEST_PARAMS, "min_child_weight": 10},
    "higher regularization": {**BEST_PARAMS, "reg_lambda": 3.0, "reg_alpha": 1.0},
    "more conservative sampling": {**BEST_PARAMS, "subsample": 0.6, "colsample_bytree": 0.6},
    "all four combined": {
        **BEST_PARAMS, "max_depth": 6, "min_child_weight": 10,
        "reg_lambda": 3.0, "reg_alpha": 1.0, "subsample": 0.6, "colsample_bytree": 0.6,
    },
}

sweep_results = []
for name, params in CANDIDATE_PARAMS.items():
    m = xgb.XGBClassifier(
        n_estimators=2000, tree_method="hist", device="cuda",
        eval_metric="aucpr", early_stopping_rounds=100, random_state=42, **params,
    )
    m.fit(X_trprime3, y_trprime, eval_set=[(X_va3, y_va)], verbose=False)

    raw_tp = m.predict_proba(X_trprime3)[:, 1]
    raw_td = m.predict_proba(X_traindev3)[:, 1]
    raw_v = m.predict_proba(X_va3)[:, 1]

    plt_sweep = LogisticRegression()
    plt_sweep.fit(raw_v.reshape(-1, 1), y_va)

    def cal_sweep(raw):
        return plt_sweep.predict_proba(raw.reshape(-1, 1))[:, 1]

    pr_tp = average_precision_score(y_trprime, cal_sweep(raw_tp))
    pr_td = average_precision_score(y_traindev, cal_sweep(raw_td))
    pr_v = average_precision_score(y_va, cal_sweep(raw_v))
    gap = pr_td - pr_v

    sweep_results.append({
        "name": name, "params": params,
        "pr_auc_trainprime": float(pr_tp), "pr_auc_traindev": float(pr_td),
        "pr_auc_val": float(pr_v), "traindev_to_val_gap": float(gap),
    })
    print(f"{name:32s}  train'={pr_tp:.4f}  traindev={pr_td:.4f}  val={pr_v:.4f}  gap={gap:+.4f}")

# Sanity check: the control row above should closely reproduce the earlier training-dev
# addendum's own numbers (0.9864 / 0.8357 / 0.6110) — same params, same split, same seed.
control_row = sweep_results[0]
print(f"\nsanity check — control vs the training-dev addendum above (should closely match):")
print(f"  this run:  train'={control_row['pr_auc_trainprime']:.4f}  "
      f"traindev={control_row['pr_auc_traindev']:.4f}  val={control_row['pr_auc_val']:.4f}")
print(f"  earlier:   train'=0.9864  traindev=0.8357  val=0.6110")

control_val = control_row["pr_auc_val"]
eligible = [r for r in sweep_results if r["pr_auc_val"] >= control_val - 0.03]
best = min(eligible, key=lambda r: r["traindev_to_val_gap"])
print(f"\nSelected (smallest traindev->val gap, val within 0.03 of control): {best['name']}")
print(f"  train'={best['pr_auc_trainprime']:.4f}  traindev={best['pr_auc_traindev']:.4f}  "
      f"val={best['pr_auc_val']:.4f}  gap={best['traindev_to_val_gap']:+.4f}")

# %% [markdown]
# ## Confirm the selected config against test — ONCE, as a check, never as the selector

# %%
best_params_candidate = best["params"]
m_final_check = xgb.XGBClassifier(
    n_estimators=2000, tree_method="hist", device="cuda",
    eval_metric="aucpr", early_stopping_rounds=100, random_state=42, **best_params_candidate,
)
m_final_check.fit(X_trprime3, y_trprime, eval_set=[(X_va3, y_va)], verbose=False)

raw_te_check = m_final_check.predict_proba(X_te3)[:, 1]
raw_va_check = m_final_check.predict_proba(X_va3)[:, 1]
plt_check = LogisticRegression()
plt_check.fit(raw_va_check.reshape(-1, 1), y_va)
cal_te_check = plt_check.predict_proba(raw_te_check.reshape(-1, 1))[:, 1]

pr_te_check = average_precision_score(y_te, cal_te_check)
roc_te_check = roc_auc_score(y_te, cal_te_check)

print(f"selected config ('{best['name']}') on TEST (checked once): "
      f"PR-AUC {pr_te_check:.4f}, ROC-AUC {roc_te_check:.4f}")
print(f"shipped model on TEST (already published): PR-AUC 0.5514, ROC-AUC 0.9077")
print(f"delta: {pr_te_check - 0.5514:+.4f} PR-AUC")

with open("/kaggle/working/artifacts/hyperparam_sweep.json", "w") as f:
    json.dump({
        "sweep_results": sweep_results,
        "selected": best["name"],
        "selected_params": best_params_candidate,
        "selected_config_test_pr_auc": float(pr_te_check),
        "selected_config_test_roc_auc": float(roc_te_check),
        "shipped_test_pr_auc": 0.5514,
        "shipped_test_roc_auc": 0.9077,
        "note": ("Selection was made using train'/training-dev/val ONLY — test was checked "
                 "exactly once, after selection, as a confirmation, not as the selector. "
                 "This diagnostic model is trained on 85% of the training period, same as "
                 "the training-dev addendum above — if this config is adopted for real, it "
                 "still needs a full refit on 100% of tr before shipping."),
    }, f, indent=2)
print("Saved: hyperparam_sweep.json")

# %% [markdown]
# ## Segment-aware calibration — does conditioning Platt scaling on ProductCD reduce the
# false-positive cluster in ProductCD='C'?
#
# WHY: `docs/eval_report.md` §8b already flagged this as a "bounded, not-pursued idea" — the
# 233 exact false positives cluster overwhelmingly in ProductCD='C' (81.1% of FPs, vs its
# 11.7% base fraud rate, the highest of 5 codes), while false negatives cluster in
# ProductCD='W' (75%, vs its 2.0% base rate, the lowest). ProductCD is ALREADY a feature the
# model sees directly (`artifacts/feature_manifest.json` lists it) — so this isn't a
# missing-feature problem. It's a question of whether ONE global Platt mapping, fit on the
# pooled val set, is mis-calibrated for a segment whose raw-score distribution differs
# systematically from the pool it was fit on.
#
# WHAT THIS CANNOT FIX, stated upfront: §8b already proved a hard zero ceiling on the
# false-negative side — 0 of 100 sampled FNs (ProductCD='W') scored anywhere near the
# operating threshold (all under 10% vs the 77.4% cutoff). No recalibration can move a
# probability that far without wrecking every other transaction's ranking. This experiment
# can therefore only plausibly help PRECISION in ProductCD='C' — fewer real genuine
# customers wrongly blocked — never recall in 'W'. Framed narrowly on purpose.
#
# METHOD: fit a SEPARATE Platt calibrator per ProductCD value, using only that segment's val
# rows (`raw_va`, `y_va`) — falling back to the existing GLOBAL calibrator (`platt`) for any
# segment with too few val-fraud rows to fit a stable curve. Apply to val and test's raw
# scores. Run the SAME value_allow/value_stepup/value_block + realized_value machinery
# already used for the headline number — nothing about the cost model changes, only which
# calibrated probability feeds it.
#
# NOTE ON THE INTEGER ENCODING: by this point in the notebook, `va["ProductCD"]` and
# `te["ProductCD"]` are already integer-encoded (the in-place `.map(mapping)` near the top),
# not raw strings — segmenting on those integers is equivalent to segmenting on the raw
# category (1:1 mapping), so no need to reconstruct true raw values the way the error-
# analysis export had to. `cat_mappings["ProductCD"]` (already built above) labels them.
#
# SELECTION ON VAL ONLY, same discipline as everywhere else in this project: segment-aware
# calibration is adopted only if it beats the global calibrator's VAL realized value AND
# reduces the ProductCD='C' false-positive count on VAL. Test is checked ONCE, after that
# decision is already locked in, purely to confirm — never to choose.

# %%
from sklearn.linear_model import LogisticRegression as _LR

MIN_SEGMENT_FRAUD = 30  # minimum val-fraud rows in a segment to fit its own Platt curve;
                         # below this a segment calibrator would itself be unstable -- the
                         # exact failure mode this experiment is trying to avoid introducing

productcd_va = va["ProductCD"].values
productcd_te = te["ProductCD"].values
y_va_arr = y_va.values.astype(int)
productcd_label = {v: k for k, v in cat_mappings["ProductCD"].items()}  # int -> "W"/"C"/...

segment_calibrators = {}
for seg in pd.unique(productcd_va):
    seg_mask = productcd_va == seg
    n_fraud_seg = int(y_va_arr[seg_mask].sum())
    if n_fraud_seg >= MIN_SEGMENT_FRAUD:
        lr = _LR()
        lr.fit(raw_va[seg_mask].reshape(-1, 1), y_va_arr[seg_mask])
        segment_calibrators[seg] = lr
    else:
        segment_calibrators[seg] = None  # falls back to the global `platt` below
    label = productcd_label.get(seg, seg)
    print(f"ProductCD={label!r}: {seg_mask.sum():,} val rows, {n_fraud_seg} fraud -> "
          f"{'own calibrator' if segment_calibrators[seg] else 'falls back to global (too few fraud rows)'}")


def segment_calibrate(raw_scores, productcd_arr):
    out = np.empty_like(raw_scores)
    for seg in pd.unique(productcd_arr):
        seg_mask = productcd_arr == seg
        cal = segment_calibrators.get(seg)
        if cal is not None:
            out[seg_mask] = cal.predict_proba(raw_scores[seg_mask].reshape(-1, 1))[:, 1]
        else:
            out[seg_mask] = platt.predict_proba(raw_scores[seg_mask].reshape(-1, 1))[:, 1]
    return out


cal_va_seg = segment_calibrate(raw_va, productcd_va)
cal_te_seg = segment_calibrate(raw_te, productcd_te)


def policy_value_and_fp(p_arr, amt_arr, is_fraud_arr, productcd_arr=None, target_segment=None):
    """Runs the real 3-way policy (identical formulas to the shipped decision engine) and
    returns (total realized value, count of wrongly-blocked genuine customers[, restricted
    to target_segment if given])."""
    va_a = value_allow(p_arr, amt_arr)
    va_s = value_stepup(p_arr, amt_arr)
    va_b = value_block(p_arr, amt_arr)
    acts = np.argmax(np.vstack([va_a, va_s, va_b]).T, axis=1)
    total = realized_value(acts, is_fraud_arr, amt_arr).sum()
    wrongly_blocked = (acts == 2) & (is_fraud_arr == 0)
    if target_segment is not None:
        seg_mask = productcd_arr == target_segment
        wrongly_blocked = wrongly_blocked & seg_mask
    return float(total), int(wrongly_blocked.sum())


PRODUCTCD_C = cat_mappings["ProductCD"]["C"]

# --- VAL: global vs segment-aware. This decides which one gets used -- test is not touched yet ---
val_global_value, val_global_fp_C = policy_value_and_fp(cal_va, amt_va_inr, y_va_arr, productcd_va, PRODUCTCD_C)
val_seg_value, val_seg_fp_C = policy_value_and_fp(cal_va_seg, amt_va_inr, y_va_arr, productcd_va, PRODUCTCD_C)

print(f"\nVAL -- global calibration:   total value Rs {val_global_value:,.0f}, "
      f"ProductCD=C wrongly-blocked: {val_global_fp_C}")
print(f"VAL -- segment calibration:  total value Rs {val_seg_value:,.0f}, "
      f"ProductCD=C wrongly-blocked: {val_seg_fp_C}")

adopt_segment = (val_seg_value >= val_global_value) and (val_seg_fp_C <= val_global_fp_C)
print(f"\n>>> DECISION (made on VAL only): "
      f"{'ADOPT segment-aware calibration' if adopt_segment else 'KEEP global calibration'}")

# --- TEST: checked ONCE, as confirmation only, regardless of which config was picked ---
test_global_value, test_global_fp_C = policy_value_and_fp(cal_te, amt_te_inr, y_te_arr, productcd_te, PRODUCTCD_C)
test_seg_value, test_seg_fp_C = policy_value_and_fp(cal_te_seg, amt_te_inr, y_te_arr, productcd_te, PRODUCTCD_C)

print(f"\nTEST (confirmation only) -- global:   total value Rs {test_global_value:,.0f}, "
      f"ProductCD=C wrongly-blocked: {test_global_fp_C}")
print(f"TEST (confirmation only) -- segment:  total value Rs {test_seg_value:,.0f}, "
      f"ProductCD=C wrongly-blocked: {test_seg_fp_C}")
print(f"delta (segment - global) on test: Rs {test_seg_value - test_global_value:,.0f}, "
      f"{test_seg_fp_C - test_global_fp_C:+d} wrongly-blocked in ProductCD=C")

with open("/kaggle/working/artifacts/segment_calibration.json", "w") as f:
    json.dump({
        "min_segment_fraud": MIN_SEGMENT_FRAUD,
        "segments_with_own_calibrator": [productcd_label.get(seg, seg)
                                          for seg, cal in segment_calibrators.items() if cal is not None],
        "val": {
            "global_total_value": val_global_value, "global_fp_productcd_c": val_global_fp_C,
            "segment_total_value": val_seg_value, "segment_fp_productcd_c": val_seg_fp_C,
        },
        "decision_made_on_val_only": "adopt_segment" if adopt_segment else "keep_global",
        "test_confirmation_only": {
            "global_total_value": test_global_value, "global_fp_productcd_c": test_global_fp_C,
            "segment_total_value": test_seg_value, "segment_fp_productcd_c": test_seg_fp_C,
        },
        "note": ("Selection made on VAL only (adopt_segment decided before test was touched). "
                 "Test numbers are a confirmation of that VAL decision, never the selector. "
                 "This changes ONLY the calibration step -- decide_action's cost formulas are "
                 "untouched. Scoped narrowly to ProductCD='C' precision per the zero-ceiling "
                 "finding in eval_report.md 8b: FN recall in ProductCD='W' is not expected to "
                 "move and is not the thing being tested here."),
    }, f, indent=2)
print("Saved: segment_calibration.json")

# %% [markdown]
# ## Ensemble diagnostic — does averaging XGBoost with LightGBM and CatBoost close any of
# the gap to the Kaggle-winning ensemble (CatBoost + LightGBM + XGBoost + a neural net)?
#
# WHY: the causal-vs-leaky comparison (component B) already showed that honoring the
# no-future-data constraint only costs +0.0066 PR-AUC — so most of the real gap to the
# winners' 0.9408 ROC-AUC isn't the leakage trick, it's something else. Ensembling different
# model families is the most standard lever nobody has tried here yet, and it's literally
# what the winners did — two of their three tree models are these same two libraries.
#
# WHAT THIS IS NOT: a hyperparameter search for LightGBM/CatBoost. Each gets ONE fair shot
# with settings broadly comparable to XGBoost's own tuned config (similar depth,
# regularization, learning rate) — not independently tuned. This is a bounded diagnostic to
# answer "is there real headroom here at all", not a from-scratch optimization. If the
# result looks promising, a real per-model tuning pass is the natural next step, not skipped
# here, just correctly out of scope for this question.
#
# CPU is fine for both — a single fit on ~352k rows takes a few minutes each; GPU only
# matters for a repeated tuning loop (like Optuna's 60 trials), not a one-shot fit. Both
# libraries ship in Kaggle's default image already.
#
# METHOD: train on the exact same feature matrix already used for the shipped XGBoost
# (X_tr3/y_tr, early-stopped on X_va3/y_va) — same features, same split, so the model family
# is the only thing that varies. Each gets its OWN Platt calibrator (fit on its own val raw
# scores, same as the shipped model). The ensemble is a plain average of the three
# calibrated probabilities — the simplest, most standard combination, not a trained
# stacking meta-learner.
#
# SELECTION ON VAL ONLY: does the 3-model average beat XGBoost-alone's val PR-AUC? This
# isn't vulnerable to the same "val is an inflated signal" concern as the hyperparameter
# sweep — that concern was specifically about capacity/regularization exploiting val's
# temporal proximity to train; averaging independently-trained models is an orthogonal
# technique (variance reduction across uncorrelated errors) with no such failure mode. Test
# is still checked ONCE afterward, as confirmation, never as the selector.

# %%
import lightgbm as lgb
from catboost import CatBoostClassifier

# Settings matched in spirit to BEST_PARAMS (depth, learning rate, regularization), not
# independently tuned per library — see the markdown above.
lgb_model = lgb.LGBMClassifier(
    n_estimators=2000, max_depth=10, num_leaves=255, learning_rate=0.0492,
    subsample=0.8155, colsample_bytree=0.8872, min_child_samples=20,
    reg_lambda=0.7815, reg_alpha=0.0232, random_state=42, verbose=-1,
)
lgb_model.fit(X_tr3, y_tr, eval_set=[(X_va3, y_va)], eval_metric="auc",
              callbacks=[lgb.early_stopping(100, verbose=False)])

cat_model = CatBoostClassifier(
    iterations=2000, depth=10, learning_rate=0.0492, l2_leaf_reg=3.0,
    subsample=0.8155, random_seed=42, eval_metric="AUC",
    early_stopping_rounds=100, verbose=False,
)
cat_model.fit(X_tr3, y_tr, eval_set=(X_va3, y_va))

raw_va_lgb = lgb_model.predict_proba(X_va3)[:, 1]
raw_te_lgb = lgb_model.predict_proba(X_te3)[:, 1]
raw_va_cat = cat_model.predict_proba(X_va3)[:, 1]
raw_te_cat = cat_model.predict_proba(X_te3)[:, 1]

platt_lgb = LogisticRegression()
platt_lgb.fit(raw_va_lgb.reshape(-1, 1), y_va)
cal_va_lgb = platt_lgb.predict_proba(raw_va_lgb.reshape(-1, 1))[:, 1]
cal_te_lgb = platt_lgb.predict_proba(raw_te_lgb.reshape(-1, 1))[:, 1]

platt_cat = LogisticRegression()
platt_cat.fit(raw_va_cat.reshape(-1, 1), y_va)
cal_va_cat = platt_cat.predict_proba(raw_va_cat.reshape(-1, 1))[:, 1]
cal_te_cat = platt_cat.predict_proba(raw_te_cat.reshape(-1, 1))[:, 1]

# Plain average of the three calibrated probabilities -- simplest, most standard ensemble
ens_va = (cal_va + cal_va_lgb + cal_va_cat) / 3
ens_te = (cal_te + cal_te_lgb + cal_te_cat) / 3

individual_scores = {
    "xgboost (shipped)": {
        "val_pr_auc": float(average_precision_score(y_va, cal_va)),
        "test_pr_auc": float(average_precision_score(y_te, cal_te)),
        "test_roc_auc": float(roc_auc_score(y_te, cal_te)),
    },
    "lightgbm": {
        "val_pr_auc": float(average_precision_score(y_va, cal_va_lgb)),
        "test_pr_auc": float(average_precision_score(y_te, cal_te_lgb)),
        "test_roc_auc": float(roc_auc_score(y_te, cal_te_lgb)),
    },
    "catboost": {
        "val_pr_auc": float(average_precision_score(y_va, cal_va_cat)),
        "test_pr_auc": float(average_precision_score(y_te, cal_te_cat)),
        "test_roc_auc": float(roc_auc_score(y_te, cal_te_cat)),
    },
    "ensemble (avg of all 3)": {
        "val_pr_auc": float(average_precision_score(y_va, ens_va)),
        "test_pr_auc": float(average_precision_score(y_te, ens_te)),
        "test_roc_auc": float(roc_auc_score(y_te, ens_te)),
    },
}
for name, scores in individual_scores.items():
    print(f"{name:26s}  val PR-AUC={scores['val_pr_auc']:.4f}  "
          f"test PR-AUC={scores['test_pr_auc']:.4f}  test ROC-AUC={scores['test_roc_auc']:.4f}")

xgb_val = individual_scores["xgboost (shipped)"]["val_pr_auc"]
ens_val = individual_scores["ensemble (avg of all 3)"]["val_pr_auc"]
worth_pursuing = ens_val > xgb_val
print(f"\n>>> DECISION (made on VAL only): ensemble "
      f"{'BEATS' if worth_pursuing else 'does NOT beat'} XGBoost-alone on val "
      f"({ens_val:.4f} vs {xgb_val:.4f}) -> "
      f"{'worth a real per-model tuning pass + shipping conversation' if worth_pursuing else 'not worth pursuing further'}")

# TEST: checked once, as confirmation only -- never as the selector
xgb_test = individual_scores["xgboost (shipped)"]["test_pr_auc"]
ens_test = individual_scores["ensemble (avg of all 3)"]["test_pr_auc"]
print(f"\nTEST (confirmation only, checked once): ensemble {ens_test:.4f} vs shipped "
      f"{xgb_test:.4f} ({ens_test - xgb_test:+.4f})")

with open("/kaggle/working/artifacts/ensemble_diagnostic.json", "w") as f:
    json.dump({
        "individual_scores": individual_scores,
        "decision_made_on_val_only": "worth_pursuing" if worth_pursuing else "not_worth_pursuing",
        "test_confirmation_delta_pr_auc": float(ens_test - xgb_test),
        "note": ("Each model got ONE fair shot with hyperparameters comparable to XGBoost's "
                 "own tuned config, not independently tuned for LightGBM/CatBoost -- a real "
                 "per-model hyperparameter search is the natural next step if this looks "
                 "promising, not done here. Ensemble is a plain average of 3 independently- "
                 "calibrated probabilities, not a trained stacking meta-learner. Selection "
                 "decided on val only, before test was touched; test checked once after, as "
                 "confirmation, never as the selector."),
    }, f, indent=2)
print("Saved: ensemble_diagnostic.json")

# %% [markdown]
# ## Is the ensemble's improvement real, or within noise? Paired bootstrap on the delta
#
# WHY: the point-estimate improvement (ensemble test PR-AUC vs. shipped) is smaller than
# the half-width of the bootstrap CI already reported on the shipped model's own PR-AUC
# (0.5350-0.5688, +-0.017) — same rigor this project gives every other headline number
# should apply here before treating this as a confirmed win, not just eyeballing a bigger
# point estimate. A PAIRED bootstrap (the same resampled row indices used for every model,
# every resample) isolates the DELTA's own uncertainty, which is much tighter than either
# model's individual CI, because shared sampling noise cancels out.
#
# Also checks a SIMPLICITY question in the same pass: CatBoost was individually the weakest
# of the three (test PR-AUC below even the shipped XGBoost alone) — does a leaner 2-model
# average (XGBoost + LightGBM only) capture most of the gain with one fewer model to deploy,
# monitor, and keep in sync? Cheap to check — both models are already fit and in memory, no
# retraining needed.

# %%
ens2_va = (cal_va + cal_va_lgb) / 2
ens2_te = (cal_te + cal_te_lgb) / 2

print(f"2-model (XGB+LGB) ensemble  -- val PR-AUC={average_precision_score(y_va, ens2_va):.4f}  "
      f"test PR-AUC={average_precision_score(y_te, ens2_te):.4f}")
print(f"3-model ensemble (from above) -- test PR-AUC={average_precision_score(y_te, ens_te):.4f}")

rng = np.random.default_rng(42)
n = len(y_te_arr)
N_BOOT = 2000

deltas_3model = np.empty(N_BOOT)
deltas_2model = np.empty(N_BOOT)
for i in range(N_BOOT):
    idx = rng.integers(0, n, n)
    y_b = y_te_arr[idx]
    if y_b.sum() == 0:  # guard: a resample with zero positives makes PR-AUC undefined
        deltas_3model[i] = np.nan
        deltas_2model[i] = np.nan
        continue
    pr_shipped = average_precision_score(y_b, cal_te[idx])
    pr_ens3 = average_precision_score(y_b, ens_te[idx])
    pr_ens2 = average_precision_score(y_b, ens2_te[idx])
    deltas_3model[i] = pr_ens3 - pr_shipped
    deltas_2model[i] = pr_ens2 - pr_shipped

deltas_3model = deltas_3model[~np.isnan(deltas_3model)]
deltas_2model = deltas_2model[~np.isnan(deltas_2model)]

ci_3model = np.percentile(deltas_3model, [2.5, 97.5])
ci_2model = np.percentile(deltas_2model, [2.5, 97.5])

print(f"\n3-model ensemble vs shipped -- delta PR-AUC 95% CI: [{ci_3model[0]:+.4f}, {ci_3model[1]:+.4f}]")
print(f"2-model ensemble vs shipped -- delta PR-AUC 95% CI: [{ci_2model[0]:+.4f}, {ci_2model[1]:+.4f}]")
print(f"\n3-model CI excludes zero (real improvement): {ci_3model[0] > 0}")
print(f"2-model CI excludes zero (real improvement): {ci_2model[0] > 0}")

with open("/kaggle/working/artifacts/ensemble_bootstrap.json", "w") as f:
    json.dump({
        "n_resamples": N_BOOT,
        "seed": 42,
        "ensemble_2model_xgb_lgb": {
            "val_pr_auc": float(average_precision_score(y_va, ens2_va)),
            "test_pr_auc": float(average_precision_score(y_te, ens2_te)),
        },
        "delta_vs_shipped_pr_auc_ci95": {
            "3model": [float(ci_3model[0]), float(ci_3model[1])],
            "2model": [float(ci_2model[0]), float(ci_2model[1])],
        },
        "note": ("Paired bootstrap (same resampled row indices used for every model every "
                 "resample), so shared sampling noise cancels and the CI reflects the "
                 "DELTA's own uncertainty, not each model's individual noise. If a CI's "
                 "lower bound is above zero, that improvement is statistically real for this "
                 "test month, not just a bigger point estimate that could be noise."),
    }, f, indent=2)
print("Saved: ensemble_bootstrap.json")

# %% [markdown]
# ## Download instructions
#
# In Kaggle's right sidebar: Output panel -> artifacts/ folder -> download each file (or
# the whole folder as a zip). Place all files into the LOCAL repo at:
#   artifacts/model.json
#   artifacts/calibrator.json
#   artifacts/feature_manifest.json
#   artifacts/sample_transactions.json
#   artifacts/dashboard_data.json
#   artifacts/test_month_raw.json
#   artifacts/error_analysis_false_positives.json
#   artifacts/error_analysis_false_negatives_sample.json
#   artifacts/training_dev_gap.json
#   artifacts/hyperparam_sweep.json
#   artifacts/segment_calibration.json
#   artifacts/ensemble_diagnostic.json
#   artifacts/ensemble_bootstrap.json

# %%
print("Files in /kaggle/working/artifacts:")
for f in os.listdir("/kaggle/working/artifacts"):
    size = os.path.getsize(f"/kaggle/working/artifacts/{f}")
    print(f"  {f}  ({size:,} bytes)")
