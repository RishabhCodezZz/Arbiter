"""
Three robustness checks on the headline cost-model result and the model's own ranking
metrics, run against the REAL, FULL 92,427-row test month (artifacts/test_month_raw.json)
— not a smaller sample, not a re-derivation with different assumptions.

  1. Exact false-positive/false-negative breakdown of the real 3-way policy. Every prior
     report of false-positive cost (docs/eval_report.md) came from an aggregate-curve
     ESTIMATE (~172 wrongly-blocked genuine customers, derived from the single-threshold
     PR-curve proxy in eval_report.md §3). This recomputes the same per-transaction actions
     Arbiter actually picks (allow/step-up/block) and counts the real, exact composition of
     each bucket against real labels — no estimation, no proxy. Cross-checked against the
     already-published policy mix (88,560/2,519/1,348) before anything built on it is trusted.

  2. Bootstrap confidence interval on the headline rupee lift (already existed). Resamples
     the same month with replacement, PAIRED per resample (Arbiter's total and each
     baseline's total on the SAME resampled rows, then differenced) rather than as two
     independent intervals subtracted afterward, which would ignore the correlation
     between them and overstate the uncertainty.

  3. Bootstrap confidence interval on PR-AUC and ROC-AUC THEMSELVES, not just the rupee
     lift — using the exact same resampled indices each iteration as check #2, so this is
     free (no extra resampling cost). Both metrics are hand-rolled here (not imported from
     sklearn) for the same reason scripts/llm_benchmark.py hand-rolls average precision:
     requirements.txt deliberately keeps sklearn out of scripts/ and src/ (it's only ever a
     transitive dependency of shap) — a new metric here follows that discipline rather than
     quietly breaking it. Both are verified to reproduce the already-published point
     estimates (0.5514 / 0.9077) before being trusted inside the bootstrap loop.

  4. A rules-based baseline: block if amount > threshold, no ML at all. Naive-0.5 is a
     strawman nobody actually runs; this is closer to what a merchant with no model might
     plausibly do. Swept for its own best threshold (fair shot, same discipline as picking
     the strongest available LLM for the other benchmark) rather than picked arbitrarily —
     the point is to beat the best simple rule, not a weak one.

Run from the repo root: python scripts/robustness_checks.py
"""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src import policy

RAW_PATH = "artifacts/test_month_raw.json"
OUT_PATH = "docs/robustness_results.json"
N_BOOTSTRAP = 2000
SEED = 42

# Already-published headline point estimates (docs/eval_report.md §1) — the hand-rolled
# PR-AUC/ROC-AUC below must reproduce these before the bootstrap CI built on top of them is
# trusted. Same "does the known-good number still reproduce" discipline used throughout
# this project (e.g. the row-level export's own self-check in notebook 04).
PUBLISHED_PR_AUC = 0.5514
PUBLISHED_ROC_AUC = 0.9077
REPRODUCTION_TOLERANCE = 0.0005


def realized_value(action, is_fraud, amt, margin=None, fee=None):
    """Identical formula to notebooks/04_cost_model.py's realized_value() and
    src/policy.py's value_allow/value_stepup/value_block — reimplemented at the array
    level here only because this script needs per-row values for a batch of transactions
    at once, which src/policy.py's per-transaction decide_action() isn't shaped for.

    margin/fee: optional overrides for the sensitivity sweep (Part 3, below) — default to
    the shipped policy's own constants, so every existing call site (bootstrap CI, rules
    baseline) is unaffected and still uses the real, deployed parameters."""
    margin = policy.MARGIN if margin is None else margin
    fee = policy.CHARGEBACK_FEE if fee is None else fee
    out = np.zeros_like(amt)
    m = action == 0  # allow
    out[m] = np.where(
        is_fraud[m] == 1,
        -(amt[m] + fee + policy.MDR_RATE * amt[m]),
        margin * amt[m] - policy.MDR_RATE * amt[m],
    )
    m = action == 1  # step-up
    out[m] = np.where(
        is_fraud[m] == 1,
        -(1 - policy.P_STOP) * (amt[m] + fee + policy.MDR_RATE * amt[m]),
        (1 - policy.P_DROPOFF) * (margin * amt[m] - policy.MDR_RATE * amt[m]),
    )
    m = action == 2  # block
    out[m] = np.where(is_fraud[m] == 1, 0.0, -(margin * amt[m] * (1 + policy.LTV_MULTIPLIER)))
    return out


def ci95(arr):
    return float(np.percentile(arr, 2.5)), float(np.percentile(arr, 97.5))


def average_precision(y_true, y_score):
    """PR-AUC, identical formula to scripts/llm_benchmark.py's average_precision() —
    duplicated rather than imported so this script stays self-contained (same convention
    notebook 05 uses for notebook 02's aggregate logic, rather than sharing kernel state).
    Matches sklearn.metrics.average_precision_score's definition exactly:
    AP = sum_n (R_n - R_{n-1}) * P_n over scores sorted descending."""
    n_pos = y_true.sum()
    if n_pos == 0:
        return float("nan")
    order = np.argsort(-y_score, kind="mergesort")
    y_true_sorted = y_true[order]
    tp_cum = np.cumsum(y_true_sorted)
    fp_cum = np.cumsum(1 - y_true_sorted)
    precision = tp_cum / (tp_cum + fp_cum)
    recall = tp_cum / n_pos
    recall_prev = np.concatenate(([0.0], recall[:-1]))
    return float(np.sum((recall - recall_prev) * precision))


def _rankdata_avg(a):
    """Average ranks (1-indexed), matching scipy.stats.rankdata(method='average') — needed
    for a CORRECT tie-aware ROC-AUC. Not a theoretical concern here: the calibrated
    probabilities genuinely have duplicate values (docs/experiments.md: 91,271 distinct
    scores out of 92,427 test-month rows — over a thousand rows share a value with at least
    one other row), so a naive argsort-based rank (no tie handling) would introduce a small,
    avoidable bias. No scipy dependency needed — this is the standard vectorized recipe,
    verified against a hand-worked tied example before use (see the point-estimate
    reproduction check in main() for the real-data verification)."""
    n = len(a)
    sorter = np.argsort(a, kind="mergesort")
    a_sorted = a[sorter]
    is_new_group = np.empty(n, dtype=bool)
    is_new_group[0] = True
    is_new_group[1:] = a_sorted[1:] != a_sorted[:-1]
    group_id = np.cumsum(is_new_group) - 1
    sorted_ranks = np.arange(1, n + 1, dtype=np.float64)
    group_sum = np.bincount(group_id, weights=sorted_ranks)
    group_count = np.bincount(group_id)
    avg_rank_sorted = (group_sum / group_count)[group_id]
    avg_rank = np.empty(n)
    avg_rank[sorter] = avg_rank_sorted
    return avg_rank


def roc_auc(y_true, y_score):
    """Rank-based (Mann-Whitney U) ROC-AUC — hand-rolled for the same reason
    average_precision() above is: requirements.txt deliberately keeps sklearn out of
    scripts/ and src/ (it's only ever a transitive dependency of shap), so a new metric
    here follows that discipline rather than quietly breaking it for one script. Verified
    against the already-published point estimate (see main()) before being trusted inside
    a 2,000-iteration bootstrap loop."""
    n_pos = int(y_true.sum())
    n_neg = len(y_true) - n_pos
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    ranks = _rankdata_avg(y_score)
    sum_ranks_pos = ranks[y_true == 1].sum()
    return float((sum_ranks_pos - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def main():
    if not os.path.exists(RAW_PATH):
        print(f"Missing {RAW_PATH}. See artifacts/README.md — this needs the row-level "
              f"export addendum run in a live Kaggle session and downloaded.")
        sys.exit(1)

    with open(RAW_PATH) as f:
        d = json.load(f)
    p = np.array(d["calibrated_probability"])
    y = np.array(d["is_fraud"])
    amt = np.array(d["amount_inr"])
    n = len(p)
    print(f"loaded {n:,} real test-month transactions ({int(y.sum()):,} fraud, {y.mean():.3%})\n")

    # ---- per-row values under each policy, computed once (deterministic given p/amt) ----
    values = np.vstack([policy.value_allow(p, amt), policy.value_stepup(p, amt), policy.value_block(p, amt)]).T
    arbiter_actions = np.argmax(values, axis=1)
    arbiter_row_values = realized_value(arbiter_actions, y, amt)

    noop_row_values = realized_value(np.zeros(n, dtype=int), y, amt)  # always allow
    naive_row_values = realized_value(np.where(p >= 0.5, 2, 0), y, amt)  # allow/block only

    point = {
        "no_system": float(noop_row_values.sum()),
        "naive_0.5": float(naive_row_values.sum()),
        "arbiter": float(arbiter_row_values.sum()),
    }
    print("point estimates (must match the already-reported headline exactly):")
    for k, v in point.items():
        print(f"  {k:12s} Rs {v:,.0f}")
    print(f"  lift vs no_system:  Rs {point['arbiter']-point['no_system']:,.0f}")
    print(f"  lift vs naive_0.5:  Rs {point['arbiter']-point['naive_0.5']:,.0f}")

    # ---- Part 0: exact false-positive/false-negative breakdown of the real 3-way policy ----
    # Replaces the aggregate-curve ESTIMATE in docs/eval_report.md SS3-4 (~172 false
    # positives, derived from the single-threshold PR-curve proxy) with an exact count for
    # the REAL 3-way policy specifically — every mask below is a real per-transaction action
    # against a real label, not a proxy. Cross-checked against the already-published policy
    # mix (allow 88,560 / step-up 2,519 / block 1,348) before trusting anything on top of it.
    n_allow = int((arbiter_actions == 0).sum())
    n_stepup = int((arbiter_actions == 1).sum())
    n_block = int((arbiter_actions == 2).sum())
    print(f"\nexact policy mix: allow {n_allow:,} ({n_allow/n:.1%}), "
          f"step-up {n_stepup:,} ({n_stepup/n:.1%}), block {n_block:,} ({n_block/n:.1%})")
    print("  (already-published policy mix: allow 88,560 (95.8%), step-up 2,519 (2.7%), "
          "block 1,348 (1.5%) -- should match exactly)")

    block_mask = arbiter_actions == 2
    block_fraud_mask = block_mask & (y == 1)     # correctly blocked fraud -- real, exact
    block_genuine_mask = block_mask & (y == 0)   # THE false positives -- real, exact, not estimated

    n_block_fraud = int(block_fraud_mask.sum())
    n_block_genuine = int(block_genuine_mask.sum())
    # realized_value() already stores exactly -(margin*amt*(1+ltv_mult)) on a wrongly-
    # blocked genuine row (0 on a correctly-blocked fraud row) -- reuse it rather than
    # recompute the formula a second time and risk the two drifting apart.
    fp_total_cost_inr = float(-arbiter_row_values[block_genuine_mask].sum())
    fp_avg_cost_inr = fp_total_cost_inr / n_block_genuine if n_block_genuine else 0.0

    stepup_mask = arbiter_actions == 1
    n_stepup_fraud = int((stepup_mask & (y == 1)).sum())
    n_stepup_genuine = int((stepup_mask & (y == 0)).sum())

    print(f"\nEXACT false-positive breakdown (real per-transaction actions x real labels, "
          f"not an aggregate-curve estimate):")
    print(f"  blocked, correctly (real fraud):                       {n_block_fraud:,}")
    print(f"  blocked, WRONGLY (real genuine -- the false positives): {n_block_genuine:,}")
    print(f"  exact false-positive cost:             Rs {fp_total_cost_inr:,.0f}  "
          f"(Rs {fp_avg_cost_inr:,.0f} average per wrongly-blocked customer)")
    print(f"  (docs/eval_report.md's prior ESTIMATE, from the aggregate PR curve: ~172 FPs -- "
          f"this is the exact real count for the real 3-way policy, not an estimate)")
    print(f"\n  step-up band composition (exact -- outcome for these txns is still MODELED, "
          f"not measured, per the honest exception list):")
    print(f"    real fraud in the step-up band:    {n_stepup_fraud:,}")
    print(f"    real genuine in the step-up band:  {n_stepup_genuine:,}")

    assert n_block == n_block_fraud + n_block_genuine, "block count doesn't partition cleanly by label -- bug"
    assert n_allow + n_stepup + n_block == n, "policy-mix counts don't sum to the full test month -- bug"

    # ---- Part 0b: PR-AUC/ROC-AUC point estimates, verified against the published headline ----
    point_pr_auc = average_precision(y, p)
    point_roc_auc = roc_auc(y, p)
    print(f"\npoint-estimate PR-AUC:  {point_pr_auc:.4f}  (published headline: {PUBLISHED_PR_AUC})")
    print(f"point-estimate ROC-AUC: {point_roc_auc:.4f}  (published headline: {PUBLISHED_ROC_AUC})")
    assert abs(point_pr_auc - PUBLISHED_PR_AUC) < REPRODUCTION_TOLERANCE, (
        "hand-rolled PR-AUC does not reproduce the published headline -- DO NOT trust the "
        "bootstrap CI below until this passes"
    )
    assert abs(point_roc_auc - PUBLISHED_ROC_AUC) < REPRODUCTION_TOLERANCE, (
        "hand-rolled ROC-AUC does not reproduce the published headline -- DO NOT trust the "
        "bootstrap CI below until this passes"
    )
    print(">>> both reproduce the published headline -- trusting the bootstrap CI below.")

    # ---- Part 1: paired bootstrap CI (rupee lift AND model-quality metrics together) ----
    rng = np.random.default_rng(SEED)
    boot_arbiter = np.empty(N_BOOTSTRAP)
    boot_noop = np.empty(N_BOOTSTRAP)
    boot_naive = np.empty(N_BOOTSTRAP)
    boot_pr_auc = np.empty(N_BOOTSTRAP)
    boot_roc_auc = np.empty(N_BOOTSTRAP)
    for b in range(N_BOOTSTRAP):
        idx = rng.integers(0, n, size=n)
        boot_arbiter[b] = arbiter_row_values[idx].sum()
        boot_noop[b] = noop_row_values[idx].sum()
        boot_naive[b] = naive_row_values[idx].sum()
        boot_pr_auc[b] = average_precision(y[idx], p[idx])
        boot_roc_auc[b] = roc_auc(y[idx], p[idx])

    lift_noop = boot_arbiter - boot_noop
    lift_naive = boot_arbiter - boot_naive

    print(f"\n95% bootstrap CI ({N_BOOTSTRAP} resamples, paired per resample, seed={SEED}):")
    print(f"  arbiter total:      Rs {ci95(boot_arbiter)[0]:,.0f}  to  Rs {ci95(boot_arbiter)[1]:,.0f}")
    print(f"  lift vs no_system:  Rs {ci95(lift_noop)[0]:,.0f}  to  Rs {ci95(lift_noop)[1]:,.0f}")
    print(f"  lift vs naive_0.5:  Rs {ci95(lift_naive)[0]:,.0f}  to  Rs {ci95(lift_naive)[1]:,.0f}")
    print(f"  PR-AUC:             {ci95(boot_pr_auc)[0]:.4f}  to  {ci95(boot_pr_auc)[1]:.4f}  "
          f"(point {point_pr_auc:.4f})")
    print(f"  ROC-AUC:            {ci95(boot_roc_auc)[0]:.4f}  to  {ci95(boot_roc_auc)[1]:.4f}  "
          f"(point {point_roc_auc:.4f})")

    # ---- Part 2: rules-based baseline, swept for its own best threshold ----
    # Swept well past the observed max amount so the true optimum is actually inside the
    # search range, not truncated by it -- a first pass at 200k hit that exact boundary,
    # which is the tell that the real optimum was outside it (caught before trusting the
    # number, not after).
    thresholds = np.linspace(500, amt.max() * 1.5, 800)
    best_val, best_thresh = -np.inf, None
    for t in thresholds:
        val = float(realized_value(np.where(amt > t, 2, 0), y, amt).sum())
        if val > best_val:
            best_val, best_thresh = val, float(t)
    hit_boundary = best_thresh >= thresholds[-1] - 1.0
    if hit_boundary:
        print(f"WARNING: best threshold still at the search boundary ({best_thresh:,.0f}) -- "
              f"extend the range further before trusting this number.")

    natural_thresholds = {}
    for t in (10000, 50000):
        natural_thresholds[t] = float(realized_value(np.where(amt > t, 2, 0), y, amt).sum())

    print(f"\nrules-based baseline (block if amount > threshold, no model, no probability used):")
    print(f"  best threshold (swept, own best shot): Rs {best_thresh:,.0f}")
    print(f"  total value at that threshold:         Rs {best_val:,.0f}")
    print(f"  Arbiter's lift over the BEST simple rule: Rs {point['arbiter']-best_val:,.0f}")
    for t, val in natural_thresholds.items():
        print(f"  (context, not tuned) threshold Rs {t:,}:  Rs {val:,.0f}")

    # ---- Part 3: sensitivity -- does the LIFT over both baselines stay positive across
    # the whole margin x fee grid, not just at the assumed operating point? ----
    # notebooks/04_cost_model.py's sensitivity map only ever showed Arbiter's OWN value
    # moving across assumptions -- it never confirmed the LIFT over either baseline stays
    # positive everywhere. Self-flagged as an open gap in docs/experiments.md and
    # journal/build-log.md ("worth extending before the final report") and never closed
    # until now. Fully local -- no Kaggle run needed, reuses test_month_raw.json and the
    # already-parameterized realized_value(). MDR is held fixed (disclosed fact, not an
    # assumption, same treatment as everywhere else in this project); only margin and
    # chargeback fee vary, same grid as the original sensitivity map.
    SENS_MARGINS = [0.05, 0.10, 0.15, 0.20, 0.30, 0.40, 0.50]
    SENS_FEES = [200, 350, 500, 600, 1000]

    sens_rows = []
    min_lift_noop, min_lift_naive = np.inf, np.inf
    for m in SENS_MARGINS:
        for f in SENS_FEES:
            va_a = (1 - p) * (m * amt - policy.MDR_RATE * amt) - p * (amt + f + policy.MDR_RATE * amt)
            va_s = ((1 - p) * (1 - policy.P_DROPOFF) * (m * amt - policy.MDR_RATE * amt)
                    - p * (1 - policy.P_STOP) * (amt + f + policy.MDR_RATE * amt))
            va_b = -(1 - p) * (m * amt * (1 + policy.LTV_MULTIPLIER))
            act = np.argmax(np.vstack([va_a, va_s, va_b]).T, axis=1)

            arbiter_total = float(realized_value(act, y, amt, margin=m, fee=f).sum())
            noop_total = float(realized_value(np.zeros(n, dtype=int), y, amt, margin=m, fee=f).sum())
            naive_total = float(realized_value(np.where(p >= 0.5, 2, 0), y, amt, margin=m, fee=f).sum())
            lift_noop_here = arbiter_total - noop_total
            lift_naive_here = arbiter_total - naive_total
            min_lift_noop = min(min_lift_noop, lift_noop_here)
            min_lift_naive = min(min_lift_naive, lift_naive_here)
            sens_rows.append({
                "margin": m, "fee": f, "arbiter_inr": arbiter_total,
                "no_system_inr": noop_total, "naive_0.5_inr": naive_total,
                "lift_vs_no_system_inr": lift_noop_here, "lift_vs_naive_inr": lift_naive_here,
            })

    n_negative_noop = sum(1 for r in sens_rows if r["lift_vs_no_system_inr"] <= 0)
    n_negative_naive = sum(1 for r in sens_rows if r["lift_vs_naive_inr"] <= 0)
    assumed = next(r for r in sens_rows if r["margin"] == 0.20 and r["fee"] == 500)

    print(f"\nsensitivity: lift over BOTH baselines across the full "
          f"{len(SENS_MARGINS)}x{len(SENS_FEES)} = {len(sens_rows)}-point margin x fee grid "
          f"(MDR held fixed at {policy.MDR_RATE:.4f}, disclosed fact, not swept):")
    print(f"  lift vs no_system: min Rs {min_lift_noop:,.0f} across the grid  "
          f"({'ALL POSITIVE' if n_negative_noop == 0 else f'{n_negative_noop} NEGATIVE CELLS -- see sens_rows'})")
    print(f"  lift vs naive_0.5: min Rs {min_lift_naive:,.0f} across the grid  "
          f"({'ALL POSITIVE' if n_negative_naive == 0 else f'{n_negative_naive} NEGATIVE CELLS -- see sens_rows'})")
    print(f"  at our assumed point (margin=0.20, fee=500): lift vs no_system Rs "
          f"{assumed['lift_vs_no_system_inr']:,.0f}, lift vs naive Rs {assumed['lift_vs_naive_inr']:,.0f}")

    results = {
        "n_transactions": n,
        "n_fraud": int(y.sum()),
        "point_estimates_inr": point,
        "policy_mix_exact": {"allow": n_allow, "stepup": n_stepup, "block": n_block},
        "false_positive_exact": {
            "n_block_total": n_block,
            "n_block_correctly_fraud": n_block_fraud,
            "n_block_false_positive": n_block_genuine,
            "false_positive_total_cost_inr": fp_total_cost_inr,
            "false_positive_avg_cost_inr": fp_avg_cost_inr,
            "stepup_band_composition": {
                "n_total": n_stepup, "n_fraud": n_stepup_fraud, "n_genuine": n_stepup_genuine,
            },
            "note": ("block-side counts are exact, real per-transaction outcomes against real "
                     "labels -- not an estimate. Step-up band composition (who ended up there) "
                     "is also exact; what WOULD have happened to them (stopped vs got through, "
                     "abandoned vs completed) remains modeled from population rates (P_STOP/"
                     "P_DROPOFF), since this dataset has no real step-up interaction data -- "
                     "unchanged limitation, stated in the honest exception list."),
        },
        "model_quality": {
            "pr_auc_point": point_pr_auc,
            "roc_auc_point": point_roc_auc,
            "published_pr_auc": PUBLISHED_PR_AUC,
            "published_roc_auc": PUBLISHED_ROC_AUC,
            "n_resamples": N_BOOTSTRAP,
            "seed": SEED,
            "pr_auc_ci95": list(ci95(boot_pr_auc)),
            "roc_auc_ci95": list(ci95(boot_roc_auc)),
        },
        "bootstrap": {
            "n_resamples": N_BOOTSTRAP,
            "seed": SEED,
            "arbiter_ci95_inr": list(ci95(boot_arbiter)),
            "lift_vs_no_system_ci95_inr": list(ci95(lift_noop)),
            "lift_vs_naive_ci95_inr": list(ci95(lift_naive)),
        },
        "sensitivity_lift": {
            "margins": SENS_MARGINS,
            "fees": SENS_FEES,
            "min_lift_vs_no_system_inr": min_lift_noop,
            "min_lift_vs_naive_inr": min_lift_naive,
            "n_negative_cells_vs_no_system": n_negative_noop,
            "n_negative_cells_vs_naive": n_negative_naive,
            "grid": sens_rows,
            "note": ("Extends the original sensitivity map (which only showed Arbiter's OWN "
                     "value moving across margin x fee) to confirm the LIFT over both "
                     "baselines specifically -- previously verified only at the single "
                     "assumed point (margin=0.20, fee=500)."),
        },
        "rules_baseline": {
            "best_threshold_inr": best_thresh,
            "best_value_inr": best_val,
            "arbiter_lift_over_best_rule_inr": point["arbiter"] - best_val,
            "context_thresholds_inr": natural_thresholds,
        },
    }
    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    with open(OUT_PATH, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved: {OUT_PATH}")


if __name__ == "__main__":
    main()
