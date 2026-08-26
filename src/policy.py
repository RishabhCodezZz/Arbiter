"""
The cost model — IDENTICAL parameters and formulas to notebooks/04_cost_model.py. If these
two ever drift apart, the notebook's measured lift (docs/experiments.md) stops describing
what the engine actually does. Kept in one place conceptually; duplicated here only because
the notebook runs on Kaggle and this runs locally — see PLAN.md for the tracked follow-up
about consolidating this once both environments can share one source file.

Every number is sourced — see notebooks/04_cost_model.py's markdown for citations
(Razorpay's own chargeback-fee range and MDR, 3D Secure studies, checkout-friction studies).
"""
CHARGEBACK_FEE = 500.0
MDR_RATE = 0.02 * 1.18  # Razorpay's disclosed 2% platform fee + 18% GST
MARGIN = 0.20
P_STOP = 0.60
P_DROPOFF = 0.15
LTV_MULTIPLIER = 3.0
USD_TO_INR = 95.41  # live, dated quote, corrected from an earlier placeholder — see journal


def value_allow(p: float, amt: float) -> float:
    return (1 - p) * (MARGIN * amt - MDR_RATE * amt) - p * (amt + CHARGEBACK_FEE + MDR_RATE * amt)


def value_stepup(p: float, amt: float) -> float:
    genuine_value = (1 - p) * (1 - P_DROPOFF) * (MARGIN * amt - MDR_RATE * amt)
    fraud_cost = p * (1 - P_STOP) * (amt + CHARGEBACK_FEE + MDR_RATE * amt)
    return genuine_value - fraud_cost


def value_block(p: float, amt: float) -> float:
    genuine_penalty = MARGIN * amt * (1 + LTV_MULTIPLIER)
    return -(1 - p) * genuine_penalty


ACTIONS = ("allow", "step-up", "block")


PROB_SHRINK_TOWARD_UNCERTAINTY = 0.05  # see decide_action's degraded= docstring


def decide_action(calibrated_prob: float, amount_usd: float, degraded: bool = False) -> tuple[str, dict]:
    """The real, per-transaction policy — not a single global threshold. See
    notebooks/04_cost_model.py's markdown for why: every cost term scales with amount, so
    the cost-optimal action genuinely depends on transaction size, not just probability.
    Returns (action, {action_name: value}) — the values are kept for the audit record, so
    a reviewer can see not just WHAT was decided but by how much it beat the alternatives.

    degraded: set True when the features feeding this decision were built via
    features.build_degraded() (a required raw field was missing/malformed and we imputed
    rather than guessed) — see engine.py. Required fallback behaviour, CLAUDE.md's failure
    table: "Required feature missing -> impute, flag degraded confidence, WIDEN THE STEP-UP
    BAND rather than guess."

    MECHANISM, and why it's implemented this way and not the obvious-looking alternative:
    the first version of this shrank allow's and block's VALUES toward step-up's value by a
    constant blend factor. Tested it directly (compared the allow/block boundary before and
    after) and found the boundary NEVER MOVED, for any blend factor — proven algebraically
    too: if a(p) == s(p) at the crossover, then a(p)*(1-k) + s(p)*k == s(p) at that exact
    same p, for any k != 1. Blending a value toward another value cannot move the point
    where they were already equal. Caught this by testing the actual boundary, not by
    trusting that the code "looked right".

    Fixed by shrinking the PROBABILITY toward 0.5 (maximum uncertainty) before it ever
    reaches the value formulas, instead: p_adjusted = p*(1-k) + 0.5*k. This genuinely moves
    both boundaries — verified directly: at k=0.05 on a Rs9,541 transaction, the confident-
    allow ceiling drops from p<0.0393 to p<0.0150, and the confident-block floor rises from
    p>0.6885 to p>0.6982. A real, modest widening, not an unchanged one.
    """
    if degraded:
        calibrated_prob = calibrated_prob * (1 - PROB_SHRINK_TOWARD_UNCERTAINTY) + 0.5 * PROB_SHRINK_TOWARD_UNCERTAINTY

    amt_inr = amount_usd * USD_TO_INR
    values = {
        "allow": value_allow(calibrated_prob, amt_inr),
        "step-up": value_stepup(calibrated_prob, amt_inr),
        "block": value_block(calibrated_prob, amt_inr),
    }
    action = max(values, key=values.get)
    return action, values


def rules_baseline(amount_usd: float) -> tuple[str, dict]:
    """Fail-closed fallback for when the model is unavailable (see model.py's
    ModelUnavailableError and engine.py's handling). Per CLAUDE.md's failure table: fail
    CLOSED, don't silently allow. "Step-up everything" is the chosen default — conservative
    (no fraud gets waved through unchecked) without being maximally disruptive (doesn't
    outright block 100% of revenue while the model is down). No probability is available
    in this path, so no cost-model values are computed — the audit record reflects that
    explicitly (see engine.py) rather than fabricating a confidence the system doesn't have.
    """
    return "step-up", {}
