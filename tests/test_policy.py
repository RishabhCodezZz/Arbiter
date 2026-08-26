"""
The cost model's decision logic. No trained model needed here — policy.py is pure
arithmetic over a probability and an amount, so these run instantly, no artifacts required.
"""
from src import policy


def test_actions_are_exactly_the_three_expected():
    assert set(policy.ACTIONS) == {"allow", "step-up", "block"}


def test_rules_baseline_is_always_stepup_with_no_probability():
    action, values = policy.rules_baseline(amount_usd=100.0)
    assert action == "step-up"
    assert values == {}


def test_decide_action_returns_one_of_the_three_actions():
    action, values = policy.decide_action(calibrated_prob=0.5, amount_usd=100.0)
    assert action in policy.ACTIONS
    assert set(values.keys()) == {"allow", "step-up", "block"}


def test_low_probability_is_never_blocked():
    action, _ = policy.decide_action(calibrated_prob=0.001, amount_usd=100.0)
    assert action != "block"


def test_very_high_probability_is_never_allowed():
    action, _ = policy.decide_action(calibrated_prob=0.999, amount_usd=100.0)
    assert action != "allow"


def test_degraded_mode_widens_the_stepup_band():
    """The regression test for the algebraic bug found and fixed while wiring engine.py:
    the original implementation blended VALUES toward step-up's value, which can be proven
    to never move the allow/block boundary for any blend factor. The fix shrinks the
    PROBABILITY toward 0.5 before the value formulas ever see it, which genuinely does move
    both boundaries. This test finds the real confident-allow ceiling on a representative
    transaction, then confirms degraded=True narrows it (moves it toward more caution) —
    not just that the numbers differ, but that they move in the conservative direction."""
    amount = 100.0

    def confident_allow_ceiling(degraded):
        # sweep down from a clearly-allowed probability until the action stops being allow
        lo, hi = 0.0, 0.5
        for _ in range(40):
            mid = (lo + hi) / 2
            action, _ = policy.decide_action(mid, amount, degraded=degraded)
            if action == "allow":
                lo = mid
            else:
                hi = mid
        return lo

    normal_ceiling = confident_allow_ceiling(degraded=False)
    degraded_ceiling = confident_allow_ceiling(degraded=True)
    assert degraded_ceiling < normal_ceiling, (
        "degraded=True must narrow the confident-allow band (move the boundary toward more "
        "caution) — if this fails, the widen-the-step-up-band fallback described in "
        "CLAUDE.md's failure table is not actually doing anything, the same way the "
        "original (reverted) implementation wasn't."
    )


def test_value_functions_are_finite_across_the_probability_range():
    for p in (0.0, 0.001, 0.25, 0.5, 0.75, 0.999, 1.0):
        for amt in (10.0, 500.0, 50000.0):
            for fn in (policy.value_allow, policy.value_stepup, policy.value_block):
                v = fn(p, amt)
                assert v == v, f"{fn.__name__}({p}, {amt}) returned NaN"  # NaN != NaN
                assert v not in (float("inf"), float("-inf"))
