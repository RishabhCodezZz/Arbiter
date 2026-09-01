"""
Audit log properties in isolation — no trained model needed, these test AuditLog and
verify_and_replay directly against hand-built records.
"""
from src.audit import AuditLog, verify_and_replay
from src import policy


def test_lookup_coerces_int_transaction_id_to_str(empty_audit_log):
    """The regression test for the bug caught while writing the engine's smoke test:
    lookup() used to compare transaction_id without coercing type, so a caller passing a
    raw pandas/numpy int (the natural way, straight from a DataFrame row) silently got a
    false 'not found' instead of the real record — a false negative on the one property
    idempotency exists to guarantee."""
    record = empty_audit_log.make_record(
        transaction_id="12345", model_version="test", feature_vector={"_amount": 100.0},
        raw_prob=0.1, calibrated_prob=0.1, cost_params={}, action="allow",
        action_values={}, latency_ms=1.0, fallbacks=[],
    )
    empty_audit_log.append(record)

    assert empty_audit_log.lookup("12345") is not None
    assert empty_audit_log.lookup(12345) is not None, (
        "lookup() must coerce a raw int the same way it handles a str"
    )


def test_lookup_returns_none_for_an_unknown_id(empty_audit_log):
    assert empty_audit_log.lookup("does-not-exist") is None


def test_untampered_record_replays_successfully(empty_audit_log):
    action, values = policy.decide_action(0.1, 100.0)
    record = empty_audit_log.make_record(
        transaction_id="1", model_version="test", feature_vector={"_amount": 100.0},
        raw_prob=0.1, calibrated_prob=0.1, cost_params={}, action=action,
        action_values=values, latency_ms=1.0, fallbacks=[],
    )
    empty_audit_log.append(record)
    stored = empty_audit_log.lookup("1")
    ok, msg = verify_and_replay(stored, policy)
    assert ok, msg


def test_tampered_action_is_detected(empty_audit_log):
    action, values = policy.decide_action(0.1, 100.0)
    record = empty_audit_log.make_record(
        transaction_id="1", model_version="test", feature_vector={"_amount": 100.0},
        raw_prob=0.1, calibrated_prob=0.1, cost_params={}, action=action,
        action_values=values, latency_ms=1.0, fallbacks=[],
    )
    empty_audit_log.append(record)
    stored = dict(empty_audit_log.lookup("1"))
    stored["action"] = "block" if stored["action"] != "block" else "allow"
    ok, msg = verify_and_replay(stored, policy)
    assert not ok


def test_tampered_feature_vector_is_detected_via_hash(empty_audit_log):
    action, values = policy.decide_action(0.1, 100.0)
    record = empty_audit_log.make_record(
        transaction_id="1", model_version="test", feature_vector={"_amount": 100.0},
        raw_prob=0.1, calibrated_prob=0.1, cost_params={}, action=action,
        action_values=values, latency_ms=1.0, fallbacks=[],
    )
    empty_audit_log.append(record)
    stored = dict(empty_audit_log.lookup("1"))
    stored["feature_vector"] = dict(stored["feature_vector"], _amount=999999.0)
    ok, msg = verify_and_replay(stored, policy)
    assert not ok
    assert "TAMPER" in msg


def test_degraded_flag_is_required_for_correct_replay(empty_audit_log):
    """The regression test for the bug found while wiring engine.py: verify_and_replay must
    replay with the SAME degraded flag the original decision used, or a decision made under
    the widened step-up band gets replayed against the normal boundaries and can produce a
    different action — a false tamper alarm on a legitimate decision."""
    # find a probability where degraded=True and degraded=False actually disagree
    p = None
    for candidate in [i / 1000 for i in range(1, 100)]:
        a_normal, _ = policy.decide_action(candidate, 100.0, degraded=False)
        a_degraded, _ = policy.decide_action(candidate, 100.0, degraded=True)
        if a_normal != a_degraded:
            p = candidate
            break
    assert p is not None, "expected to find at least one probability where degraded mode changes the action"

    action, values = policy.decide_action(p, 100.0, degraded=True)
    record = empty_audit_log.make_record(
        transaction_id="1", model_version="test", feature_vector={"_amount": 100.0},
        raw_prob=p, calibrated_prob=p, cost_params={}, action=action,
        action_values=values, latency_ms=1.0, fallbacks=[], degraded=True,
    )
    empty_audit_log.append(record)
    stored = empty_audit_log.lookup("1")
    ok, msg = verify_and_replay(stored, policy)
    assert ok, f"legitimate degraded decision falsely flagged as tampered: {msg}"


def test_fallback_path_record_with_no_probability_still_replays(empty_audit_log):
    """A fail-closed record (model unavailable) has calibrated_probability=None — replay
    can't re-run the cost model against nothing, but the tamper-hash check must still work."""
    record = empty_audit_log.make_record(
        transaction_id="1", model_version="test", feature_vector={"_amount": 100.0},
        raw_prob=None, calibrated_prob=None, cost_params={}, action="step-up",
        action_values={}, latency_ms=1.0, fallbacks=["model_unavailable_fail_closed"],
    )
    empty_audit_log.append(record)
    stored = empty_audit_log.lookup("1")
    ok, msg = verify_and_replay(stored, policy)
    assert ok, msg


# --- P1-3: replay against the STORED cost params, and hash the decision fields --------

_FULL_COST_PARAMS = {
    "chargeback_fee": 500.0, "mdr_rate": 0.0236, "margin": 0.20, "p_stop": 0.60,
    "p_dropoff": 0.15, "ltv_multiplier": 3.0, "usd_to_inr": 95.41,
}


def test_replay_uses_stored_cost_params_not_current_module_constants(empty_audit_log):
    """A decision made under one set of cost parameters must replay against THOSE parameters
    (stored in the record), not policy.py's current module constants — the record is meant
    to be re-derivable from itself even after a merchant re-tunes margin or an FX rate is
    corrected. Caught in review: verify_and_replay ignored record['cost_params'] entirely."""
    amt_usd, p = 100.0, 0.35
    # An extreme chargeback fee flips this borderline transaction's cost-optimal action.
    weird_params = dict(_FULL_COST_PARAMS, chargeback_fee=5_000_000.0)
    action_weird, values_weird = policy.decide_action(p, amt_usd, cost_params=weird_params)
    action_default, _ = policy.decide_action(p, amt_usd)
    assert action_weird != action_default, (
        "test needs a probability/amount where the modified params change the action"
    )

    record = empty_audit_log.make_record(
        transaction_id="cp1", model_version="test", feature_vector={"_amount": amt_usd},
        raw_prob=p, calibrated_prob=p, cost_params=weird_params, action=action_weird,
        action_values=values_weird, latency_ms=1.0, fallbacks=[],
    )
    empty_audit_log.append(record)
    stored = empty_audit_log.lookup("cp1")

    ok, msg = verify_and_replay(stored, policy)
    assert ok, f"replay must use the stored cost params and reproduce '{action_weird}': {msg}"


def test_tampered_cost_params_are_detected_by_record_hash(empty_audit_log):
    action, values = policy.decide_action(0.2, 100.0, cost_params=_FULL_COST_PARAMS)
    record = empty_audit_log.make_record(
        transaction_id="cp2", model_version="test", feature_vector={"_amount": 100.0},
        raw_prob=0.2, calibrated_prob=0.2, cost_params=_FULL_COST_PARAMS, action=action,
        action_values=values, latency_ms=1.0, fallbacks=[],
    )
    empty_audit_log.append(record)
    stored = dict(empty_audit_log.lookup("cp2"))
    stored["cost_params"] = dict(stored["cost_params"], margin=0.99)  # altered after the fact
    ok, msg = verify_and_replay(stored, policy)
    assert not ok and "TAMPER" in msg


def test_tampered_probability_alone_is_detected_by_record_hash(empty_audit_log):
    """Changing only the stored probability (leaving the feature vector and action alone)
    was previously undetectable — the old hash covered the feature vector only."""
    action, values = policy.decide_action(0.05, 100.0)
    record = empty_audit_log.make_record(
        transaction_id="cp3", model_version="test", feature_vector={"_amount": 100.0},
        raw_prob=0.05, calibrated_prob=0.05, cost_params={}, action=action,
        action_values=values, latency_ms=1.0, fallbacks=[],
    )
    empty_audit_log.append(record)
    stored = dict(empty_audit_log.lookup("cp3"))
    stored["calibrated_probability"] = 0.99  # action left untouched
    ok, msg = verify_and_replay(stored, policy)
    assert not ok, "a lone probability edit must be caught by the record hash"


def test_tampered_action_values_are_detected(empty_audit_log):
    """The allow/step-up/block rupee breakdown shown to a reviewer as the rationale must be
    tamper-evident too — editing it while leaving the action, probability and feature vector
    alone was previously undetectable (action_values was stored but not hashed or replayed).
    Caught in review."""
    action, values = policy.decide_action(0.2, 100.0, cost_params=_FULL_COST_PARAMS)
    record = empty_audit_log.make_record(
        transaction_id="av1", model_version="test", feature_vector={"_amount": 100.0},
        raw_prob=0.2, calibrated_prob=0.2, cost_params=_FULL_COST_PARAMS, action=action,
        action_values=values, latency_ms=1.0, fallbacks=[],
    )
    empty_audit_log.append(record)
    stored = dict(empty_audit_log.lookup("av1"))
    # inflate the "block" rationale by an order of magnitude, everything else untouched
    stored["action_values"] = dict(stored["action_values"], block=stored["action_values"]["block"] * 10 - 1)
    ok, msg = verify_and_replay(stored, policy)
    assert not ok and ("TAMPER" in msg or "MISMATCH" in msg), msg
