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
