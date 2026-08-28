"""
The core engine proof — same properties scripts/demo_engine.py demonstrates as one linear
narrative, here as independent, named tests. Run against the real trained model.
"""
import shutil

import pytest

from src.audit import verify_and_replay
from src import policy
from tests.conftest import requires_real_artifacts


@requires_real_artifacts
def test_model_loads_from_saved_artifact(engine):
    assert engine.model is not None


@requires_real_artifacts
def test_decide_returns_a_valid_action(engine, real_transactions):
    d = engine.decide(real_transactions[0])
    assert d.action in policy.ACTIONS
    assert d.idempotent_replay is False
    assert d.calibrated_probability is not None
    assert 0.0 <= d.calibrated_probability <= 1.0


@requires_real_artifacts
def test_client_history_accumulates_across_calls(engine, real_transactions):
    """samples[0] and samples[1] are the same client's first two transactions in the test
    month (see notebooks/04_cost_model.py's export) — proves history is genuinely
    accumulated across two separate decide() calls, not faked within one."""
    first = engine.decide(real_transactions[0])
    uid = engine.audit.lookup(first.transaction_id)["feature_vector"]["_uid"]
    stats_after_first = engine.store.get_fine(uid)
    assert stats_after_first.n >= 1

    second = engine.decide(real_transactions[1])
    fv_second = engine.audit.lookup(second.transaction_id)["feature_vector"]
    assert fv_second["_uid"] == uid, "sample transactions 0 and 1 are expected to share a client"
    assert fv_second["uid_amt_mean_prior"] is not None, (
        "second transaction should see real prior history, not a cold start"
    )


@requires_real_artifacts
def test_idempotency_same_transaction_id_twice(engine, real_transactions):
    txn = real_transactions[0]
    first = engine.decide(txn)
    second = engine.decide(txn)
    assert second.idempotent_replay is True
    assert second.action == first.action
    assert second.calibrated_probability == first.calibrated_probability


@requires_real_artifacts
def test_replay_reproduces_the_decision(engine, real_transactions):
    txn = real_transactions[0]
    d = engine.decide(txn)
    record = engine.audit.lookup(d.transaction_id)
    ok, msg = verify_and_replay(record, policy)
    assert ok, msg


@requires_real_artifacts
def test_tampered_record_is_rejected_on_replay(engine, real_transactions):
    txn = real_transactions[0]
    d = engine.decide(txn)
    record = dict(engine.audit.lookup(d.transaction_id))
    record["action"] = "block" if record["action"] != "block" else "allow"
    ok, msg = verify_and_replay(record, policy)
    assert not ok, "a tampered record must be rejected on replay, not silently accepted"


@requires_real_artifacts
def test_fail_closed_when_model_artifact_missing(engine, real_transactions, tmp_path):
    """Hides the real model.json, constructs a fresh Engine pointed at the now-incomplete
    artifacts dir, and confirms it degrades to the conservative step-up default instead of
    crashing or silently allowing everything through."""
    from src.engine import Engine as _Engine

    broken_dir = tmp_path / "artifacts_missing_model"
    broken_dir.mkdir()
    for name in ("calibrator.json", "feature_manifest.json"):
        src_path = f"artifacts/{name}"
        shutil.copy(src_path, broken_dir / name)
    # deliberately do NOT copy model.json

    broken_engine = _Engine(
        artifacts_dir=str(broken_dir),
        store_path=str(tmp_path / "store.json"),
        audit_path=str(tmp_path / "audit.jsonl"),
    )
    assert broken_engine.model is None, "engine must construct even with no model present"

    d = broken_engine.decide(real_transactions[3])
    assert d.action == "step-up", "fail-closed default must be step-up, never a silent allow"
    assert "model_unavailable_fail_closed" in d.fallbacks_triggered, (
        "the fallback must be logged explicitly, not hidden"
    )
    assert d.calibrated_probability is None


@requires_real_artifacts
def test_engine_starts_with_no_model_at_all(tmp_path):
    """Constructing an Engine against a directory with none of the model files present must
    not raise — this is the cold-start version of the fail-closed path (proven by
    test_fail_closed_when_model_artifact_missing above, mid-session)."""
    from src.engine import Engine as _Engine

    empty_dir = tmp_path / "empty_artifacts"
    empty_dir.mkdir()
    e = _Engine(
        artifacts_dir=str(empty_dir),
        store_path=str(tmp_path / "store.json"),
        audit_path=str(tmp_path / "audit.jsonl"),
    )
    assert e.model is None


def _no_model_engine(tmp_path):
    from src.engine import Engine as _Engine
    empty_dir = tmp_path / "empty_artifacts_boundary"
    empty_dir.mkdir()
    return _Engine(
        artifacts_dir=str(empty_dir),
        store_path=str(tmp_path / "store.json"),
        audit_path=str(tmp_path / "audit.jsonl"),
    )


@pytest.mark.parametrize("bad_txn", [{}, {"TransactionAmt": 100.0}, {"TransactionID": None}, "not-a-dict"])
def test_missing_transaction_id_raises_a_clear_error(tmp_path, bad_txn):
    """A request with no TransactionID can't be made idempotent or audited — the boundary
    must reject it with a clear ValueError, not a bare KeyError from deep in str(txn[...])."""
    e = _no_model_engine(tmp_path)
    with pytest.raises(ValueError):
        e.decide(bad_txn)


@pytest.mark.parametrize("bad_amt", [float("nan"), float("inf"), -5.0, "100", [1], True])
def test_non_finite_or_negative_amount_rejected_at_boundary(tmp_path, bad_amt):
    """A NaN/string/negative TransactionAmt would otherwise reach the cost-model arithmetic
    (after a successful model score, outside the fail-closed try) — a NaN amount makes every
    action value NaN and max() picks 'allow'. Reject it at the boundary instead."""
    e = _no_model_engine(tmp_path)
    with pytest.raises(ValueError):
        e.decide({"TransactionID": "amt-1", "TransactionAmt": bad_amt})


def test_absent_amount_is_still_allowed(tmp_path):
    """TransactionAmt is optional — a transaction without one must NOT be rejected (it's
    treated as 0.0 downstream), only a present-but-invalid value is."""
    e = _no_model_engine(tmp_path)
    d = e.decide({"TransactionID": "amt-2", "TransactionDT": 86400 * 100})
    assert d.action in ("step-up", "block")  # no model -> fail closed, but no ValueError


@requires_real_artifacts
def test_unrecoverable_feature_build_still_fails_closed(engine):
    """A non-numeric TransactionDT breaks features.build() AND features.build_degraded().
    The engine must still return a fail-closed decision (never crash, never silent-allow),
    with the double failure logged."""
    d = engine.decide({"TransactionID": "boundary-1", "TransactionDT": "not-a-number",
                        "TransactionAmt": 250.0})
    assert d.action in ("step-up", "block")
    assert any(f.startswith("degraded_build_failed:") for f in d.fallbacks_triggered)
    assert d.calibrated_probability is None
