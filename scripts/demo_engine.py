"""
Engine-build proof script. Run from the repo root:

    python scripts/demo_engine.py

Demonstrates, in order:
  1. decide() works from a saved artifact, on plain CPU, in a fresh process — no notebook,
     no GPU, no training. (Gate G7.)
  2. The client-history store genuinely accumulates state between calls — a repeat
     customer's second transaction sees their first one as prior history.
  3. Idempotency — calling decide() twice with the same transaction_id returns the
     IDENTICAL decision the second time, not a freshly recomputed one.
  4. Replayability — an audit record contains enough to re-derive its own decision, and a
     tampered record is detected before it's trusted.
  5. The fail-closed path — with the model artifact temporarily hidden, decide() degrades
     to the conservative rules baseline instead of crashing or silently allowing everything
     through. (Part of Gate G8, completed properly alongside the LLM fallbacks.)

The shipped model is a 2-model ensemble (XGBoost + LightGBM, simple-averaged after
independent Platt calibration — see docs/experiments.md). Requires artifacts/{model.json,
calibrator.json, model_lgb.txt, calibrator_lgb.json, feature_manifest.json,
sample_transactions.json} — produced by the engine-build addenda in
notebooks/06_cost_model_refined.py, downloaded from Kaggle's Output panel.
"""
import json
import os
import shutil
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.audit import verify_and_replay
from src.engine import Engine
from src import policy

ARTIFACTS_DIR = "artifacts"
SAMPLE_PATH = os.path.join(ARTIFACTS_DIR, "sample_transactions.json")


def line(title=""):
    print("\n" + "=" * 70)
    if title:
        print(title)
        print("=" * 70)


def check(condition: bool, label: str):
    mark = "PASS" if condition else "FAIL"
    print(f"  [{mark}] {label}")
    return condition


def main():
    all_ok = True

    if not os.path.exists(SAMPLE_PATH):
        print(f"Missing {SAMPLE_PATH}.")
        print("Run the engine-build addenda in notebooks/06_cost_model_refined.py in your")
        print("Kaggle session, then download the artifacts/ folder from the Output panel —")
        print("needs model.json, calibrator.json, model_lgb.txt, calibrator_lgb.json, and")
        print("feature_manifest.json, alongside this sample file, at the repo's artifacts/.")
        sys.exit(1)

    with open(SAMPLE_PATH) as f:
        samples = json.load(f)
    print(f"loaded {len(samples)} real held-out transactions from {SAMPLE_PATH}")

    # fresh state each demo run, so the story is reproducible
    for p in ("data/client_history.json", "data/audit_log.jsonl"):
        if os.path.exists(p):
            os.remove(p)

    # ---------------------------------------------------------------- 1. basic decide()
    line("1. decide() from a saved artifact — CPU, no notebook, no training")
    engine = Engine(artifacts_dir=ARTIFACTS_DIR)
    all_ok &= check(engine.model is not None, "model artifact loaded")
    if engine.model is not None:
        print("  artifact SHA-256 (record these — they fingerprint the exact model in use):")
        for name, digest in engine.model.artifact_sha256.items():
            print(f"    {name:24} {digest}")

    first = samples[0]
    d = engine.decide(first)
    print(f"  transaction {d.transaction_id}: {d.explanation}  ({d.latency_ms:.1f}ms)")
    all_ok &= check(d.action in policy.ACTIONS, "action is one of allow/step-up/block")
    all_ok &= check(not d.idempotent_replay, "first call is NOT a replay")

    raw = engine.audit.lookup(d.transaction_id)["raw_probability"]
    print(f"  raw probabilities by ensemble member: {raw}")
    all_ok &= check(
        isinstance(raw, dict) and "xgboost" in raw and "lightgbm" in raw,
        "audit record shows BOTH ensemble members' raw scores, not just the blended result",
    )

    # ------------------------------------------------------ 2. history store accumulates
    line("2. Client-history store accumulates real state between calls")
    # samples[0] and samples[1] are the SAME client's first two transactions in the test
    # month (see notebook 04's export) — samples[0] was already decided in step 1 above,
    # which is exactly why this step reuses it rather than a fresh pair: it proves the
    # accumulation is real across the two steps, not just within one carefully-staged call.
    uid_of_first = engine.audit.lookup(samples[0]["TransactionID"])["feature_vector"]["_uid"]
    stats_after_first = engine.store.get_fine(uid_of_first)
    print(f"  after client's 1st transaction (from step 1): prior-transaction count = {stats_after_first.n}")

    d_second = engine.decide(samples[1])
    fv_second = engine.audit.lookup(d_second.transaction_id)["feature_vector"]
    same_client = fv_second["_uid"] == uid_of_first
    print(f"  2nd transaction is the same client: {same_client}")
    print(f"  2nd transaction saw uid_amt_mean_prior = {fv_second['uid_amt_mean_prior']} "
          f"(should be non-None: it's now seen the 1st transaction)")
    all_ok &= check(same_client, "sample transactions 0 and 1 share a client, as exported")
    all_ok &= check(stats_after_first.n >= 1, "store recorded the 1st transaction")
    all_ok &= check(fv_second["uid_amt_mean_prior"] is not None,
                     "2nd transaction's features used real prior history, not a cold start")

    # ---------------------------------------------------------------- 3. idempotency
    line("3. Idempotency — same transaction_id twice, second call is NOT recomputed")
    d_repeat = engine.decide(first)  # same transaction as step 1
    print(f"  first call:  action={d.action}, idempotent_replay={d.idempotent_replay}")
    print(f"  second call: action={d_repeat.action}, idempotent_replay={d_repeat.idempotent_replay}")
    all_ok &= check(d_repeat.idempotent_replay, "second call flagged as an idempotent replay")
    all_ok &= check(d_repeat.action == d.action, "action is byte-identical across both calls")

    # ---------------------------------------------------------------- 4. replay + tamper
    line("4. Replayability — recompute a decision from its audit record, detect tampering")
    record = engine.audit.lookup(first["TransactionID"])
    ok, msg = verify_and_replay(record, policy)
    print(f"  {msg}")
    all_ok &= check(ok, "untampered record replays to the same decision")

    tampered = dict(record)
    tampered["action"] = "block" if record["action"] != "block" else "allow"
    ok2, msg2 = verify_and_replay(tampered, policy)
    print(f"  tampered record: {msg2}")
    all_ok &= check(not ok2, "tampered record is correctly rejected on replay")

    # ---------------------------------------------------------------- 5. fail-closed
    line("5. Fail-closed — model artifact unavailable, engine degrades, doesn't crash")
    hidden_path = os.path.join(ARTIFACTS_DIR, "model.json")
    backup_path = hidden_path + ".bak"
    shutil.move(hidden_path, backup_path)
    try:
        broken_engine = Engine(artifacts_dir=ARTIFACTS_DIR)
        all_ok &= check(broken_engine.model is None, "engine starts even with no model (doesn't crash)")
        fallback_txn = samples[3]
        d_fallback = broken_engine.decide(fallback_txn)
        print(f"  decision with no model: {d_fallback.explanation}")
        print(f"  fallbacks triggered: {d_fallback.fallbacks_triggered}")
        all_ok &= check(d_fallback.action == "step-up",
                         "fail-closed default is step-up (never a silent allow)")
        all_ok &= check("model_unavailable_fail_closed" in d_fallback.fallbacks_triggered,
                         "fallback is logged explicitly, not hidden")
    finally:
        shutil.move(backup_path, hidden_path)

    # ---------------------------------------------------------------- summary
    line("SUMMARY")
    print("ALL CHECKS PASSED" if all_ok else "SOME CHECKS FAILED — see [FAIL] lines above")
    sys.exit(0 if all_ok else 1)


if __name__ == "__main__":
    main()
