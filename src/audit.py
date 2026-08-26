"""
Append-only audit log. Every decide() call writes exactly one record here — this IS the
"full paper trail behind every decision" the whole project keeps promising. Two properties
matter more than the storage format: (1) idempotency — looking a transaction_id up here
happens BEFORE any scoring, so a duplicate/replayed transaction returns the original
decision rather than being re-decided; (2) replayability — a record must contain enough to
re-derive its own decision, not just describe it after the fact.

JSONL (one JSON object per line), append-only, never rewritten in place — a real audit
trail doesn't get edited.
"""
import hashlib
import json
import os
import time
from dataclasses import dataclass, asdict
from typing import Optional


@dataclass
class AuditRecord:
    transaction_id: str
    timestamp: float
    model_version: str
    feature_vector: dict          # human-readable causal features, for the replay check
    feature_vector_hash: str      # tamper-evidence: recomputed and checked before replay
    raw_probability: Optional[dict]   # {"xgboost": ..., "lightgbm": ...} — each ensemble
                                       # member's own raw score, kept for audit transparency
                                       # (not just the blended result). None on the
                                       # fail-closed path, same as before.
    calibrated_probability: Optional[float]   # the ensemble AVERAGE — this is what
                                                # decide_action() actually saw
    cost_params: dict             # the exact parameters in force at decision time
    action: str
    action_values: dict           # value_allow/stepup/block, or {} on the fallback path
    latency_ms: float
    fallbacks_triggered: list
    idempotent_replay: bool = False   # True if this call returned an EXISTING decision
    degraded: bool = False            # True if decided via policy.decide_action(degraded=True)
                                       # MUST be replayed with the same flag — see verify_and_replay
    shap_contributions: Optional[list] = None   # top-k, flagged transactions only
    narrative: Optional[str] = None
    used_llm: bool = False


def _hash_features(feature_vector: dict) -> str:
    # sorted keys -> stable hash regardless of dict insertion order
    encoded = json.dumps(feature_vector, sort_keys=True, default=str).encode()
    return hashlib.sha256(encoded).hexdigest()


class AuditLog:
    def __init__(self, path: str):
        self.path = path
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        if not os.path.exists(path):
            open(path, "w").close()

    def lookup(self, transaction_id) -> Optional[dict]:
        """Linear scan — fine for a hackathon demo's data volume. A production version
        swaps this for an indexed store without engine.py or features.py changing at all,
        same interface-isolation reasoning as store.py.

        Coerces to str defensively: engine.decide() always stores transaction_id as a
        str, but a caller handing this a raw pandas/numpy int (e.g. straight from a
        DataFrame row) would otherwise silently get a false "not found" — caught this
        exact mismatch while writing the engine's smoke test, fixed here rather than only in
        the one call site that happened to trip over it."""
        transaction_id = str(transaction_id)
        if not os.path.exists(self.path):
            return None
        with open(self.path) as f:
            for line in f:
                if not line.strip():
                    continue
                rec = json.loads(line)
                if rec["transaction_id"] == transaction_id:
                    return rec
        return None

    def append(self, record: AuditRecord) -> None:
        with open(self.path, "a") as f:
            f.write(json.dumps(asdict(record)) + "\n")

    def make_record(self, transaction_id, model_version, feature_vector, raw_prob,
                     calibrated_prob, cost_params, action, action_values, latency_ms,
                     fallbacks, idempotent_replay=False, degraded=False,
                     shap_contributions=None, narrative=None, used_llm=False) -> AuditRecord:
        return AuditRecord(
            transaction_id=transaction_id,
            timestamp=time.time(),
            model_version=model_version,
            feature_vector=feature_vector,
            feature_vector_hash=_hash_features(feature_vector),
            raw_probability=raw_prob,
            calibrated_probability=calibrated_prob,
            cost_params=cost_params,
            action=action,
            action_values=action_values,
            latency_ms=latency_ms,
            fallbacks_triggered=fallbacks,
            idempotent_replay=idempotent_replay,
            degraded=degraded,
            shap_contributions=shap_contributions,
            narrative=narrative,
            used_llm=used_llm,
        )


def verify_and_replay(record: dict, policy_module) -> tuple[bool, str]:
    """THE replay proof: given a stored record, (1) recompute the feature-vector hash and
    confirm it matches what was stored — proves the record wasn't tampered with after the
    fact — then (2) re-run the SAME policy decision from the stored probability and amount,
    and assert it matches the stored action. Returns (ok, message).

    Deliberately does NOT re-run the model (that would need the exact model artifact and
    feature vector reconstruction) — it replays the POLICY layer, which is the part that
    must be perfectly deterministic and auditable. The model-scoring path is proven
    separately by the correctness checks in notebook 02/03.
    """
    recomputed_hash = _hash_features(record["feature_vector"])
    if recomputed_hash != record["feature_vector_hash"]:
        return False, "TAMPER DETECTED: feature vector hash does not match stored hash"

    if record["calibrated_probability"] is None:
        # fallback-path record — nothing to replay against the cost model, but the
        # tamper check above still applies and already passed if we reach here.
        return True, "fallback-path record: hash verified, no policy replay applicable"

    amount_usd = record["feature_vector"].get("_amount")
    # MUST pass the same `degraded` flag the original decision used — a decision made
    # under a widened step-up band would otherwise be replayed against the NORMAL
    # boundaries and could produce a different action, falsely flagged as tampered even
    # though nothing was actually altered. Caught this while wiring degraded-mode support
    # into engine.py: verify_and_replay was written before `degraded` existed.
    replayed_action, replayed_values = policy_module.decide_action(
        record["calibrated_probability"], amount_usd, degraded=record.get("degraded", False)
    )
    if replayed_action != record["action"]:
        return False, (f"REPLAY MISMATCH: stored action was '{record['action']}', "
                        f"replay produced '{replayed_action}'")
    return True, f"replay OK: '{replayed_action}' reproduced exactly from stored inputs"
