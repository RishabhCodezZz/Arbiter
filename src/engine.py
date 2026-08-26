"""
decide(transaction) -> Decision

The one function everything else in src/ exists to support. Deliberately small — the real
engineering is in features.py/model.py/policy.py/audit.py/explain.py/narrative.py; this
file just sequences them in the right order and makes sure nothing here can silently do
the wrong thing.

INVARIANTS THIS FILE ENFORCES:
  - idempotency check happens before ANY scoring work
  - the history store is updated AFTER scoring, never before (a transaction must not see
    its own effect on its own score)
  - SHAP + narrative are computed ONLY for flagged (non-"allow") transactions
  - the LLM's own failure can NEVER change the action — narrative.narrate() never raises,
    and the action is decided before narrate() is ever called
Get any of these orderings wrong and CLAUDE.md's "causal, replayable, idempotent, the LLM
never decides" claims become false in the one place that actually matters.
"""
import time
from dataclasses import dataclass
from typing import Optional

from . import features as features_mod
from . import narrative as narrative_mod
from . import policy
from .audit import AuditLog
from .explain import Explainer
from .model import FraudModel, ModelUnavailableError
from .store import ClientHistoryStore

MODEL_VERSION = "v6-ensemble-xgb-lgb-platt"
_EMPTY_MANIFEST = {"categorical_columns": [], "categorical_mappings": {}}


@dataclass
class Decision:
    transaction_id: str
    action: str
    calibrated_probability: Optional[float]
    latency_ms: float
    fallbacks_triggered: list
    idempotent_replay: bool
    explanation: str
    narrative: Optional[str] = None
    used_llm: bool = False


class Engine:
    def __init__(self, artifacts_dir: str = "artifacts",
                 store_path: str = "data/client_history.json",
                 audit_path: str = "data/audit_log.jsonl"):
        self.artifacts_dir = artifacts_dir
        self.store = ClientHistoryStore(store_path)
        self.audit = AuditLog(audit_path)
        try:
            self.model = FraudModel(artifacts_dir)
            self.manifest = self.model.manifest
        except ModelUnavailableError:
            # Deliberately does NOT raise here. A model that's unavailable at STARTUP is
            # exactly the scenario decide() has to survive per transaction — constructing
            # the engine with no model proves the fail-closed path works from a cold start,
            # not just mid-session. self.model stays None; decide() checks for this.
            self.model = None
            self.manifest = None
        self._explainer = None  # built lazily — see _get_explainer()

    def _get_explainer(self) -> Explainer:
        """SHAP TreeExplainer construction has a real one-time cost, now for both ensemble
        members. Building it lazily means traffic that's entirely "allow" (the large
        majority — 95.8% measured pre-ensemble in notebooks/04) never pays for it at all."""
        if self._explainer is None:
            self._explainer = Explainer(self.model.xgb_booster, self.model.lgb_booster)
        return self._explainer

    def decide(self, transaction: dict) -> Decision:
        txn_id = str(transaction["TransactionID"])

        # --- idempotency: check BEFORE any scoring work. A duplicate/replayed transaction
        # returns the ORIGINAL decision, not a freshly (possibly differently) computed one.
        existing = self.audit.lookup(txn_id)
        if existing is not None:
            return Decision(
                transaction_id=txn_id,
                action=existing["action"],
                calibrated_probability=existing["calibrated_probability"],
                latency_ms=0.0,
                fallbacks_triggered=existing["fallbacks_triggered"],
                idempotent_replay=True,
                explanation=f"idempotent replay of a decision already made for {txn_id}",
                narrative=existing.get("narrative"),
                used_llm=existing.get("used_llm", False),
            )

        t0 = time.monotonic()
        fallbacks = []
        degraded = False

        # --- feature build, with a degraded fallback if the primary path fails ---
        try:
            fv = features_mod.build(transaction, self.store, self.manifest or _EMPTY_MANIFEST)
        except Exception as e:
            fallbacks.append(f"feature_build_failed:{type(e).__name__}")
            fv = features_mod.build_degraded(transaction, self.manifest or _EMPTY_MANIFEST)
            degraded = True

        # --- score, with the fail-closed rules baseline if the model is unavailable ---
        shap_contributions = None
        narrative_text, used_llm = None, False

        if self.model is None:
            fallbacks.append("model_unavailable_fail_closed")
            raw_prob, calibrated_prob = None, None
            action, values = policy.rules_baseline(fv.get("_amount") or 0.0)
        else:
            try:
                X = self.model.to_dataframe(fv)
                raw_prob, calibrated_prob = self.model.score_df(X)
            except Exception as e:
                fallbacks.append(f"scoring_failed_fail_closed:{type(e).__name__}")
                raw_prob, calibrated_prob = None, None
                action, values = policy.rules_baseline(fv.get("_amount") or 0.0)
            else:
                action, values = policy.decide_action(
                    calibrated_prob, fv.get("_amount") or 0.0, degraded=degraded
                )

                # --- SHAP + narrative: ONLY for flagged transactions, per CLAUDE.md sec 6
                # ("nobody reads 570,000 explanations"). A failure here NEVER touches the
                # action above — it was already decided — only the prose degrades.
                if action != "allow":
                    try:
                        shap_contributions = self._get_explainer().explain(X, top_k=5)
                    except Exception as e:
                        fallbacks.append(f"shap_failed:{type(e).__name__}")
                        shap_contributions = []
                    amount_inr = (fv.get("_amount") or 0.0) * policy.USD_TO_INR
                    narrative_text, used_llm, narrative_fallbacks = narrative_mod.narrate(
                        action, calibrated_prob, amount_inr, shap_contributions
                    )
                    fallbacks.extend(narrative_fallbacks)

        latency_ms = (time.monotonic() - t0) * 1000

        cost_params = {
            "chargeback_fee": policy.CHARGEBACK_FEE, "mdr_rate": policy.MDR_RATE,
            "margin": policy.MARGIN, "p_stop": policy.P_STOP,
            "p_dropoff": policy.P_DROPOFF, "ltv_multiplier": policy.LTV_MULTIPLIER,
            "usd_to_inr": policy.USD_TO_INR,
        }
        record = self.audit.make_record(
            transaction_id=txn_id, model_version=MODEL_VERSION, feature_vector=fv,
            raw_prob=raw_prob, calibrated_prob=calibrated_prob, cost_params=cost_params,
            action=action, action_values=values, latency_ms=latency_ms, fallbacks=fallbacks,
            degraded=degraded, shap_contributions=shap_contributions,
            narrative=narrative_text, used_llm=used_llm,
        )
        self.audit.append(record)

        # --- update history store AFTER the decision, never before: this transaction
        # must not be able to see its own effect on its own score. ---
        if fv.get("_uid") is not None:
            self.store.update(
                uid=fv["_uid"], coarse_uid=fv["_coarse_uid"], amount=fv["_amount"] or 0.0,
                txn_dt=fv["_dt"], device=fv.get("_device"), email=fv.get("_email"),
                d15n=fv.get("_d15n"),
            )

        explanation = (f"p={calibrated_prob:.4f} -> {action}" if calibrated_prob is not None
                        else f"model unavailable -> fail-closed default: {action}")
        return Decision(
            transaction_id=txn_id, action=action, calibrated_probability=calibrated_prob,
            latency_ms=latency_ms, fallbacks_triggered=fallbacks,
            idempotent_replay=False, explanation=explanation,
            narrative=narrative_text, used_llm=used_llm,
        )
