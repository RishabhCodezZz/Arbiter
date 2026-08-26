"""
Feature engineering for ONE transaction, matching notebooks/01-03 exactly.

Every formula here has a sibling in the notebooks. Nothing new is invented — this module's
only job is turning the batch/vectorized versions into per-transaction versions that a live
engine can actually call. Where a notebook comment explains WHY a design choice was made,
it isn't repeated here in full — see the notebooks for the reasoning; this stays close to
the mechanics.
"""
from typing import Optional

from .store import ClientHistoryStore

SECONDS_PER_DAY = 24 * 60 * 60


class FeatureBuildError(Exception):
    pass


def _uid(card1, addr1, d1n, d4n) -> str:
    return f"{card1}_{addr1}_{d1n}_{d4n}"


def _coarse_uid(card1, addr1, d1n) -> str:
    return f"{card1}_{addr1}_{d1n}"


def build(transaction: dict, store: ClientHistoryStore, manifest: dict) -> dict:
    """Returns a dict of {feature_name: value} in the SAME semantics as features_v3.
    Caller is responsible for turning this into a model-ready row (see model.py) — kept
    separate so this function is independently testable and its output is human-readable
    for the audit log.
    """
    txn = dict(transaction)  # don't mutate the caller's dict

    dt = txn.get("TransactionDT")
    if dt is None:
        raise FeatureBuildError("TransactionDT is required")

    day = dt // SECONDS_PER_DAY
    hour = (dt // 3600) % 24
    dow = day % 7

    d1, d4, d10, d15 = txn.get("D1"), txn.get("D4"), txn.get("D10"), txn.get("D15")
    d1n = (day - d1) if d1 is not None else None
    d4n = (day - d4) if d4 is not None else None
    d10n = (day - d10) if d10 is not None else None
    d15n = (day - d15) if d15 is not None else None

    card1, addr1 = txn.get("card1"), txn.get("addr1")
    uid = _uid(card1, addr1, d1n, d4n)
    coarse_uid = _coarse_uid(card1, addr1, d1n)

    # --- causal client-history features: PRIOR state only, looked up BEFORE this
    # transaction is scored. The store is updated separately, after scoring (see engine.py)
    # — this ordering is the entire causal-honesty guarantee, enforced structurally here,
    # not just by convention. ---
    prior = store.get_fine(uid)
    coarse_prior = store.get_coarse(coarse_uid)

    amount = txn.get("TransactionAmt")
    device = txn.get("DeviceInfo")
    email = txn.get("P_emaildomain")

    uid_amt_mean_prior = prior.mean()
    uid_amt_std_prior = prior.std()
    uid_amt_sum_prior = prior.amt_sum if prior.n > 0 else None
    uid_amt_sumsq_prior = prior.amt_sumsq if prior.n > 0 else None
    uid_time_since_prev = (dt - prior.last_txn_dt) if prior.last_txn_dt is not None else None
    uid_device_seen_before = 1 if (device is not None and device in prior.devices_seen) else 0
    uid_email_seen_before = 1 if (email is not None and email in prior.emails_seen) else 0
    uid_ambiguity_std_prior = coarse_prior.std()  # None if <2 prior D15n values, matches training

    features = dict(txn)
    features.update({
        "hour": hour, "dow": dow,
        "uid_amt_mean_prior": uid_amt_mean_prior,
        "uid_amt_std_prior": uid_amt_std_prior,
        "uid_amt_sum_prior": uid_amt_sum_prior,
        "uid_amt_sumsq_prior": uid_amt_sumsq_prior,
        "uid_time_since_prev": uid_time_since_prev,
        "uid_device_seen_before": uid_device_seen_before,
        "uid_email_seen_before": uid_email_seen_before,
        "uid_ambiguity_std_prior": uid_ambiguity_std_prior,
    })

    # categorical encoding, using the mapping FIT ON TRAIN and exported in the manifest —
    # unseen category or missing value -> -1, exactly matching notebooks 01-04.
    for col in manifest["categorical_columns"]:
        raw_val = features.get(col)
        mapping = manifest["categorical_mappings"].get(col, {})
        features[col] = mapping.get(str(raw_val), -1) if raw_val is not None else -1

    # stash identity/context fields the engine needs downstream (store update, audit log)
    # that are NOT model features themselves — namespaced with a leading underscore so
    # model.py can filter them out cleanly rather than by exclusion-list duplication.
    features["_uid"] = uid
    features["_coarse_uid"] = coarse_uid
    features["_day"] = day
    features["_d15n"] = d15n
    features["_amount"] = amount
    features["_device"] = device
    features["_email"] = email
    features["_dt"] = dt

    return features


def build_degraded(transaction: dict, manifest: dict) -> dict:
    """Fallback feature builder for when the primary path fails (e.g. the history store is
    unreachable). Every uid_* causal feature becomes None (XGBoost routes missing values
    natively — see notebooks/02's design note) rather than guessing a value. This widens
    effective uncertainty rather than pretending confidence the engine doesn't have —
    engine.py additionally narrows the step-up band when this path is used (see policy.py).
    """
    txn = dict(transaction)
    dt = txn.get("TransactionDT", 0)
    day = dt // SECONDS_PER_DAY
    features = dict(txn)
    features.update({
        "hour": (dt // 3600) % 24, "dow": day % 7,
        "uid_amt_mean_prior": None, "uid_amt_std_prior": None,
        "uid_amt_sum_prior": None, "uid_amt_sumsq_prior": None,
        "uid_time_since_prev": None, "uid_device_seen_before": 0,
        "uid_email_seen_before": 0, "uid_ambiguity_std_prior": None,
    })
    for col in manifest["categorical_columns"]:
        raw_val = features.get(col)
        mapping = manifest["categorical_mappings"].get(col, {})
        features[col] = mapping.get(str(raw_val), -1) if raw_val is not None else -1
    features["_uid"] = None
    features["_coarse_uid"] = None
    features["_day"] = day
    features["_d15n"] = None
    features["_amount"] = txn.get("TransactionAmt")
    features["_device"] = None
    features["_email"] = None
    features["_dt"] = dt
    return features
