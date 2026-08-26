"""
Shared fixtures. Tests run against the REAL trained model in artifacts/ (not a synthetic
stand-in) — same discipline as scripts/demo_engine.py, just converted into named,
independently-runnable pytest functions instead of one linear script.

Every test gets its own tmp_path for the audit log / client-history store, so tests never
share state with each other or with the real data/ directory a manual run of the engine
would use.
"""
import json
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.engine import Engine
from src.audit import AuditLog

ARTIFACTS_DIR = "artifacts"
SAMPLE_PATH = os.path.join(ARTIFACTS_DIR, "sample_transactions.json")
# The shipped model is a 2-model ensemble (XGBoost + LightGBM) — checking for the LightGBM
# artifacts too, not just the older XGBoost ones, so a repo that has the pre-ensemble
# artifacts but not yet the new ones skips cleanly with a clear reason instead of failing
# confusingly on a half-present artifacts/ directory.
_LGB_MODEL_PATH = os.path.join(ARTIFACTS_DIR, "model_lgb.txt")
_LGB_CALIB_PATH = os.path.join(ARTIFACTS_DIR, "calibrator_lgb.json")

requires_real_artifacts = pytest.mark.skipif(
    not (os.path.exists(SAMPLE_PATH) and os.path.exists(_LGB_MODEL_PATH)
         and os.path.exists(_LGB_CALIB_PATH)),
    reason=(
        f"{SAMPLE_PATH}, {_LGB_MODEL_PATH}, or {_LGB_CALIB_PATH} not found — download the "
        "real artifacts from a Kaggle run first (see artifacts/README.md). These tests "
        "deliberately run against the real trained ensemble, not a synthetic stand-in, so "
        "they can't run without all of it."
    ),
)


@pytest.fixture
def real_transactions():
    with open(SAMPLE_PATH) as f:
        return json.load(f)


@pytest.fixture
def engine(tmp_path):
    """A real Engine against the real artifacts, but with a throwaway store/audit path so
    tests never touch data/ or interfere with each other."""
    return Engine(
        artifacts_dir=ARTIFACTS_DIR,
        store_path=str(tmp_path / "client_history.json"),
        audit_path=str(tmp_path / "audit_log.jsonl"),
    )


@pytest.fixture
def empty_audit_log(tmp_path):
    return AuditLog(str(tmp_path / "audit_log.jsonl"))
