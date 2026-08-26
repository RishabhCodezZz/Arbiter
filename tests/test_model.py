"""
Version-mismatch guards on FraudModel — proven by actually triggering them, not just
reading the code and assuming it works. Mirrors the fail-closed pattern already used in
tests/test_engine.py (copy real artifacts into a tmp dir, then break exactly one thing).

WHY THIS FILE EXISTS: the XGBoost version guard was previously verified only manually
("deliberately faked a mismatched manifest, confirmed it fires" — see journal/), never
captured as an automated regression test. Closing that gap here, for both ensemble
members, while the code is already being touched for the 2-model rebuild.
"""
import json
import shutil

import pytest

from src.model import FraudModel, LightGBMVersionMismatchError, XGBoostVersionMismatchError
from tests.conftest import requires_real_artifacts

_ARTIFACT_FILES = ("model.json", "calibrator.json", "model_lgb.txt", "calibrator_lgb.json",
                   "feature_manifest.json")


def _copy_real_artifacts(dest_dir):
    for name in _ARTIFACT_FILES:
        shutil.copy(f"artifacts/{name}", dest_dir / name)


def _corrupt_manifest_version(dest_dir, **overrides):
    manifest_path = dest_dir / "feature_manifest.json"
    with open(manifest_path) as f:
        manifest = json.load(f)
    manifest.update(overrides)
    with open(manifest_path, "w") as f:
        json.dump(manifest, f)


@requires_real_artifacts
def test_both_ensemble_members_load_with_correct_versions(tmp_path):
    """Sanity check the two mismatch tests below depend on: an UNMODIFIED copy of the real
    artifacts must load cleanly, with no override needed. Proves the mismatch tests are
    actually exercising the version check, not failing for some unrelated reason."""
    clean_dir = tmp_path / "artifacts_clean_copy"
    clean_dir.mkdir()
    _copy_real_artifacts(clean_dir)

    model = FraudModel(str(clean_dir))
    assert model.xgb_booster is not None
    assert model.lgb_booster is not None


@requires_real_artifacts
def test_xgboost_version_mismatch_refuses_to_score(tmp_path):
    broken_dir = tmp_path / "artifacts_bad_xgb_version"
    broken_dir.mkdir()
    _copy_real_artifacts(broken_dir)
    _corrupt_manifest_version(broken_dir, xgboost_version="0.0.0-deliberately-wrong")

    with pytest.raises(XGBoostVersionMismatchError):
        FraudModel(str(broken_dir))


@requires_real_artifacts
def test_lightgbm_version_mismatch_refuses_to_score(tmp_path):
    broken_dir = tmp_path / "artifacts_bad_lgb_version"
    broken_dir.mkdir()
    _copy_real_artifacts(broken_dir)
    _corrupt_manifest_version(broken_dir, lightgbm_version="0.0.0-deliberately-wrong")

    with pytest.raises(LightGBMVersionMismatchError):
        FraudModel(str(broken_dir))


@requires_real_artifacts
def test_xgboost_version_mismatch_override_allows_loading(tmp_path):
    """The explicit override exists for a real reason (an operator who has independently
    confirmed a mismatch is safe) — this proves it's actually usable, not just documented."""
    broken_dir = tmp_path / "artifacts_bad_xgb_version_override"
    broken_dir.mkdir()
    _copy_real_artifacts(broken_dir)
    _corrupt_manifest_version(broken_dir, xgboost_version="0.0.0-deliberately-wrong")

    model = FraudModel(str(broken_dir), allow_version_mismatch=True)
    assert model.xgb_booster is not None


@requires_real_artifacts
def test_lightgbm_version_mismatch_override_allows_loading(tmp_path):
    broken_dir = tmp_path / "artifacts_bad_lgb_version_override"
    broken_dir.mkdir()
    _copy_real_artifacts(broken_dir)
    _corrupt_manifest_version(broken_dir, lightgbm_version="0.0.0-deliberately-wrong")

    model = FraudModel(str(broken_dir), allow_version_mismatch=True)
    assert model.lgb_booster is not None
