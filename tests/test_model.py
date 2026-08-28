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

from src.model import (
    FraudModel,
    LightGBMVersionMismatchError,
    ModelUnavailableError,
    XGBoostVersionMismatchError,
)
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


# --- malformed (not just missing) artifacts must ALSO fail closed --------------------
# These don't need the real model files: the manifest schema is validated before any
# model is loaded, so a bad manifest raises ModelUnavailableError first. Caught in review:
# `self.manifest["features"]` was outside the try/except, so a `{}` manifest raised a bare
# KeyError past engine.py's `except ModelUnavailableError`, crashing engine construction.

@pytest.mark.parametrize("bad_manifest", [
    {},                                                   # empty object
    {"categorical_columns": [], "categorical_mappings": {}},   # missing "features"
    {"features": [], "categorical_columns": [], "categorical_mappings": {}},  # empty list
    {"features": "not-a-list", "categorical_columns": [], "categorical_mappings": {}},
    [1, 2, 3],                                            # not even an object
])
def test_malformed_manifest_raises_model_unavailable(tmp_path, bad_manifest):
    d = tmp_path / "artifacts_bad_manifest"
    d.mkdir()
    with open(d / "feature_manifest.json", "w") as f:
        json.dump(bad_manifest, f)
    with pytest.raises(ModelUnavailableError):
        FraudModel(str(d))


def test_unparseable_manifest_raises_model_unavailable(tmp_path):
    d = tmp_path / "artifacts_unparseable_manifest"
    d.mkdir()
    (d / "feature_manifest.json").write_text("{ this is not json")
    with pytest.raises(ModelUnavailableError):
        FraudModel(str(d))


def test_engine_survives_a_malformed_manifest(tmp_path):
    """The whole point: a `{}` manifest must land the Engine in the fail-closed state
    (self.model is None), not crash __init__ with an uncaught KeyError."""
    from src.engine import Engine

    d = tmp_path / "artifacts_bad_manifest_engine"
    d.mkdir()
    (d / "feature_manifest.json").write_text("{}")
    e = Engine(
        artifacts_dir=str(d),
        store_path=str(tmp_path / "store.json"),
        audit_path=str(tmp_path / "audit.jsonl"),
    )
    assert e.model is None


@requires_real_artifacts
@pytest.mark.parametrize("bad_calib", ['{}', '{"coef": 1.0}', '{"coef": "x", "intercept": 0.0}', 'nope'])
def test_malformed_calibrator_raises_model_unavailable(tmp_path, bad_calib):
    d = tmp_path / "artifacts_bad_calib"
    d.mkdir()
    _copy_real_artifacts(d)
    (d / "calibrator.json").write_text(bad_calib)
    with pytest.raises(ModelUnavailableError):
        FraudModel(str(d))


@requires_real_artifacts
@pytest.mark.parametrize("bad_calib", [
    '{"coef": NaN, "intercept": 0.0}',          # json.load parses NaN by default
    '{"coef": Infinity, "intercept": 0.0}',
    '{"coef": 1.0, "intercept": -Infinity}',
])
def test_non_finite_calibrator_raises_model_unavailable(tmp_path, bad_calib):
    """A NaN/Inf coef would sail through `float(...)` and make every calibrated probability
    NaN — which decide_action()'s max() resolves to 'allow'. Must fail closed at load."""
    d = tmp_path / "artifacts_nonfinite_calib"
    d.mkdir()
    _copy_real_artifacts(d)
    (d / "calibrator.json").write_text(bad_calib)
    with pytest.raises(ModelUnavailableError):
        FraudModel(str(d))


@pytest.mark.parametrize("bad_manifest", [
    {"features": ["a"], "categorical_columns": "not-a-list", "categorical_mappings": {}},
    {"features": ["a"], "categorical_columns": [], "categorical_mappings": []},
    {"features": ["a"], "categorical_columns": ["X"], "categorical_mappings": {}},          # X has no mapping
    {"features": ["a"], "categorical_columns": ["X"], "categorical_mappings": {"X": "nope"}},  # non-object mapping
])
def test_corrupt_categorical_schema_raises_model_unavailable(tmp_path, bad_manifest):
    """A categorical column with no (or a non-object) mapping would silently turn every
    value of that column into the -1 'unseen' sentinel. Fail closed instead."""
    d = tmp_path / "artifacts_bad_cat_schema"
    d.mkdir()
    with open(d / "feature_manifest.json", "w") as f:
        json.dump(bad_manifest, f)
    with pytest.raises(ModelUnavailableError):
        FraudModel(str(d))


@requires_real_artifacts
def test_non_finite_ensemble_probability_fails_closed(tmp_path):
    """If a sub-model ever emits a NaN (degenerate row, corrupt booster), score_df() must
    raise rather than return it — engine.py then fails closed. Simulated by poisoning one
    calibrator coefficient after load."""
    clean = tmp_path / "artifacts_clean"
    clean.mkdir()
    _copy_real_artifacts(clean)
    model = FraudModel(str(clean))
    model._xgb._coef = float("nan")  # any raw score -> NaN calibrated -> NaN average
    X = model.to_dataframe({k: None for k in model.feature_order})
    with pytest.raises(ModelUnavailableError):
        model.score_df(X)
