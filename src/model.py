"""
Loads the trained models + calibrators from the artifacts exported by notebooks/06's
engine-build addendum. CPU only — no GPU device is ever requested here, on purpose: this is
what proves the notebooks and this module are genuinely separable, per CLAUDE.md sec 7
("the cloned repo must run without a GPU").

Ships as a 2-model ensemble (XGBoost + LightGBM), both trained untuned on the exact same
feature set, simple-averaged after independent Platt calibration — confirmed as a real,
statistically significant rupee-value improvement over the single-XGBoost baseline (95% CI
[+Rs6.55L, +Rs21.24L] on the test month; see docs/experiments.md's ensemble sections for the
full evidence trail, including why a 3rd model (CatBoost) and further tuning were both
tried and NOT adopted).

Platt's two numbers are applied by hand (a plain sigmoid) rather than reloading a pickled
sklearn object, for both models — see notebook 06's engine-build addendum comment for why:
no sklearn-version fragility, and the calibration step stays auditable by anyone reading
this file.
"""
import json
import math
import os

import lightgbm as lgb
import pandas as pd
import xgboost as xgb


class ModelUnavailableError(Exception):
    """Raised when a model artifact is missing or fails to load. engine.py catches this
    specifically to trigger the fail-closed path — see policy.py's rules_baseline()."""
    pass


class XGBoostVersionMismatchError(ModelUnavailableError):
    """A subclass of ModelUnavailableError on purpose: engine.py treats it identically —
    fail closed to the rules baseline — rather than needing a second code path. See the
    journal: xgboost 3.0.0 vs the training version (3.2.0) gave a 23x different RAW
    probability (0.169 vs 0.007) on byte-identical input, with no error or warning of any
    kind on its own. That gap is large enough, and silent enough, that shipping a
    plausible-looking wrong number is worse than degrading to the conservative default."""
    pass


class LightGBMVersionMismatchError(ModelUnavailableError):
    """Same fail-closed reasoning as XGBoostVersionMismatchError. The LightGBM component
    hasn't been independently verified to produce a large cross-version probability gap the
    way the XGBoost 3.0/3.2 case was measured to — but an unverified cross-version
    assumption isn't a safe one to make silently either, so this refuses to score on a
    version mismatch rather than assume it's fine."""
    pass


def _sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x))


def _load_calibrator(calib_path: str, which: str) -> tuple[float, float]:
    """Load and fully validate a Platt calibrator JSON. Any problem — missing file,
    unparseable JSON, missing/non-numeric coef or intercept — becomes ModelUnavailableError
    so engine.py's single fail-closed catch handles it, instead of a KeyError/TypeError
    escaping FraudModel.__init__ and crashing engine construction."""
    try:
        with open(calib_path) as f:
            calib = json.load(f)
    except Exception as e:
        raise ModelUnavailableError(f"{which} calibrator failed to load ({calib_path}): {e}") from e
    if not isinstance(calib, dict) or "coef" not in calib or "intercept" not in calib:
        raise ModelUnavailableError(
            f"{which} calibrator {calib_path} is missing 'coef'/'intercept' — "
            f"got keys {sorted(calib) if isinstance(calib, dict) else type(calib).__name__}"
        )
    try:
        return float(calib["coef"]), float(calib["intercept"])
    except (TypeError, ValueError) as e:
        raise ModelUnavailableError(
            f"{which} calibrator {calib_path} has non-numeric coef/intercept: {e}"
        ) from e


class _CalibratedSubModel:
    """One member of the ensemble (XGBoost or LightGBM), holding its own raw-scoring
    function and its own independently-fit Platt calibrator. Not exposed outside this
    module — FraudModel composes two of these and averages their calibrated output."""

    def __init__(self, name: str, raw_fn, coef: float, intercept: float):
        self.name = name
        self._raw_fn = raw_fn
        self._coef = coef
        self._intercept = intercept

    def raw(self, X: pd.DataFrame) -> float:
        return self._raw_fn(X)

    def calibrate(self, raw: float) -> float:
        return _sigmoid(self._coef * raw + self._intercept)


class FraudModel:
    """The 2-model ensemble: XGBoost + LightGBM, each independently calibrated, then
    simple-averaged — exactly the configuration validated in notebooks/06, not a stacking
    meta-learner and not weighted. The averaged calibrated probability is what drives
    every decide_action() call; both models' own raw scores are kept for audit
    transparency (see engine.py/audit.py) rather than only exposing the average."""

    def __init__(self, artifacts_dir: str, allow_version_mismatch: bool = False):
        manifest_path = os.path.join(artifacts_dir, "feature_manifest.json")
        if not os.path.exists(manifest_path):
            raise ModelUnavailableError(f"missing artifact: {manifest_path}")
        try:
            with open(manifest_path) as f:
                self.manifest = json.load(f)
        except Exception as e:
            raise ModelUnavailableError(f"artifact failed to load: {e}") from e

        # Validate the manifest schema here — a syntactically valid but incomplete manifest
        # (e.g. `{}`) must fail closed, not raise a bare KeyError past engine.py's catch.
        if not isinstance(self.manifest, dict):
            raise ModelUnavailableError(f"manifest {manifest_path} is not a JSON object")
        for key in ("features", "categorical_columns", "categorical_mappings"):
            if key not in self.manifest:
                raise ModelUnavailableError(f"manifest {manifest_path} is missing required key '{key}'")
        if not isinstance(self.manifest["features"], list) or not self.manifest["features"]:
            raise ModelUnavailableError(f"manifest {manifest_path} 'features' must be a non-empty list")

        self.feature_order = self.manifest["features"]

        # Both sub-models are trained on the exact same feature_order (LightGBM was fit on
        # the identical X_tr3 as XGBoost — see notebooks/06's ensemble diagnostic), so one
        # shared feature build below is correct for both, not an approximation.
        self._xgb = self._load_xgb(artifacts_dir, allow_version_mismatch)
        self._lgb = self._load_lgb(artifacts_dir, allow_version_mismatch)

    def _load_xgb(self, artifacts_dir: str, allow_version_mismatch: bool) -> _CalibratedSubModel:
        model_path = os.path.join(artifacts_dir, "model.json")
        calib_path = os.path.join(artifacts_dir, "calibrator.json")
        for p in (model_path, calib_path):
            if not os.path.exists(p):
                raise ModelUnavailableError(f"missing artifact: {p}")
        try:
            booster = xgb.XGBClassifier()
            booster.load_model(model_path)
        except Exception as e:
            raise ModelUnavailableError(f"xgboost artifact failed to load: {e}") from e
        coef, intercept = _load_calibrator(calib_path, "xgboost")

        trained_version = self.manifest.get("xgboost_version")
        running_version = xgb.__version__
        if trained_version is not None and trained_version != running_version and not allow_version_mismatch:
            raise XGBoostVersionMismatchError(
                f"XGBoost component was trained with xgboost=={trained_version}, but "
                f"xgboost=={running_version} is running locally. Verified empirically that "
                f"this specific gap produces a 23x different probability on identical "
                f"input — refusing to score rather than serve a silently wrong number. "
                f"Fix: pip install xgboost=={trained_version} (see requirements.txt)."
            )

        self.xgb_booster = booster  # exposed for src/explain.py's SHAP TreeExplainer
        return _CalibratedSubModel(
            "xgboost",
            lambda X: float(booster.predict_proba(X)[0, 1]),
            coef=coef, intercept=intercept,
        )

    def _load_lgb(self, artifacts_dir: str, allow_version_mismatch: bool) -> _CalibratedSubModel:
        model_path = os.path.join(artifacts_dir, "model_lgb.txt")
        calib_path = os.path.join(artifacts_dir, "calibrator_lgb.json")
        for p in (model_path, calib_path):
            if not os.path.exists(p):
                raise ModelUnavailableError(f"missing artifact: {p}")
        try:
            booster = lgb.Booster(model_file=model_path)
        except Exception as e:
            raise ModelUnavailableError(f"lightgbm artifact failed to load: {e}") from e
        coef, intercept = _load_calibrator(calib_path, "lightgbm")

        trained_version = self.manifest.get("lightgbm_version")
        running_version = lgb.__version__
        if trained_version is not None and trained_version != running_version and not allow_version_mismatch:
            raise LightGBMVersionMismatchError(
                f"LightGBM component was trained with lightgbm=={trained_version}, but "
                f"lightgbm=={running_version} is running locally — refusing to score on an "
                f"unverified cross-version assumption, same fail-closed principle as the "
                f"XGBoost guard. Fix: pip install lightgbm=={trained_version} (see "
                f"requirements.txt)."
            )

        self.lgb_booster = booster  # exposed for src/explain.py's SHAP TreeExplainer
        # LightGBM's Booster.predict() returns the sigmoid-applied probability directly for
        # a binary objective (the model's own saved config, not something set here) — the
        # same convention as XGBoost's predict_proba()[:, 1], so both feed their calibrator
        # a genuine probability, not a raw margin.
        return _CalibratedSubModel(
            "lightgbm",
            lambda X: float(booster.predict(X)[0]),
            coef=coef, intercept=intercept,
        )

    def to_dataframe(self, feature_dict: dict) -> pd.DataFrame:
        """Takes the dict produced by features.build() — filters out the underscore-
        prefixed context fields, orders the rest to match training exactly. One shared
        DataFrame for both ensemble members (see class docstring for why that's correct,
        not an approximation). Exposed separately from score() so src/explain.py's SHAP
        explainers can run against the EXACT SAME row each model just scored, rather than
        rebuilding it and risking drift."""
        row = {k: feature_dict.get(k) for k in self.feature_order}
        X = pd.DataFrame([row], columns=self.feature_order)
        # None -> NaN so each model's native missing-value routing applies, exactly as in
        # training. pd.to_numeric (not .astype) because it coerces cleanly without the
        # version-dependent behavior of astype's errors= parameter.
        return X.apply(pd.to_numeric, errors="coerce")

    def score_df(self, X: pd.DataFrame) -> tuple[dict, float]:
        """Scores an already-built row (see to_dataframe). Returns
        ({"xgboost": raw_xgb, "lightgbm": raw_lgb}, calibrated_ensemble_average) — the
        averaged, calibrated value is what actually drives decide_action(); the raw dict is
        kept so the audit record shows each component's own contribution, not just the
        blended result (CLAUDE.md sec 8's audit trail is meant to show HOW a number was
        reached, not just what it was). Split out from score() so engine.py can build X
        once and feed it to both this AND the SHAP explainers."""
        raw = {"xgboost": self._xgb.raw(X), "lightgbm": self._lgb.raw(X)}
        calibrated = (self._xgb.calibrate(raw["xgboost"]) + self._lgb.calibrate(raw["lightgbm"])) / 2
        return raw, calibrated

    def score(self, feature_dict: dict) -> tuple[dict, float]:
        """Returns (raw_probabilities_dict, calibrated_ensemble_probability). Convenience
        wrapper around to_dataframe + score_df for callers that don't need the intermediate
        X (most tests, notebooks). engine.py itself calls the two steps separately."""
        return self.score_df(self.to_dataframe(feature_dict))
