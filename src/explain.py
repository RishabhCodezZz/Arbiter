"""
SHAP explanations for the 2-model ensemble — computed ONLY for flagged transactions
(step-up/block), never for allowed ones. Per CLAUDE.md sec 6: "compute SHAP only for
flagged transactions, not the full stream — nobody reads 570,000 explanations." It's also
what makes narrative.py's LLM call safe: the LLM never invents a reason, it only describes
real per-transaction Shapley contributions computed here. Delete this module and the LLM
has nothing true left to say.

ENSEMBLE NOTE: the shipped decision is the AVERAGE of two independently-calibrated models
(see model.py). Explaining it averages each model's own SHAP contribution per feature —
standard, defensible practice for tree-ensemble explanation (both XGBoost and LightGBM are
gradient-boosted trees on a log-loss objective, so their TreeExplainer values live in a
comparable additive log-odds-ish space), not a claim that this is an exact causal
decomposition of the blended, calibrated probability. Each model's own contribution is kept
alongside the average, not hidden, so a reviewer can see the two views agree or disagree.
"""
import numpy as np
import pandas as pd
import shap


class Explainer:
    def __init__(self, xgb_booster, lgb_booster):
        # TreeExplainer walks the tree structure directly — exact Shapley values for tree
        # models, no background dataset needed the way KernelExplainer would require. One
        # explainer per ensemble member.
        self._xgb_explainer = shap.TreeExplainer(xgb_booster)
        self._lgb_explainer = shap.TreeExplainer(lgb_booster)

    def explain(self, X: pd.DataFrame, top_k: int = 5) -> list[dict]:
        """X is the SAME single-row DataFrame model.py just scored for BOTH ensemble
        members — same feature order, same values, so the explanation can never drift from
        what was actually predicted. Averages each model's per-feature contribution, then
        returns the top_k by |average contribution|, sorted descending."""
        xgb_values = _flatten_to_1d(self._xgb_explainer.shap_values(X))
        lgb_values = _flatten_to_1d(self._lgb_explainer.shap_values(X))

        contributions = []
        for feat, val, xgb_c, lgb_c in zip(X.columns, X.iloc[0].values, xgb_values, lgb_values):
            avg_contrib = float((xgb_c + lgb_c) / 2)
            contributions.append({
                "feature": feat,
                "value": _clean(val),
                "contribution": avg_contrib,
                "xgboost_contribution": float(xgb_c),
                "lightgbm_contribution": float(lgb_c),
            })
        contributions.sort(key=lambda c: abs(c["contribution"]), reverse=True)
        return contributions[:top_k]


def _flatten_to_1d(shap_values) -> np.ndarray:
    """SHAP's return shape varies by version and model type — sometimes a list (one array
    per class), sometimes a single (n_rows, n_features) array, occasionally an explicit
    3D (n_classes, n_rows, n_features) array. Normalise all of them to one 1D array for
    the single row and single (positive) class we care about."""
    if isinstance(shap_values, list):
        # binary classification: [negative_class_values, positive_class_values]
        arr = np.asarray(shap_values[-1])
    else:
        arr = np.asarray(shap_values)

    if arr.ndim == 3:
        arr = arr[-1, 0, :]
    elif arr.ndim == 2:
        arr = arr[0, :]
    return arr.ravel()


def _clean(v):
    if v is None:
        return None
    try:
        if np.isnan(v):
            return None
    except TypeError:
        pass
    if isinstance(v, (int, float, np.floating, np.integer)):
        return float(v)
    return v
