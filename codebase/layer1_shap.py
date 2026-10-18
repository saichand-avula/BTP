#!/usr/bin/env python3
"""Held-out TreeSHAP analysis for the selected Layer 1 model."""
from pathlib import Path
import pickle
import numpy as np
import pandas as pd
import shap
from scipy.stats import spearmanr

BASE = Path(__file__).parent
DATA = BASE / "data_processed" / "diabetic_data_cleaned.csv"
MODELS = BASE / "models"
OUT_IMPORTANCE = MODELS / "shap_global_importance.csv"
OUT_CASE = MODELS / "shap_high_risk_case.csv"
OUT_VALUES = MODELS / "shap_values_heldout.npy"
OUT_ROBUSTNESS = MODELS / "explainer_robustness.csv"
ROBUSTNESS_ROWS = 2000
KERNEL_BACKGROUND_ROWS = 20
KERNEL_NSAMPLES = 256

def _test_data(df):
    from layer1 import _split
    _, _, test = _split(df)
    return test


def _as_positive_class_values(explanation):
    """Normalise SHAP's version-dependent binary-class output shape."""
    values = explanation.values if hasattr(explanation, "values") else explanation
    if isinstance(values, list):
        values = values[1]
    values = np.asarray(values)
    if values.ndim == 3:
        values = values[:, :, 1]
    return values


def _rank(values, features):
    return pd.Series(np.abs(values).mean(axis=0), index=features).sort_values(ascending=False)


def _comparison_row(reference_name, reference_rank, method_name, method_rank, n_rows):
    reference_top = set(reference_rank.head(15).index)
    method_top = set(method_rank.head(15).index)
    rho, _ = spearmanr(reference_rank.reindex(reference_rank.index),
                       method_rank.reindex(reference_rank.index))
    return {
        "reference_method": reference_name,
        "comparison_method": method_name,
        "n_heldout_rows": n_rows,
        "top15_overlap_count": len(reference_top & method_top),
        "top15_overlap_fraction": round(len(reference_top & method_top) / 15, 4),
        "spearman_mean_abs_shap": round(float(rho), 4),
        "kernel_background_rows": KERNEL_BACKGROUND_ROWS if method_name == "KernelSHAP" else "",
        "kernel_nsamples": KERNEL_NSAMPLES if method_name == "KernelSHAP" else "",
    }


def _write_explainer_robustness(model, x, features):
    """Compare global ranks on a fixed 2,000-row held-out subset.

    KernelSHAP is Monte-Carlo approximated (``kernel_nsamples`` is written to
    the CSV) so the artefact does not imply an exact exhaustive calculation.
    """
    n = min(ROBUSTNESS_ROWS, len(x))
    x_eval = x.iloc[:n].copy()
    path_values = _as_positive_class_values(shap.TreeExplainer(model)(x_eval))
    path_rank = _rank(path_values, features)

    background = x.iloc[:min(100, len(x))]
    interventional_values = _as_positive_class_values(
        shap.TreeExplainer(model, data=background,
                           feature_perturbation="interventional")(x_eval))
    interventional_rank = _rank(interventional_values, features)

    kernel_background = x.iloc[:min(KERNEL_BACKGROUND_ROWS, len(x))]
    # A wrapper avoids SHAP mutating XGBoost's read-only feature_names_in_.
    def kernel_predict(array):
        return model.predict_proba(pd.DataFrame(array, columns=features))
    kernel = shap.KernelExplainer(kernel_predict, kernel_background)
    kernel_values = _as_positive_class_values(
        kernel.shap_values(x_eval, nsamples=KERNEL_NSAMPLES, silent=True))
    kernel_rank = _rank(kernel_values, features)

    rows = [
        _comparison_row("TreeSHAP_path_dependent", path_rank,
                        "TreeSHAP_interventional", interventional_rank, n),
        _comparison_row("TreeSHAP_path_dependent", path_rank,
                        "KernelSHAP", kernel_rank, n),
        _comparison_row("TreeSHAP_interventional", interventional_rank,
                        "KernelSHAP", kernel_rank, n),
    ]
    pd.DataFrame(rows).to_csv(OUT_ROBUSTNESS, index=False)
    return pd.DataFrame(rows)

def run():
    with open(MODELS / "layer1_model.pkl", "rb") as f:
        model = pickle.load(f)
    with open(MODELS / "layer1_features.pkl", "rb") as f:
        features_safe = pickle.load(f)   # safe (sanitized) names
    raw_df = pd.read_csv(DATA)

    # Build safe-name mapping so we can restore readable names in output
    safe_map = {c: c.replace("[","_").replace("]","_").replace("<","lt").replace(">","gt")
                for c in raw_df.columns}
    inv_map  = {v: k for k, v in safe_map.items()}  # safe -> original
    features_orig = [inv_map.get(f, f) for f in features_safe]

    test_raw = _test_data(raw_df)
    test_safe = test_raw.rename(columns=safe_map)
    x = test_safe[features_safe]

    explanation = shap.TreeExplainer(model)(x)
    values = _as_positive_class_values(explanation)
    np.save(OUT_VALUES, values)
    # Mean absolute SHAP gives ranking; correlation gives directional sign.
    directions = []
    for index in range(len(features_safe)):
        xv = x.iloc[:, index].to_numpy(dtype=float)
        sv = values[:, index]
        if np.std(xv) == 0 or np.std(sv) == 0:
            directions.append(np.nan)
        else:
            directions.append(float(np.corrcoef(xv, sv)[0, 1]))

    importance = pd.DataFrame({
        "feature": features_orig,              # readable original names
        "mean_abs_shap": np.abs(values).mean(axis=0),
        "feature_shap_correlation": directions,
    }).sort_values("mean_abs_shap", ascending=False)
    importance["shap_direction"] = np.where(
        importance.feature_shap_correlation >= 0,
        "higher value raises model risk",
        "higher value lowers model risk")
    importance.to_csv(OUT_IMPORTANCE, index=False)

    risk = model.predict_proba(x)[:, 1]
    row  = int(np.argmax(risk))
    case = pd.DataFrame({
        "feature": features_orig,
        "feature_value": x.iloc[row].to_numpy(),
        "shap_value": values[row],
    })
    case["abs_shap"] = case.shap_value.abs()
    case.sort_values("abs_shap", ascending=False).head(15).to_csv(OUT_CASE, index=False)
    robustness = _write_explainer_robustness(model, x, features_safe)
    print("Saved held-out TreeSHAP outputs; top features:")
    print(importance.head(10).to_string(index=False))
    print("Saved explainer robustness comparison:")
    print(robustness.to_string(index=False))
    return importance

if __name__ == "__main__":
    run()
