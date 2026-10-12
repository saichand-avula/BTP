#!/usr/bin/env python3
"""Held-out TreeSHAP analysis for the selected Layer 1 model."""
from pathlib import Path
import pickle
import numpy as np
import pandas as pd
import shap

BASE = Path(__file__).parent
DATA = BASE / "data_processed" / "diabetic_data_cleaned.csv"
MODELS = BASE / "models"
OUT_IMPORTANCE = MODELS / "shap_global_importance.csv"
OUT_CASE = MODELS / "shap_high_risk_case.csv"
OUT_VALUES = MODELS / "shap_values_heldout.npy"

def _test_data(df):
    from layer1 import _split
    _, _, test = _split(df)
    return test

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
    values = explanation.values
    if values.ndim == 3:
        values = values[:, :, 1]
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
    print("Saved held-out TreeSHAP outputs; top features:")
    print(importance.head(10).to_string(index=False))
    return importance

if __name__ == "__main__":
    run()
