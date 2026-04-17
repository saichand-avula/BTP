#!/usr/bin/env python3
"""
Layer 3 - Safe Counterfactual Explanations
===========================================
For HIGH RISK patients, finds the minimum actionable changes
that would reduce readmission risk — using ONLY clinically
actionable features that are NOT confounded by disease severity.

The key difference from standard counterfactuals:
  - Confounded features (e.g. time_in_hospital) are BLOCKED
  - Only features a doctor can actually act on are used
  - Output is human-readable: "If you do X, risk drops from 78% to 45%"
"""

import pickle
import warnings
import numpy as np
import pandas as pd
from pathlib import Path

warnings.filterwarnings('ignore')

BASE         = Path(__file__).parent
CLEANED_DATA = BASE / "data_processed" / "diabetic_data_cleaned.csv"
MODELS_DIR   = BASE / "models"
MODEL_PKL    = MODELS_DIR / "layer1_model.pkl"
FEATURES_PKL = MODELS_DIR / "layer1_features.pkl"
CAUSAL_CSV   = MODELS_DIR / "layer2_causal_results.csv"

# ── Human-readable labels for features ──────────────────────────────────────
FEATURE_LABELS = {
    "insulin_prescribed":        "Prescribe insulin",
    "insulin_increased":         "Increase insulin dose",
    "diabetesMed_Yes":           "Prescribe diabetes medication",
    "change_No":                 "Keep medication unchanged",
    "metformin_prescribed":      "Prescribe metformin",
    "glipizide_prescribed":      "Prescribe glipizide",
    "glyburide_prescribed":      "Prescribe glyburide",
    "pioglitazone_prescribed":   "Prescribe pioglitazone",
    "rosiglitazone_prescribed":  "Prescribe rosiglitazone",
    "A1Cresult_>8":              "HbA1c result is above 8 (poor control)",
    "A1Cresult_Norm":            "HbA1c result is normal",
    "discharge_disposition_1":   "Discharge patient to home",
    "discharge_disposition_2":   "Transfer to short-term hospital",
    "discharge_disposition_3":   "Discharge to skilled nursing facility (SNF)",
    "discharge_disposition_6":   "Discharge home with home health service",
    "discharge_disposition_22":  "Discharge to rehab facility",
}

def get_action_sentence(feat, from_val, to_val):
    """
    Converts a feature change into a plain English clinical action.
    e.g. discharge_disposition_22: 0→1 = "Arrange discharge to rehab facility"
    """
    label = FEATURE_LABELS.get(feat, feat.replace("_", " ").title())

    # Binary features (0/1)
    if to_val == 1.0 and from_val == 0.0:
        return f"→ **{label}** (currently not done — recommend doing this)"
    elif to_val == 0.0 and from_val == 1.0:
        return f"→ **Stop: {label}** (currently being done — consider stopping)"
    else:
        direction = "Increase" if to_val > from_val else "Decrease"
        return f"→ **{direction} {label.lower()}** from {from_val:.0f} to {to_val:.0f}"


# ── Always-actionable features (can be changed by a doctor) ──────────────────
# These are NOT severity markers — a doctor can directly intervene on these
ALWAYS_ACTIONABLE = [
    "insulin_prescribed",
    "insulin_increased",
    "diabetesMed_Yes",
    "change_No",
    "metformin_prescribed",
    "glipizide_prescribed",
    "glyburide_prescribed",
    "pioglitazone_prescribed",
    "rosiglitazone_prescribed",
    "A1Cresult_>8",
    "A1Cresult_Norm",
    "discharge_disposition_1",
    "discharge_disposition_2",
    "discharge_disposition_3",
    "discharge_disposition_6",
    "discharge_disposition_22",
]


def get_safe_actionable_features(feature_names):
    """
    Returns features that are safe to change:
    1. Features explicitly in ALWAYS_ACTIONABLE list
    2. PLUS any features marked ROBUST in the causal audit
    Excludes all FRAGILE (confounded) and severity markers.
    """
    safe = set(ALWAYS_ACTIONABLE)

    # Also include any causal ROBUST features
    if CAUSAL_CSV.exists():
        causal_df = pd.read_csv(CAUSAL_CSV)
        robust = causal_df[causal_df["Status"] == "ROBUST"]["Feature"].tolist()
        safe.update(robust)

    # Match against actual feature names in the model
    actionable_idx = [
        i for i, f in enumerate(feature_names)
        if f in safe
    ]
    return actionable_idx


def greedy_counterfactual(model, patient_row, feature_names, df_ref, actionable_idx):
    """
    Greedy search: find minimum changes to actionable features
    that reduce risk below 50% (flip HIGH→LOW).
    Returns changes as human-readable sentences.
    """
    x       = patient_row.copy()
    changes = []

    orig_prob = float(model.predict_proba(x.reshape(1, -1))[0, 1])

    # Sort by model feature importance (try most impactful first)
    try:
        fi = model.feature_importances_
        actionable_idx = sorted(actionable_idx, key=lambda i: fi[i], reverse=True)
    except Exception:
        pass

    for idx in actionable_idx:
        feat     = feature_names[idx]
        orig_val = x[idx]

        # Try all unique values this feature takes in the dataset
        col_vals = sorted(df_ref[feat].dropna().unique())

        best_val  = orig_val
        best_prob = float(model.predict_proba(x.reshape(1, -1))[0, 1])

        for candidate in col_vals:
            if candidate == orig_val:
                continue
            x_try      = x.copy()
            x_try[idx] = candidate
            prob_try   = float(model.predict_proba(x_try.reshape(1, -1))[0, 1])
            if prob_try < best_prob:
                best_prob = prob_try
                best_val  = candidate

        if best_val != orig_val:
            x[idx] = best_val
            sentence = get_action_sentence(feat, orig_val, best_val)
            changes.append({
                "feature":    feat,
                "from_val":   orig_val,
                "to_val":     best_val,
                "sentence":   sentence,
                "prob_after": float(model.predict_proba(x.reshape(1, -1))[0, 1]),
            })

        # Stop once risk is below 50%
        if model.predict(x.reshape(1, -1))[0] == 0:
            break

    final_prob = float(model.predict_proba(x.reshape(1, -1))[0, 1])
    flipped    = int(final_prob < 0.5)

    return {
        "original_prob": orig_prob,
        "final_prob":    final_prob,
        "flipped":       flipped,
        "num_changes":   len(changes),
        "changes":       changes,
    }


def run(n_patients=20):
    print("\n" + "=" * 60)
    print("LAYER 3: SAFE CLINICAL COUNTERFACTUALS")
    print("=" * 60)

    df    = pd.read_csv(CLEANED_DATA)
    with open(MODEL_PKL,    "rb") as f: model         = pickle.load(f)
    with open(FEATURES_PKL, "rb") as f: feature_names = pickle.load(f)

    actionable_idx = get_safe_actionable_features(feature_names)
    print(f"\nSafe actionable features available: {len(actionable_idx)}")
    for i in actionable_idx[:10]:
        print(f"  - {feature_names[i]}")

    X  = df[feature_names].values
    y  = df["Readmitted_30d"].values
    yp = (model.predict_proba(X)[:, 1] >= 0.5).astype(int)

    high_risk_idx = np.where((yp == 1) & (y == 1))[0]
    if len(high_risk_idx) == 0:
        high_risk_idx = np.where(yp == 1)[0]

    sample_idx = np.random.choice(
        high_risk_idx, size=min(n_patients, len(high_risk_idx)), replace=False
    )

    records    = []
    flip_count = 0

    print(f"\nGenerating counterfactuals for {len(sample_idx)} high-risk patients...\n")

    for idx in sample_idx:
        result = greedy_counterfactual(
            model, X[idx], feature_names, df, actionable_idx
        )
        flip_count += result["flipped"]

        status = "✓ FLIPPED" if result["flipped"] else "✗ not flipped"
        print(
            f"  Patient {idx}: {result['original_prob']*100:.1f}% → "
            f"{result['final_prob']*100:.1f}%  ({result['num_changes']} changes)  {status}"
        )
        for ch in result["changes"]:
            print(f"    {ch['sentence']}  → risk becomes {ch['prob_after']*100:.1f}%")

        records.append({
            "patient_idx":   int(idx),
            "original_prob": round(result["original_prob"], 4),
            "cf_prob":       round(result["final_prob"], 4),
            "flipped":       result["flipped"],
            "num_changes":   result["num_changes"],
            "changes_str":   str([c["sentence"] for c in result["changes"]]),
        })

    print(f"\nFlip rate: {flip_count}/{len(sample_idx)} "
          f"({flip_count/len(sample_idx)*100:.0f}%)")

    out_df = pd.DataFrame(records)
    out_df.to_csv(MODELS_DIR / "layer3_counterfactuals.csv", index=False)
    print(f"Saved → {MODELS_DIR}/layer3_counterfactuals.csv")
    return out_df


def generate_for_patient(patient_dict, feature_names, model, df_ref):
    """
    Called live from the dashboard for a single patient.
    Returns the counterfactual result with human-readable sentences.
    """
    actionable_idx = get_safe_actionable_features(feature_names)
    patient_row    = np.array([patient_dict.get(f, 0.0) for f in feature_names])
    return greedy_counterfactual(model, patient_row, feature_names, df_ref, actionable_idx)


if __name__ == "__main__":
    run(n_patients=20)