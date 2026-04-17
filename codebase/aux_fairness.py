#!/usr/bin/env python3
"""
Layer 2 - Fairness Audit
  - Loads Layer 1 model
  - Measures disparate impact across demographic groups (age, race, gender)
  - Computes per-group: accuracy, FPR, FNR, demographic parity, equalised odds
  - Saves fairness report CSV
"""

import pickle
import warnings
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, confusion_matrix
)

warnings.filterwarnings('ignore')

BASE         = Path(__file__).parent
CLEANED_DATA = BASE / "data_processed" / "diabetic_data_cleaned.csv"
MODELS_DIR   = BASE / "models"

MODEL_PKL    = MODELS_DIR / "layer1_model.pkl"
FEATURES_PKL = MODELS_DIR / "layer1_features.pkl"


def _load_model():
    with open(MODEL_PKL, 'rb') as f:
        model = pickle.load(f)
    with open(FEATURES_PKL, 'rb') as f:
        features = pickle.load(f)
    return model, features


def group_metrics(y_true, y_pred, y_prob, group_name, group_val):
    """Compute fairness-relevant metrics for one group slice."""
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    n = len(y_true)
    pos_rate = y_pred.mean()          # Demographic parity: P(ŷ=1 | group)
    tpr      = tp / (tp + fn) if (tp + fn) > 0 else 0.0   # True Positive Rate
    fpr      = fp / (fp + tn) if (fp + tn) > 0 else 0.0   # False Positive Rate
    fnr      = fn / (tp + fn) if (tp + fn) > 0 else 0.0   # False Negative Rate

    return {
        "group":       group_name,
        "value":       group_val,
        "n":           int(n),
        "prevalence":  float(y_true.mean()),
        "pos_pred_rate": float(pos_rate),   # Demographic parity numerator
        "accuracy":    float(accuracy_score(y_true, y_pred)),
        "precision":   float(precision_score(y_true, y_pred, zero_division=0)),
        "recall_tpr":  float(tpr),
        "fpr":         float(fpr),
        "fnr":         float(fnr),
        "f1":          float(f1_score(y_true, y_pred, zero_division=0)),
        "mean_prob":   float(y_prob.mean()),
    }


def find_demographic_columns(df, feature_names):
    """
    Identify one-hot encoded demographic columns from the feature list.
    Returns dict: {group_prefix: [column_list]}
    """
    groups = {}
    for prefix in ['age_', 'race_', 'gender_']:
        cols = [c for c in feature_names if c.startswith(prefix)]
        if cols:
            groups[prefix.rstrip('_')] = cols
    return groups


def run():
    print("\n" + "=" * 55)
    print("AUXILIARY - FAIRNESS AUDIT")
    print("=" * 55)

    df = pd.read_csv(CLEANED_DATA)
    model, feature_names = _load_model()

    X = df[feature_names].values
    y = df['Readmitted_30d'].values

    y_pred = model.predict(X)
    y_prob = model.predict_proba(X)[:, 1]

    print(f"\nOverall — n={len(y):,}  |  "
          f"Accuracy={accuracy_score(y, y_pred):.4f}  |  "
          f"Pos. rate={y_pred.mean():.3f}")

    # Identify demographic groups from one-hot columns
    dem_groups = find_demographic_columns(df, feature_names)

    all_results = []

    for group_name, cols in dem_groups.items():
        print(f"\n  ── {group_name.upper()} ──")

        group_rows = {}
        for col in cols:
            label = col.replace(f"{group_name}_", "")
            mask  = df[col].values == 1
            if mask.sum() < 30:          # skip tiny groups
                continue
            group_rows[label] = mask

        # Handle "both zero" for one-hot (rare but possible)
        assigned   = np.zeros(len(df), dtype=bool)
        for mask in group_rows.values():
            assigned |= mask
        if (~assigned).sum() >= 30:
            group_rows["unknown"] = ~assigned

        for label, mask in group_rows.items():
            m = group_metrics(
                y[mask], y_pred[mask], y_prob[mask],
                group_name, label
            )
            all_results.append(m)
            print(
                f"    {label:<20}  n={m['n']:>5}  "
                f"acc={m['accuracy']:.3f}  "
                f"pos_rate={m['pos_pred_rate']:.3f}  "
                f"TPR={m['recall_tpr']:.3f}  "
                f"FPR={m['fpr']:.3f}"
            )

    # Disparate Impact ratio (per group vs reference group)
    report_df = pd.DataFrame(all_results)

    print("\n  ── DISPARATE IMPACT (pos_rate_group / pos_rate_reference) ──")
    for group_name, cols in dem_groups.items():
        sub = report_df[report_df['group'] == group_name].copy()
        if sub.empty:
            continue
        # Reference = most frequent group
        ref_row  = sub.loc[sub['n'].idxmax()]
        ref_rate = ref_row['pos_pred_rate']
        ref_val  = ref_row['value']
        print(f"\n    {group_name}  (reference = '{ref_val}', rate={ref_rate:.3f})")
        for _, row in sub.iterrows():
            di = row['pos_pred_rate'] / ref_rate if ref_rate > 0 else float('nan')
            flag = "⚠ " if di < 0.80 or di > 1.25 else "  "
            print(f"    {flag}{row['value']:<20}  DI={di:.3f}")

    out_path = MODELS_DIR / "aux_fairness_report.csv"
    report_df.to_csv(out_path, index=False)
    print(f"\nFairness report saved → {out_path}")

    return report_df


if __name__ == "__main__":
    run()