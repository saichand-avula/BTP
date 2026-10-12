#!/usr/bin/env python3
"""
Layer 2 - Naive Model vs Causal Model Comparison
=================================================
This is the HEART of the confounding detection in this BTP project.

HOW IT WORKS:
  Model 1 (Naive): A simple Logistic Regression trained directly on the
  top SHAP features. It sees raw correlations — no confounder adjustment.
  It will wrongly blame insulin for readmissions.

  Model 2 (Causal): The DoubleML results already computed in layer2_causal.py.
  This adjusts for confounders using Random Forests for both the treatment
  and outcome models. It finds the TRUE causal effect.

  CONFOUNDING DETECTED when: |Naive effect - Causal effect| is large.
  This proves the naive ML model was fooled by a hidden confounder.
"""

import pickle
import warnings
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
import shap

warnings.filterwarnings("ignore")

BASE         = Path(__file__).parent
CLEANED_DATA = BASE / "data_processed" / "diabetic_data_cleaned.csv"
MODELS_DIR   = BASE / "models"
CAUSAL_CSV   = MODELS_DIR / "layer2_causal_results.csv"
OUTPUT_CSV   = MODELS_DIR / "layer2_comparison.csv"

N_TOP_FEATURES = 7


def load_top_shap_features():
    """Auto-reads top N features from Layer 1's saved SHAP explainer."""
    try:
        with open(MODELS_DIR / "layer1_features.pkl", "rb") as f:
            all_features = pickle.load(f)
        with open(MODELS_DIR / "layer1_model.pkl", "rb") as f:
            model = pickle.load(f)

        explainer = None
        try:
            with open(MODELS_DIR / "layer1_explainer.pkl", "rb") as f:
                explainer = pickle.load(f)
        except Exception:
            explainer = None

        if explainer is None:
            explainer = shap.TreeExplainer(model)

        df = pd.read_csv(CLEANED_DATA)
        X_sample = df[all_features].head(2000).values

        shap_vals = explainer.shap_values(X_sample)
        if isinstance(shap_vals, list):
            shap_vals = shap_vals[1]

        mean_abs_shap = np.abs(shap_vals).mean(axis=0)
        top_idx = np.argsort(mean_abs_shap)[::-1][:N_TOP_FEATURES]
        top_features = [all_features[i] for i in top_idx]

        print(f"  Auto-loaded top {N_TOP_FEATURES} SHAP features:")
        for i, f in enumerate(top_features):
            print(f"    {i+1}. {f}  (mean |SHAP| = {mean_abs_shap[top_idx[i]]:.4f})")
        return top_features

    except Exception as e:
        print(f"  Warning: Could not auto-load SHAP features ({e}). Using domain defaults.")
        return [
            "time_in_hospital", "num_lab_procedures", "num_procedures",
            "num_medications", "number_outpatient", "number_emergency", "number_inpatient"
        ]


def compute_naive_effects(df, features, outcome="Readmitted_30d"):
    """
    Model 1 — Naive Logistic Regression.
    Trains a logistic regression directly on the top SHAP features.
    The coefficient for each feature = naive (confounded) effect estimate.
    This is what a standard ML explanation gives you — correlation, not causation.
    """
    print("\n  Running Model 1: Naive Logistic Regression...")

    valid_features = [f for f in features if f in df.columns]
    X = df[valid_features].fillna(0).values
    y = df[outcome].values

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    lr = LogisticRegression(max_iter=1000, random_state=42)
    lr.fit(X_scaled, y)

    naive_effects = {}
    for feat, coef in zip(valid_features, lr.coef_[0]):
        # Convert log-odds coefficient to approximate probability change
        # using the formula: effect ≈ coef * p̄ * (1 - p̄) where p̄ = base rate
        base_rate = y.mean()
        prob_effect = coef * base_rate * (1 - base_rate)
        naive_effects[feat] = round(prob_effect, 4)

    print(f"  Model 1 done. Base readmission rate: {base_rate:.3f}")
    return naive_effects


def run():
    print("\n" + "=" * 55)
    print("NAIVE vs CAUSAL MODEL COMPARISON")
    print("(Confounding Detection Core)")
    print("=" * 55)

    # Step 1: Load data
    df = pd.read_csv(CLEANED_DATA)
    print(f"\nLoaded {len(df):,} records")

    # Step 2: Get top SHAP features from Layer 1
    print("\nStep 1: Loading top features from Layer 1 SHAP explainer...")
    top_features = load_top_shap_features()

    # Step 3: Model 1 — Naive effects
    naive_effects = compute_naive_effects(df, top_features)

    # Step 4: Model 2 — Load existing DoubleML causal results
    print("\n  Loading Model 2: DoubleML Causal Results (already computed)...")
    if not CAUSAL_CSV.exists():
        print("  ERROR: Run layer2_causal.py first!")
        return None

    causal_df = pd.read_csv(CAUSAL_CSV)
    print(f"  Loaded {len(causal_df)} causal estimates.")

    # Step 5: Build comparison table
    print("\n" + "=" * 75)
    print(f"  {'Feature':<28} {'Naive Effect':>14} {'Causal CATE':>14} {'Gap':>10} {'Verdict'}")
    print("  " + "-" * 73)

    rows = []
    for feat in top_features:
        if feat not in df.columns:
            continue

        naive_val = naive_effects.get(feat, None)
        causal_row = causal_df[causal_df["Feature"] == feat]

        if causal_row.empty or naive_val is None:
            continue

        causal_val = causal_row["CATE"].values[0]
        e_value    = causal_row["E_Value"].values[0]
        status     = causal_row["Status"].values[0]

        gap = abs(naive_val - causal_val)

        # Confounding verdict: large gap = confounded
        if status == "PROTECTED":
            verdict = "PROTECTED (demographic)"
        elif gap > 0.015:
            verdict = "HEAVILY CONFOUNDED"
        elif gap > 0.005:
            verdict = "MILDLY CONFOUNDED"
        else:
            verdict = "CLEAN (minimal confounding)"

        print(
            f"  {feat:<28} {naive_val:>+14.4f} {causal_val:>+14.4f} "
            f"{gap:>10.4f}  {verdict}"
        )

        rows.append({
            "Feature":       feat,
            "Naive_Effect":  naive_val,
            "Causal_CATE":   causal_val,
            "Gap":           round(gap, 4),
            "E_Value":       e_value,
            "Status":        status,
            "Verdict":       verdict,
        })

    print("=" * 75)

    # Step 6: Highlight the key finding (the Insulin Paradox equivalent)
    print("\nKEY FINDINGS:")
    for row in rows:
        if row["Gap"] > 0.005:
            direction_naive  = "increases" if row["Naive_Effect"] > 0 else "decreases"
            direction_causal = "increases" if row["Causal_CATE"]  > 0 else "decreases"
            print(
                f"  [{row['Feature']}] Naive model says it {direction_naive} risk by "
                f"{abs(row['Naive_Effect']):.4f}, but causal model says {direction_causal} "
                f"by {abs(row['Causal_CATE']):.4f} after confounder adjustment. "
                f"Gap = {row['Gap']:.4f} → {row['Verdict']}"
            )

    # Step 7: Save
    result_df = pd.DataFrame(rows)
    result_df.to_csv(OUTPUT_CSV, index=False)
    print(f"\nComparison saved → {OUTPUT_CSV}")

    return result_df


if __name__ == "__main__":
    run()