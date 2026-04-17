#!/usr/bin/env python3
"""
Layer 1 - Readmission Prediction Model
  - Trains on the full cleaned dataset
  - Saves the trained model
  - Terminal loop: enter patient features → get prediction + SHAP feature importance
"""

import sys
import pickle
import warnings
import numpy as np
import pandas as pd
from pathlib import Path

warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score, f1_score
import shap

# ── Paths ────────────────────────────────────────────────────────────────────
BASE         = Path(__file__).parent
CLEANED_DATA = BASE / "data_processed" / "diabetic_data_cleaned.csv"
MODEL_PKL    = BASE / "models" / "layer1_model.pkl"
EXPLAINER_PKL= BASE / "models" / "layer1_explainer.pkl"
FEATURES_PKL = BASE / "models" / "layer1_features.pkl"

MODEL_PKL.parent.mkdir(parents=True, exist_ok=True)


# ── Train ─────────────────────────────────────────────────────────────────────
def train():
    if not CLEANED_DATA.exists():
        print("Cleaned data not found. Running data_preprocessing.py first...")
        import data_preprocessing
        data_preprocessing.run()

    print("\n" + "=" * 55)
    print("LAYER 1: BLACK-BOX PREDICTION & ASSOCIATIVE EXPLANATION")
    print("=" * 55)

    df = pd.read_csv(CLEANED_DATA)
    print(f"Dataset: {len(df):,} records  |  {len(df.columns)-1} features")

    X = df.drop(columns=['Readmitted_30d'])
    y = df['Readmitted_30d']
    feature_names = X.columns.tolist()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"Train: {len(X_train):,}  |  Test: {len(X_test):,}")

    model = GradientBoostingClassifier(
        n_estimators=200, learning_rate=0.1,
        max_depth=5, subsample=0.8, random_state=42
    )
    print("Training... (this may take a few minutes)")
    model.fit(X_train, y_train)

    y_pred       = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]

    print("\nTest Performance:")
    print(f"  Accuracy   : {accuracy_score(y_test, y_pred):.4f}")
    print(f"  Precision  : {precision_score(y_test, y_pred, zero_division=0):.4f}")
    print(f"  Recall     : {recall_score(y_test, y_pred, zero_division=0):.4f}")
    print(f"  F1-Score   : {f1_score(y_test, y_pred, zero_division=0):.4f}")
    print(f"  AUC-ROC    : {roc_auc_score(y_test, y_pred_proba):.4f}")

    print("\nBuilding SHAP explainer...")
    explainer = shap.TreeExplainer(model)

    with open(MODEL_PKL,     'wb') as f: pickle.dump(model,        f)
    with open(EXPLAINER_PKL, 'wb') as f: pickle.dump(explainer,    f)
    with open(FEATURES_PKL,  'wb') as f: pickle.dump(feature_names,f)

    print(f"\nModel saved  →  {MODEL_PKL.parent}")
    return model, explainer, feature_names


# ── Load ──────────────────────────────────────────────────────────────────────
def load():
    with open(MODEL_PKL,     'rb') as f: model         = pickle.load(f)
    with open(EXPLAINER_PKL, 'rb') as f: explainer     = pickle.load(f)
    with open(FEATURES_PKL,  'rb') as f: feature_names = pickle.load(f)
    return model, explainer, feature_names


# ── Predict & explain ─────────────────────────────────────────────────────────
def predict_and_explain(model, explainer, feature_names, patient_row, top_n):
    """
    patient_row : 1-D numpy array aligned to feature_names (no target column)
    Returns prediction, probability, and a DataFrame of top_n SHAP features.
    """
    X = patient_row.reshape(1, -1)

    pred_class = int(model.predict(X)[0])
    pred_proba = model.predict_proba(X)[0]

    shap_vals = explainer.shap_values(X)
    if isinstance(shap_vals, list):
        shap_vals = shap_vals[pred_class]
    shap_vals = shap_vals.flatten()

    df_exp = pd.DataFrame({
        'feature': feature_names,
        'value'  : patient_row,
        'shap'   : shap_vals,
    })
    df_exp['abs_shap'] = df_exp['shap'].abs()
    top = df_exp.nlargest(top_n, 'abs_shap').reset_index(drop=True)

    return pred_class, pred_proba, top


# ── Terminal input ─────────────────────────────────────────────────────────────
def _ask(prompt, default=""):
    """Safe input() that returns default on EOF (e.g. piped/test runs)."""
    try:
        return input(prompt).strip()
    except EOFError:
        return default


def get_patient(feature_names, df_reference):
    """
    Ask the user to either pick a dataset patient or enter feature values.
    Returns a 1-D numpy array aligned to feature_names.
    """
    print("\n" + "-" * 55)
    print("PATIENT INPUT")
    print("-" * 55)
    print("  1  –  Pick a random patient from the dataset")
    print("  2  –  Enter a patient row number from the dataset")
    print("  3  –  Manually enter feature values")

    choice = _ask("Choice: ", default="3")

    if choice == "1":
        idx = np.random.randint(0, len(df_reference))
        row = df_reference.iloc[idx]
        print(f"  Using dataset patient #{idx}")
        _show_key_fields(row, feature_names)
        return row[feature_names].values.astype(float)

    elif choice == "2":
        raw_idx = _ask(f"  Row number (0 – {len(df_reference)-1}): ")
        try:
            idx = int(raw_idx)
            row = df_reference.iloc[idx]
            print(f"  Using dataset patient #{idx}")
            _show_key_fields(row, feature_names)
            return row[feature_names].values.astype(float)
        except (ValueError, IndexError):
            print("  Invalid row number, using row 0.")
            row = df_reference.iloc[0]
            return row[feature_names].values.astype(float)

    elif choice == "3":
        print(f"\n  There are {len(feature_names)} features.")
        print("  You can enter values for the key clinical fields below.")
        print("  (Press Enter to accept the shown default for any field.)\n")

        # Use row 0 as defaults
        defaults = df_reference.iloc[0][feature_names].to_dict()
        values   = defaults.copy()

        # Mappings for coded categorical fields (shown as one-hot columns)
        ADMISSION_TYPE = {
            '1': 'Emergency', '2': 'Urgent', '3': 'Elective',
            '4': 'Newborn',   '7': 'Trauma Center',
        }
        DISCHARGE_DISP = {
            '1':  'Discharged to home',
            '2':  'Transferred to short-term hospital',
            '3':  'Transferred to SNF',
            '6':  'Discharged home with home-health service',
            '7':  'Left AMA',
            '22': 'Transferred to rehab facility',
            '23': 'Transferred to long-term care hospital',
        }
        ADMISSION_SRC = {
            '1': 'Physician Referral', '2': 'Clinic Referral',
            '3': 'HMO Referral',       '4': 'Transfer from hospital',
            '7': 'Emergency Room',
        }

        # Continuous/count fields the user should enter directly
        key_fields = [
            'time_in_hospital', 'num_lab_procedures', 'num_procedures',
            'num_medications', 'number_diagnoses',
            'number_outpatient', 'number_emergency', 'number_inpatient',
        ]
        key_fields = [f for f in key_fields if f in feature_names]

        for field in key_fields:
            default_val = defaults[field]
            raw = _ask(f"  {field} [{default_val}]: ")
            if raw:
                try:
                    values[field] = float(raw)
                except ValueError:
                    pass  # keep default

        # One-hot binary fields the user can toggle
        print()
        print("  --- Categorical fields (enter the code number) ---")

        def _pick_onehot(prefix, mapping, feature_names, values, defaults):
            """Set the correct one-hot column for a categorical ID."""
            print(f"  {prefix.replace('_', ' ')}:")
            for k, v in mapping.items():
                print(f"    {k} = {v}")
            raw = _ask("  Enter code [press Enter to keep default]: ")
            if raw in mapping:
                col_name = f"{prefix}_{raw}"
                for f in feature_names:
                    if f.startswith(prefix + '_') and not any(
                        f.startswith(prefix + '_group') for _ in [None]
                    ):
                        values[f] = 1.0 if f == col_name else 0.0

        _pick_onehot('admission_type',        ADMISSION_TYPE, feature_names, values, defaults)
        _pick_onehot('discharge_disposition', DISCHARGE_DISP, feature_names, values, defaults)
        _pick_onehot('admission_source',      ADMISSION_SRC,  feature_names, values, defaults)

        # Age group
        age_groups = ['[0-10)', '[10-20)', '[20-30)', '[30-40)', '[40-50)',
                      '[50-60)', '[60-70)', '[70-80)', '[80-90)', '[90-100)']
        print("  age group:")
        for i, g in enumerate(age_groups, 1):
            print(f"    {i} = {g}")
        raw = _ask("  Enter number [press Enter to keep default]: ")
        try:
            chosen_age = age_groups[int(raw) - 1]
            for f in feature_names:
                if f.startswith('age_'):
                    values[f] = 1.0 if f == f'age_{chosen_age}' else 0.0
        except (ValueError, IndexError):
            pass

        # HbA1c result
        a1c_options = {'1': '>8 (poor control)', '2': 'Norm (normal)'}
        print("  HbA1c (A1Cresult):")
        for k, v in a1c_options.items(): print(f"    {k} = {v}")
        raw = _ask("  Enter code [press Enter to keep default]: ")
        if raw == '1':
            values['A1Cresult_>8']   = 1.0; values['A1Cresult_Norm'] = 0.0
        elif raw == '2':
            values['A1Cresult_>8']   = 0.0; values['A1Cresult_Norm'] = 1.0

        # Insulin
        print("  Insulin:")
        print("    1 = Prescribed   2 = Not prescribed   3 = Dose increased")
        raw = _ask("  Enter code [press Enter to keep default]: ")
        if raw == '1':
            values['insulin_prescribed'] = 1.0; values['insulin_increased'] = 0.0
        elif raw == '2':
            values['insulin_prescribed'] = 0.0; values['insulin_increased'] = 0.0
        elif raw == '3':
            values['insulin_prescribed'] = 1.0; values['insulin_increased'] = 1.0

        # Diabetes medication overall
        raw = _ask("  On any diabetes medication? (1=Yes / 2=No) [press Enter to keep default]: ")
        if raw == '1':
            values['diabetesMed_Yes'] = 1.0
        elif raw == '2':
            values['diabetesMed_Yes'] = 0.0

        return np.array([values[f] for f in feature_names], dtype=float)

    else:
        print("  Unrecognised choice, using random patient.")
        idx = np.random.randint(0, len(df_reference))
        row = df_reference.iloc[idx]
        return row[feature_names].values.astype(float)


def _show_key_fields(row, feature_names):
    key = ['time_in_hospital', 'num_medications', 'num_lab_procedures',
           'number_diagnoses', 'number_inpatient', 'number_emergency']
    print("  Key fields:")
    for f in key:
        if f in feature_names:
            print(f"    {f}: {row[f]}")


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    # Train or load
    if MODEL_PKL.exists() and EXPLAINER_PKL.exists() and FEATURES_PKL.exists():
        print("=" * 55)
        print("Model found – loading existing model.")
        print("(Run with  --retrain  to train from scratch.)")
        print("=" * 55)
        model, explainer, feature_names = load()
    else:
        model, explainer, feature_names = train()

    # Load data once for reference during input
    if not CLEANED_DATA.exists():
        print("Cleaned data missing. Cannot proceed.")
        sys.exit(1)
    df_ref = pd.read_csv(CLEANED_DATA)

    # Interaction loop
    while True:
        print("\n" + "=" * 55)
        print("OPTIONS")
        print("  1  –  Make a prediction")
        print("  2  –  Retrain model")
        print("  3  –  Exit")
        print("=" * 55)

        try:
            choice = _ask("Choice: ", default="3")
        except KeyboardInterrupt:
            print("\nGoodbye.")
            break

        if choice == "1":
            try:
                top_n = int(_ask("How many top features to show? [default 10]: ") or "10")
            except ValueError:
                top_n = 10

            patient_row = get_patient(feature_names, df_ref)

            pred_class, pred_proba, top_features = predict_and_explain(
                model, explainer, feature_names, patient_row, top_n
            )

            label = "READMITTED" if pred_class == 1 else "NOT READMITTED"
            prob  = pred_proba[1] * 100

            print("\n" + "=" * 55)
            print("PREDICTION")
            print("=" * 55)
            print(f"  Result      :  {label}")
            print(f"  Probability :  {prob:.1f}%  readmission")
            print(f"\nTop {top_n} features that influenced this prediction:")
            print(f"  {'#':<4} {'Feature':<45} {'Direction':<15} {'Impact':>7}")
            print("  " + "-" * 73)
            for i, row in top_features.iterrows():
                direction = "increases risk" if row['shap'] > 0 else "decreases risk"
                print(f"  {i+1:<4} {row['feature']:<45} {direction:<15} {abs(row['shap'])*100:>6.2f}%")
                print(f"       patient value: {row['value']}")
            print("=" * 55)

        elif choice == "2":
            model, explainer, feature_names = train()

        elif choice == "3":
            print("Goodbye.")
            break

        else:
            print("Invalid choice.")


if __name__ == "__main__":
    if "--retrain" in sys.argv:
        train()
    else:
        main()
