import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import cross_val_predict
import scipy.stats as st
import math
import pickle
import os
import warnings

warnings.filterwarnings("ignore")

# Constants
DATA_PATH = "data_processed/diabetic_data_cleaned.csv"
OUTCOME = "Readmitted_30d"
MODELS_DIR = "models"
IMMUTABLE_PREFIXES = ['age_', 'race_', 'gender_']

def load_top_features(n=12):
    """
    Returns the top N features by actual model feature importance.
    Excludes demographic/immutable features (age, race, gender).
    This ensures we audit features that actually drive predictions.
    """
    try:
        with open(f"{MODELS_DIR}/layer1_features.pkl", "rb") as f:
            all_features = pickle.load(f)
        with open(f"{MODELS_DIR}/layer1_model.pkl", "rb") as f:
            model = pickle.load(f)

        importances = model.feature_importances_
        sorted_idx = np.argsort(importances)[::-1]

        top = []
        for i in sorted_idx:
            feat = all_features[i]
            if any(feat.startswith(p) for p in IMMUTABLE_PREFIXES):
                continue
            top.append(feat)
            if len(top) >= n:
                break

        print(f"  Top {len(top)} features selected by model importance:")
        for f in top:
            print(f"    - {f}")
        return top

    except Exception as e:
        print(f"Could not auto-load features ({e}), using domain defaults.")
        return ["time_in_hospital", "num_medications", "num_lab_procedures",
                "insulin_prescribed", "number_inpatient", "number_diagnoses"]

def is_protected(feature_name):
    """Checks if a feature is immutable/demographic based on architecture rules."""
    return any(feature_name.startswith(prefix) for prefix in IMMUTABLE_PREFIXES)

def doubly_robust_cate(df, treatment, outcome, confounders):
    """
    Implements Doubly Robust Estimation (DoubleML / AIPW) using Random Forests.
    Stage 1: Propensity Score via cross-fitted RF Classifier
    Stage 2: Outcome prediction via RF Regressors
    Stage 3: AIPW Score calculation
    """
    Y = df[outcome].values
    X = df[confounders].values
    
    # Step 3B: Treatment Binarization
    # DoubleML requires a binary treatment. If numeric, split at median for "High vs Low" exposure.
    if df[treatment].nunique() > 2:
        median_val = df[treatment].median()
        T = (df[treatment] > median_val).astype(int).values
        print(f"    (Binarized '{treatment}' at > {median_val} for causal estimation)")
    else:
        T = df[treatment].values

    # Stage 1: Treatment Model m(X) = P(T=1 | X)
    m_model = RandomForestClassifier(n_estimators=50, max_depth=5, random_state=42, n_jobs=-1)
    # Cross-fitting to prevent overfitting the propensity scores
    m_x = cross_val_predict(m_model, X, T, cv=5, method='predict_proba')[:, 1]
    # Positivity Assumption: Clip probabilities to ensure common support
    m_x = np.clip(m_x, 0.05, 0.95)

    # Stage 2: Outcome Models g(X, T)
    X_T0, Y_T0 = X[T == 0], Y[T == 0]
    X_T1, Y_T1 = X[T == 1], Y[T == 1]
    
    g0_model = RandomForestRegressor(n_estimators=50, max_depth=5, random_state=42, n_jobs=-1).fit(X_T0, Y_T0)
    g1_model = RandomForestRegressor(n_estimators=50, max_depth=5, random_state=42, n_jobs=-1).fit(X_T1, Y_T1)
    
    g0_x = g0_model.predict(X)
    g1_x = g1_model.predict(X)

    # Stage 3: Doubly Robust Score (AIPW) calculation
    dr_scores = (g1_x - g0_x) + \
                (T / m_x) * (Y - g1_x) - \
                ((1 - T) / (1 - m_x)) * (Y - g0_x)

    cate = np.mean(dr_scores)
    se = np.std(dr_scores) / np.sqrt(len(dr_scores))
    
    return cate, se

def calculate_e_value(cate, se):
    """
    Calculates E-values based on VanderWeele 2017 formulation provided in architecture.md.
    Uses the lower bound of the CI (or upper bound if protective).
    """
    ci_lower = cate - 1.96 * se
    ci_upper = cate + 1.96 * se
    p_value = 2 * (1 - st.norm.cdf(abs(cate / se))) if se > 0 else 1.0

    # If the effect is not statistically significant, E-value defaults to 1.0
    if (cate > 0 and ci_lower <= 0) or (cate < 0 and ci_upper >= 0):
        return 1.0, ci_lower, ci_upper, p_value

    if cate > 0: # Risk factor
        rr = math.exp(ci_lower)
        e_value = rr + math.sqrt(rr * (rr - 1))
    else:        # Protective factor
        rr = math.exp(ci_upper) 
        rr_inv = 1.0 / rr if rr > 0 else 1.0
        e_value = rr_inv + math.sqrt(rr_inv * (rr_inv - 1))
        
    return round(e_value, 2), ci_lower, ci_upper, p_value

def classify_status(e_value, is_prot):
    if is_prot: return "PROTECTED"
    if e_value > 3.0: return "ROBUST"
    if e_value > 2.0: return "UNCERTAIN"
    return "FRAGILE"

def run():
    print("=======================================================")
    print("LAYER 2: CAUSAL AUDIT (CORE NOVELTY)")
    print("=======================================================")
    
    df = pd.read_csv(DATA_PATH)
    top_features = load_top_features(n=7)
    valid_features = [f for f in top_features if f in df.columns]
    
    print(f"Auditing Top {len(valid_features)} SHAP Features...\n")
    
    results = []
    
    for feature in valid_features:
        print(f"Analyzing: {feature}")
        
        # 1. Check if Demographic/Immutable
        if is_protected(feature):
            print("    Status: PROTECTED (Demographic/Immutable)\n")
            results.append({
                "Feature": feature, "CATE": np.nan, "CI_Lower": np.nan, 
                "CI_Upper": np.nan, "P_Value": np.nan, "E_Value": np.nan, "Status": "PROTECTED"
            })
            continue
            
        # 2. Define Confounders (Everything except outcome and current treatment)
        confounders = [c for c in df.columns if c != OUTCOME and c != feature]
        
        # 3. Estimate CATE using DoubleML
        cate, se = doubly_robust_cate(df, feature, OUTCOME, confounders)
        
        # 4. Calculate E-value and Classify
        e_val, ci_l, ci_u, p_val = calculate_e_value(cate, se)
        status = classify_status(e_val, False)
        
        print(f"    CATE: {cate:+.4f}  |  95% CI: [{ci_l:+.4f}, {ci_u:+.4f}]")
        print(f"    E-Value: {e_val:<5} |  Status: {status}\n")
        
        results.append({
            "Feature": feature,
            "CATE": round(cate, 4),
            "CI_Lower": round(ci_l, 4),
            "CI_Upper": round(ci_u, 4),
            "P_Value": round(p_val, 4),
            "E_Value": e_val,
            "Status": status
        })
        
    results_df = pd.DataFrame(results)
    
    # Save for Layer 4 (Counterfactuals) to read
    os.makedirs(MODELS_DIR, exist_ok=True)
    results_df.to_csv(f"{MODELS_DIR}/layer2_causal_results.csv", index=False)
    print(f"Audit Complete. Results saved to {MODELS_DIR}/layer2_causal_results.csv")
    print("=======================================================")
    
    return results_df

if __name__ == "__main__":
    run()