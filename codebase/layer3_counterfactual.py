#!/usr/bin/env python3
"""Actionability-first recourse safety audit (not a treatment recommender)."""
from pathlib import Path
import pickle
import numpy as np
import pandas as pd

BASE = Path(__file__).parent
MODELS = BASE / "models"
PREDICTIONS = MODELS / "heldout_predictions.csv"
AUDIT = MODELS / "causal_audit_full.csv"
OUT = MODELS / "recourse_audit.csv"
POLICY = MODELS / "recourse_policy.csv"

# A feature is allowed only after clinician governance confirms it is a direct,
# feasible intervention. The historical UCI extract has no such approved field.
# Entries below document why sensitivity-qualified features remain blocked.
ACTIONABILITY_POLICY = {
    "number_inpatient": ("blocked", "Historical utilization; cannot be changed for the current encounter."),
    "number_emergency": ("blocked", "Historical utilization; cannot be changed for the current encounter."),
    "insulin_prescribed": ("conditional", "Medication decision requires clinician review and treatment-specific causal evidence."),
    "metformin_prescribed": ("conditional", "Medication decision requires clinician review and treatment-specific causal evidence."),
    "diabetesMed_Yes": ("conditional", "Medication decision requires clinician review and treatment-specific causal evidence."),
}

def run(n_cases=20):
    audit = pd.read_csv(AUDIT)
    eligible = audit.feature.tolist()
    policy_rows = []
    allowed = []
    for feature in eligible:
        decision, reason = ACTIONABILITY_POLICY.get(
            feature, ("blocked", "No clinician-approved feasibility constraint is available."))
        evidence = audit.loc[audit.feature.eq(feature), "sensitivity_category"].iloc[0]
        policy_rows.append({"feature": feature, "actionability_class": decision,
                            "evidence_status": evidence, "recourse_eligible": False,
                            "reason": reason})
        if decision == "allowed" and evidence == "ROBUST":
            allowed.append(feature)
    policy = pd.DataFrame(policy_rows)
    policy.to_csv(POLICY, index=False)

    with open(MODELS / "layer1_model.pkl", "rb") as f: model = pickle.load(f)
    with open(MODELS / "layer1_features.pkl", "rb") as f: features = pickle.load(f)
    # Recourse is scoped to held-out encounters only.  The prediction artefact
    # contains the exact model features and the held-out patient rows.
    df = pd.read_csv(PREDICTIONS)
    # Layer 1 sanitises feature names for XGBoost compatibility before fitting.
    # Apply the identical mapping before scoring here.
    safe_df = df.rename(columns={c: c.replace("[", "_").replace("]", "_")
                                  .replace("<", "lt").replace(">", "gt")
                                 for c in df.columns})
    probabilities = model.predict_proba(safe_df[features])[:, 1]
    high = np.where(probabilities >= .5)[0][:n_cases]
    records = []
    for idx in high:
        records.append({
            "patient_row": int(idx), "original_risk": float(probabilities[idx]),
            "audited_features": ";".join(eligible),
            "allowed_action_features": ";".join(allowed),
            "status": "no_validated_recourse" if not allowed else "research_scenario_pending_review",
            "reason": "No feature was both clinically approved and robust to unmeasured confounding." if not allowed else "Requires clinician review before any scenario generation.",
        })
    result = pd.DataFrame(records)
    result.to_csv(OUT, index=False)
    print(f"Recourse audit: {len(eligible)} audited features, {len(allowed)} approved robust actionable features.")
    print("No patient-level recommendation is emitted unless both conditions are met.")
    return result, policy

if __name__ == "__main__":
    run()
