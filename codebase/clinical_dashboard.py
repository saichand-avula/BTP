"""
Clinical Decision Support with Causal Auditing — BTP Project
Run: streamlit run clinical_dashboard.py
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import streamlit as st
import pandas as pd
import numpy as np
import pickle

st.set_page_config(page_title="Clinical AI Auditor", layout="wide", page_icon="🏥")

BASE       = Path(__file__).parent
MODELS_DIR = BASE / "models"

# ── Plain English feature labels ─────────────────────────────────────────────
FEATURE_LABELS = {
    "time_in_hospital":         "Time Spent in Hospital (days)",
    "num_lab_procedures":       "Number of Lab Tests Done",
    "num_medications":          "Number of Medications",
    "num_procedures":           "Number of Medical Procedures",
    "number_inpatient":         "Prior Hospital Admissions in the Last Year",
    "number_emergency":         "Emergency Room Visits in the Last Year",
    "number_outpatient":        "Outpatient Visits in the Last Year",
    "number_diagnoses":         "Number of Diagnoses",
    "insulin_prescribed":       "Insulin Prescribed",
    "insulin_increased":        "Insulin Dose Increased",
    "diabetesMed_Yes":          "Diabetes Medication Prescribed",
    "change_No":                "No Medication Change Made",
    "metformin_prescribed":     "Metformin Prescribed",
    "glipizide_prescribed":     "Glipizide Prescribed",
    "glyburide_prescribed":     "Glyburide Prescribed",
    "pioglitazone_prescribed":  "Pioglitazone Prescribed",
    "rosiglitazone_prescribed": "Rosiglitazone Prescribed",
    "A1Cresult_>8":             "HbA1c Test Result Above 8 (Poor Blood Sugar Control)",
    "A1Cresult_Norm":           "HbA1c Test Result Normal (Good Blood Sugar Control)",
    "discharge_disposition_1":  "Discharged to Home",
    "discharge_disposition_2":  "Transferred to Another Hospital",
    "discharge_disposition_3":  "Discharged to a Skilled Nursing Facility",
    "discharge_disposition_5":  "Discharged to Inpatient Care",
    "discharge_disposition_6":  "Discharged Home with Home Health Nurse",
    "discharge_disposition_22": "Discharged to a Rehabilitation Facility",
    "discharge_disposition_28": "Discharged to Psychiatric Hospital",
}

BINARY_FEATURES = {
    "insulin_prescribed","insulin_increased","diabetesMed_Yes","change_No",
    "metformin_prescribed","glipizide_prescribed","glyburide_prescribed",
    "pioglitazone_prescribed","rosiglitazone_prescribed",
    "A1Cresult_>8","A1Cresult_Norm",
}

def get_label(feat):
    return FEATURE_LABELS.get(feat, feat.replace("_", " ").title())

def plain_value(feat, val, shap):
    """Returns a plain English description of a feature's current value."""
    if feat in BINARY_FEATURES or feat.startswith("discharge_disposition_") or feat.startswith("diag_"):
        return "Yes" if val == 1.0 else "No"
    # Numeric — describe relative level
    return f"{val:.0f} (which is {'above' if shap > 0 else 'below'} average)"

def plain_direction(shap):
    return "increasing" if shap > 0 else "reducing"

def opposite_direction(shap):
    return "decrease" if shap > 0 else "increase"

@st.cache_resource
def load_assets():
    with open(MODELS_DIR / "layer1_model.pkl",    "rb") as f: model         = pickle.load(f)
    with open(MODELS_DIR / "layer1_features.pkl", "rb") as f: feature_names = pickle.load(f)
    with open(MODELS_DIR / "layer1_explainer.pkl","rb") as f: explainer     = pickle.load(f)
    df_ref       = pd.read_csv(BASE / "data_processed" / "diabetic_data_cleaned.csv")
    causal_rules = pd.read_csv(MODELS_DIR / "layer2_causal_results.csv") \
        if (MODELS_DIR / "layer2_causal_results.csv").exists() else pd.DataFrame()
    comparison_df = pd.read_csv(MODELS_DIR / "layer2_comparison.csv") \
        if (MODELS_DIR / "layer2_comparison.csv").exists() else pd.DataFrame()
    fairness_df  = pd.read_csv(MODELS_DIR / "aux_fairness_report.csv") \
        if (MODELS_DIR / "aux_fairness_report.csv").exists() else pd.DataFrame()
    return model, feature_names, explainer, df_ref, causal_rules, comparison_df, fairness_df

model, feature_names, explainer, df_ref, causal_rules, comparison_df, fairness_df = load_assets()

def get_status(feat):
    if causal_rules.empty: return "UNKNOWN"
    r = causal_rules[causal_rules["Feature"] == feat]
    return r["Status"].values[0] if not r.empty else "UNKNOWN"

def get_cate(feat):
    if causal_rules.empty: return 0.0
    r = causal_rules[causal_rules["Feature"] == feat]
    return float(r["CATE"].values[0]) if not r.empty else 0.0

def get_eval(feat):
    if causal_rules.empty: return "N/A"
    r = causal_rules[causal_rules["Feature"] == feat]
    return float(r["E_Value"].values[0]) if not r.empty else "N/A"

def build_shap_table(X_input, n=7):
    """Returns top N features by SHAP magnitude as a DataFrame."""
    shap_vals = explainer.shap_values(X_input)
    if isinstance(shap_vals, list): shap_vals = shap_vals[1]
    shap_vals = shap_vals.flatten()
    df = pd.DataFrame({
        "Feature": feature_names,
        "SHAP":    shap_vals,
        "Value":   X_input.flatten(),
    })
    df["Abs_SHAP"] = df["SHAP"].abs()
    return df.nlargest(n, "Abs_SHAP").copy()

def get_clean_features(X_input):
    """
    Dynamically finds features satisfying two rules:
      Rule 1: Show at least 10 features total.
      Rule 2: At least 4 of those must be non-confounded (ROBUST or UNKNOWN).
    Expands from 10 → 15 → 20 → 25 until both rules are satisfied.
    If we reach 25 and still cannot find 4 clean ones, we show what we have.
    """
    for n in [10, 15, 20, 25]:
        top = build_shap_table(X_input, n=n)
        top["Status"] = top["Feature"].apply(get_status)
        top["CATE"]   = top["Feature"].apply(get_cate)
        top["EValue"] = top["Feature"].apply(get_eval)

        fragile   = top[top["Status"] == "FRAGILE"]
        protected = top[top["Status"] == "PROTECTED"]
        clean     = top[top["Status"].isin(["ROBUST", "UNKNOWN"])]

        # Both rules satisfied, or we have reached max expansion
        if (len(top) >= 10 and len(clean) >= 4) or n == 25:
            return top, clean, fragile, protected

    return top, clean, fragile, protected


st.sidebar.title("🏥 Clinical AI Auditor")
page = st.sidebar.radio("Navigate", [
    "Patient Risk Assessment",
    "Naive vs Causal Comparison",
    "Fairness Audit"
])
st.sidebar.divider()


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 1 — Patient Risk Assessment
# ══════════════════════════════════════════════════════════════════════════════
if page == "Patient Risk Assessment":

    st.title("🏥 Clinical Decision Support with Causal Auditing")
    st.markdown(
        "Enter patient details on the left. The system will **predict** readmission risk, "
        "**explain** why, **flag** misleading AI reasons, show the **true causes**, "
        "and suggest **safe clinical actions**."
    )
    st.divider()

    # ── Sidebar inputs ────────────────────────────────────────────────────────
    st.sidebar.header("Enter Patient Details")
    time_in_hospital   = st.sidebar.number_input("Time in Hospital (Days)",      min_value=1,  max_value=14,  value=4,  step=1)
    num_lab_procedures = st.sidebar.number_input("Number of Lab Tests Done",     min_value=1,  max_value=132, value=44, step=1)
    num_medications    = st.sidebar.number_input("Number of Medications",        min_value=1,  max_value=81,  value=15, step=1)
    num_procedures     = st.sidebar.number_input("Number of Procedures",         min_value=0,  max_value=6,   value=1,  step=1)
    number_inpatient   = st.sidebar.number_input("Prior Hospital Admissions",    min_value=0,  max_value=21,  value=0,  step=1)
    number_emergency   = st.sidebar.number_input("Emergency Visits (Past Year)", min_value=0,  max_value=76,  value=0,  step=1)
    insulin_val        = st.sidebar.radio("Insulin Prescribed?", ["Yes", "No"])
    predict_btn        = st.sidebar.button("🔍 Predict & Audit", type="primary")

    if not predict_btn:
        st.info("👈 Fill in the patient details on the left and click **Predict & Audit**.")
    else:
        # ── Build patient row ─────────────────────────────────────────────────
        patient_dict = dict.fromkeys(feature_names, 0.0)
        patient_dict["time_in_hospital"]   = float(time_in_hospital)
        patient_dict["num_lab_procedures"] = float(num_lab_procedures)
        patient_dict["num_medications"]    = float(num_medications)
        patient_dict["num_procedures"]     = float(num_procedures)
        patient_dict["number_inpatient"]   = float(number_inpatient)
        patient_dict["number_emergency"]   = float(number_emergency)
        if "insulin_prescribed" in patient_dict:
            patient_dict["insulin_prescribed"] = 1.0 if insulin_val == "Yes" else 0.0

        X_input = np.array([patient_dict[f] for f in feature_names]).reshape(1, -1)
        prob    = float(model.predict_proba(X_input)[0, 1])
        is_high = prob >= 0.5

        # ── Get features dynamically (expand if all confounded) ───────────────
        all_top, clean_feats, fragile_feats, protected_feats = get_clean_features(X_input)
        n_shown = len(all_top)

        # ════════════════════════════════════
        # STEP 1 — PREDICTION
        # ════════════════════════════════════
        st.markdown("## Step 1 — Prediction Result")
        col_score, col_verdict = st.columns([1, 2])
        with col_score:
            color = "#d62728" if is_high else "#2ca02c"
            st.markdown(
                f"<div style='text-align:center;padding:28px;border-radius:14px;"
                f"background:{color}18;border:2.5px solid {color};'>"
                f"<p style='font-size:3.8rem;font-weight:800;color:{color};margin:0'>{prob*100:.1f}%</p>"
                f"<p style='font-size:1.1rem;color:{color};margin:6px 0 0;font-weight:600'>"
                f"{'HIGH RISK' if is_high else 'LOW RISK'}</p></div>",
                unsafe_allow_html=True
            )
        with col_verdict:
            st.markdown("&nbsp;")
            if is_high:
                st.error(
                    f"**This patient is likely to be readmitted to hospital within 30 days.**\n\n"
                    f"The model predicts a **{prob*100:.1f}% chance of readmission**. "
                    f"This is above the 50% decision threshold. "
                    f"This patient needs careful discharge planning and close follow-up after leaving hospital."
                )
            else:
                st.success(
                    f"**This patient is NOT likely to be readmitted to hospital within 30 days.**\n\n"
                    f"The model predicts only a **{prob*100:.1f}% chance of readmission**. "
                    f"This is below the 50% decision threshold. "
                    f"Standard discharge process can be followed."
                )
        st.divider()

        # ════════════════════════════════════
        # STEP 2 — WHAT THE AI SEES
        # ════════════════════════════════════
        st.markdown(f"## Step 2 — What the AI Model Says (Top {n_shown} Factors)")
        st.markdown(
            "Below are the factors the AI model identified as influencing this prediction — "
            "shown exactly as a standard hospital AI would report them. "
            "**⚠️ Some of these may be misleading.** Step 3 will identify which ones."
        )

        for _, row in all_top.iterrows():
            feat  = row["Feature"]
            label = get_label(feat)
            val   = row["Value"]
            shap  = row["SHAP"]
            pval  = plain_value(feat, val, shap)
            bc    = "#d62728" if shap > 0 else "#2ca02c"
            direction_text = "This is pushing the risk UP" if shap > 0 else "This is keeping the risk DOWN"
            st.markdown(
                f"<div style='padding:11px 16px;margin:6px 0;border-radius:8px;"
                f"border-left:4px solid {bc};background:{bc}0f;'>"
                f"<b>{label}</b> = <b>{pval}</b><br>"
                f"<span style='color:{bc};font-size:0.92rem'>{direction_text}</span>"
                f"</div>",
                unsafe_allow_html=True
            )
        st.divider()

        # ════════════════════════════════════
        # STEP 3 — CONFOUNDING WARNING
        # ════════════════════════════════════
        st.markdown("## Step 3 — ⚠️ Which AI Reasons Are Misleading?")
        st.markdown(
            "Our **Causal Audit** examined each factor above. "
            "Some features appear important to the AI only because sicker patients naturally "
            "score higher on them. These are called **confounded** features — "
            "the AI is seeing a *sign of sickness*, not a *cause of readmission*."
        )

        if fragile_feats.empty and protected_feats.empty:
            st.success("✅ All AI reasons above are causally reliable. No confounding detected.")
        else:
            for _, row in fragile_feats.iterrows():
                label = get_label(row["Feature"])
                val   = row["Value"]
                shap  = row["SHAP"]
                e_val = row["EValue"]
                cate  = row["CATE"]
                pval  = plain_value(row["Feature"], val, shap)
                # Plain explanation of WHY it is confounded
                reason = (
                    f"Sicker patients tend to have more {label.lower().replace('(which is above average)','').strip()}. "
                    f"So the AI sees this and thinks it causes readmission — but actually, it is just "
                    f"a reflection of how sick the patient already is."
                )
                st.error(
                    f"⚠️ **{label} = {pval} — This reason is MISLEADING**\n\n"
                    f"The AI flagged this as important. But our causal audit found it is confounded.\n\n"
                    f"{reason}\n\n"
                    f"The true causal effect of this feature (after accounting for severity) is only "
                    f"**{abs(cate):.4f}** — much smaller than what the AI thinks.\n\n"
                    f"❌ **Do not try to change this to reduce the risk score. It will not actually help the patient.**"
                )
            for _, row in protected_feats.iterrows():
                label = get_label(row["Feature"])
                st.warning(
                    f"⚠️ **{label} — This is a demographic attribute**\n\n"
                    f"Age, race, and gender cannot be changed. "
                    f"They are excluded from any clinical recommendations."
                )
        st.divider()

        # ════════════════════════════════════
        # STEP 4 — TRUE CAUSAL EXPLANATION
        # ════════════════════════════════════
        st.markdown("## Step 4 — What is Actually Causing the Risk?")
        st.markdown(
            "After removing the misleading (confounded) features, "
            "here are the factors that are **genuinely contributing** to this patient's readmission risk."
        )

        if clean_feats.empty:
            # All top features were confounded — give honest, plain English explanation
            st.info(
                f"**The AI model's top {n_shown} features were all found to be confounded.**\n\n"
                f"This means there is no single, clean, fixable reason driving this patient's "
                f"{'high' if is_high else 'low'} risk. The risk is coming from the patient's "
                f"**overall medical complexity** — they have multiple conditions, a history of "
                f"frequent hospital contact, and a high overall disease burden.\n\n"
                f"**What this means in practice:** No single medication change or procedure will "
                f"significantly reduce the risk. The focus should be on comprehensive discharge "
                f"planning, strong follow-up, and ensuring the patient has good support at home."
            )
        else:
            # Show clean features in plain English
            st.markdown("The following factors are **not confounded** and represent real clinical signals:")

            increasing = clean_feats[clean_feats["SHAP"] > 0]
            decreasing = clean_feats[clean_feats["SHAP"] < 0]

            for _, row in increasing.iterrows():
                label = get_label(row["Feature"])
                val   = row["Value"]
                pval  = plain_value(row["Feature"], val, row["SHAP"])
                st.markdown(
                    f"<div style='padding:11px 16px;margin:6px 0;border-radius:8px;"
                    f"border-left:4px solid #d62728;background:#d6272808;'>"
                    f"<b>↑ {label}</b> = {pval}<br>"
                    f"<span style='color:#d62728;font-size:0.92rem'>"
                    f"This is genuinely increasing the readmission risk.</span>"
                    f"</div>",
                    unsafe_allow_html=True
                )

            for _, row in decreasing.iterrows():
                label = get_label(row["Feature"])
                val   = row["Value"]
                pval  = plain_value(row["Feature"], val, row["SHAP"])
                st.markdown(
                    f"<div style='padding:11px 16px;margin:6px 0;border-radius:8px;"
                    f"border-left:4px solid #2ca02c;background:#2ca02c08;'>"
                    f"<b>↓ {label}</b> = {pval}<br>"
                    f"<span style='color:#2ca02c;font-size:0.92rem'>"
                    f"This is genuinely reducing the readmission risk.</span>"
                    f"</div>",
                    unsafe_allow_html=True
                )

            # Plain English summary — no jargon
            inc_labels = [get_label(r["Feature"]) for _, r in increasing.iterrows()]
            dec_labels = [get_label(r["Feature"]) for _, r in decreasing.iterrows()]

            summary_parts = []
            if inc_labels:
                joined = " and ".join(inc_labels)
                summary_parts.append(f"the patient's **{joined}** {'is' if len(inc_labels)==1 else 'are'} high")
            if dec_labels:
                joined = " and ".join(dec_labels)
                summary_parts.append(f"**{joined}** {'is' if len(dec_labels)==1 else 'are'} in a favourable range")

            if summary_parts:
                st.markdown("---")
                risk_level = "high" if is_high else "at this level"
                st.markdown(
                    f"**In plain English:** The readmission risk is {risk_level} mainly because "
                    + ", and ".join(summary_parts) + "."
                )
        st.divider()

        # ════════════════════════════════════
        # STEP 5 — WHAT THE DOCTOR CAN DO
        # ════════════════════════════════════
        st.markdown("## Step 5 — What Can Be Done to Reduce the Risk?")

        if not is_high:
            st.success(
                "This patient is already **LOW RISK**. No urgent intervention is needed.\n\n"
                "Follow standard discharge protocol with a routine follow-up at 2–4 weeks."
            )
        else:
            st.markdown(
                "The system now looks for **safe, actionable changes** a doctor can make "
                "— using only features that are **not confounded** "
                "(medications, discharge planning, HbA1c management). "
                "It finds the minimum change that would bring the risk below 50%."
            )

            from layer3_counterfactual import generate_for_patient
            with st.spinner("Finding safe clinical interventions..."):
                cf = generate_for_patient(patient_dict, feature_names, model, df_ref)

            orig_pct  = cf["original_prob"] * 100
            final_pct = cf["final_prob"]    * 100
            changes   = cf["changes"]
            flipped   = cf["flipped"]

            if changes:
                if flipped:
                    st.success(
                        f"✅ **The readmission risk CAN be reduced from {orig_pct:.1f}% → {final_pct:.1f}%** "
                        f"with {len(changes)} intervention{'s' if len(changes) > 1 else ''}. "
                        f"Here are the 'What If' scenarios:"
                    )
                else:
                    st.warning(
                        f"⚠️ The following interventions can partially reduce risk from {orig_pct:.1f}% to {final_pct:.1f}%, "
                        f"but it remains above 50%. This patient likely needs comprehensive support."
                    )

                running_risk = orig_pct
                for i, ch in enumerate(changes, 1):
                    feat       = ch["feature"]
                    label      = get_label(feat)
                    from_val   = ch["from_val"]
                    to_val     = ch["to_val"]
                    risk_after = ch["prob_after"] * 100
                    drop       = running_risk - risk_after

                    # ── "What if / If only" counterfactual sentence ──────────
                    if feat in BINARY_FEATURES:
                        if to_val == 1.0 and from_val == 0.0:
                            what_if_q  = f"What if the doctor **prescribes {label}** for this patient?"
                            what_if_a  = (
                                f"If {label.lower()} is prescribed (changing from Not Prescribed → Prescribed), "
                                f"the readmission risk would drop from **{running_risk:.1f}% to {risk_after:.1f}%** "
                                f"— a reduction of **{drop:.1f} percentage points**."
                            )
                        else:
                            what_if_q  = f"What if **{label}** is stopped for this patient?"
                            what_if_a  = (
                                f"If {label.lower()} is stopped (changing from Prescribed → Not Prescribed), "
                                f"the readmission risk would drop from **{running_risk:.1f}% to {risk_after:.1f}%** "
                                f"— a reduction of **{drop:.1f} percentage points**."
                            )
                    elif feat.startswith("discharge_disposition_"):
                        what_if_q = f"What if the patient is **discharged to {label.lower()}**?"
                        what_if_a = (
                            f"If the discharge plan is changed to '{label}' (instead of the current plan), "
                            f"the readmission risk would drop from **{running_risk:.1f}% to {risk_after:.1f}%** "
                            f"— a reduction of **{drop:.1f} percentage points**."
                        )
                    elif feat.startswith("diag_"):
                        what_if_q = f"What if the primary diagnosis is noted as **{label}**?"
                        what_if_a = (
                            f"Noting this diagnosis context changes the risk from **{running_risk:.1f}% to {risk_after:.1f}%** "
                            f"— a reduction of **{drop:.1f} percentage points**."
                        )
                    else:
                        direction = "reduced" if to_val < from_val else "increased"
                        what_if_q = (
                            f"What if **{label}** is {direction} from {from_val:.0f} to {to_val:.0f}?"
                        )
                        what_if_a = (
                            f"If {label.lower()} is {direction} from {from_val:.0f} to {to_val:.0f}, "
                            f"the readmission risk would drop from **{running_risk:.1f}% to {risk_after:.1f}%** "
                            f"— a reduction of **{drop:.1f} percentage points**."
                        )

                    running_risk = risk_after

                    border_color = "#1a6e3c" if i == len(changes) and flipped else "#2166a0"
                    bg_color     = "#e6f4ec" if i == len(changes) and flipped else "#e8f0f8"
                    txt_color    = "#155724" if i == len(changes) and flipped else "#0c3461"

                    st.markdown(
                        f"<div style='padding:16px 20px;margin:10px 0;border-radius:10px;"
                        f"border:1.5px solid {border_color};background:{bg_color};'>"
                        f"<p style='margin:0 0 6px;font-size:1rem;font-weight:600;color:{border_color};'>"
                        f"💡 Scenario {i}: {what_if_q}</p>"
                        f"<p style='margin:0;font-size:0.95rem;color:{txt_color};'>{what_if_a}</p>"
                        f"</div>",
                        unsafe_allow_html=True
                    )

                if flipped:
                    st.markdown(
                        f"<div style='padding:14px 20px;margin:12px 0;border-radius:10px;"
                        f"border:2px solid #1a6e3c;background:#d4edda;'>"
                        f"<p style='margin:0;font-size:1rem;font-weight:600;color:#155724;'>"
                        f"✅ Final outcome: If all {len(changes)} scenario{'s are' if len(changes)>1 else ' is'} "
                        f"acted upon, the readmission risk drops from <b>{orig_pct:.1f}%</b> → <b>{final_pct:.1f}%</b> "
                        f"— moving this patient from <b>HIGH RISK</b> to <b>LOW RISK</b>.</p>"
                        f"</div>",
                        unsafe_allow_html=True
                    )
            else:
                st.warning(
                    f"The system could not find a single safe actionable change that reduces risk below 50% "
                    f"for this patient (current risk: {orig_pct:.1f}%).\n\n"
                    f"This patient's risk is driven by their overall disease complexity "
                    f"rather than any single modifiable factor. "
                    f"**Recommended approach:** Arrange comprehensive post-discharge support — "
                    f"a follow-up appointment within 7 days, home health nurse visits if possible, "
                    f"a clear medication review, and ensure the patient and family understand the discharge plan."
                )

        st.divider()

        # ════════════════════════════════════
        # STEP 6 — FINAL RECOMMENDATION
        # ════════════════════════════════════
        st.markdown("## Step 6 — Final Clinical Recommendation")
        if is_high:
            st.error(
                "**This patient is HIGH RISK for 30-day readmission. Recommended actions:**\n\n"
                "- Arrange a **follow-up appointment within 7 days** of discharge\n"
                "- Review and clearly explain the **full discharge plan** with the patient\n"
                "- Consider arranging **home health nurse visits** or a **skilled nursing facility**\n"
                "- Make sure the patient understands all their medications and dosages\n"
                "- **Flag this patient** in the hospital system for readmission monitoring\n\n"
                "> ⚠️ Do not adjust medications or procedures just to lower the AI risk score. "
                "The causal audit showed those features are confounded — changing them will not "
                "actually reduce the chance of the patient coming back."
            )
        else:
            st.success(
                "**This patient is LOW RISK. Standard discharge is appropriate:**\n\n"
                "- Provide clear discharge instructions\n"
                "- Schedule a routine follow-up at 2–4 weeks\n"
                "- Educate the patient on warning signs to watch for at home\n\n"
                "> Continue standard monitoring protocol."
            )


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 2 — Naive vs Causal Comparison
# ══════════════════════════════════════════════════════════════════════════════
elif page == "Naive vs Causal Comparison":
    st.title("📊 Naive AI vs Causal Model — Confounding Detection Proof")
    st.markdown("This is the **academic core** of the BTP project. Side-by-side proof of how standard AI is fooled by confounders.")
    st.divider()

    st.markdown("""
### The Core Problem — In Plain English

Imagine the hospital AI says: *"This patient is on insulin, and our model shows insulin increases readmission risk."*
A doctor might think: *"Let's reduce the insulin dose to reduce readmission."* **That is dangerous.**

Here is why:
- Doctors give insulin to the **most severely diabetic patients** — those with dangerously high blood sugar
- Those same severely ill patients are also the ones most likely to come back to hospital — **because they are very sick**, not because of the insulin
- The AI saw: patients on insulin → higher readmission. It wrongly concluded: *insulin causes readmission*
- The truth: **disease severity → doctor prescribes insulin** AND **disease severity → patient gets readmitted**
- Insulin is just a *signal* that the patient is seriously ill. Not the cause of readmission.

This mistake is called **Confounding by Indication**.

### How We Detect It

| | Model 1 — Standard AI | Model 2 — Our Causal Audit |
|---|---|---|
| Method | Plain logistic regression on top features | DoubleML — mathematically removes severity confounding |
| What it measures | Raw correlations (gets fooled by severity) | True causal effect only |
| Safe for a doctor to act on? | ❌ No | ✅ Yes |

A **large gap** between the two numbers means the AI was being fooled.
    """)
    st.divider()

    if comparison_df.empty:
        st.warning("⚠️ Run `python layer2_naive_vs_causal.py` first to generate this data.")
    else:
        n_conf  = len(comparison_df[comparison_df["Verdict"].str.contains("CONFOUNDED")])
        n_clean = len(comparison_df[comparison_df["Verdict"] == "CLEAN (minimal confounding)"])
        c1, c2, c3 = st.columns(3)
        c1.metric("Features Audited", len(comparison_df))
        c2.metric("Confounded (AI was fooled)", n_conf, delta="dangerous to act on", delta_color="inverse")
        c3.metric("Clean (reliable)", n_clean)
        st.divider()

        st.subheader("Feature-by-Feature Comparison Table")
        st.caption(
            "Naive Effect = what standard AI says. "
            "Causal Effect = true effect after removing confounders. "
            "Gap = how much the AI was fooled. Bigger gap = more dangerous."
        )

        def color_v(val):
            if "HEAVILY" in str(val):  return "background-color:#ffcccc;color:#7a0000;font-weight:bold"
            elif "MILDLY" in str(val): return "background-color:#fff3cd;color:#6b4a00"
            elif "CLEAN"  in str(val): return "background-color:#d4edda;color:#155724"
            return ""

        disp = comparison_df.copy()
        disp["Feature"]      = disp["Feature"].apply(lambda x: FEATURE_LABELS.get(x, x.replace("_"," ").title()))
        disp["Naive_Effect"] = disp["Naive_Effect"].apply(lambda x: f"{x:+.4f}")
        disp["Causal_CATE"]  = disp["Causal_CATE"].apply(lambda x: f"{x:+.4f}")
        disp["Gap"]          = disp["Gap"].apply(lambda x: f"{x:.4f}")
        st.dataframe(
            disp[["Feature","Naive_Effect","Causal_CATE","Gap","E_Value","Status","Verdict"]]
                .style.applymap(color_v, subset=["Verdict"]),
            use_container_width=True, hide_index=True
        )
        st.divider()

        st.subheader("Visual Comparison")
        chart_df = comparison_df.copy()
        chart_df.index = chart_df["Feature"].apply(
            lambda x: FEATURE_LABELS.get(x, x.replace("_"," ").title())
        )
        st.bar_chart(chart_df[["Naive_Effect","Causal_CATE"]].rename(
            columns={"Naive_Effect":"Standard AI (Naive)","Causal_CATE":"Causal Audit (DoubleML)"}
        ))
        st.caption(
            "For each feature, if the two bars look very different — the AI was being fooled. "
            "If they look similar — the AI reason is trustworthy."
        )
        st.divider()

        st.subheader("Plain English Summary")
        for _, row in comparison_df.iterrows():
            label = FEATURE_LABELS.get(row["Feature"], row["Feature"].replace("_"," ").title())
            if "CONFOUNDED" in row["Verdict"]:
                st.error(
                    f"**{label}**\n\n"
                    f"The standard AI says this feature changes readmission risk by `{row['Naive_Effect']:+.4f}` — which sounds significant.\n\n"
                    f"But after the causal audit removes the effect of disease severity, the true causal effect is only `{row['Causal_CATE']:+.4f}` — much smaller.\n\n"
                    f"Gap = `{row['Gap']:.4f}`. The AI was confusing **disease severity** with **causation**. "
                    f"This feature is a *marker of how sick the patient is*, not a *cause of readmission*."
                )
            else:
                st.success(
                    f"**{label}** — Standard AI: `{row['Naive_Effect']:+.4f}` | "
                    f"Causal Audit: `{row['Causal_CATE']:+.4f}` | Gap: `{row['Gap']:.4f}`. "
                    f"Minimal difference — this AI reason is relatively trustworthy."
                )


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 3 — Fairness Audit
# ══════════════════════════════════════════════════════════════════════════════
elif page == "Fairness Audit":
    st.title("⚖️ Fairness Audit — Is the Model Treating All Patients Equally?")
    st.markdown(
        "This checks whether the AI's readmission predictions are fair across different patient groups. "
        "A fair AI should flag HIGH RISK at similar rates for all age groups, races, and genders — "
        "when those patients have similar actual health conditions."
    )
    st.divider()

    st.markdown("""
### How We Measure Fairness — The 80% Rule

We compare how often the AI flags each demographic group as HIGH RISK, relative to the largest group (the reference).

- If Group A is flagged at **less than 80%** of the reference group's rate → the AI is **under-estimating** risk for Group A (potentially missing patients who need help)
- If Group A is flagged at **more than 125%** of the reference group's rate → the AI is **over-predicting** risk for Group A (potentially over-treating or unfairly burdening that group)

Both are unfair. Both are flagged.
    """)
    st.divider()

    if fairness_df.empty:
        st.warning("⚠️ Run `python aux_fairness.py` first to generate fairness data.")
    else:
        for group_name in fairness_df["group"].unique():
            st.subheader(f"Fairness by {group_name.upper()}")
            sub      = fairness_df[fairness_df["group"] == group_name].copy()
            ref_row  = sub.loc[sub["n"].idxmax()]
            ref_rate = ref_row["pos_pred_rate"]
            ref_val  = ref_row["value"]
            sub["Disparate Impact Ratio"] = (sub["pos_pred_rate"] / ref_rate).round(3)
            sub["Fairness Verdict"]       = sub["Disparate Impact Ratio"].apply(
                lambda x: "⚠️ UNFAIR" if (x < 0.80 or x > 1.25) else "✅ Fair"
            )
            st.caption(
                f"Reference group (largest, used as baseline): **{ref_val}**. "
                f"The AI flags {ref_rate*100:.1f}% of this group as HIGH RISK."
            )
            st.dataframe(
                sub[["value","n","accuracy","pos_pred_rate","recall_tpr","fpr",
                     "Disparate Impact Ratio","Fairness Verdict"]].rename(columns={
                    "value":"Group","n":"Number of Patients",
                    "accuracy":"Accuracy","pos_pred_rate":"% Flagged HIGH RISK",
                    "recall_tpr":"True Positive Rate","fpr":"False Positive Rate",
                }),
                use_container_width=True, hide_index=True
            )
            for _, row in sub.iterrows():
                di = row["pos_pred_rate"] / ref_rate if ref_rate > 0 else 1.0
                if di < 0.80:
                    st.warning(
                        f"⚠️ **{row['value']}** patients are flagged HIGH RISK at only "
                        f"{di:.2f}× the rate of {ref_val} patients. "
                        f"The model may be **under-estimating risk** for this group — "
                        f"some high-risk patients in this group may not be getting the attention they need."
                    )
                elif di > 1.25:
                    st.warning(
                        f"⚠️ **{row['value']}** patients are flagged HIGH RISK at "
                        f"{di:.2f}× the rate of {ref_val} patients. "
                        f"The model may be **over-predicting risk** for this group — "
                        f"some low-risk patients may be receiving unnecessary interventions."
                    )
            st.divider()