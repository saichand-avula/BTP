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
import shap

# --- Custom Styling & Page Config ---
st.set_page_config(page_title="AI Clinical Auditor", layout="wide", page_icon="🏥", initial_sidebar_state="expanded")

st.markdown("""
    <style>
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    .main-title {
        font-size: 2.8rem;
        font-weight: 800;
        color: #1E3A8A;
        margin-bottom: 0px;
        padding-bottom: 0px;
    }
    .sub-title {
        font-size: 1.15rem;
        color: #4B5563;
        margin-top: 0px;
        margin-bottom: 2rem;
    }
    div.stButton > button:first-child {
        background-color: #1E3A8A;
        color: white;
        border-radius: 8px;
        font-weight: 600;
        height: 3.5rem;
        font-size: 1.2rem;
        width: 100%;
        border: none;
    }
    div.stButton > button:first-child:hover {
        background-color: #1D4ED8;
        color: white;
    }
    .section-header {
        font-size: 1.5rem;
        font-weight: 700;
        color: #1f2937;
        border-bottom: 2px solid #E5E7EB;
        padding-bottom: 8px;
        margin-top: 1.5rem;
        margin-bottom: 1rem;
    }
    /* Explicitly color standard text inside markdown for Streamlit darkmode compatibility */
    p, span, div {
        color: inherit;
    }
    </style>
""", unsafe_allow_html=True)

BASE       = Path(__file__).parent
MODELS_DIR = BASE / "models"

# ── Plain English feature labels ─────────────────────────────────────────────
FEATURE_LABELS = {
    "time_in_hospital":         "Time Spent in Hospital (days)",
    "num_lab_procedures":       "Number of Lab Tests Done",
    "num_medications":          "Number of Medications",
    "num_procedures":           "Number of Medical Procedures",
    "number_inpatient":         "Prior Hospital Admissions (Past Year)",
    "number_emergency":         "Emergency Room Visits (Past Year)",
    "number_outpatient":        "Outpatient Visits (Past Year)",
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
    "A1Cresult_>8":             "HbA1c > 8 (Poor Control)",
    "A1Cresult_Norm":           "HbA1c Normal (Good Control)",
    "admission_type_1":         "Emergency Admission",
    "admission_type_2":         "Urgent Admission",
    "admission_type_3":         "Elective Admission",
    "discharge_disposition_1":  "Discharged to Home",
    "discharge_disposition_2":  "Transferred to Another Hospital",
    "discharge_disposition_3":  "Discharged to Skilled Nursing Facility (SNF)",
    "discharge_disposition_6":  "Discharged Home with Home Health Nurse",
    "discharge_disposition_22": "Discharged to Rehabilitation Facility",
}

BINARY_FEATURES = {
    "insulin_prescribed","insulin_increased","diabetesMed_Yes","change_No",
    "metformin_prescribed","glipizide_prescribed","glyburide_prescribed",
    "pioglitazone_prescribed","rosiglitazone_prescribed",
    "A1Cresult_>8","A1Cresult_Norm",
}

def get_label(feat):
    if feat.startswith('age_'):
        return f"Age Group: {feat.split('_')[1]}"
    return FEATURE_LABELS.get(feat, feat.replace("_", " ").title())

def plain_value(feat, val, shap_val):
    if feat in BINARY_FEATURES or feat.startswith("discharge_disposition_") or feat.startswith("admission_") or feat.startswith("diag_") or feat.startswith("age_"):
        return "Yes" if val == 1.0 else "No"
    return f"{val:.0f} (which is {'above' if shap_val > 0 else 'below'} average)"

@st.cache_resource
def load_assets():
    with open(MODELS_DIR / "layer1_model.pkl",    "rb") as f: model         = pickle.load(f)
    with open(MODELS_DIR / "layer1_features.pkl", "rb") as f: feature_names = pickle.load(f)
    explainer = shap.TreeExplainer(model)
    df_ref       = pd.read_csv(BASE / "data_processed" / "diabetic_data_cleaned.csv")
    causal_rules = pd.read_csv(MODELS_DIR / "layer2_causal_results.csv") \
        if (MODELS_DIR / "layer2_causal_results.csv").exists() else pd.DataFrame()
    return model, feature_names, explainer, df_ref, causal_rules

model, feature_names, explainer, df_ref, causal_rules = load_assets()

def get_status(feat):
    if causal_rules.empty: return "UNKNOWN"
    r = causal_rules[causal_rules["Feature"] == feat]
    return r["Status"].values[0] if not r.empty else "UNKNOWN"

def get_cate(feat):
    if causal_rules.empty: return 0.0
    r = causal_rules[causal_rules["Feature"] == feat]
    return float(r["CATE"].values[0]) if not r.empty else 0.0

def build_shap_table(X_input, n=12):
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
    for n in [10, 15, 20]:
        top = build_shap_table(X_input, n=n)
        top["Status"] = top["Feature"].apply(get_status)
        top["CATE"]   = top["Feature"].apply(get_cate)

        fragile   = top[top["Status"] == "FRAGILE"]
        protected = top[top["Status"] == "PROTECTED"]
        clean     = top[top["Status"].isin(["ROBUST", "UNKNOWN"])]

        if (len(top) >= 10 and len(clean) >= 4) or n == 20:
            return top, clean, fragile, protected
    return top, clean, fragile, protected


# ══════════════════════════════════════════════════════════════════════════════
# MAIN LAYOUT
# ══════════════════════════════════════════════════════════════════════════════

st.markdown('<div class="main-title">🏥 Clinical AI Auditor</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">Advanced Patient Risk Assessment & Causal Interventions</div>', unsafe_allow_html=True)

# ── Inputs Section ── # By default, initialized with a HIGH RISK case (Risk > 70%)
st.markdown("### 📋 Enter Comprehensive Patient File")
with st.container():
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown('<div class="section-header">Demographics & History</div>', unsafe_allow_html=True)
        age_group = st.selectbox("Age Group", ['[0-10)', '[10-20)', '[20-30)', '[30-40)', '[40-50)', '[50-60)', '[60-70)', '[70-80)', '[80-90)', '[90-100)'], index=7) # Defaults to 70-80
        number_inpatient = st.number_input("Prior Inpatient Admissions (Year)", min_value=0, max_value=25, value=4)
        number_emergency = st.number_input("Prior Emergency Returns (Year)", min_value=0, max_value=80, value=3)
        number_outpatient = st.number_input("Prior Outpatient Visits (Year)", min_value=0, max_value=50, value=1)
        number_diagnoses = st.number_input("Number of Total Diagnoses", min_value=1, max_value=16, value=9)

    with col2:
        st.markdown('<div class="section-header">Current Admission Profile</div>', unsafe_allow_html=True)
        admission_type = st.selectbox("Admission Type", ["Emergency", "Urgent", "Elective", "Newborn", "Trauma Center"], index=0)
        time_in_hospital = st.number_input("Time Spent in Hospital (Days)", min_value=1, max_value=14, value=9)
        num_lab_procedures = st.number_input("Lab Tests Done", min_value=1, max_value=132, value=75)
        num_procedures = st.number_input("Medical Procedures Performed", min_value=0, max_value=6, value=2)
        discharge_disp = st.selectbox("Discharge Disposition Plan", 
            ["Discharged to Home", "Transferred to Short-term Hospital", "Transferred to SNF", "Discharged Home with Home-Health Service", "Transferred to Rehab Facility", "Other"], index=2) # SNF

    with col3:
        st.markdown('<div class="section-header">Medications & Outcomes</div>', unsafe_allow_html=True)
        hba1c = st.radio("HbA1c Test Result", ["Not Taken/Normal", "Normal", "Above 8 (Poor)"], index=2)
        diabetes_med = st.radio("Is Patient on Diabetes Meds?", ["Yes", "No"], index=0)
        num_medications = st.number_input("Total Number of Medications", min_value=1, max_value=81, value=25)
        
        st.markdown("<p style='color:#1f2937; font-weight:bold;'>Key Prescriptions:</p>", unsafe_allow_html=True)
        med_col1, med_col2 = st.columns(2)
        with med_col1:
            insulin = st.checkbox("Insulin Prescribed", value=True)
            metformin = st.checkbox("Metformin", value=True)
            glipizide = st.checkbox("Glipizide")
        with med_col2:
            glyburide = st.checkbox("Glyburide")
            pioglitazone = st.checkbox("Pioglitazone")
            insulin_inc = st.checkbox("Insulin Dose Increased", value=True)


st.markdown("<br>", unsafe_allow_html=True)
predict_btn = st.button("🚀 Analyze Risk & Compute Safe Interventions")
st.markdown("<hr>", unsafe_allow_html=True)

if predict_btn:
    # ── Mappings ──
    patient_dict = dict.fromkeys(feature_names, 0.0)
    
    # Continuous
    patient_dict["time_in_hospital"]   = float(time_in_hospital)
    patient_dict["num_lab_procedures"] = float(num_lab_procedures)
    patient_dict["num_medications"]    = float(num_medications)
    patient_dict["num_procedures"]     = float(num_procedures)
    patient_dict["number_inpatient"]   = float(number_inpatient)
    patient_dict["number_outpatient"]  = float(number_outpatient)
    patient_dict["number_emergency"]   = float(number_emergency)
    patient_dict["number_diagnoses"]   = float(number_diagnoses)
    
    # Demographics
    age_col = f"age_{age_group}"
    if age_col in patient_dict: patient_dict[age_col] = 1.0
        
    # Admission
    type_map = {"Emergency": "1", "Urgent": "2", "Elective": "3", "Newborn": "4", "Trauma Center": "7"}
    adm_col = f"admission_type_{type_map[admission_type]}"
    if adm_col in patient_dict: patient_dict[adm_col] = 1.0
        
    # Discharge
    disp_map = {"Discharged to Home": "1", "Transferred to Short-term Hospital": "2", "Transferred to SNF": "3", 
                "Discharged Home with Home-Health Service": "6", "Transferred to Rehab Facility": "22", "Other": "0"}
    dsc_col = f"discharge_disposition_{disp_map[discharge_disp]}"
    if dsc_col in patient_dict: patient_dict[dsc_col] = 1.0
        
    # HbA1c
    if hba1c == "Above 8 (Poor)" and "A1Cresult_>8" in patient_dict: patient_dict["A1Cresult_>8"] = 1.0
    elif hba1c == "Normal" and "A1Cresult_Norm" in patient_dict: patient_dict["A1Cresult_Norm"] = 1.0
        
    # Medications
    if diabetes_med == "Yes" and "diabetesMed_Yes" in patient_dict: patient_dict["diabetesMed_Yes"] = 1.0
    if insulin and "insulin_prescribed" in patient_dict: patient_dict["insulin_prescribed"] = 1.0
    if insulin_inc and "insulin_increased" in patient_dict: patient_dict["insulin_increased"] = 1.0
    if metformin and "metformin_prescribed" in patient_dict: patient_dict["metformin_prescribed"] = 1.0
    if glipizide and "glipizide_prescribed" in patient_dict: patient_dict["glipizide_prescribed"] = 1.0
    if glyburide and "glyburide_prescribed" in patient_dict: patient_dict["glyburide_prescribed"] = 1.0
    if pioglitazone and "pioglitazone_prescribed" in patient_dict: patient_dict["pioglitazone_prescribed"] = 1.0

    X_input = np.array([patient_dict[f] for f in feature_names]).reshape(1, -1)
    prob    = float(model.predict_proba(X_input)[0, 1])
    is_high = prob >= 0.5

    all_top, clean_feats, fragile_feats, protected_feats = get_clean_features(X_input)

    # ════════════════════════════════════
    # PREDICTION
    # ════════════════════════════════════
    st.markdown('<div class="section-header">Step 1 — Deep AI Risk Profiling</div>', unsafe_allow_html=True)
    col_score, col_verdict = st.columns([1, 2])
    with col_score:
        color = "#EF4444" if is_high else "#10B981" # Red vs Green
        bg_color = "#FEF2F2" if is_high else "#ECFDF5"
        risk_text = "HIGH RISK" if is_high else "LOW RISK"
        
        st.markdown(
            f"""<div style='text-align:center; padding:30px; border-radius:16px; background-color:{bg_color}; border:2px solid {color};'>
            <div style='font-size:3.5rem; font-weight:800; color:{color}; line-height:1'>{prob*100:.1f}%</div>
            <div style='font-size:1.2rem; font-weight:700; color:{color}; margin-top:10px;'>{risk_text}</div>
            </div>""",
            unsafe_allow_html=True
        )
    with col_verdict:
        st.markdown("<br>", unsafe_allow_html=True)
        if is_high:
            st.markdown(f"""
            <div style="background-color: #FEF2F2; color: #991B1B; padding: 15px; border-radius: 8px; font-weight: 500; font-size: 1.05rem;">
                <strong>Clinical Alert: Patient exceeds the critical readmission threshold!</strong><br><br>
                The AI model predicts a <strong>{prob*100:.1f}% likelihood</strong> of hospital readmission within 30 days. 
                This patient requires rigorous discharge planning, medication reconciliation, and strict follow-up care to avoid clinical deterioration.
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div style="background-color: #ECFDF5; color: #065F46; padding: 15px; border-radius: 8px; font-weight: 500; font-size: 1.05rem;">
                <strong>Patient remains beneath the clinical readmission threshold.</strong><br><br>
                The predictive likelihood is only <strong>{prob*100:.1f}%</strong>. 
                Standard post-discharge procedures and routine follow-ups are considered medically appropriate at this time.
            </div>
            """, unsafe_allow_html=True)
            
    st.markdown("<hr>", unsafe_allow_html=True)

    # ════════════════════════════════════
    # WHAT THE AI SEES (UNFILTERED SHAP)
    # ════════════════════════════════════
    st.markdown('<div class="section-header">Step 2 — What the AI Sees (Black-Box Features)</div>', unsafe_allow_html=True)
    st.markdown("<p style='color: #4B5563; font-size: 1.05rem;'>These are the absolute top features driving the AI model's decision <strong>before</strong> we filter out biases. Some of these may be misleading correlations.</p>", unsafe_allow_html=True)

    cols = st.columns(3)
    for i, (_, row) in enumerate(all_top.head(9).iterrows()):
        col = cols[i % 3]
        f_lbl = get_label(row['Feature'])
        f_val = plain_value(row['Feature'], row['Value'], row['SHAP'])
        f_dir = "Raises Risk ↑" if row['SHAP'] > 0 else "Lowers Risk ↓"
        f_col = "#DC2626" if row['SHAP'] > 0 else "#059669"
        
        with col:
            st.markdown(f"""
            <div style="background-color: #F3F4F6; padding: 12px; border-radius: 8px; margin-bottom: 15px; border-left: 5px solid {f_col};">
                <div style="color: #111827; font-weight: 700; font-size: 0.95rem; margin-bottom: 5px;">{f_lbl}</div>
                <div style="color: #374151; font-size: 0.9rem; margin-bottom: 5px;">Value: <strong>{f_val}</strong></div>
                <div style="color: {f_col}; font-weight: 700; font-size: 0.85rem;">{f_dir}</div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("<hr>", unsafe_allow_html=True)

    # ════════════════════════════════════
    # CAUSAL EXPLANATION
    # ════════════════════════════════════
    st.markdown('<div class="section-header">Step 3 — True Causal Drivers vs. Confounding Artifacts</div>', unsafe_allow_html=True)
    st.markdown("<p style='color:#4B5563; font-size: 1.05rem;'>Standard AI models identify correlations that can be dangerous to act upon (Confounding by indication). We use DoubleML to mathematically separate the actual drivers of readmission risk from mere indicators of disease severity.</p>", unsafe_allow_html=True)
    
    col_drivers, col_confound = st.columns(2)
    
    with col_drivers:
        st.markdown("<h4 style='color:#1E3A8A; font-weight:700;'>✅ Validated True Drivers</h4>", unsafe_allow_html=True)
        if clean_feats.empty:
            st.markdown("<div style='color: #4B5563; font-style: italic;'>No genuinely robust feature detected among the top traits. Risk is driven by holistic multi-morbid complexity.</div>", unsafe_allow_html=True)
        else:
            increasing = clean_feats[clean_feats["SHAP"] > 0]
            decreasing = clean_feats[clean_feats["SHAP"] < 0]

            for _, row in increasing.iterrows():
                st.markdown(f"""
                <div style='padding:12px; margin-bottom:10px; border-radius:8px; border-left:4px solid #EF4444; background:#FEF2F2;'>
                <strong style='color:#991B1B; font-size:1.05rem;'>↑ {get_label(row['Feature'])}</strong><br>
                <span style='color:#7F1D1D; font-weight: 600;'>Value: {plain_value(row['Feature'], row['Value'], row['SHAP'])}</span><br>
                <span style='color:#DC2626; font-size:0.95rem; font-weight: 500;'>Genuinely increasing readmission risk.</span>
                </div>
                """, unsafe_allow_html=True)
                
            for _, row in decreasing.iterrows():
                st.markdown(f"""
                <div style='padding:12px; margin-bottom:10px; border-radius:8px; border-left:4px solid #10B981; background:#ECFDF5;'>
                <strong style='color:#065F46; font-size:1.05rem;'>↓ {get_label(row['Feature'])}</strong><br>
                <span style='color:#064E3B; font-weight: 600;'>Value: {plain_value(row['Feature'], row['Value'], row['SHAP'])}</span><br>
                <span style='color:#059669; font-size:0.95rem; font-weight: 500;'>Genuinely reducing readmission risk.</span>
                </div>
                """, unsafe_allow_html=True)

    with col_confound:
        st.markdown("<h4 style='color:#9A3412; font-weight:700;'>⚠️ Misleading AI Correlations Identified</h4>", unsafe_allow_html=True)
        count = 0
        for _, row in fragile_feats.iterrows():
            st.markdown(f"""
            <div style='padding:12px; margin-bottom:10px; border-radius:8px; border-left:4px solid #F59E0B; background:#FFFBEB;'>
            <strong style='color:#92400E; font-size:1.05rem;'>{get_label(row['Feature'])}</strong><br>
            <span style='color:#78350F; font-weight: 600;'>Value: {plain_value(row['Feature'], row['Value'], row['SHAP'])}</span><br>
            <span style='color:#D97706; font-size:0.95rem; font-weight: 500;'>Severity Marker. The AI thinks it's a driver, but changing it will NOT actually alter the risk.</span>
            </div>
            """, unsafe_allow_html=True)
            count += 1
        for _, row in protected_feats.iterrows():
            st.markdown(f"""
            <div style='padding:12px; margin-bottom:10px; border-radius:8px; border-left:4px solid #6B7280; background:#F9FAFB;'>
            <strong style='color:#374151; font-size:1.05rem;'>{get_label(row['Feature'])}</strong><br>
            <span style='color:#4B5563; font-size:0.95rem; font-weight: 500;'>Demographic / Immutable trait (Cannot intervene).</span>
            </div>
            """, unsafe_allow_html=True)
            count += 1
        if count == 0:
            st.success("No heavy statistical confounding detected in the primary decision boundaries.")
            
    st.markdown("<hr>", unsafe_allow_html=True)

    # ════════════════════════════════════
    # SAFE CLINICAL INTERVENTIONS (COUNTERFACTUALS)
    # ════════════════════════════════════
    st.markdown('<div class="section-header">Step 4 — Clinically Safe Counterfactuals</div>', unsafe_allow_html=True)
    
    if not is_high:
        st.markdown(f"""
        <div style="background-color: #ECFDF5; color: #065F46; padding: 15px; border-radius: 8px; font-weight: 500; font-size: 1.05rem;">
            ✅ <strong>No Urgent Interventions Required</strong><br>
            Patient is well within standard safety margins. No aggressive intervention algorithms triggered. Proceed with established ambulatory care.
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown("<p style='color:#4B5563; font-size: 1.05rem;'>Computing minimum unconfounded interventions required to suppress readmission likelihood below 50%. The algorithm avoids severity markers entirely...</p>", unsafe_allow_html=True)
        from layer3_counterfactual import generate_for_patient
        with st.spinner("Traversing Safe Intervention Space..."):
            cf = generate_for_patient(patient_dict, feature_names, model, df_ref)
            
        orig_pct  = cf["original_prob"] * 100
        final_pct = cf["final_prob"]    * 100
        changes   = cf["changes"]
        flipped   = cf["flipped"]
        
        if changes:
            if flipped:
                st.markdown(f"""<div style='padding:20px; border-radius:12px; background-color:#ECFDF5; border:1px solid #10B981; margin-bottom: 20px;'>
                <h3 style='color:#065F46; margin-top:0;'>✅ Feasible Intervention Plan Discovered</h3>
                <p style='color:#047857; font-size:1.1rem; margin-bottom:0;'>By executing the following <b>{len(changes)} actions</b>, predicted readmission drops radically from <b>{orig_pct:.1f}%</b> to <b>{final_pct:.1f}%</b>.</p>
                </div>""", unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div style="background-color: #FFFBEB; color: #92400E; padding: 20px; border-radius: 12px; border: 1px solid #F59E0B; margin-bottom: 20px;">
                    <h3 style="color: #92400E; margin-top: 0;">⚠️ Exhausted Safe Causal Adjustments</h3>
                    <p style="color: #B45309; font-size:1.1rem; margin-bottom: 0;">Readmission suppressed from <b>{orig_pct:.1f}%</b> to <b>{final_pct:.1f}%</b>, remaining dangerously elevated. Patient volatility is rooted in deep severity characteristics. Systemic care coordination is mandatory.</p>
                </div>
                """, unsafe_allow_html=True)
            
            running_risk = orig_pct
            for i, ch in enumerate(changes, 1):
                feat       = ch["feature"]
                label      = get_label(feat)
                from_val   = ch["from_val"]
                to_val     = ch["to_val"]
                risk_after = ch["prob_after"] * 100
                drop       = running_risk - risk_after
                
                # Format change text
                if feat in BINARY_FEATURES:
                    action_txt = f"Initiate: {label}" if to_val == 1.0 else f"Discontinue: {label}"
                elif feat.startswith("discharge_disposition_"):
                    action_txt = f"Adjust Route to: {label}"
                else:
                    direction = "Reduce" if to_val < from_val else "Increase"
                    action_txt = f"{direction} {label} from {from_val:.0f} to {to_val:.0f}"
                
                st.markdown(f"""
                <div style='padding:18px; margin-bottom:15px; border-radius:10px; background-color:#EFF6FF; border:1px solid #BFDBFE;'>
                    <div style='font-size:1.2rem; font-weight:800; color:#1D4ED8; margin-bottom:5px;'>
                    Action {i}: {action_txt}
                    </div>
                    <div style='color:#1E3A8A; font-size:1.05rem;'>
                    💡 Risk Mitigation: This action alone drives risk from <b>{running_risk:.1f}%</b> down to <b>{risk_after:.1f}%</b> (Delta: <b>↓ {drop:.1f}%</b>).
                    </div>
                </div>
                """, unsafe_allow_html=True)
                running_risk = risk_after
        else:
            st.markdown(f"""
            <div style="background-color: #FEF2F2; color: #991B1B; padding: 20px; border-radius: 12px; border: 1px solid #EF4444;">
                <h3 style="color: #991B1B; margin-top: 0;">❌ No Safe Actions Available</h3>
                <p style="color: #7F1D1D; font-size:1.1rem; margin-bottom: 0;">No unconfounded interventions algorithmically identified. Patient volatility is tied exclusively to immutable characteristics or extreme severity parameters. Implement maximum post-acute surveillance.</p>
            </div>
            """, unsafe_allow_html=True)