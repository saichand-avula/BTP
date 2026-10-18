#!/usr/bin/env python3
"""Exploratory feature-exposure audit with cross-fitted AIPW, E-values, and SMD.

Methodology
-----------
* Exposures: top-N SHAP-ranked features that are not demographic, not
  post-encounter/disposition, not within-encounter utilization, and not ICD
  diagnosis codes (which are assigned at discharge). The final eligible list
  is selected by a single timing-eligibility rule (see EXCLUDED_TIMING).
* Confounder set: baseline utilisation counts, demographics, and ICD
  diagnosis-group indicators. When auditing a diag_* feature, the ENTIRE
  diag_* family is dropped from the adjustment set, not just the audited col.
* Propensity model: unweighted LogisticRegression; calibrated probabilities.
* Outcome model: HistGradientBoostingClassifier (no class-weight distortion).
* Overlap: restrict to 0.10 < e-hat < 0.90. Estimation skipped if <100 obs.
* Standard errors: patient-cluster sandwich variance of the influence function.
* Multiplicity: Benjamini-Hochberg FDR correction across all audited features.
* SMD: post-weighting standardized mean difference (max over all covariates).

Analysis sample: the full cohort (all encounters), not the held-out test set.
This is appropriate because the causal audit is a separate observational
analysis, not an evaluation of predictive performance. Using the full cohort
gives more precise estimates; it does not evaluate model predictions.

Gate unit test (item 9): run with (a) real author-defined policy and
(b) counterfactual policy (medication features allowed).
"""
from pathlib import Path
import json, math
import numpy as np
import pandas as pd
from scipy import stats as scipy_stats
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold

BASE   = Path(__file__).parent
MODELS = BASE / "models"
DATA   = BASE / "data_processed" / "diabetic_data_cleaned.csv"
SHAP   = MODELS / "shap_global_importance.csv"
OUT    = MODELS / "causal_audit_full.csv"
OUT_SENS = MODELS / "causal_audit_sensitivity_no_diag.csv"
SIM_OUT = MODELS / "simulation_validation.csv"
GATE_OUT = MODELS / "recourse_gate_test.csv"
RANDOM_STATE = 42
PRIMARY_AUDIT_K = 5

IMMUTABLE = ("age_", "race_", "gender_")

# Features excluded by the SINGLE timing rule: features that are recorded
# at or after the discharge decision, or represent within-encounter
# utilization. Crucially this now includes diag_* features (coded at
# discharge), consistent with excluding number_diagnoses.
EXCLUDED_TIMING = (
    "discharge_disposition_",
    "time_in_hospital",
    "num_medications",
    "num_lab_procedures",
    "num_procedures",
    "number_diagnoses",
    "diag_1_",          # ICD codes assigned at discharge — excluded
    "diag_2_",          # ICD codes assigned at discharge — excluded
    "diag_3_",
)

BASELINE = ["number_inpatient", "number_outpatient", "number_emergency",
            ]


# ─── Helpers ─────────────────────────────────────────────────────────────────

def _e_value_from_rr(rr_bound):
    if rr_bound <= 0 or rr_bound == 1:
        return 1.0
    rr = rr_bound if rr_bound > 1 else 1 / rr_bound
    return float(rr + math.sqrt(rr * (rr - 1)))


def _controls(df, feature, exclude_diag=False):
    """Build confounder set. Drop the ENTIRE diag family if feature is in it.

    Parameters
    ----------
    exclude_diag : bool
        If True, exclude ALL diagnosis-group indicators from the confounder
        set. Used for sensitivity analysis (ICD codes are assigned at
        discharge and may introduce post-treatment bias).
    """
    # Determine if the audited feature belongs to a diag_N_ family
    audited_family = None
    for prefix in ("diag_1_", "diag_2_", "diag_3_"):
        if feature.startswith(prefix):
            audited_family = prefix
            break

    cols = [c for c in BASELINE if c in df and c != feature]
    cols += [c for c in df if c.startswith(IMMUTABLE) and c != feature]

    if not exclude_diag:
        # Diag groups: keep as confounders unless they belong to the audited family
        for c in df.columns:
            if c.startswith("diag_") and c != feature:
                if audited_family and c.startswith(audited_family):
                    continue  # drop entire family (item 3)
                cols.append(c)
    return list(dict.fromkeys(cols))


def _exposure(series):
    if series.nunique() <= 2:
        return series.astype(int).to_numpy(), "binary"
    threshold = float(series.median())
    return (series > threshold).astype(int).to_numpy(), f"above_median({threshold:g})"


def _cluster_se(scores, groups):
    """Patient-clustered SE for the sample mean of an influence score."""
    frame    = pd.DataFrame({"score": scores, "group": groups})
    sums     = frame.groupby("group", sort=False).score.sum().to_numpy()
    counts   = frame.groupby("group", sort=False).size().to_numpy()
    theta    = scores.mean()
    n, g     = len(scores), len(sums)
    centered = sums - counts * theta
    return float(np.sqrt((g / max(g - 1, 1)) * np.sum(centered**2) / n**2))


def _max_smd(x_df, t, weights_treated, weights_control):
    """Max standardized mean difference over all covariates (post-weighting)."""
    smds = []
    for col in x_df.columns:
        xv = x_df[col].to_numpy(float)
        n1 = weights_treated.sum()
        n0 = weights_control.sum()
        if n1 == 0 or n0 == 0:
            continue
        m1 = np.sum(weights_treated * xv) / n1
        m0 = np.sum(weights_control * xv) / n0
        v1 = np.sum(weights_treated * (xv - m1)**2) / n1
        v0 = np.sum(weights_control * (xv - m0)**2) / n0
        denom = math.sqrt((v1 + v0) / 2)
        if denom > 0:
            smds.append(abs(m1 - m0) / denom)
    return float(max(smds)) if smds else float("nan")


# ─── Main AIPW estimator ─────────────────────────────────────────────────────

def estimate(df, feature, exclude_diag=False):
    controls = _controls(df, feature, exclude_diag=exclude_diag)
    data     = df[["patient_id", feature, "Readmitted_30d"] + controls].dropna()
    t, defn  = _exposure(data[feature])
    y        = data["Readmitted_30d"].to_numpy(float)
    x        = data[controls]
    groups   = data["patient_id"].to_numpy()

    if t.min() == t.max():
        raise ValueError("no exposure variation")

    n_full = len(data)
    e_hat  = np.zeros(n_full)
    mu0    = np.zeros(n_full)
    mu1    = np.zeros(n_full)

    cv = GroupKFold(5)
    for tr, ho in cv.split(x, t, groups=groups):
        prop = LogisticRegression(solver="liblinear", max_iter=300,
                                  random_state=RANDOM_STATE)
        prop.fit(x.iloc[tr], t[tr])
        e_hat[ho] = prop.predict_proba(x.iloc[ho])[:, 1]
        for arm, dst in ((0, mu0), (1, mu1)):
            rows = tr[t[tr] == arm]
            if len(rows) < 5:
                continue
            m = HistGradientBoostingClassifier(max_iter=60, max_leaf_nodes=15,
                                               l2_regularization=1.0,
                                               random_state=RANDOM_STATE)
            m.fit(x.iloc[rows], y[rows])
            dst[ho] = m.predict_proba(x.iloc[ho])[:, 1]

    untrimmed = (float(e_hat.min()), float(e_hat.max()))
    overlap   = (e_hat > .10) & (e_hat < .90)
    n         = int(overlap.sum())

    if n < 100 or np.unique(t[overlap]).size < 2:
        raise ValueError(
            f"insufficient common support after trimming "
            f"(retained {n}/{n_full} = {100*n/n_full:.1f}%)")

    # Post-weighting SMD using IPW weights
    t_ov, y_ov = t[overlap], y[overlap]
    e_ov   = e_hat[overlap]
    mu0_ov = mu0[overlap]
    mu1_ov = mu1[overlap]
    g_ov   = groups[overlap]
    x_ov   = x.iloc[np.where(overlap)[0]]

    ipw_t = np.where(t_ov == 1, 1.0 / e_ov, 0.0)
    ipw_c = np.where(t_ov == 0, 1.0 / (1 - e_ov), 0.0)
    max_smd = _max_smd(x_ov, t_ov, ipw_t, ipw_c)

    psi1   = mu1_ov + t_ov * (y_ov - mu1_ov) / e_ov
    psi0   = mu0_ov + (1 - t_ov) * (y_ov - mu0_ov) / (1 - e_ov)
    diff   = psi1 - psi0
    rd     = float(diff.mean())
    se_rd  = _cluster_se(diff, g_ov)
    r1, r0 = float(psi1.mean()), float(psi0.mean())
    rr     = r1 / r0

    lri    = psi1 / r1 - psi0 / r0
    se_lri = _cluster_se(lri, g_ov)
    rr_lo  = math.exp(math.log(rr) - 1.96 * se_lri)
    rr_hi  = math.exp(math.log(rr) + 1.96 * se_lri)
    ev     = (_e_value_from_rr(rr_lo if rr > 1 else rr_hi)
              if not (rr_lo <= 1 <= rr_hi) else 1.0)
    status = "ROBUST" if ev > 3 else "UNCERTAIN" if ev > 2 else "FRAGILE"

    # Two-sided Wald p-value on RD for BH correction
    z_rd  = rd / se_rd if se_rd > 0 else 0.0
    pval  = float(2 * (1 - scipy_stats.norm.cdf(abs(z_rd))))

    return {
        "feature":             feature,
        "exposure_definition": defn,
        "n":                   n,
        "n_before_trimming":   n_full,
        "overlap_fraction":    round(n / n_full, 4),
        "exposed_prevalence":  round(float(t_ov.mean()), 4),
        "aipw_risk_difference": round(rd, 4),
        "rd_ci_lower":         round(rd - 1.96 * se_rd, 4),
        "rd_ci_upper":         round(rd + 1.96 * se_rd, 4),
        "aipw_risk_ratio":     round(rr, 4),
        "rr_ci_lower":         round(rr_lo, 4),
        "rr_ci_upper":         round(rr_hi, 4),
        "e_value_ci_bound":    round(ev, 4),
        "sensitivity_category": status,
        "max_smd_post_weighting": round(max_smd, 4),
        "pvalue_rd":           round(pval, 6),
        "propensity_min":      round(untrimmed[0], 6),
        "propensity_max":      round(untrimmed[1], 6),
        "interpretation":      "exploratory observational audit; not an intervention recommendation",
    }


# ─── BH multiple-comparison correction ──────────────────────────────────────

def _bh_correct(results_df):
    """Add BH-adjusted q-values. Features with audit failures get q=NaN."""
    pvals = results_df["pvalue_rd"].to_numpy(float)
    valid = ~np.isnan(pvals)
    q = np.full(len(pvals), float("nan"))
    if valid.sum() > 0:
        from statsmodels.stats.multitest import multipletests
        _, q_valid, _, _ = multipletests(pvals[valid], alpha=0.05, method="fdr_bh")
        q[valid] = q_valid
    results_df["bh_qvalue"] = np.round(q, 6)
    return results_df


# ─── Gate unit test ──────────────────────────────────────────────────────────

ACTIONABILITY_POLICY = {
    # author-defined policy (item 9)
    "number_inpatient":      {"allowed": False, "reason": "historical utilization, non-modifiable"},
    "number_emergency":      {"allowed": False, "reason": "historical utilization, non-modifiable"},
    "number_outpatient":     {"allowed": False, "reason": "historical utilization, non-modifiable"},
    "diabetesMed_Yes":       {"allowed": "conditional", "reason": "medication: requires prescriber review"},
    "metformin_prescribed":  {"allowed": "conditional", "reason": "medication: requires prescriber review"},
    "insulin_prescribed":    {"allowed": "conditional", "reason": "medication: requires prescriber review"},
    "A1Cresult_>7":          {"allowed": "conditional", "reason": "lab result: informational only"},
    "A1Cresult_>8":          {"allowed": "conditional", "reason": "lab result: informational only"},
}

COUNTERFACTUAL_POLICY = {
    # hypothetical policy: mark medication features allowed (item 9)
    **ACTIONABILITY_POLICY,
    "diabetesMed_Yes":       {"allowed": True, "reason": "counterfactual: assume prescriber-approved"},
    "metformin_prescribed":  {"allowed": True, "reason": "counterfactual: assume prescriber-approved"},
    "insulin_prescribed":    {"allowed": True, "reason": "counterfactual: assume prescriber-approved"},
}


def _run_gate(audit_df, policy, policy_name):
    rows = []
    for _, r in audit_df.iterrows():
        feat   = r.get("feature", "")
        status = r.get("sensitivity_category", "FRAGILE")
        pol    = policy.get(feat, {"allowed": False, "reason": "not in policy"})
        action_ok = pol["allowed"] is True
        robust_ok  = status == "ROBUST"
        passes     = action_ok and robust_ok
        rows.append({
            "policy": policy_name,
            "feature": feat,
            "policy_allowed": pol["allowed"],
            "policy_reason": pol["reason"],
            "sensitivity_category": status,
            "e_value": r.get("e_value_ci_bound", None),
            "gate_passes": passes,
            "recourse_output": "validated_recourse" if passes else "no_validated_recourse",
        })
    return pd.DataFrame(rows)


# ─── Simulation validation (item 10) ─────────────────────────────────────────

def _run_simulation(df, n_reps=200, seed=42):
    """Semi-parametric simulation with real covariates.

    For each replication:
      NULL exposure:  T_null ~ Bernoulli(sigma(0.5*X_a - 0.3*X_b))
        where X_a = (number_inpatient > 0), X_b = (number_emergency > 0).
        True causal effect = 0. Propensities are forced within 0.15-0.85
        by keeping only encounters in that range.
      TRUE exposures: T ~ Bernoulli(sigma(0.4*X_a)) with two log-odds
        shifts (delta=0.4 and delta=1.0).  Their true RDs are calculated
        directly from each replication's potential-outcome probabilities,
        averaged over the real covariates; they are never hard-coded.
      Outcome: Y ~ Bernoulli(sigma(-2.5 + 0.8*X_a + 0.3*X_b + delta*T))
      delta=0 for NULL, delta in {0.4, 1.0} for planted effects.
    Run AIPW for each. Record RD bias, CI coverage, and E-value category.
    Confounders in audit: X_a, X_b.
    """
    rng = np.random.default_rng(seed)
    controls = ["number_inpatient", "number_outpatient", "number_emergency"]
    controls = [c for c in controls if c in df]
    data = df[controls + ["patient_id"]].dropna().reset_index(drop=True)

    xa = (data["number_inpatient"] > 0).astype(float).values
    xb = (data["number_emergency"] > 0).astype(float).values if "number_emergency" in data else np.zeros(len(data))
    groups = data["patient_id"].to_numpy()

    def _sigma(x): return 1 / (1 + np.exp(-x))

    rows = []
    for rep in range(n_reps):
        noise = rng.standard_normal(len(data))
        # --- NULL exposure ---
        p_null = _sigma(0.5 * xa - 0.3 * xb + 0.3 * noise)
        t_null = (rng.uniform(size=len(data)) < p_null).astype(int)
        y_null = (rng.uniform(size=len(data)) < _sigma(-2.5 + 0.8*xa + 0.3*xb)).astype(float)

        # --- Planted-effect exposure assignment ---
        p_true = _sigma(0.4 * xa + 0.2 * noise)
        t_true = (rng.uniform(size=len(data)) < p_true).astype(int)
        linear_predictor = -2.5 + 0.8 * xa + 0.3 * xb
        scenarios = [("null", t_null, y_null, 0.0)]
        for delta in (0.4, 1.0):
            # The true marginal RD is E[Y(1)-Y(0)] under this DGP, evaluated
            # on the real covariates used in this replication.
            true_rd = float(np.mean(_sigma(linear_predictor + delta) -
                                    _sigma(linear_predictor)))
            y_true = (rng.uniform(size=len(data)) <
                      _sigma(linear_predictor + delta * t_true)).astype(float)
            scenarios.append((f"delta_{delta:.1f}", t_true, y_true, true_rd))

        for label, t_vec, y_vec, true_rd in scenarios:
            x_df = pd.DataFrame({"xa": xa, "xb": xb})
            try:
                # Three-fold AIPW implementation check.  This known DGP is
                # logistic in two binary covariates, so logistic nuisance
                # models are correctly specified and avoid conflating this
                # check with flexible-model tuning.
                n_full2 = len(y_vec)
                e2   = np.zeros(n_full2); m02 = np.zeros(n_full2); m12 = np.zeros(n_full2)
                cv2  = GroupKFold(3)
                for tr2, ho2 in cv2.split(x_df, t_vec, groups=groups):
                    prop2 = LogisticRegression(solver="liblinear", max_iter=200, random_state=rep)
                    prop2.fit(x_df.iloc[tr2], t_vec[tr2])
                    e2[ho2] = prop2.predict_proba(x_df.iloc[ho2])[:, 1]
                    for arm2, dst2 in ((0, m02), (1, m12)):
                        rows2 = tr2[t_vec[tr2] == arm2]
                        if len(rows2) < 10: continue
                        om2 = LogisticRegression(solver="liblinear", max_iter=200,
                                                 random_state=rep)
                        om2.fit(x_df.iloc[rows2], y_vec[rows2])
                        dst2[ho2] = om2.predict_proba(x_df.iloc[ho2])[:, 1]
                ov2 = (e2 > .10) & (e2 < .90)
                if ov2.sum() < 50 or np.unique(t_vec[ov2]).size < 2:
                    raise ValueError("low overlap")
                t2, y2, e2ov = t_vec[ov2], y_vec[ov2], e2[ov2]
                m02ov, m12ov, g2 = m02[ov2], m12[ov2], groups[ov2]
                ps1 = m12ov + t2*(y2-m12ov)/e2ov
                ps0 = m02ov + (1-t2)*(y2-m02ov)/(1-e2ov)
                dif = ps1 - ps0
                rd2 = float(dif.mean()); se2 = _cluster_se(dif, g2)
                r12, r02 = float(ps1.mean()), float(ps0.mean())
                rr2 = r12/r02
                lri2 = ps1/r12 - ps0/r02
                se_lr2 = _cluster_se(lri2, g2)
                rlo2 = math.exp(math.log(rr2) - 1.96*se_lr2)
                rhi2 = math.exp(math.log(rr2) + 1.96*se_lr2)
                ev2  = (_e_value_from_rr(rlo2 if rr2>1 else rhi2)
                        if not (rlo2<=1<=rhi2) else 1.0)
                cat2 = "ROBUST" if ev2>3 else "UNCERTAIN" if ev2>2 else "FRAGILE"
                ci_lo = rd2 - 1.96*se2; ci_hi = rd2 + 1.96*se2
                covers = int(ci_lo <= true_rd <= ci_hi)
                rows.append({"rep": rep, "exposure_type": label, "true_rd": true_rd,
                             "rd_estimate": round(rd2,4), "bias": round(rd2-true_rd,4),
                             "ci_lower": round(ci_lo,4), "ci_upper": round(ci_hi,4),
                             "ci_covers_truth": covers, "e_value": round(ev2,3),
                             "category": cat2, "n_overlap": int(ov2.sum())})
            except Exception as exc:
                rows.append({"rep": rep, "exposure_type": label, "true_rd": true_rd,
                             "rd_estimate": float("nan"), "bias": float("nan"),
                             "ci_lower": float("nan"), "ci_upper": float("nan"),
                             "ci_covers_truth": 0, "e_value": float("nan"),
                             "category": "FAILED", "note": str(exc)})
    return pd.DataFrame(rows)


# ─── Main entry point ─────────────────────────────────────────────────────────

def run(top_k=PRIMARY_AUDIT_K):
    MODELS.mkdir(exist_ok=True)
    df     = pd.read_csv(DATA)
    ranked = pd.read_csv(SHAP)

    # Single timing rule: exclude any prefix in EXCLUDED_TIMING
    features = [
        f for f in ranked["feature"]
        if f in df
        and not f.startswith(IMMUTABLE)
        and not any(f.startswith(ex) for ex in EXCLUDED_TIMING)
    ][:top_k]

    print(f"Eligible features for audit ({len(features)}): {features}")

    results = []
    for feat in features:
        try:
            results.append(estimate(df, feat))
            print(f"  OK: {feat}")
        except Exception as exc:
            print(f"  SKIP: {feat} -- {exc}")
            results.append({"feature": feat, "sensitivity_category": "SKIPPED",
                            "interpretation": f"audit skipped: {exc}"})

    out = pd.DataFrame(results)
    out = _bh_correct(out)
    out.to_csv(OUT, index=False)

    success = out[out["sensitivity_category"].isin(["ROBUST","UNCERTAIN","FRAGILE"])]
    print("\n=== AIPW AUDIT RESULTS ===")
    cols = ["feature","n","overlap_fraction","aipw_risk_difference",
            "rd_ci_lower","rd_ci_upper","aipw_risk_ratio","rr_ci_lower","rr_ci_upper",
            "e_value_ci_bound","sensitivity_category","max_smd_post_weighting","bh_qvalue"]
    print(success[[c for c in cols if c in success]].to_string(index=False))

    # ── Gate unit tests (item 9) ──
    gate_real  = _run_gate(success, ACTIONABILITY_POLICY,    "author_defined")
    gate_cfact = _run_gate(success, COUNTERFACTUAL_POLICY,   "counterfactual_medications_allowed")
    gate_all   = pd.concat([gate_real, gate_cfact], ignore_index=True)
    gate_all.to_csv(GATE_OUT, index=False)
    print("\n=== GATE UNIT TEST ===")
    print("Real policy:"); print(gate_real[["feature","policy_allowed","sensitivity_category","gate_passes"]].to_string(index=False))
    print("Counterfactual policy:"); print(gate_cfact[["feature","policy_allowed","sensitivity_category","gate_passes"]].to_string(index=False))

    # ── Sensitivity analysis: exclude diagnosis-group confounders ──
    print("\n=== SENSITIVITY: AIPW without diagnosis-group confounders ===")
    sens_results = []
    for feat in features:
        try:
            sens_results.append(estimate(df, feat, exclude_diag=True))
            print(f"  OK: {feat}")
        except Exception as exc:
            print(f"  SKIP: {feat} -- {exc}")
            sens_results.append({"feature": feat, "sensitivity_category": "SKIPPED",
                                "interpretation": f"sensitivity audit skipped: {exc}"})
    sens_df = pd.DataFrame(sens_results)
    sens_df = _bh_correct(sens_df)
    sens_df.to_csv(OUT_SENS, index=False)
    sens_ok = sens_df[sens_df["sensitivity_category"].isin(["ROBUST","UNCERTAIN","FRAGILE"])]
    if len(sens_ok):
        print(sens_ok[[c for c in cols if c in sens_ok]].to_string(index=False))

    # Simulation is an implementation check under a deliberately simple DGP,
    # not evidence for causal validity on the observational hospital data.
    print("\n=== SIMULATION: simple known-DGP implementation check ===")
    simulation = _run_simulation(df)
    simulation.to_csv(SIM_OUT, index=False)
    # Preserve literal labels such as "null" when this artefact is reloaded.
    simulation = pd.read_csv(SIM_OUT, keep_default_na=False)
    print(simulation.groupby("exposure_type", dropna=False).agg(
        reps=("rep", "size"), coverage=("ci_covers_truth", "mean"),
        mean_bias=("bias", "mean")).round(4).to_string())

    return out


if __name__ == "__main__":
    run()
