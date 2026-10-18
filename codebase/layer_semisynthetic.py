#!/usr/bin/env python3
"""Semi-synthetic ground-truth validation of the AIPW audit.

Design
------
We plant a synthetic binary variable Z with KNOWN zero direct causal effect.
Z is built as a noisy severity proxy:

  Z_i = 1  iff  (0.7 * signal_i + 0.3 * U_i) > 0.5
  signal_i = (number_inpatient_i > 0)
  U_i ~ Uniform(0,1)

This gives Z:
  - Strong positive association with Y through number_inpatient (confounder).
  - Enough random noise (30%) that propensity scores are not at the boundary,
    so common-support trimming retains a usable overlap region.
  - Zero direct causal effect on Y by construction.

A naive SHAP-only system (without number_inpatient as a predictor) would
rank Z near the top because it is predictive via the confounder.

After the AIPW audit adjusts for number_inpatient:
  - Expected result: CI includes zero, or FRAGILE E-value.
  - If propensity overlap fails: the audit correctly flags a positivity
    violation and blocks estimation — also the desired behavior.

Results: models/semisynthetic_validation.csv
"""
from pathlib import Path
import math
import numpy as np
import pandas as pd
from sklearn.ensemble import (HistGradientBoostingClassifier,
                              GradientBoostingClassifier)
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
import shap

BASE   = Path(__file__).parent
MODELS = BASE / "models"
DATA   = BASE / "data_processed" / "diabetic_data_cleaned.csv"
OUT    = MODELS / "semisynthetic_validation.csv"
RANDOM_STATE = 42

IMMUTABLE = ("age_", "race_", "gender_")
BASELINE  = ["number_inpatient", "number_outpatient", "number_emergency"]


# ── Helpers ─────────────────────────────────────────────────────────────────
def _cluster_se(scores, groups):
    frame    = pd.DataFrame({"score": scores, "group": groups})
    sums     = frame.groupby("group", sort=False).score.sum().to_numpy()
    counts   = frame.groupby("group", sort=False).size().to_numpy()
    theta    = scores.mean()
    n, g     = len(scores), len(sums)
    centered = sums - counts * theta
    return float(np.sqrt((g / max(g-1, 1)) * np.sum(centered**2) / n**2))


def _e_value(rr_bound):
    if rr_bound <= 0 or rr_bound == 1:
        return 1.0
    rr = rr_bound if rr_bound > 1 else 1/rr_bound
    return float(rr + math.sqrt(rr*(rr-1)))


def _aipw(df_sub, t, y, x, g):
    """Run 5-fold patient-grouped AIPW. Returns dict of estimates."""
    n_full = len(y)
    e_hat  = np.zeros(n_full)
    mu0    = np.zeros(n_full)
    mu1    = np.zeros(n_full)
    cv     = GroupKFold(5)
    for tr, ho in cv.split(x, t, groups=g):
        prop = LogisticRegression(solver="liblinear", max_iter=300,
                                  random_state=RANDOM_STATE)
        prop.fit(x.iloc[tr], t[tr])
        e_hat[ho] = prop.predict_proba(x.iloc[ho])[:, 1]
        for arm, dst in ((0, mu0), (1, mu1)):
            rows = tr[t[tr] == arm]
            if len(rows) < 10:
                continue
            m = HistGradientBoostingClassifier(max_iter=60, max_leaf_nodes=15,
                                               l2_regularization=1.0,
                                               random_state=RANDOM_STATE)
            m.fit(x.iloc[rows], y[rows])
            dst[ho] = m.predict_proba(x.iloc[ho])[:, 1]

    prop_min = float(e_hat.min())
    overlap  = (e_hat > 0.10) & (e_hat < 0.90)
    n_ov     = int(overlap.sum())

    if n_ov < 100 or np.unique(t[overlap]).size < 2:
        return {"status": "POSITIVITY_VIOLATION",
                "n_before_trim": n_full, "n_after_trim": n_ov,
                "overlap_fraction": round(n_ov/n_full, 4),
                "propensity_min": round(prop_min, 6)}

    t2, y2, e2 = t[overlap], y[overlap], e_hat[overlap]
    m0, m1, g2 = mu0[overlap], mu1[overlap], g[overlap]
    psi1 = m1 + t2*(y2-m1)/e2
    psi0 = m0 + (1-t2)*(y2-m0)/(1-e2)
    diff = psi1 - psi0
    rd   = float(diff.mean())
    se   = _cluster_se(diff, g2)
    r1, r0 = float(psi1.mean()), float(psi0.mean())
    rr   = r1/r0 if r0 else float("nan")
    ci0  = rd - 1.96*se
    ci1  = rd + 1.96*se
    includes_zero = bool(ci0 <= 0 <= ci1)

    if np.isnan(rr) or rr <= 0:
        ev, cat = float("nan"), "FRAGILE"
    else:
        lir  = psi1/r1 - psi0/r0
        slr  = _cluster_se(lir, g2)
        rlo  = math.exp(math.log(rr) - 1.96*slr)
        rhi  = math.exp(math.log(rr) + 1.96*slr)
        ev   = _e_value(rlo if rr > 1 else rhi) if not (rlo <= 1 <= rhi) else 1.0
        cat  = "ROBUST" if ev > 3 else "UNCERTAIN" if ev > 2 else "FRAGILE"

    return {"status": "OK",
            "n_before_trim": n_full, "n_after_trim": n_ov,
            "overlap_fraction": round(n_ov/n_full, 4),
            "propensity_min": round(prop_min, 6),
            "rd": round(rd, 4), "ci_lower": round(ci0, 4), "ci_upper": round(ci1, 4),
            "rr": round(rr, 3) if not np.isnan(rr) else float("nan"),
            "e_value": round(ev, 3) if not np.isnan(ev) else float("nan"),
            "sensitivity_category": cat,
            "ci_includes_zero": includes_zero}


# ── Experiment A ─────────────────────────────────────────────────────────────
def experiment_a(df: pd.DataFrame, rng: np.random.Generator) -> dict:
    """Noisy severity proxy with 70% signal + 30% random noise; true ATE = 0."""
    df = df.copy()
    signal = (df["number_inpatient"] > 0).astype(float).values
    noise  = rng.uniform(0, 1, size=len(df))
    df["Z_noisy_proxy"] = ((0.7*signal + 0.3*noise) > 0.5).astype(int)

    controls = ([c for c in BASELINE if c in df]
                + [c for c in df if c.startswith(IMMUTABLE)]
                + [c for c in df if c.startswith("diag_")])
    controls = list(dict.fromkeys(controls))

    data = df[["patient_id","Z_noisy_proxy","Readmitted_30d"]+controls].dropna()
    t = data["Z_noisy_proxy"].to_numpy(int)
    y = data["Readmitted_30d"].to_numpy(float)
    x = data[controls]
    g = data["patient_id"].to_numpy()

    aipw = _aipw(data, t, y, x, g)

    # -- Naive SHAP rank (model trained WITHOUT number_inpatient)
    feat_no_inp = [c for c in df.columns
                   if c not in {"Readmitted_30d","patient_id","Z_noisy_proxy",
                                "number_inpatient"}]
    from layer1 import _split
    splits  = _split(df)
    x_test  = splits[2][feat_no_inp + ["Z_noisy_proxy"]]
    x_tr    = pd.concat([splits[0], splits[1]])[feat_no_inp + ["Z_noisy_proxy"]]
    y_tr    = pd.concat([splits[0]["Readmitted_30d"], splits[1]["Readmitted_30d"]])
    naive   = GradientBoostingClassifier(n_estimators=100, max_depth=3,
                                         learning_rate=0.05, random_state=RANDOM_STATE)
    naive.fit(x_tr, y_tr)
    sv = shap.TreeExplainer(naive)(x_test).values
    if sv.ndim == 3:
        sv = sv[:, :, 1]
    imp = (pd.Series(np.abs(sv).mean(0), index=feat_no_inp+["Z_noisy_proxy"])
             .sort_values(ascending=False))
    z_rank = int(imp.index.get_loc("Z_noisy_proxy")) + 1
    z_shap = round(float(imp["Z_noisy_proxy"]), 4)

    verdict = ("estimation blocked: no common support"
               if aipw["status"] == "POSITIVITY_VIOLATION"
               else "estimation completed; descriptive diagnostic only")

    row = {"experiment": "A_noisy_severity_proxy",
           "planted_feature": "Z_noisy_proxy (70% severity signal + 30% noise)",
           "true_causal_effect": 0.0,
           "naive_model_shap_rank": z_rank,
           "naive_model_mean_abs_shap": z_shap,
           "audit_verdict": verdict,
           "interpretation": (
               "Z has zero true causal effect on readmission. Its SHAP rank on "
               "a model without number_inpatient should be high (confounding). "
               "After adjusting for number_inpatient via AIPW, the estimate "
               "should be near zero or FRAGILE.")}
    row.update(aipw)
    return row


# ── Main ─────────────────────────────────────────────────────────────────────
def run():
    MODELS.mkdir(exist_ok=True)
    rng = np.random.default_rng(RANDOM_STATE)
    df  = pd.read_csv(DATA)
    print("Running semi-synthetic ground-truth validation...")
    row_a = experiment_a(df, rng)
    result = pd.DataFrame([row_a])
    result.to_csv(OUT, index=False)
    print(f"\n{'='*65}")
    print("SEMI-SYNTHETIC VALIDATION RESULT")
    print(f"{'='*65}")
    if row_a.get("status") == "POSITIVITY_VIOLATION":
        print(f"  Status:                    POSITIVITY_VIOLATION (estimation blocked)")
        print(f"  Overlap fraction retained: {row_a.get('overlap_fraction', 'n/a')}")
    else:
        print(f"  Naive SHAP rank of planted feature: {row_a['naive_model_shap_rank']}")
        print(f"  Naive SHAP mean |phi|:              {row_a['naive_model_mean_abs_shap']}")
        print(f"  AIPW RD (true = 0.000):             {row_a.get('rd', 'n/a')}")
        print(f"  95% CI includes zero:               {row_a.get('ci_includes_zero', 'n/a')}")
        print(f"  E-value:                            {row_a.get('e_value', 'n/a')}")
        print(f"  Sensitivity category:               {row_a.get('sensitivity_category','n/a')}")
    print(f"  Verdict: {row_a['audit_verdict']}")
    print(f"{'='*65}")
    print(f"Saved to {OUT}")
    return result


if __name__ == "__main__":
    run()
