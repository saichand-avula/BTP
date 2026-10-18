#!/usr/bin/env python3
"""Held-out subgroup performance report (descriptive; not a fairness certificate)."""
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score, recall_score

BASE = Path(__file__).parent
PRED = BASE / "models" / "heldout_predictions.csv"
OUT = BASE / "models" / "heldout_subgroup_metrics.csv"
OUT_DI = BASE / "models" / "disparate_impact.csv"
REFERENCE_LABELS = {"race": "AfricanAmerican", "gender": "Female", "age": "[0-10)"}
RANDOM_STATE = 42
N_BOOTSTRAP = 2000


def _bootstrap_di_ci(df, group_col_mask, ref_mask, alert_col, n_boot=N_BOOTSTRAP,
                     seed=RANDOM_STATE):
    """Patient-level bootstrap 95% CI for a disparate-impact ratio.

    Resamples patient_ids with replacement to respect within-patient
    correlation, then computes the DI ratio on each bootstrap sample.
    Returns (ci_lower, ci_upper) or (NaN, NaN) if not computable.
    """
    rng = np.random.default_rng(seed)
    # Use patient_id if available, else row-level bootstrap
    if "patient_id" in df.columns:
        grp_pids = df.loc[group_col_mask, "patient_id"].unique()
        ref_pids = df.loc[ref_mask, "patient_id"].unique()
    else:
        grp_pids = df.loc[group_col_mask].index.to_numpy()
        ref_pids = df.loc[ref_mask].index.to_numpy()

    if len(grp_pids) < 10 or len(ref_pids) < 10:
        return float("nan"), float("nan")

    ratios = []
    for _ in range(n_boot):
        g_samp = rng.choice(grp_pids, len(grp_pids), replace=True)
        r_samp = rng.choice(ref_pids, len(ref_pids), replace=True)
        if "patient_id" in df.columns:
            g_rows = df.loc[df["patient_id"].isin(g_samp) & group_col_mask, alert_col]
            r_rows = df.loc[df["patient_id"].isin(r_samp) & ref_mask, alert_col]
        else:
            g_rows = df.loc[g_samp, alert_col]
            r_rows = df.loc[r_samp, alert_col]
        g_rate = g_rows.mean() if len(g_rows) > 0 else 0.0
        r_rate = r_rows.mean() if len(r_rows) > 0 else 0.0
        if r_rate > 0:
            ratios.append(g_rate / r_rate)
    if len(ratios) < 100:
        return float("nan"), float("nan")
    return tuple(float(v) for v in np.percentile(ratios, [2.5, 97.5]))


def run():
    df = pd.read_csv(PRED)
    # A 0.5 probability cutoff produces almost no alerts for this imbalanced
    # outcome.  Use a single held-out-set operating point that alerts the top
    # 20% by risk for the descriptive selection-rate (DI) comparison.
    alert_threshold = float(df["predicted_probability"].quantile(.80))
    df["predicted_label_top20pct"] = (df["predicted_probability"] >= alert_threshold).astype(int)
    rows = []
    # Build subgroup masks for later bootstrap use
    group_masks = {}  # (prefix, value) -> boolean mask
    for prefix in ("age_", "race_", "gender_"):
        columns = [c for c in df if c.startswith(prefix)]
        assigned = pd.Series(False, index=df.index)
        for col in columns:
            mask = df[col].eq(1)
            assigned |= mask
            if mask.sum() < 50:
                continue
            group_masks[(prefix[:-1], col[len(prefix):])] = mask
            y, p = df.loc[mask, "Readmitted_30d"], df.loc[mask, "predicted_probability"]
            rows.append({"group": prefix[:-1], "value": col[len(prefix):], "n": int(mask.sum()),
                         "prevalence": y.mean(), "roc_auc": roc_auc_score(y, p) if y.nunique() == 2 else None,
                         "pr_auc": average_precision_score(y, p) if y.nunique() == 2 else None,
                         "sensitivity_at_0_5": recall_score(y, df.loc[mask, "predicted_label_0_5"], zero_division=0),
                         "positive_prediction_rate_at_0_5": df.loc[mask, "predicted_label_0_5"].mean(),
                         "alert_rate_top20pct": df.loc[mask, "predicted_label_top20pct"].mean(),
                         "mean_predicted_risk": p.mean()})
        unknown = ~assigned
        if unknown.sum() >= 50:
            y, p = df.loc[unknown, "Readmitted_30d"], df.loc[unknown, "predicted_probability"]
            group_name = prefix[:-1]
            ref_label = REFERENCE_LABELS.get(group_name, "reference")
            group_masks[(group_name, ref_label)] = unknown
            rows.append({"group": group_name, "value": ref_label,
                         "n": int(unknown.sum()),
                         "prevalence": y.mean(), "roc_auc": roc_auc_score(y, p) if y.nunique() == 2 else None,
                         "pr_auc": average_precision_score(y, p) if y.nunique() == 2 else None,
                         "sensitivity_at_0_5": recall_score(y, df.loc[unknown, "predicted_label_0_5"], zero_division=0),
                         "positive_prediction_rate_at_0_5": df.loc[unknown, "predicted_label_0_5"].mean(),
                         "alert_rate_top20pct": df.loc[unknown, "predicted_label_top20pct"].mean(),
                         "mean_predicted_risk": p.mean()})
    result = pd.DataFrame(rows)
    result.to_csv(OUT, index=False)
    di_rows = []
    for group, sub in result.groupby("group"):
        reference = sub.loc[sub.n.idxmax()]
        ref_rate = float(reference.alert_rate_top20pct)
        ref_mask = group_masks.get((group, reference.value))
        for _, row in sub.iterrows():
            rate = float(row.alert_rate_top20pct)
            ratio = rate / ref_rate if ref_rate else float("nan")
            # Bootstrap CI for the DI ratio
            grp_mask = group_masks.get((group, row.value))
            if grp_mask is not None and ref_mask is not None and row.value != reference.value:
                ci_lo, ci_hi = _bootstrap_di_ci(df, grp_mask, ref_mask,
                                                "predicted_label_top20pct")
            else:
                ci_lo, ci_hi = float("nan"), float("nan")
            di_rows.append({"group": group, "value": row.value, "reference": reference.value,
                            "operating_point": "top_20_percent_heldout_risk",
                            "risk_threshold": alert_threshold, "alert_rate": rate,
                            "disparate_impact_ratio": ratio,
                            "di_ci_lower": round(ci_lo, 4) if not np.isnan(ci_lo) else float("nan"),
                            "di_ci_upper": round(ci_hi, 4) if not np.isnan(ci_hi) else float("nan"),
                            "flag": "review" if ratio < .80 or ratio > 1.25 else "within_0.80_1.25"})
    pd.DataFrame(di_rows).to_csv(OUT_DI, index=False)
    print(f"Saved held-out subgroup metrics: {OUT}")
    print(f"Saved disparate impact with bootstrap CIs: {OUT_DI}")
    return result


if __name__ == "__main__":
    run()
