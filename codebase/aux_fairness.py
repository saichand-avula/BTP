#!/usr/bin/env python3
"""Held-out subgroup performance report (descriptive; not a fairness certificate)."""
from pathlib import Path
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score, recall_score

BASE = Path(__file__).parent
PRED = BASE / "models" / "heldout_predictions.csv"
OUT = BASE / "models" / "heldout_subgroup_metrics.csv"
OUT_DI = BASE / "models" / "disparate_impact.csv"
REFERENCE_LABELS = {"race": "AfricanAmerican", "gender": "Female", "age": "[0-10)"}


def run():
    df = pd.read_csv(PRED)
    # A 0.5 probability cutoff produces almost no alerts for this imbalanced
    # outcome.  Use a single held-out-set operating point that alerts the top
    # 20% by risk for the descriptive selection-rate (DI) comparison.
    alert_threshold = float(df["predicted_probability"].quantile(.80))
    df["predicted_label_top20pct"] = (df["predicted_probability"] >= alert_threshold).astype(int)
    rows = []
    for prefix in ("age_", "race_", "gender_"):
        columns = [c for c in df if c.startswith(prefix)]
        assigned = pd.Series(False, index=df.index)
        for col in columns:
            mask = df[col].eq(1)
            assigned |= mask
            if mask.sum() < 50:
                continue
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
            rows.append({"group": group_name, "value": REFERENCE_LABELS.get(group_name, "reference"),
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
        for _, row in sub.iterrows():
            rate = float(row.alert_rate_top20pct)
            ratio = rate / ref_rate if ref_rate else float("nan")
            di_rows.append({"group": group, "value": row.value, "reference": reference.value,
                            "operating_point": "top_20_percent_heldout_risk",
                            "risk_threshold": alert_threshold, "alert_rate": rate,
                            "disparate_impact_ratio": ratio,
                            "flag": "review" if ratio < .80 or ratio > 1.25 else "within_0.80_1.25"})
    pd.DataFrame(di_rows).to_csv(OUT_DI, index=False)
    print(f"Saved held-out subgroup metrics: {OUT}")
    return result


if __name__ == "__main__":
    run()
