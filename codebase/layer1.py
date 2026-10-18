#!/usr/bin/env python3
"""Reproducible patient-level readmission prediction experiment.

Modelling is kept separate from causal analysis. This uses grouped
train/validation/test splits so repeat encounters never cross a split.
"""
from __future__ import annotations

import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (average_precision_score, brier_score_loss,
                             precision_score, recall_score, roc_auc_score)
from sklearn.model_selection import GroupShuffleSplit, GroupKFold

BASE = Path(__file__).parent
DATA = BASE / "data_processed" / "diabetic_data_cleaned.csv"
MODELS = BASE / "models"
MODEL_PKL = MODELS / "layer1_model.pkl"
FEATURES_PKL = MODELS / "layer1_features.pkl"
METRICS_CSV = MODELS / "prediction_metrics.csv"
PREDICTIONS_CSV = MODELS / "heldout_predictions.csv"
IMPORTANCE_CSV = MODELS / "permutation_importance.csv"
SPLIT_JSON = MODELS / "split_summary.json"
RANDOM_STATE = 42
N_BOOTSTRAP = 1000


def _split(df: pd.DataFrame):
    """Create 60/20/20 patient-grouped splits (GroupShuffleSplit is not stratified)."""
    outer = GroupShuffleSplit(n_splits=1, test_size=0.20, random_state=RANDOM_STATE)
    dev_idx, test_idx = next(outer.split(df, groups=df["patient_id"].astype(str)))
    dev, test = df.iloc[dev_idx].copy(), df.iloc[test_idx].copy()
    inner = GroupShuffleSplit(n_splits=1, test_size=0.25, random_state=RANDOM_STATE)
    train_idx, val_idx = next(inner.split(dev, groups=dev["patient_id"].astype(str)))
    return dev.iloc[train_idx].copy(), dev.iloc[val_idx].copy(), test


def _calibration(y, p):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    logit = np.log(p / (1 - p)).reshape(-1, 1)
    lr = LogisticRegression(C=1e6, max_iter=2000).fit(logit, y)
    return float(lr.coef_[0, 0]), float(lr.intercept_[0])


def _cluster_bootstrap_auc_ci(y, p, patient_ids, n_bootstrap=N_BOOTSTRAP):
    """Percentile 95% CI using patient-level resampling of the held-out set."""
    y, p = np.asarray(y), np.asarray(p)
    groups = np.asarray(patient_ids)
    unique_groups, inverse = np.unique(groups, return_inverse=True)
    member_indices = [np.flatnonzero(inverse == group) for group in range(len(unique_groups))]
    rng = np.random.default_rng(RANDOM_STATE)
    aucs = []
    for _ in range(n_bootstrap):
        sampled_groups = rng.integers(0, len(unique_groups), len(unique_groups))
        indices = np.concatenate([member_indices[group] for group in sampled_groups])
        if np.unique(y[indices]).size == 2:
            aucs.append(roc_auc_score(y[indices], p[indices]))
    if len(aucs) < 100:
        raise RuntimeError("too few valid cluster-bootstrap resamples for ROC-AUC CI")
    return tuple(float(value) for value in np.percentile(aucs, [2.5, 97.5]))


def _metrics(name, y, p):
    pred = (p >= 0.5).astype(int)
    slope, intercept = _calibration(y, p)
    return {"model": name, "roc_auc": roc_auc_score(y, p),
            "pr_auc": average_precision_score(y, p), "brier": brier_score_loss(y, p),
            "precision_at_0_5": precision_score(y, pred, zero_division=0),
            "recall_at_0_5": recall_score(y, pred, zero_division=0),
            "calibration_slope": slope, "calibration_intercept": intercept,
            "roc_auc_ci_lower": None, "roc_auc_ci_upper": None}


def _top20_metrics(y, p):
    """Return sensitivity/specificity/PPV at the 80th-pctile score threshold."""
    threshold = float(np.percentile(p, 80))
    pred = (p >= threshold).astype(int)
    tp = int(((pred==1)&(y==1)).sum())
    fp = int(((pred==1)&(y==0)).sum())
    fn = int(((pred==0)&(y==1)).sum())
    tn = int(((pred==0)&(y==0)).sum())
    sens  = tp/(tp+fn) if (tp+fn)>0 else 0.0
    spec  = tn/(tn+fp) if (tn+fp)>0 else 0.0
    ppv   = tp/(tp+fp) if (tp+fp)>0 else 0.0
    return {"threshold_top20pct": round(threshold,4), "sensitivity_top20": round(sens,4),
            "specificity_top20": round(spec,4), "ppv_top20": round(ppv,4)}


def _safe_colnames(df: pd.DataFrame):
    """Return (renamed_df, col_map). XGBoost rejects [, ], < in column names."""
    mapping = {c: c.replace("[", "_").replace("]", "_").replace("<", "lt").replace(">", "gt")
               for c in df.columns}
    return df.rename(columns=mapping), mapping


def train():
    if not DATA.exists():
        from data_preprocessing import run
        run()
    MODELS.mkdir(exist_ok=True)
    df = pd.read_csv(DATA)
    if "patient_id" not in df:
        raise RuntimeError("Re-run data_preprocessing.py: patient_id is required for grouped evaluation.")
    train_df, val_df, test_df = _split(df)
    # Sanitize column names (XGBoost rejects [, ], <)
    df_safe, col_map = _safe_colnames(df)
    train_df_s, _  = _safe_colnames(train_df)
    val_df_s,   _  = _safe_colnames(val_df)
    test_df_s,  _  = _safe_colnames(test_df)
    feature_names_safe = [c for c in df_safe.columns if c not in {"Readmitted_30d", "patient_id"}]
    feature_names      = [c for c in df.columns       if c not in {"Readmitted_30d", "patient_id"}]
    x_train, y_train = train_df_s[feature_names_safe], train_df_s.Readmitted_30d
    x_val,   y_val   = val_df_s[feature_names_safe],   val_df_s.Readmitted_30d
    # ── Build candidate models ───────────────────────────────────────────────
    try:
        from xgboost import XGBClassifier
        _xgb = XGBClassifier(n_estimators=200, learning_rate=0.05, max_depth=3,
                             subsample=0.8, n_jobs=1, eval_metric="logloss", verbosity=0,
                             random_state=RANDOM_STATE, use_label_encoder=False)
    except ImportError:
        _xgb = None
    try:
        from lightgbm import LGBMClassifier
        _lgbm = LGBMClassifier(n_estimators=200, learning_rate=0.05, max_depth=3,
                               subsample=0.8, n_jobs=1, verbose=-1, random_state=RANDOM_STATE)
    except ImportError:
        _lgbm = None

    base_candidates = {
        "Logistic regression":  LogisticRegression(solver="liblinear", max_iter=500,
                                                    random_state=RANDOM_STATE),
        "Random forest":        RandomForestClassifier(n_estimators=120, min_samples_leaf=5,
                                                       n_jobs=1, random_state=RANDOM_STATE),
        "Gradient boosting":    GradientBoostingClassifier(n_estimators=200, learning_rate=0.05,
                                                           max_depth=3, subsample=0.8,
                                                           random_state=RANDOM_STATE),
    }
    if _xgb is not None:  base_candidates["XGBoost"]   = _xgb
    if _lgbm is not None: base_candidates["LightGBM"]  = _lgbm

    # ── 5-fold patient-grouped CV on dev set for model selection ─────────────
    # We use base (uncalibrated) models for selection — AUC-ROC is rank-based
    # and unaffected by calibration. We apply a single post-hoc calibration to
    # the SELECTED model only, before held-out evaluation.
    dev_s  = pd.concat([train_df_s, val_df_s]).copy()
    x_dev  = dev_s[feature_names_safe]
    y_dev  = dev_s["Readmitted_30d"]
    g_dev  = pd.concat([train_df["patient_id"], val_df["patient_id"]]).astype(str).reset_index(drop=True)

    cv_aucs = {name: [] for name in base_candidates}
    from sklearn.base import clone
    gkf = GroupKFold(n_splits=5)
    for tr_idx, va_idx in gkf.split(x_dev, y_dev, groups=g_dev):
        for name, m in base_candidates.items():
            m_clone = clone(m)
            m_clone.fit(x_dev.iloc[tr_idx], y_dev.iloc[tr_idx])
            p_va = m_clone.predict_proba(x_dev.iloc[va_idx])[:, 1]
            try:
                cv_aucs[name].append(roc_auc_score(y_dev.iloc[va_idx], p_va))
            except Exception:
                pass
    cv_mean = {n: float(np.mean(v)) for n, v in cv_aucs.items() if v}
    selected_name = max(cv_mean, key=cv_mean.get)
    print(f"CV AUCs (5-fold grouped): {cv_mean}")
    print(f"Selected: {selected_name}")

    # Retrain all base models on train split for validation-table metrics
    rows = []
    for name, m in base_candidates.items():
        m_clone = clone(m)
        m_clone.fit(x_train, y_train)
        p_val = m_clone.predict_proba(x_val)[:, 1]
        row = _metrics(name, y_val.to_numpy(), p_val)
        row["cv_auc_mean"] = round(cv_mean.get(name, float("nan")), 4)
        rows.append(row)
    validation = pd.DataFrame(rows).sort_values("cv_auc_mean", ascending=False)

    # Retrain the selected, unweighted candidate on the full development set.
    x_tv = pd.concat([x_train, x_val])
    y_tv = pd.concat([y_train, y_val])
    selected = clone(base_candidates[selected_name])
    selected.fit(x_tv, y_tv)

    x_test  = test_df_s[feature_names_safe]
    y_test  = test_df_s["Readmitted_30d"].to_numpy()
    test_prob = selected.predict_proba(x_test)[:, 1]
    final = _metrics(selected_name, y_test, test_prob)
    final["roc_auc_ci_lower"], final["roc_auc_ci_upper"] = _cluster_bootstrap_auc_ci(
        y_test, test_prob, test_df["patient_id"].to_numpy())
    final.update(_top20_metrics(y_test, test_prob))
    metrics = pd.concat([validation.assign(split="validation"),
                         pd.DataFrame([{**final, "split": "held_out_test"}])], ignore_index=True)
    metrics.to_csv(METRICS_CSV, index=False)

    # Persist the exact held-out covariate rows with their predictions.  This
    # makes downstream subgroup and recourse analyses provably held-out-only.
    heldout = test_df[["patient_id", "Readmitted_30d"] + feature_names].copy()
    heldout["predicted_probability"] = test_prob
    heldout["predicted_label_0_5"] = (test_prob >= 0.5).astype(int)
    heldout.to_csv(PREDICTIONS_CSV, index=False)

    perm = permutation_importance(selected, x_test, y_test, scoring="roc_auc",
                                  n_repeats=5, random_state=RANDOM_STATE, n_jobs=-1)
    # Save permutation importance with ORIGINAL (unsanitized) feature names for paper
    pd.DataFrame({"feature": feature_names,
                  "mean_auc_decrease": perm.importances_mean,
                  "std_auc_decrease": perm.importances_std}
                 ).sort_values("mean_auc_decrease", ascending=False
                 ).to_csv(IMPORTANCE_CSV, index=False)

    with open(MODEL_PKL, "wb") as f: pickle.dump(selected, f)
    # Pickle safe feature names so SHAP can call predict_proba on sanitized DFs
    with open(FEATURES_PKL, "wb") as f: pickle.dump(feature_names_safe, f)

    SPLIT_JSON.write_text(json.dumps({
        "train_encounters": len(train_df), "validation_encounters": len(val_df),
        "test_encounters":  len(test_df),
        "train_patients":   train_df.patient_id.nunique(),
        "validation_patients": val_df.patient_id.nunique(),
        "test_patients":    test_df.patient_id.nunique(),
        "selected_model":   selected_name,
        "col_map":          col_map,
    }, indent=2))
    print(f"Selected model: {selected_name}")
    print(pd.DataFrame([final]).round(4).to_string(index=False))
    return selected, feature_names


def load():
    with open(MODEL_PKL, "rb") as f: model = pickle.load(f)
    with open(FEATURES_PKL, "rb") as f: features = pickle.load(f)
    return model, features


if __name__ == "__main__":
    train()
