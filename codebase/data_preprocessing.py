#!/usr/bin/env python3
"""
Data Preprocessing for Diabetic Patient Readmission Dataset
Run this first before layer1.py if cleaned data doesn't exist.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

RAW_DATA  = Path(__file__).parent.parent / "diabetes+130-us+hospitals+for+years+1999-2008 (1)" / "diabetic_data.csv"
OUT_FILE  = Path(__file__).parent / "data_processed" / "diabetic_data_cleaned.csv"

DISCHARGE_EXCLUDE = [11, 13, 14, 19, 20, 21]   # death / hospice

MEDICATIONS = [
    'metformin', 'repaglinide', 'nateglinide', 'chlorpropamide', 'glimepiride',
    'acetohexamide', 'glipizide', 'glyburide', 'tolbutamide', 'pioglitazone',
    'rosiglitazone', 'acarbose', 'miglitol', 'troglitazone', 'tolazamide',
    'examide', 'citoglipton', 'insulin', 'glyburide-metformin', 'glipizide-metformin',
    'glimepiride-pioglitazone', 'metformin-rosiglitazone', 'metformin-pioglitazone'
]

DIAG_GROUPS = {
    'Circulatory':     list(range(390, 460)) + [785],
    'Respiratory':     list(range(460, 520)) + [786],
    'Diabetes':        [250],
    'Injury':          list(range(800, 1000)),
    'Musculoskeletal': list(range(710, 740)),
    'Genitourinary':   list(range(580, 630)) + [788],
    'Neoplasms':       list(range(140, 240)),
}


def group_icd9(code):
    """Map an ICD-9 code string to a broad diagnosis category."""
    if pd.isna(code) or str(code).strip() in ('?', ''):
        return 'Other'
    s = str(code).strip()
    try:
        num = int(float(s.replace('E', '').replace('V', '').split('.')[0]))
    except ValueError:
        return 'Other'
    for group, codes in DIAG_GROUPS.items():
        if num in codes:
            return group
    return 'Other'


def run():
    print("=" * 55)
    print("DATA PREPROCESSING")
    print("=" * 55)

    print(f"\nLoading raw data from:\n  {RAW_DATA}")
    df = pd.read_csv(RAW_DATA)
    print(f"  {len(df):,} records  |  {len(df.columns)} columns")

    # ── Remove deaths / hospice ──────────────────────────────────
    before = len(df)
    df = df[~df['discharge_disposition_id'].isin(DISCHARGE_EXCLUDE)].copy()
    print(f"\nRemoved {before - len(df):,} death/hospice records → {len(df):,} remain")

    # ── Target variable ──────────────────────────────────────────
    df['Readmitted_30d'] = (df['readmitted'] == '<30').astype(int)
    pos = df['Readmitted_30d'].sum()
    print(f"\nTarget  →  Readmitted <30d: {pos:,} ({pos/len(df)*100:.1f}%)  |  Not: {len(df)-pos:,}")

    # ── Drop useless columns ─────────────────────────────────────
    # Keep a non-model patient key solely for grouped evaluation.  It is removed
    # before modelling, but prevents encounters from the same patient appearing
    # in both development and evaluation sets.
    df['patient_id'] = df['patient_nbr'].astype(str)
    drop_cols = ['encounter_id', 'patient_nbr', 'readmitted', 'weight',
                 'payer_code', 'medical_specialty']
    df.drop(columns=[c for c in drop_cols if c in df.columns], inplace=True)
    # ── Cast ID columns to string so they get one-hot encoded below ───────────
    # Rename first to remove _id suffix so final cols are e.g. admission_type_1
    for id_col in ['admission_type_id', 'discharge_disposition_id', 'admission_source_id']:
        if id_col in df.columns:
            new_col = id_col.replace('_id', '')          # 'admission_type' etc.
            df[new_col] = df[id_col].astype(str)         # keep as string for dummies
            df.drop(columns=[id_col], inplace=True)
    # ── Replace '?' with NaN then fill with mode ─────────────────
    df.replace('?', np.nan, inplace=True)
    for col in df.select_dtypes(include='object').columns:
        df[col].fillna(df[col].mode()[0], inplace=True)

    # ── ICD-9 diagnosis grouping ─────────────────────────────────
    for col in ['diag_1', 'diag_2', 'diag_3']:
        if col in df.columns:
            df[f'{col}_group'] = df[col].apply(group_icd9)
    df.drop(columns=[c for c in ['diag_1', 'diag_2', 'diag_3'] if c in df.columns], inplace=True)

    # ── Medication → binary (prescribed / increased) ─────────────
    for med in MEDICATIONS:
        if med in df.columns:
            df[f'{med}_prescribed'] = (df[med] != 'No').astype(int)
            df[f'{med}_increased']  = (df[med] == 'Up').astype(int)
    df.drop(columns=[m for m in MEDICATIONS if m in df.columns], inplace=True)

    # ── One-hot encode remaining categoricals ────────────────────
    cat_cols = [c for c in df.select_dtypes(include='object').columns
                if c not in {'Readmitted_30d', 'patient_id'}]
    df = pd.get_dummies(df, columns=cat_cols, drop_first=True, dtype=int)

    # ── Move target to last column ───────────────────────────────
    target = df.pop('Readmitted_30d')
    df['Readmitted_30d'] = target

    OUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT_FILE, index=False)

    print(f"\nSaved cleaned data → {OUT_FILE}")
    print(f"  {len(df):,} records  |  {len(df.columns) - 1} features  |  1 target")
    print("\nPreprocessing complete. Run layer1.py next.")


if __name__ == "__main__":
    run()
