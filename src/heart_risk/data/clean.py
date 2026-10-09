"""
src/heart_risk/data/clean.py
─────────────────────────────────────────────────────────────────────────────
Phase 3: Clean the raw data based on configured rules.

Steps:
1. Drop exact duplicates (keeping the lowest row_id).
2. Convert explicit missing tokens to NaN.
3. Convert fake zeros to NaN (e.g. chol=0, trestbps=0) as per config.
4. Binarize the target column to 0 vs >0.
5. Save the cleaned data to `patients_clean` table.
"""

import pandas as pd
import numpy as np
import yaml
from pathlib import Path

from heart_risk.config import Settings, get_settings, PROJECT_ROOT
from heart_risk.db.engine import build_engine
from heart_risk.db.repository import read_raw, write_clean
from heart_risk.data.profile import parse_in_memory

def run_cleaning(settings: Settings) -> pd.DataFrame:
    """Executes the cleaning pipeline and returns the cleaned dataframe."""
    engine = build_engine(settings)
    df_raw = read_raw(engine)
    
    # 1. Drop Exact Duplicates
    print(f"[INFO] Initial rows: {len(df_raw)}")
    df_raw = df_raw.sort_values("row_id")
    check_cols = [c for c in settings.expected_raw_columns if c != "id"]
    
    dupes_mask = df_raw.duplicated(subset=check_cols, keep="first")
    dropped_rows = df_raw[dupes_mask]
    
    if not dropped_rows.empty:
        print(f"[INFO] Dropping {len(dropped_rows)} exact duplicates. IDs: {dropped_rows['id'].tolist()}")
    
    df_dedup = df_raw[~dupes_mask].copy()
    print(f"[INFO] Rows after dedup: {len(df_dedup)}")
    
    # 2. Parse basic missing strings to NaN
    df_clean = parse_in_memory(df_dedup, settings.expected_raw_columns)
    
    # 3. Apply Zero -> NaN rules from config
    features_yaml = PROJECT_ROOT / "configs" / "features.yaml"
    with open(features_yaml, "r") as f:
        feat_config = yaml.safe_load(f)
        
    for col, rules in feat_config.get("features", {}).items():
        if col in df_clean.columns and rules.get("treat_0_as_missing", False):
            zeros_count = (df_clean[col] == 0).sum()
            if zeros_count > 0:
                print(f"[INFO] Converting {zeros_count} zeros to NaN in column '{col}'")
            df_clean[col] = df_clean[col].replace(0, np.nan)
            
    # 4. Re-check duplicates (just logging, not dropping automatically if there are new ones)
    dupes_mask_after = df_clean.duplicated(subset=check_cols, keep=False)
    new_dupes_count = dupes_mask_after.sum()
    if new_dupes_count > 0:
        print(f"[WARNING] Found {new_dupes_count} duplicate rows AFTER NaN conversion.")
        
    # 5. Target binarization
    print("[INFO] Binarizing target 'num' -> 'target'")
    df_clean['target'] = (pd.to_numeric(df_clean['num'], errors='coerce') > 0).astype(int)
    
    # 6. Save to DB
    print("[INFO] Saving to `patients_clean` table...")
    write_clean(engine, df_clean)
    print(f"[OK] Saved {len(df_clean)} rows to patients_clean.")
    
    return df_clean

if __name__ == "__main__":
    settings = get_settings()
    run_cleaning(settings)
