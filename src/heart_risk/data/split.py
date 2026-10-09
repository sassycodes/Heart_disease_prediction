"""
src/heart_risk/data/split.py
─────────────────────────────────────────────────────────────────────────────
Phase 4 (Prep): Split the cleaned data into appropriate training and evaluation sets.

Decisions:
1. Primary Dataset: Cleveland + Hungary.
2. External Sets: Switzerland + VA Long Beach.
3. 80/20 train/test split on Primary, stratified by `hospital × target`.
4. Fixed seed 42.
5. Create a 'naive pooled' split for the reference baseline.
"""

import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split

from heart_risk.config import Settings, get_settings, PROJECT_ROOT
from heart_risk.db.engine import build_engine

def run_split(settings: Settings) -> None:
    # Read patients_clean using pandas direct SQL
    engine = build_engine(settings)
    df = pd.read_sql_table("patients_clean", con=engine.connect())
    
    # Ensure artifacts/data directory exists
    data_dir = PROJECT_ROOT / "artifacts" / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"[INFO] Initial rows in patients_clean: {len(df)}")
    
    # External Sets
    switz = df[df['dataset'] == 'Switzerland'].copy()
    va = df[df['dataset'] == 'VA Long Beach'].copy()
    
    switz.to_csv(data_dir / "external_switzerland.csv", index=False)
    va.to_csv(data_dir / "external_va.csv", index=False)
    
    print(f"\n--- EXTERNAL SETS ---")
    print(f"Switzerland: {len(switz)} rows. Class counts:")
    print(switz['target'].value_counts().to_dict())
    print(f"VA Long Beach: {len(va)} rows. Class counts:")
    print(va['target'].value_counts().to_dict())
    
    # Primary Dataset
    primary = df[df['dataset'].isin(['Cleveland', 'Hungary'])].copy()
    print(f"\n--- PRIMARY SET (Cleveland + Hungary) ---")
    print(f"Primary set rows: {len(primary)}")
    
    # Create stratification key: hospital + target
    primary['stratify_key'] = primary['dataset'] + "_" + primary['target'].astype(str)
    
    train, test = train_test_split(
        primary, 
        test_size=0.2, 
        stratify=primary['stratify_key'], 
        random_state=settings.random_seed
    )
    
    # Drop the temporary stratify key before saving
    train = train.drop(columns=['stratify_key'])
    test = test.drop(columns=['stratify_key'])
    
    train.to_csv(data_dir / "train.csv", index=False)
    test.to_csv(data_dir / "test.csv", index=False)
    
    print(f"\nPrimary Train rows: {len(train)}")
    print("Train hospital x target counts:")
    print(train.groupby(['dataset', 'target']).size().to_dict())
    
    print(f"\nPrimary Test (Hold-out) rows: {len(test)}")
    print("Test hospital x target counts:")
    print(test.groupby(['dataset', 'target']).size().to_dict())
    
    # Naive Pooled Split
    df['stratify_key'] = df['dataset'] + "_" + df['target'].astype(str)
    pooled_train, pooled_test = train_test_split(
        df,
        test_size=0.2,
        stratify=df['stratify_key'],
        random_state=settings.random_seed
    )
    pooled_train = pooled_train.drop(columns=['stratify_key'])
    pooled_test = pooled_test.drop(columns=['stratify_key'])
    
    pooled_train.to_csv(data_dir / "pooled_train.csv", index=False)
    pooled_test.to_csv(data_dir / "pooled_test.csv", index=False)
    
    print(f"\n[OK] Split complete. Files written to {data_dir}")

if __name__ == "__main__":
    settings = get_settings()
    run_split(settings)
