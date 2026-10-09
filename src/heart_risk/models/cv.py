"""
src/heart_risk/models/cv.py
─────────────────────────────────────────────────────────────────────────────
Generates or loads the fixed 15 folds (5-fold, 3 repeats) for cross-validation.
"""

import pandas as pd
import json
import hashlib
from sklearn.model_selection import RepeatedStratifiedKFold
from heart_risk.config import PROJECT_ROOT, Settings

def get_train_sha256(train_path) -> str:
    with open(train_path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()

def get_or_create_cv_folds(settings: Settings) -> list:
    """
    Returns a list of 15 tuples: (train_indices, val_indices).
    Saves the folds to artifacts/cv_folds.json so they are identical across models.
    Fails loudly if train.csv has changed since generation.
    """
    train_path = PROJECT_ROOT / "artifacts" / "data" / "train.csv"
    folds_path = PROJECT_ROOT / "artifacts" / "cv_folds.json"
    
    if not train_path.exists():
        raise FileNotFoundError("train.csv not found. Run `make split` first.")
        
    current_sha = get_train_sha256(train_path)
    df = pd.read_csv(train_path)
    
    if folds_path.exists():
        with open(folds_path, "r") as f:
            data = json.load(f)
        if data["train_sha256"] != current_sha:
            raise RuntimeError(
                f"Hash mismatch! train.csv has changed since cv_folds.json was created.\n"
                f"Expected {data['train_sha256']}, got {current_sha}.\n"
                f"Delete cv_folds.json to regenerate."
            )
        # Convert saved lists back to list of tuples for scikit-learn
        folds = [(train_idx, val_idx) for train_idx, val_idx in data["folds"]]
        return folds
        
    # Generate folds
    print("[INFO] Generating new RepeatedStratifiedKFold indices...")
    df['stratify_key'] = df['dataset'] + "_" + df['target'].astype(str)
    
    rskf = RepeatedStratifiedKFold(n_splits=5, n_repeats=3, random_state=settings.random_seed)
    folds = []
    for train_idx, val_idx in rskf.split(df, df['stratify_key']):
        folds.append((train_idx.tolist(), val_idx.tolist()))
        
    # Save for reproducibility
    data = {
        "train_sha256": current_sha,
        "folds": folds,
        "row_ids_reference": df['id'].tolist() # Save stable row_ids just in case
    }
    with open(folds_path, "w") as f:
        json.dump(data, f)
        
    print(f"[OK] Saved 15 folds to {folds_path.name}")
    return folds
