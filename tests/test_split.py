import pandas as pd
from pathlib import Path
from heart_risk.config import get_settings, PROJECT_ROOT

def test_split_integrity():
    settings = get_settings()
    data_dir = PROJECT_ROOT / "artifacts" / "data"
    
    # We require the split files to exist
    train_path = data_dir / "train.csv"
    test_path = data_dir / "test.csv"
    switz_path = data_dir / "external_switzerland.csv"
    va_path = data_dir / "external_va.csv"
    
    if not train_path.exists():
        return # Skip if split hasn't been run
        
    train = pd.read_csv(train_path)
    test = pd.read_csv(test_path)
    switz = pd.read_csv(switz_path)
    va = pd.read_csv(va_path)
    
    # 1. No row appears in both train and test
    train_ids = set(train['id'].astype(str))
    test_ids = set(test['id'].astype(str))
    overlap = train_ids.intersection(test_ids)
    assert len(overlap) == 0, f"Found {len(overlap)} IDs in both train and test!"
    
    # 2. External set rows never appear in the primary split (train or test)
    switz_ids = set(switz['id'].astype(str))
    va_ids = set(va['id'].astype(str))
    ext_ids = switz_ids.union(va_ids)
    
    primary_ids = train_ids.union(test_ids)
    ext_overlap = primary_ids.intersection(ext_ids)
    assert len(ext_overlap) == 0, f"Found {len(ext_overlap)} external IDs in primary split!"
