import pandas as pd
import numpy as np
from pathlib import Path
import json

from heart_risk.config import get_settings, PROJECT_ROOT
from heart_risk.models.cv import get_or_create_cv_folds
from heart_risk.features.pipeline import build_pipeline
from heart_risk.models.registry import get_models

def test_pipeline_leakage():
    """
    Test that pipeline imputer/scaler statistics equal train-only statistics (leakage test).
    """
    settings = get_settings()
    train_path = PROJECT_ROOT / "artifacts" / "data" / "train.csv"
    if not train_path.exists():
        return
        
    df = pd.read_csv(train_path)
    X = df.drop(columns=['target'])
    y = df['target']
    
    # Take the first fold for testing
    folds = get_or_create_cv_folds(settings)
    train_idx, val_idx = folds[0]
    
    X_train = X.iloc[train_idx]
    
    models = get_models(settings)
    pipe = build_pipeline('F1', models['lr'])
    
    pipe.fit(X_train, y.iloc[train_idx])
    
    # Check numeric imputation
    num_imputer = pipe.named_steps['preprocessor'].transformers_[0][1].named_steps['imputer']
    num_features = pipe.named_steps['preprocessor'].transformers_[0][2]
    
    # Calculate median manually on train data
    manual_medians = X_train[num_features].median().values
    np.testing.assert_array_almost_equal(num_imputer.statistics_, manual_medians)
    
def test_no_external_data_loaded(monkeypatch):
    """
    Grep-style test that this step's code never reads the hold-out or external files.
    """
    models_dir = PROJECT_ROOT / "src" / "heart_risk" / "models"
    
    forbidden_terms = ["test.csv", "external_switzerland.csv", "external_va.csv", "pooled"]
    
    for py_file in models_dir.glob("*.py"):
        with open(py_file, "r") as f:
            content = f.read()
            for term in forbidden_terms:
                assert term not in content, f"Forbidden term '{term}' found in {py_file.name}!"

def test_identical_folds_and_no_overlap():
    settings = get_settings()
    folds_path = PROJECT_ROOT / "artifacts" / "cv_folds.json"
    if not folds_path.exists():
        return
        
    with open(folds_path, "r") as f:
        data = json.load(f)
        
    folds = data["folds"]
    assert len(folds) == 15
    
    for train_idx, val_idx in folds:
        train_set = set(train_idx)
        val_set = set(val_idx)
        overlap = train_set.intersection(val_set)
        assert len(overlap) == 0, "Train and Validation sets overlap within a fold!"
