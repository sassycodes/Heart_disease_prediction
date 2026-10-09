"""
src/heart_risk/models/registry.py
─────────────────────────────────────────────────────────────────────────────
Registry for models and their hyperparameter search spaces.
"""

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.dummy import DummyClassifier
from xgboost import XGBClassifier
import yaml
from pathlib import Path
from heart_risk.config import PROJECT_ROOT, Settings

def get_models(settings: Settings) -> dict:
    """Returns the initialized models for cross-validation."""
    return {
        "lr": LogisticRegression(
            class_weight=None,
            max_iter=1000,
            random_state=settings.random_seed
        ),
        "rf": RandomForestClassifier(
            class_weight=None,
            random_state=settings.random_seed
        ),
        "xgb": XGBClassifier(
            scale_pos_weight=1,
            random_state=settings.random_seed,
            eval_metric="logloss"
        ),
        "dummy": DummyClassifier(
            strategy="prior",
            random_state=settings.random_seed
        )
    }

def get_search_spaces() -> dict:
    """Reads the search spaces from configs/models.yaml."""
    models_yaml = PROJECT_ROOT / "configs" / "models.yaml"
    with open(models_yaml, "r") as f:
        config = yaml.safe_load(f)
    
    # Prepend the model step name if needed in a pipeline.
    # The models will be wrapped in a pipeline as `clf`.
    spaces = {}
    for model_name, space in config.get("models", {}).items():
        spaces[model_name] = {f"clf__{k}": v for k, v in space.items()}
    return spaces
