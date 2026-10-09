"""
src/heart_risk/features/pipeline.py
─────────────────────────────────────────────────────────────────────────────
Builds the preprocessing pipeline based on the feature sets in features.yaml.
"""

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
import yaml
from heart_risk.config import PROJECT_ROOT

def build_pipeline(feature_set: str, model) -> Pipeline:
    """
    Builds a complete sklearn Pipeline that pre-processes features and runs the model.
    Reads `configs/features.yaml` to dynamically separate features into numeric vs categorical.
    """
    features_yaml = PROJECT_ROOT / "configs" / "features.yaml"
    with open(features_yaml, "r") as f:
        config = yaml.safe_load(f)
        
    all_features = config.get("features", {})
    feature_list = config.get("feature_sets", {}).get(feature_set, [])
    
    numeric_features = []
    categorical_features = []
    
    for f_name in feature_list:
        f_type = all_features.get(f_name, {}).get("type", "categorical")
        if f_type == "numeric":
            numeric_features.append(f_name)
        else: # categorical and boolean
            categorical_features.append(f_name)
            
    # Define transformers
    numeric_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])
    
    categorical_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
    ])
    
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features)
        ],
        remainder="drop" # Drops id, dataset, target, etc. if they accidentally enter
    )
    
    pipeline = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("clf", model)
    ])
    
    return pipeline
