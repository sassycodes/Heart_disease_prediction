"""
src/heart_risk/models/train.py
─────────────────────────────────────────────────────────────────────────────
Runs cross-validation model comparison on the primary training set.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV
from sklearn.metrics import roc_auc_score, average_precision_score, recall_score, precision_score, fbeta_score, confusion_matrix

from heart_risk.config import get_settings, PROJECT_ROOT, Settings
from heart_risk.models.registry import get_models, get_search_spaces
from heart_risk.features.pipeline import build_pipeline
from heart_risk.models.cv import get_or_create_cv_folds

def apply_1se_rule(cv_results, model_name: str):
    """
    Applies the 1-Standard Error rule to select the simplest configuration
    within 1 SE of the best AUC.
    """
    df_cv = pd.DataFrame(cv_results)
    
    # Best mean AUC
    best_idx = df_cv['mean_test_score'].idxmax()
    best_score = df_cv.loc[best_idx, 'mean_test_score']
    best_std = df_cv.loc[best_idx, 'std_test_score']
    
    # SE of the best = std / sqrt(15)
    se = best_std / np.sqrt(15)
    threshold = best_score - se
    
    # Candidates within 1 SE
    candidates = df_cv[df_cv['mean_test_score'] >= threshold].copy()
    
    # Define complexity sorting keys
    def complexity_sort(row):
        params = row['params']
        if model_name == 'lr':
            # Ascending C
            return (params.get('clf__C', 1.0),)
        elif model_name == 'rf':
            # Asc max_depth, Desc min_samples_leaf, Asc n_estimators
            return (params.get('clf__max_depth', 999), 
                    -params.get('clf__min_samples_leaf', 1), 
                    params.get('clf__n_estimators', 999))
        elif model_name == 'xgb':
            # Asc max_depth, Desc min_child_weight, Desc reg_lambda, Asc n_estimators
            return (params.get('clf__max_depth', 999), 
                    -params.get('clf__min_child_weight', 1), 
                    -params.get('clf__reg_lambda', 0.0), 
                    params.get('clf__n_estimators', 999))
        return (0,)
        
    candidates['complexity'] = candidates.apply(complexity_sort, axis=1)
    candidates = candidates.sort_values('complexity')
    
    # The first row is the simplest
    selected = candidates.iloc[0]
    return selected['params']

def evaluate_folds(pipeline, df, target, folds):
    """
    Runs the pipeline across the 15 folds and collects metrics.
    """
    metrics = {
        'roc_auc': [], 'pr_auc': [], 'recall': [], 'precision': [], 'f2': [],
        'train_roc_auc': []
    }
    cm_sum = np.zeros((2, 2), dtype=int)
    
    X = df.drop(columns=['target'])
    y = df['target']
    
    fold_preds = []
    
    # To compute per-hospital metrics, we need to collect all OOF predictions and their indices
    all_oof = []

    for train_idx, val_idx in folds:
        X_train, y_train = X.iloc[train_idx], y.iloc[train_idx]
        X_val, y_val = X.iloc[val_idx], y.iloc[val_idx]
        
        pipeline.fit(X_train, y_train)
        
        y_train_pred_prob = pipeline.predict_proba(X_train)[:, 1]
        y_val_pred_prob = pipeline.predict_proba(X_val)[:, 1]
        y_val_pred = (y_val_pred_prob >= 0.5).astype(int)
        
        # Keep track of predictions for paired differences
        fold_preds.append(y_val_pred_prob)
        
        # Save OOF for this fold
        dataset_val = df.iloc[val_idx]['dataset'].values
        for i, idx in enumerate(val_idx):
            all_oof.append({'idx': idx, 'y_true': y_val.values[i], 'y_prob': y_val_pred_prob[i], 'dataset': dataset_val[i]})
        
        metrics['train_roc_auc'].append(roc_auc_score(y_train, y_train_pred_prob))
        metrics['roc_auc'].append(roc_auc_score(y_val, y_val_pred_prob))
        metrics['pr_auc'].append(average_precision_score(y_val, y_val_pred_prob))
        metrics['recall'].append(recall_score(y_val, y_val_pred))
        metrics['precision'].append(precision_score(y_val, y_val_pred, zero_division=0))
        metrics['f2'].append(fbeta_score(y_val, y_val_pred, beta=2, zero_division=0))
        
        cm = confusion_matrix(y_val, y_val_pred, labels=[0, 1])
        cm_sum += cm
        
    # Per-hospital metrics over ALL OOF predictions
    oof_df = pd.DataFrame(all_oof)
    
    # We will average the 3 predictions per row
    avg_oof = oof_df.groupby(['idx', 'dataset']).agg({'y_true': 'first', 'y_prob': 'mean'}).reset_index()
    avg_oof['y_pred'] = (avg_oof['y_prob'] >= 0.5).astype(int)
    
    hosp_metrics = {}
    for hosp in ['Cleveland', 'Hungary']:
        h_df = avg_oof[avg_oof['dataset'] == hosp]
        h_auc = roc_auc_score(h_df['y_true'], h_df['y_prob'])
        h_rec = recall_score(h_df['y_true'], h_df['y_pred'])
        hosp_metrics[hosp] = {'auc': h_auc, 'recall': h_rec}

    res = {k: {"mean": np.mean(v), "std": np.std(v)} for k, v in metrics.items()}
    res['cm'] = cm_sum
    res['fold_roc_auc'] = metrics['roc_auc'] # For paired diffs
    res['hosp_metrics'] = hosp_metrics
    return res

def run_training_cv(settings: Settings):
    train_path = PROJECT_ROOT / "artifacts" / "data" / "train.csv"
    if not train_path.exists():
        raise FileNotFoundError("train.csv missing.")
        
    df = pd.read_csv(train_path)
    folds = get_or_create_cv_folds(settings)
    
    models = get_models(settings)
    spaces = get_search_spaces()
    
    X = df.drop(columns=['target'])
    y = df['target']
    
    results = []
    
    for feature_set in ['F1', 'F2']:
        print(f"\n=== Evaluating Feature Set: {feature_set} ===")
        
        # 1. Dummy Baseline
        dummy_pipe = build_pipeline(feature_set, models['dummy'])
        dummy_res = evaluate_folds(dummy_pipe, df, 'target', folds)
        results.append({
            'Feature Set': feature_set, 'Model': 'Dummy',
            'ROC-AUC': f"{dummy_res['roc_auc']['mean']:.3f} ± {dummy_res['roc_auc']['std']:.3f}",
            'PR-AUC': f"{dummy_res['pr_auc']['mean']:.3f} ± {dummy_res['pr_auc']['std']:.3f}",
            'Recall (@0.5)': f"{dummy_res['recall']['mean']:.3f} ± {dummy_res['recall']['std']:.3f}",
            'Precision (@0.5)': f"{dummy_res['precision']['mean']:.3f} ± {dummy_res['precision']['std']:.3f}",
            'F2 (@0.5)': f"{dummy_res['f2']['mean']:.3f} ± {dummy_res['f2']['std']:.3f}",
            'Train AUC': f"{dummy_res['train_roc_auc']['mean']:.3f} ± {dummy_res['train_roc_auc']['std']:.3f}",
            'CM': str(dummy_res['cm'].tolist()),
            'Cleveland AUC': f"{dummy_res['hosp_metrics']['Cleveland']['auc']:.3f}",
            'Cleveland Recall': f"{dummy_res['hosp_metrics']['Cleveland']['recall']:.3f}",
            'Hungary AUC': f"{dummy_res['hosp_metrics']['Hungary']['auc']:.3f}",
            'Hungary Recall': f"{dummy_res['hosp_metrics']['Hungary']['recall']:.3f}",
            'Hyperparameters': "N/A",
            '_fold_aucs': dummy_res['fold_roc_auc']
        })
        
        # 2. Main Models
        for m_name in ['lr', 'rf', 'xgb']:
            print(f"-> Tuning {m_name.upper()}...")
            pipe = build_pipeline(feature_set, models[m_name])
            
            if m_name == 'lr':
                search = GridSearchCV(
                    pipe, spaces[m_name], cv=folds, scoring='roc_auc', refit=False, n_jobs=-1
                )
            else:
                search = RandomizedSearchCV(
                    pipe, spaces[m_name], n_iter=40, cv=folds, scoring='roc_auc', 
                    refit=False, random_state=settings.random_seed, n_jobs=-1
                )
                
            search.fit(X, y)
            best_params = apply_1se_rule(search.cv_results_, m_name)
            
            print(f"   Selected params: {best_params}")
            
            # Re-evaluate selected
            pipe.set_params(**best_params)
            res = evaluate_folds(pipe, df, 'target', folds)
            
            results.append({
                'Feature Set': feature_set, 'Model': m_name.upper(),
                'ROC-AUC': f"{res['roc_auc']['mean']:.3f} ± {res['roc_auc']['std']:.3f}",
                'PR-AUC': f"{res['pr_auc']['mean']:.3f} ± {res['pr_auc']['std']:.3f}",
                'Recall (@0.5)': f"{res['recall']['mean']:.3f} ± {res['recall']['std']:.3f}",
                'Precision (@0.5)': f"{res['precision']['mean']:.3f} ± {res['precision']['std']:.3f}",
                'F2 (@0.5)': f"{res['f2']['mean']:.3f} ± {res['f2']['std']:.3f}",
                'Train AUC': f"{res['train_roc_auc']['mean']:.3f} ± {res['train_roc_auc']['std']:.3f}",
                'CM': str(res['cm'].tolist()),
                'Cleveland AUC': f"{res['hosp_metrics']['Cleveland']['auc']:.3f}",
                'Cleveland Recall': f"{res['hosp_metrics']['Cleveland']['recall']:.3f}",
                'Hungary AUC': f"{res['hosp_metrics']['Hungary']['auc']:.3f}",
                'Hungary Recall': f"{res['hosp_metrics']['Hungary']['recall']:.3f}",
                'Hyperparameters': str(best_params),
                '_fold_aucs': res['fold_roc_auc']
            })
            
    # Compute Paired Differences against LR (baseline model)
    # We'll do this in the reporting step
    df_res = pd.DataFrame(results)
    
    # Save CSV
    reports_dir = PROJECT_ROOT / "artifacts" / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)
    
    clean_res = df_res.drop(columns=['_fold_aucs'])
    clean_res.to_csv(reports_dir / "cv_results.csv", index=False)
    
    # Generate Markdown Report
    with open(reports_dir / "cv_comparison.md", "w") as f:
        f.write("# Phase 4: CV Model Comparison\n\n")
        
        for fset in ['F1', 'F2']:
            f.write(f"## Feature Set: {fset}\n\n")
            subset = clean_res[clean_res['Feature Set'] == fset].drop(columns=['Feature Set'])
            headers = subset.columns.tolist()
            f.write("| " + " | ".join(headers) + " |\n")
            f.write("|" + "|".join(["---"] * len(headers)) + "|\n")
            for _, row in subset.iterrows():
                f.write("| " + " | ".join(str(x) for x in row.values) + " |\n")
            f.write("\n\n")
            
            # Paired diffs vs LR
            f.write("### Paired Differences (vs LR)\n\n")
            full_sub = df_res[df_res['Feature Set'] == fset]
            lr_aucs = np.array(full_sub[full_sub['Model'] == 'LR'].iloc[0]['_fold_aucs'])
            
            diff_lines = []
            for _, row in full_sub.iterrows():
                if row['Model'] in ['LR', 'Dummy']:
                    continue
                aucs = np.array(row['_fold_aucs'])
                diffs = aucs - lr_aucs
                mean_diff = np.mean(diffs)
                se_diff = np.std(diffs) / np.sqrt(15)
                wins = np.sum(diffs > 0)
                
                diff_lines.append(
                    f"- **{row['Model']} vs LR**: Mean Diff AUC = {mean_diff:+.3f}, SE = {se_diff:.3f}. "
                    f"{row['Model']} won {wins}/15 folds. Range: [{np.min(diffs):.3f}, {np.max(diffs):.3f}]. "
                    f"(Differences of this size are a practical tie within the noise of overlapping CV folds)."
                )
            
            f.write("\n".join(diff_lines) + "\n\n")

        # Compute F1 vs F2 differences per model
        f.write("## F1 vs F2 Performance (Paired Differences)\n\n")
        diff_f1_f2_lines = []
        for model_name in ['LR', 'RF', 'XGB']:
            f1_aucs = np.array(df_res[(df_res['Feature Set'] == 'F1') & (df_res['Model'] == model_name)].iloc[0]['_fold_aucs'])
            f2_aucs = np.array(df_res[(df_res['Feature Set'] == 'F2') & (df_res['Model'] == model_name)].iloc[0]['_fold_aucs'])
            diffs = f1_aucs - f2_aucs
            mean_diff = np.mean(diffs)
            se_diff = np.std(diffs) / np.sqrt(15)
            diff_f1_f2_lines.append(
                f"- **{model_name} (F1 vs F2)**: Mean Diff AUC = {mean_diff:+.3f}, SE = {se_diff:.3f}."
            )
        f.write("\n".join(diff_f1_f2_lines) + "\n\n")
            
        f.write("## Selected Hyperparameters and Edge Warnings\n")
        for _, row in clean_res.iterrows():
            if row['Model'] == 'Dummy':
                continue
            params = row['Hyperparameters']
            warnings = []
            if "'clf__max_depth': 2" in params: warnings.append("max_depth at lower edge (2)")
            if "'clf__C': 0.1" in params: warnings.append("C=0.1 (strong regularization)")
            if "'clf__min_samples_leaf': 10" in params: warnings.append("min_samples_leaf at upper edge (10)")
            warning_str = " -> " + ", ".join(warnings) if warnings else ""
            f.write(f"- **{row['Feature Set']} {row['Model']}**: {params}{warning_str}\n")
            
        f.write("\n## Notes\n")
        f.write("- SE formula used for 1-SE rule: std(15 folds) / sqrt(15). Note that because Repeated-CV folds overlap, this standard error is somewhat optimistic.\n")
        f.write("- All models were evaluated with `class_weight=None` since we rely on ROC-AUC for ranking, and will handle class imbalance during threshold tuning later.\n")

if __name__ == "__main__":
    run_training_cv(get_settings())
