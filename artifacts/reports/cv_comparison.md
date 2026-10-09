# Phase 4: CV Model Comparison

## Feature Set: F1

| Model | ROC-AUC | PR-AUC | Recall (@0.5) | Precision (@0.5) | F2 (@0.5) | Train AUC | CM | Cleveland AUC | Cleveland Recall | Hungary AUC | Hungary Recall | Hyperparameters |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Dummy | 0.500 ± 0.000 | 0.412 ± 0.002 | 0.000 ± 0.000 | 0.000 ± 0.000 | 0.000 ± 0.000 | 0.500 ± 0.000 | [[840, 0], [588, 0]] | 0.511 | 0.000 | 0.484 | 0.000 | N/A |
| LR | 0.889 ± 0.031 | 0.866 ± 0.035 | 0.717 ± 0.085 | 0.804 ± 0.053 | 0.731 ± 0.070 | 0.901 ± 0.006 | [[734, 106], [166, 422]] | 0.857 | 0.712 | 0.913 | 0.741 | {'clf__C': 0.1} |
| RF | 0.879 ± 0.029 | 0.856 ± 0.031 | 0.649 ± 0.067 | 0.822 ± 0.044 | 0.677 ± 0.060 | 0.895 ± 0.006 | [[756, 84], [206, 382]] | 0.834 | 0.604 | 0.913 | 0.694 | {'clf__n_estimators': 200, 'clf__min_samples_leaf': 10, 'clf__max_depth': 2} |
| XGB | 0.881 ± 0.027 | 0.857 ± 0.032 | 0.709 ± 0.074 | 0.788 ± 0.059 | 0.721 ± 0.060 | 0.924 ± 0.004 | [[724, 116], [171, 417]] | 0.849 | 0.703 | 0.903 | 0.694 | {'clf__subsample': 0.9, 'clf__reg_lambda': 50.0, 'clf__n_estimators': 100, 'clf__min_child_weight': 2, 'clf__max_depth': 2, 'clf__learning_rate': 0.15, 'clf__colsample_bytree': 1.0} |


### Paired Differences (vs LR)

- **RF vs LR**: Mean Diff AUC = -0.010, SE = 0.003. RF won 2/15 folds. Range: [-0.041, 0.003]. (Differences of this size are a practical tie within the noise of overlapping CV folds).
- **XGB vs LR**: Mean Diff AUC = -0.008, SE = 0.003. XGB won 4/15 folds. Range: [-0.024, 0.009]. (Differences of this size are a practical tie within the noise of overlapping CV folds).

## Feature Set: F2

| Model | ROC-AUC | PR-AUC | Recall (@0.5) | Precision (@0.5) | F2 (@0.5) | Train AUC | CM | Cleveland AUC | Cleveland Recall | Hungary AUC | Hungary Recall | Hyperparameters |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Dummy | 0.500 ± 0.000 | 0.412 ± 0.002 | 0.000 ± 0.000 | 0.000 ± 0.000 | 0.000 ± 0.000 | 0.500 ± 0.000 | [[840, 0], [588, 0]] | 0.511 | 0.000 | 0.484 | 0.000 | N/A |
| LR | 0.886 ± 0.032 | 0.863 ± 0.038 | 0.707 ± 0.077 | 0.793 ± 0.051 | 0.721 ± 0.065 | 0.898 ± 0.007 | [[729, 111], [172, 416]] | 0.858 | 0.721 | 0.909 | 0.694 | {'clf__C': 0.1} |
| RF | 0.880 ± 0.027 | 0.857 ± 0.032 | 0.643 ± 0.063 | 0.824 ± 0.044 | 0.671 ± 0.057 | 0.896 ± 0.005 | [[758, 82], [210, 378]] | 0.838 | 0.613 | 0.912 | 0.671 | {'clf__n_estimators': 200, 'clf__min_samples_leaf': 6, 'clf__max_depth': 2} |
| XGB | 0.881 ± 0.030 | 0.852 ± 0.036 | 0.660 ± 0.067 | 0.812 ± 0.048 | 0.684 ± 0.056 | 0.906 ± 0.006 | [[747, 93], [200, 388]] | 0.845 | 0.649 | 0.897 | 0.671 | {'clf__subsample': 1.0, 'clf__reg_lambda': 50.0, 'clf__n_estimators': 100, 'clf__min_child_weight': 1, 'clf__max_depth': 2, 'clf__learning_rate': 0.05, 'clf__colsample_bytree': 0.6} |


### Paired Differences (vs LR)

- **RF vs LR**: Mean Diff AUC = -0.006, SE = 0.003. RF won 6/15 folds. Range: [-0.032, 0.008]. (Differences of this size are a practical tie within the noise of overlapping CV folds).
- **XGB vs LR**: Mean Diff AUC = -0.005, SE = 0.003. XGB won 6/15 folds. Range: [-0.025, 0.010]. (Differences of this size are a practical tie within the noise of overlapping CV folds).

## F1 vs F2 Performance (Paired Differences)

- **LR (F1 vs F2)**: Mean Diff AUC = +0.002, SE = 0.002.
- **RF (F1 vs F2)**: Mean Diff AUC = -0.002, SE = 0.001.
- **XGB (F1 vs F2)**: Mean Diff AUC = -0.000, SE = 0.002.

## Selected Hyperparameters and Edge Warnings
- **F1 LR**: {'clf__C': 0.1} -> C=0.1 (strong regularization)
- **F1 RF**: {'clf__n_estimators': 200, 'clf__min_samples_leaf': 10, 'clf__max_depth': 2} -> max_depth at lower edge (2), min_samples_leaf at upper edge (10)
- **F1 XGB**: {'clf__subsample': 0.9, 'clf__reg_lambda': 50.0, 'clf__n_estimators': 100, 'clf__min_child_weight': 2, 'clf__max_depth': 2, 'clf__learning_rate': 0.15, 'clf__colsample_bytree': 1.0} -> max_depth at lower edge (2)
- **F2 LR**: {'clf__C': 0.1} -> C=0.1 (strong regularization)
- **F2 RF**: {'clf__n_estimators': 200, 'clf__min_samples_leaf': 6, 'clf__max_depth': 2} -> max_depth at lower edge (2)
- **F2 XGB**: {'clf__subsample': 1.0, 'clf__reg_lambda': 50.0, 'clf__n_estimators': 100, 'clf__min_child_weight': 1, 'clf__max_depth': 2, 'clf__learning_rate': 0.05, 'clf__colsample_bytree': 0.6} -> max_depth at lower edge (2)

## Notes
- SE formula used for 1-SE rule: std(15 folds) / sqrt(15). Note that because Repeated-CV folds overlap, this standard error is somewhat optimistic.
- All models were evaluated with `class_weight=None` since we rely on ROC-AUC for ranking, and will handle class imbalance during threshold tuning later.
