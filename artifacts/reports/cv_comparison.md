# Phase 4: CV Model Comparison

## Feature Set: F1

| Model | ROC-AUC | PR-AUC | Recall (@0.5) | Precision (@0.5) | F2 (@0.5) | Train AUC | CM |
|---|---|---|---|---|---|---|---|
| Dummy | 0.500 ± 0.000 | 0.412 ± 0.002 | 0.000 ± 0.000 | 0.000 ± 0.000 | 0.000 ± 0.000 | 0.500 ± 0.000 | [[840, 0], [588, 0]] |
| LR | 0.889 ± 0.031 | 0.866 ± 0.035 | 0.717 ± 0.085 | 0.804 ± 0.053 | 0.731 ± 0.070 | 0.901 ± 0.006 | [[734, 106], [166, 422]] |
| RF | 0.879 ± 0.029 | 0.856 ± 0.031 | 0.649 ± 0.067 | 0.822 ± 0.044 | 0.677 ± 0.060 | 0.895 ± 0.006 | [[756, 84], [206, 382]] |
| XGB | 0.881 ± 0.027 | 0.857 ± 0.032 | 0.709 ± 0.074 | 0.788 ± 0.059 | 0.721 ± 0.060 | 0.924 ± 0.004 | [[724, 116], [171, 417]] |


### Paired Differences (vs LR)

- **RF vs LR**: Mean Diff AUC = -0.010 ± 0.003 (Significant). RF won 2/15 folds. Range: [-0.041, 0.003].
- **XGB vs LR**: Mean Diff AUC = -0.008 ± 0.003 (Significant). XGB won 4/15 folds. Range: [-0.024, 0.009].

## Feature Set: F2

| Model | ROC-AUC | PR-AUC | Recall (@0.5) | Precision (@0.5) | F2 (@0.5) | Train AUC | CM |
|---|---|---|---|---|---|---|---|
| Dummy | 0.500 ± 0.000 | 0.412 ± 0.002 | 0.000 ± 0.000 | 0.000 ± 0.000 | 0.000 ± 0.000 | 0.500 ± 0.000 | [[840, 0], [588, 0]] |
| LR | 0.886 ± 0.032 | 0.863 ± 0.038 | 0.707 ± 0.077 | 0.793 ± 0.051 | 0.721 ± 0.065 | 0.898 ± 0.007 | [[729, 111], [172, 416]] |
| RF | 0.880 ± 0.027 | 0.857 ± 0.032 | 0.643 ± 0.063 | 0.824 ± 0.044 | 0.671 ± 0.057 | 0.896 ± 0.005 | [[758, 82], [210, 378]] |
| XGB | 0.881 ± 0.030 | 0.852 ± 0.036 | 0.660 ± 0.067 | 0.812 ± 0.048 | 0.684 ± 0.056 | 0.906 ± 0.006 | [[747, 93], [200, 388]] |


### Paired Differences (vs LR)

- **RF vs LR**: Mean Diff AUC = -0.006 ± 0.003 (Significant). RF won 6/15 folds. Range: [-0.032, 0.008].
- **XGB vs LR**: Mean Diff AUC = -0.005 ± 0.003 (Significant). XGB won 6/15 folds. Range: [-0.025, 0.010].

## Notes
- SE formula used for 1-SE rule: std(15 folds) / sqrt(15). Note that because Repeated-CV folds overlap, this standard error is somewhat optimistic.
- Paired differences are computed on the same folds. Because final hyperparameters were selected using these same folds, these differences might be slightly optimistic (descriptive only).
- All models were evaluated with `class_weight=None` since we rely on ROC-AUC for ranking, and will handle class imbalance during threshold tuning later.
