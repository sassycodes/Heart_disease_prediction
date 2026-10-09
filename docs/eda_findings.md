# EDA Findings & Proposed Cleaning Rules

**Note:** All treatments below are **PROPOSED, awaiting approval**. No data has been modified.

## Issue 1: High Missingness in `ca`, `thal`, and `slope` (Informative Missingness)
- **Description:** These three columns are missing for more than a third of the dataset. More importantly, the missingness is almost entirely tied to the hospital. For example, `ca` is missing in 99% of Hungary and VA Long Beach rows, and 95.9% of Switzerland rows.
- **Evidence:** `ca` has 611 blanks (66%), `thal` has 486 blanks (53%), `slope` has 309 blanks (33%).
- **Severity:** Blocking (affects imputation strategy and model fairness across hospitals).
- **PROPOSED Treatment:** Do not blindly impute with the mean/mode. For categorical variables, treat "Missing" as its own distinct category. 
  - *Alternatives:* Drop the columns entirely (Trade-off: loss of predictive signal for Cleveland, which has them). Drop all rows with missing values (Trade-off: we would lose 3 entire hospitals, which is unacceptable).

## Issue 2: Fake Zeros in Cholesterol (`chol`)
- **Description:** Cholesterol cannot biologically be zero. The dataset encodes missing `chol` as `0` in certain hospitals, alongside genuine blanks (`''`).
- **Evidence:** There are 172 rows with `chol = 0`. Exactly 123 of them are in Switzerland (100% of their data) and 49 in VA Long Beach.
- **Severity:** Blocking.
- **PROPOSED Treatment:** Convert `chol = 0` to `NaN` (missing) before imputation.
  - *Alternatives:* Leave them as 0 (Trade-off: the model will learn that "0 cholesterol" predicts disease since Switzerland has a high disease rate, which is a dangerous shortcut).

## Issue 3: Target Variable (`num`) has Rare Classes
- **Description:** The target `num` ranges from 0 (no disease) to 4, but severity levels 3 and 4 are very rare.
- **Evidence:** `num` = 0 (411), 1 (265), 2 (109), 3 (107), 4 (28).
- **Severity:** Important.
- **PROPOSED Treatment:** Binarize the target: `0` means no disease, `>0` means disease.
  - *Alternatives:* Keep all 5 classes (Trade-off: severe class imbalance makes it very hard to predict classes 3 and 4 accurately without massive data).

## Issue 4: Impossible Values in `trestbps` and `oldpeak`
- **Description:** Resting blood pressure (`trestbps`) has a 0 value. ST depression (`oldpeak`) has negative values.
- **Evidence:** `trestbps` has 1 zero. `oldpeak` has 12 negative values (min: -2.6).
- **Severity:** Minor.
- **PROPOSED Treatment:** Replace `trestbps = 0` with `NaN`. For `oldpeak`, check literature to see if negative ST depression is clinically valid; if not, treat as `NaN`. (Recommendation: Treat `<0` as `NaN`).
  - *Alternatives:* Drop the rows (Trade-off: unnecessary data loss for a single feature error).

## Issue 5: Duplicate Rows
- **Description:** 4 rows have identical features (excluding the `id` column).
- **Evidence:** `df.duplicated()` found 4 rows.
- **Severity:** Minor.
- **PROPOSED Treatment:** Drop duplicate rows (keep the first occurrence).
  - *Alternatives:* Keep them (Trade-off: minor data leakage if identical patients are split across train/test sets).

---

## Decisions Awaiting Your Approval
Please reply with your decisions on the following so I can configure Phase 3:
1. **Treat `chol=0` as missing?** (Recommended: Yes; evidence: 172 zeros, all in Switzerland and VA Long Beach).
2. **Treat `trestbps=0` as missing?** (Recommended: Yes; evidence: 1 zero value).
3. **Treat negative `oldpeak` as missing?** (Recommended: Yes, or confirm if clinically possible).
4. **Target Formulation:** Should we binarize the target `num` to (0 vs >0)? (Recommended: Yes).
5. **Handling massive missingness (`ca`, `thal`, `slope`):** Should we treat "Missing" as a distinct categorical level? (Recommended: Yes).
6. **Drop duplicates?** (Recommended: Yes).
