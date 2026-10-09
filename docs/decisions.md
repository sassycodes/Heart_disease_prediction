# Architecture Decision Records — Heart Disease Risk Project

Each entry records: the decision, why we made it, and what alternative we rejected (and why).

---

## ADR-001: Raw Data is Stored Untouched in `patients_raw`

**Decision:** Ingest the CSV verbatim into `patients_raw` with no renaming, filtering, or value changes.

**Why:**
Cleaning decisions (what to impute, what to drop) are analytical choices that must be reproducible and auditable. If we clean during ingest, we can never re-derive the original data, and any mistake in cleaning is permanent. Keeping raw data immutable means we can re-run the entire pipeline from a clean slate without re-downloading data.

This also follows the principle of "data lineage" — every downstream artifact should be traceable to its raw source.

**Alternative rejected:** Clean-on-ingest (rename columns, impute blanks, cast types). Rejected because any error in those decisions requires restarting from the CSV download.

---

## ADR-002: All Columns Ingested as Strings (`dtype=str`)

**Decision:** `pd.read_csv(..., dtype=str, keep_default_na=False)` — every cell becomes a Python string.

**Why:**
The raw CSV contains:
- `"TRUE"`/`"FALSE"` (not Python booleans)
- `""` (blank) to mean missing (not `NaN`)
- `"0"` in `chol` (a known data quirk — zero cholesterol is used as a missing-value sentinel in some source hospitals)

If pandas auto-infers types, it silently converts `"TRUE"` → `True` (bool), `""` → `NaN` (float), and `"0"` may be parsed as integer `0`. These coercions are irreversible at this stage and would corrupt the raw record. Storing everything as text is the only type that accepts all values without loss.

**Alternative rejected:** Let pandas infer types. Rejected because it introduces silent data loss that is extremely hard to detect downstream.

---

## ADR-003: Truncate-and-Reload for Idempotency

**Decision:** Before inserting, issue `DELETE FROM patients_raw` inside the same transaction. If the insert fails, the delete also rolls back.

**Why:**
Re-running `make ingest` is a normal operation (after a schema change, after adding new data). A simple `INSERT` would double the rows on every run, corrupting the table silently. Truncating first, inside one transaction, means:
- Success: old rows gone, new rows in → exactly 920 rows.
- Failure: transaction rolls back → table is unchanged from before the run.

**Alternative rejected:** Upsert (`INSERT ... ON CONFLICT UPDATE`). Rejected because it requires a natural key. The dataset has no guaranteed unique key per row across all 4 hospital sources, making upsert fragile.

---

## ADR-004: Repository Layer for All DB Access

**Decision:** All database reads/writes go through functions in `src/heart_risk/db/repository.py`. No raw SQL or ORM calls anywhere else.

**Why:**
Without a repository layer, SQL strings spread across many files. This causes:
- Security risk: harder to audit for injection vulnerabilities.
- Coupling: every file that touches the DB must know the schema.
- Testability: you can't swap Postgres for SQLite without changing every file.

The repository pattern gives us one stable interface. Changing the database backend (Postgres → SQLite for tests, or → BigQuery in future) means changing only one file.

**Alternative rejected:** Direct SQLAlchemy ORM calls in `ingest.py`. Rejected because it violates separation of concerns and makes testing harder.

---

## ADR-005: PostgreSQL for a ~920-Row Dataset

**Decision:** Use PostgreSQL 16 via Docker, even though SQLite would be technically sufficient for 920 rows.

**Why (honest answer):**
SQLite would absolutely work for this data size. We chose Postgres because:
1. **Production parity:** The goal is to learn and demonstrate production-quality practices. Real ML pipelines use proper databases.
2. **Logging value:** Postgres supports concurrent readers, JSONB, and proper transaction isolation — useful when multiple pipeline stages run in parallel (future phases).
3. **Auditability:** Docker Compose gives us a reproducible, versioned database environment that other developers can spin up identically.
4. **Learning value:** Setting this up teaches Docker, connection pooling, and environment configuration — all real-world skills.

**Alternative rejected:** SQLite as the primary database. Rejected not for technical reasons but for learning and production-readiness reasons. SQLite is still used for tests (no Docker needed there).

---

## ADR-006: Single `config.py` Entry Point for All Settings

**Decision:** All code imports settings from `src/heart_risk/config.py` via `get_settings()`. No other module reads `.env` or YAML directly.

**Why:**
If every module loads its own config, settings can drift (module A reads `POSTGRES_PORT=5432`, module B defaults to `5433`). A single entry point guarantees consistency. It also means adding a new setting requires changing only one file.

**Alternative rejected:** Each module loads its own env vars with `os.getenv()`. Rejected because it creates hidden coupling and makes testing painful (you'd have to set env vars in many places).

---

## ADR-007: Retaining Negative `oldpeak` Values

**Decision:** Negative values in the `oldpeak` feature (ST depression) are retained as-is, not treated as missing or invalid.

**Why:**
Negative ST depression (clinically ST elevation) might be a valid physiological measurement indicating severe ischemia. Treating these as errors or missing data would discard legitimate clinical indicators. There are 12 negative values overall (11 in Switzerland, 1 in VA Long Beach). While they are concentrated, they are retained but flagged. We will revisit this decision if leave-one-hospital-out cross-validation on Switzerland behaves oddly.

**Alternative rejected:** Treat negative `oldpeak` as missing (`NaN`). Rejected because it assumes the data is erroneous without clinical justification.

---

## ADR-008: Exact Duplicate Dropping Before Split

**Decision:** Exact duplicate rows (matching across all features except the generated `id`) are dropped before train/test splitting.

**Why:**
We found 4 duplicate rows in the raw dataset. If identical patients are randomly split across the train and test sets, the model's performance will be artificially inflated because it will have memorized those specific patients during training. Dropping exact duplicates ensures the test set evaluates true generalization.

**Alternative rejected:** Keep all rows verbatim. Rejected because of the confounding / shortcut learning risk between train and test sets.

---

## ADR-009: No Missing-Indicators During Cleaning (ca, thal, slope)

**Decision:** Do not add `is_missing` indicator columns during the Phase 3 cleaning step. Leave `ca`, `thal`, and `slope` as true `NaN` and keep all columns.

**Why:**
If we add missing indicators (e.g. `ca_is_missing`), the model might use them as a proxy for the hospital (since missingness is heavily tied to the source hospital). This leads to confounding / shortcut learning via a site proxy, where the model predicts the hospital and uses its base disease rate, rather than learning true physiological risk factors.
The decision on whether `ca`, `thal`, and `slope` become model features will be left to the preprocessing phase. We will likely train two feature sets: "all features" vs "deployable" (which drops hospital-skewed features). This is an OPEN DECISION.

**Alternative rejected:** Add missing indicators in cleaning. Rejected because it hardcodes confounding features into the cleaned data table.

---

## ADR-010: Feature Sets and External Evaluation Rules

**Decision:**
1. **Primary Dataset:** Cleveland and Hungary.
2. **External Sets:** Switzerland and VA Long Beach (never used for training/tuning).
3. **Feature Sets:**
   - `F1`: age, sex, cp, trestbps, chol, fbs, restecg, thalch, exang, oldpeak. `ca`, `thal`, and `slope` are dropped due to massive missingness patterns (NOT target relationships).
   - `F2`: F1 minus `chol` and `fbs`.
4. **Evaluation:** Switzerland is evaluated with F2 only. VA Long Beach is evaluated with F1 on (a) complete-case rows, and (b) all rows with imputation.

**Why:**
Switzerland has 100% missing `chol` and 61% missing `fbs`. Evaluating it on F1 would mean evaluating mostly on imputed data, rendering the test invalid. F2 drops these to provide a robust check for Switzerland. VA Long Beach has ~28% missingness in several F1 features, so checking both complete-case and imputed performance reveals imputation degradation.

**Alternative rejected:** Pooled dataset with all features. Rejected because it enables shortcut learning via missingness proxies (confounding), falsely inflating model performance.

---

## ADR-011: Hyperparameter Tuning and Threshold Strategy

**Decision:**
1. **Metric:** We tune models using RandomizedSearchCV scoring strictly on **ROC-AUC**, not Recall or Precision.
2. **Class Weights:** Set `class_weight=None` (no class weighting) for all models, including XGBoost (`scale_pos_weight=1`).

**Why:**
Tuning on a hard-classification metric like Recall or F-score at a default threshold of 0.5 is inherently unstable, especially when base rates differ or when we care about specific operating points. ROC-AUC evaluates the model's ability to rank patients correctly regardless of the threshold. Because our classes are reasonably balanced (~41/59) in the primary training set, class weighting is unnecessary. We will handle class imbalance and maximize Recall by explicitly picking an optimal threshold *after* the model is trained, rather than forcing the model to distort its probabilities during training.

---

## ADR-012: The 1-Standard Error (1-SE) Selection Rule

**Decision:**
When selecting the final hyperparameter configuration from cross-validation, we will use the 1-SE rule: among all configurations whose mean AUC is within one standard error of the absolute best mean AUC, pick the **simplest** configuration.

Complexity ordering (from simplest to most complex):
- **Logistic Regression:** Ascending `C` (higher regularization).
- **Random Forest:** Ascending `max_depth`, then Descending `min_samples_leaf`, then Ascending `n_estimators`.
- **XGBoost:** Ascending `max_depth`, then Descending `min_child_weight`, then Descending `reg_lambda` (higher = more regularized = simpler), then Ascending `n_estimators`. (Subsample/colsample are ignored for complexity ordering).

**Why:**
CV scores have variance (especially with overlapping repeated-CV folds). Picking the absolute highest AUC often selects over-complex models that fit to noise. The 1-SE rule formally defines the noise threshold and biases us toward simpler, more robust models that will generalize better to the external sets.
