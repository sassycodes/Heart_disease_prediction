# AGENTS.md — Rulebook for All Future Agents

> Every agent, human or AI, working on this repo MUST follow these rules.
> Violation of any rule is grounds to stop and ask before proceeding.

---

## 1. Raw Data is Immutable

- `data/raw/` and the database table `patients_raw` are **read-only after ingest**.
- Never modify, delete, update, or overwrite any row or column in `patients_raw`.
- Never rename CSV columns or change values during ingest; the data must land **exactly** as it appears in the source file.
- All transformations happen downstream in separate tables/pipelines.

## 2. No Hardcoding — Everything Lives in Config

- Never hardcode: seeds, file paths, column names, thresholds, credentials, table names, or port numbers.
- All settings come from `configs/project.yaml` (non-secret) and `.env` (secrets).
- `src/heart_risk/config.py` is the single entry point for all config; import from there.

## 3. No Leakage — Fit Only on Training Data

- Every transformer (scaler, imputer, encoder) and every model must live inside a `sklearn.Pipeline`.
- Pipelines are **fitted on training data only**, never on the full dataset or test data.
- The test set is a black box until the final evaluation module runs — once, at the very end.

## 4. The Test Set is Sacred

- No EDA, no feature inspection, no model selection decisions may touch the test set.
- The test set is opened **once**, by `src/heart_risk/evaluate.py` (future phase), and never again.

## 5. Primary Metric is Recall

- Model selection and hyperparameter tuning use **recall** as the primary metric.
- Accuracy is never the selection metric; it is allowed in reports for context only.
- Reason: missing a sick patient (false negative) is clinically worse than a false alarm (false positive).

## 6. Phases are Sequential and Gate-Locked

- Each phase has explicit acceptance criteria in `docs/phases/`.
- A phase is not complete until ALL its criteria pass and `make test` is green.
- Never begin the next phase until the current one is gate-approved.

## 7. Code Must Be Readable and Explained

- Every non-trivial line or block must have a "why" comment, not just a "what" comment.
- Every function must have a docstring explaining its purpose, arguments, and return value.
- Prefer boring, explicit, readable code over clever one-liners.

## 8. Credentials are Never Exposed

- Never print, log, or write credentials to any file (including reports or test output).
- `.env` is gitignored. Use `.env.example` with placeholder values for documentation.
- Database URLs (which contain passwords) must never appear in logs.

## 9. When in Doubt, Ask

- If a specification is ambiguous, **stop and ask** rather than assuming.
- Assumptions that lead to incorrect behavior downstream are more expensive than a short pause.

## 10. Database Access via Repository Only

- All database reads and writes go through functions in `src/heart_risk/db/repository.py`.
- No raw SQL or ORM calls anywhere else in the codebase.
- This rule makes the DB layer swappable (e.g., Postgres → SQLite for tests).
