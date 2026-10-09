# Phase 1: Raw Data Ingestion

**Goal:** Ingest the UCI Heart Disease CSV into a PostgreSQL table (`patients_raw`) with zero transformation, establish the project scaffold, and make the test suite green.

---

## Inputs

| Item | Location | Notes |
|------|----------|-------|
| Raw CSV | `data/raw/heart_disease_uci.csv` | Kaggle: `redwankarimsony/heart-disease-data` |
| Config | `configs/project.yaml` | Column names, row count, paths |
| Credentials | `.env` | Postgres connection; never committed |

## Outputs

| Item | Location | Description |
|------|----------|-------------|
| Raw table | `patients_raw` (Postgres) | 920 rows, all TEXT columns |
| Audit table | `ingestion_log` (Postgres) | One row per ingest run |
| Report | `artifacts/reports/ingest_report.md` | Hash, row counts, empty-cell stats |

---

## Constraints

- **Raw data is immutable.** Once ingested, `patients_raw` is never modified.
- **All values are strings.** No type coercion, imputation, or renaming during ingest.
- **Ingest is idempotent.** Running it twice must produce 920 rows, not 1840.
- **No credentials in any file** except `.env` (gitignored).
- **Tests must pass without Docker** using SQLite.

---

## Acceptance Criteria

| # | Criterion | How to verify |
|---|-----------|---------------|
| 1 | `make db-up && make db-check` succeeds | Exit code 0, prints ✓ |
| 2 | `make ingest` loads exactly 920 rows | Command output + `SELECT COUNT(*) FROM patients_raw` |
| 3 | Read-back from Postgres matches CSV | Compare sample values; no type coercion |
| 4 | Second `make ingest` leaves 920 rows, not 1840 | Run twice, query count |
| 5 | `make test` passes — no Docker needed | All pytest tests green |
| 6 | No credentials in repo, logs, or reports | `git grep` for passwords; `.env` is gitignored |
| 7 | No cleaning or transformation of values | `SELECT chol FROM patients_raw WHERE chol = '0'` returns rows |
| 8 | `ingest_report.md` exists with per-column empty-cell counts | Open `artifacts/reports/ingest_report.md` |
