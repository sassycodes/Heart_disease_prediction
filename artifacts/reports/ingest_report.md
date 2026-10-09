# Ingest Report

**Timestamp (UTC):** 2026-10-09T18:55:44.568353+00:00
**Source file:** `heart_disease_uci.csv`
**SHA-256:** `574f2fa2b43012fa25fca4fcdb36bd7c6bccdd0af4242f6c0e4c8633c5a072ae`
**Rows read from CSV:** 920
**Rows written to DB:** 920

## Columns

| Column | Empty-string cells |
|--------|-------------------|
| `id` | 0 |
| `age` | 0 |
| `sex` | 0 |
| `dataset` | 0 |
| `cp` | 0 |
| `trestbps` | 59 |
| `chol` | 30 |
| `fbs` | 90 |
| `restecg` | 2 |
| `thalch` | 55 |
| `exang` | 55 |
| `oldpeak` | 62 |
| `slope` | 309 |
| `ca` | 611 |
| `thal` | 486 |
| `num` | 0 |

## Notes

- Empty-string counts are **observations only**. No values were modified.
- All columns were ingested as plain text (no type coercion).
- This report was generated automatically by `src/heart_risk/data/ingest.py`.