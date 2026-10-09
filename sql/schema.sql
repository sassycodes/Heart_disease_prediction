-- sql/schema.sql
-- ─────────────────────────────────────────────────────────────────────────────
-- This file is run automatically by Docker on first container startup
-- (mounted at /docker-entrypoint-initdb.d/).
-- It can also be applied manually: psql -f sql/schema.sql
--
-- Design choices:
--   • All columns in patients_raw use TEXT type.
--     WHY: The raw CSV has mixed values — "TRUE"/"FALSE", blanks, "0" as
--     cholesterol, and other quirks. TEXT lets every value land exactly as-is
--     with no silent coercion. Cleaning happens in a later pipeline stage.
--   • We add metadata columns (row_id, ingested_at, source_file_sha256)
--     so we can always trace a row back to its ingest run.
-- ─────────────────────────────────────────────────────────────────────────────

-- ── Raw patient data (immutable after ingest) ─────────────────────────────────
CREATE TABLE IF NOT EXISTS patients_raw (
    -- ── Surrogate primary key (auto-increment; not from the CSV) ─────────────
    row_id              SERIAL PRIMARY KEY,

    -- ── Original CSV columns (all TEXT — no type coercion) ───────────────────
    id                  TEXT,
    age                 TEXT,
    sex                 TEXT,
    dataset             TEXT,   -- hospital source (Cleveland, Hungary, etc.)
    cp                  TEXT,   -- chest pain type
    trestbps            TEXT,   -- resting blood pressure
    chol                TEXT,   -- serum cholesterol (may be "0" for missing)
    fbs                 TEXT,   -- fasting blood sugar > 120 ("TRUE"/"FALSE")
    restecg             TEXT,   -- resting ECG results
    thalch              TEXT,   -- max heart rate achieved
    exang               TEXT,   -- exercise-induced angina ("TRUE"/"FALSE")
    oldpeak             TEXT,   -- ST depression induced by exercise
    slope               TEXT,   -- slope of peak exercise ST segment
    ca                  TEXT,   -- number of major vessels (0-3)
    thal                TEXT,   -- thalassemia type
    num                 TEXT,   -- target: diagnosis (0 = no disease, 1-4 = disease)

    -- ── Ingest metadata ───────────────────────────────────────────────────────
    ingested_at         TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    source_file_sha256  TEXT        NOT NULL  -- SHA-256 of the CSV file at ingest time
);

-- ── Ingestion audit log ───────────────────────────────────────────────────────
-- Every time `make ingest` runs, one row is written here.
-- This lets us answer: "When was this data loaded? From which file? Did it succeed?"
CREATE TABLE IF NOT EXISTS ingestion_log (
    log_id          SERIAL PRIMARY KEY,
    run_at          TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    file_name       TEXT        NOT NULL,
    file_sha256     TEXT        NOT NULL,
    rows_read       INTEGER     NOT NULL,
    rows_written    INTEGER     NOT NULL,
    status          TEXT        NOT NULL  -- "success" or "failure"
);
