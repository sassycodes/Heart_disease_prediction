"""
src/heart_risk/data/ingest.py
─────────────────────────────────────────────────────────────────────────────
Reads the raw CSV and loads it into `patients_raw`, with strict validation.

Key design decisions (details in docs/decisions.md):
  1. All columns read as str/object  → nothing is silently coerced.
  2. Validate column names and row count BEFORE touching the database.
  3. SHA-256 computed ONCE from the file bytes; stored with every row.
  4. Truncate-and-reload inside a single transaction → idempotent + atomic.
  5. Post-write verification → read back and confirm row count.
  6. Write ingestion_log and a human-readable report for auditing.
"""

from __future__ import annotations

import datetime
import hashlib
import logging
from pathlib import Path
from typing import Dict

import pandas as pd
from sqlalchemy import Engine

from heart_risk.config import Settings
from heart_risk.db import repository

logger = logging.getLogger(__name__)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _sha256_of_file(path: Path) -> str:
    """
    Compute the SHA-256 hex digest of a file.

    WHY SHA-256?
      It gives us a fingerprint. If someone swaps the CSV with a different
      version, the hash will change. We store it so every row in the DB can
      be traced back to the exact file that produced it.

    Args:
        path: Path to the file.

    Returns:
        Lowercase hex string, e.g. "a3f2...".
    """
    h = hashlib.sha256()
    with path.open("rb") as fh:
        # Read in 64 KB chunks; avoids loading a huge file all at once
        for chunk in iter(lambda: fh.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def _count_empty_strings(df: pd.DataFrame) -> Dict[str, int]:
    """
    Count cells that are the empty string "" per column.

    WHY count but not fix?
      Phase 1 is raw ingest only. Blanks and empty strings are data; we report
      them so Phase 2 (cleaning) knows what it is dealing with, but we do not
      touch them here.

    Args:
        df: DataFrame to inspect.

    Returns:
        Dict mapping column name → number of empty-string cells.
    """
    return {col: int((df[col] == "").sum()) for col in df.columns}


# ── Validation ────────────────────────────────────────────────────────────────

class IngestionError(Exception):
    """Raised when the CSV fails validation. Ingest is aborted."""
    pass


def validate_dataframe(df: pd.DataFrame, settings: Settings) -> None:
    """
    Ensure the DataFrame has exactly the expected columns and row count.

    WHY fail loudly?
      The Kaggle mirror has multiple versions of this dataset. A silent
      mismatch would corrupt downstream models. We prefer a clear error over
      a quiet wrong answer.

    Args:
        df:       DataFrame loaded from the CSV.
        settings: Project settings with expected_raw_columns and expected_row_count.

    Raises:
        IngestionError: With a human-readable explanation of what is wrong.
    """
    expected = set(settings.expected_raw_columns)
    actual   = set(df.columns.tolist())

    extra   = actual - expected
    missing = expected - actual

    if extra or missing:
        msg_parts = ["Column mismatch detected in the CSV:"]
        if missing:
            msg_parts.append(f"  Missing columns : {sorted(missing)}")
        if extra:
            msg_parts.append(f"  Unexpected columns: {sorted(extra)}")
        msg_parts.append(
            f"\n  Expected exactly: {sorted(expected)}"
            f"\n  → Check that you have the correct dataset version."
        )
        raise IngestionError("\n".join(msg_parts))

    if len(df) != settings.expected_row_count:
        raise IngestionError(
            f"Row count mismatch:\n"
            f"  Expected : {settings.expected_row_count}\n"
            f"  Actual   : {len(df)}\n"
            f"  → The file may be a different version of the dataset. "
            f"Update expected_row_count in configs/project.yaml if intentional."
        )


# ── Report writer ─────────────────────────────────────────────────────────────

def _write_report(
    settings: Settings,
    file_path: Path,
    file_sha256: str,
    df: pd.DataFrame,
    rows_written: int,
    timestamp: datetime.datetime,
) -> Path:
    """
    Write a human-readable Markdown ingest report to artifacts/reports/.

    Args:
        settings:     Project settings (for reports_dir path).
        file_path:    Path to the ingested CSV.
        file_sha256:  SHA-256 of the file.
        df:           The DataFrame as loaded from the CSV.
        rows_written: Number of rows written to the database.
        timestamp:    UTC timestamp of this ingest run.

    Returns:
        Path to the written report file.
    """
    empty_counts = _count_empty_strings(df)
    report_dir = settings.reports_dir
    report_dir.mkdir(parents=True, exist_ok=True)
    report_path = report_dir / "ingest_report.md"

    lines = [
        "# Ingest Report",
        "",
        f"**Timestamp (UTC):** {timestamp.isoformat()}",
        f"**Source file:** `{file_path.name}`",
        f"**SHA-256:** `{file_sha256}`",
        f"**Rows read from CSV:** {len(df)}",
        f"**Rows written to DB:** {rows_written}",
        "",
        "## Columns",
        "",
        "| Column | Empty-string cells |",
        "|--------|-------------------|",
    ]
    for col in settings.expected_raw_columns:
        lines.append(f"| `{col}` | {empty_counts.get(col, 0)} |")

    lines += [
        "",
        "## Notes",
        "",
        "- Empty-string counts are **observations only**. No values were modified.",
        "- All columns were ingested as plain text (no type coercion).",
        "- This report was generated automatically by `src/heart_risk/data/ingest.py`.",
    ]

    report_path.write_text("\n".join(lines), encoding="utf-8")
    logger.info("Ingest report written to: %s", report_path)
    return report_path


# ── Main ingest function ──────────────────────────────────────────────────────

def run_ingest(engine: Engine, settings: Settings) -> None:
    """
    Full ingest pipeline: read → validate → write → verify → report.

    Steps:
      1. Read CSV with every column as string (dtype=str).
      2. Validate columns and row count.
      3. Compute SHA-256 of the file.
      4. Write to DB (truncate-and-reload, single transaction).
      5. Read back and verify row count.
      6. Write ingestion_log row.
      7. Write ingest_report.md.

    Args:
        engine:   SQLAlchemy Engine (Postgres in production, SQLite in tests).
        settings: Loaded project settings.

    Raises:
        IngestionError: If validation fails (column mismatch or wrong row count).
        FileNotFoundError: If the CSV does not exist at the configured path.
    """
    csv_path = settings.raw_csv_path
    timestamp = datetime.datetime.now(tz=datetime.timezone.utc)

    # ── Step 1: Check file exists ─────────────────────────────────────────────
    if not csv_path.exists():
        raise FileNotFoundError(
            f"CSV not found at: {csv_path}\n"
            f"  → Place the file there and re-run `make ingest`."
        )

    logger.info("Reading CSV: %s", csv_path.name)

    # ── Step 2: Read every column as a plain string ───────────────────────────
    # WHY dtype=str and keep_default_na=False?
    #   dtype=str tells pandas to never guess a column's type.
    #   keep_default_na=False stops pandas from silently converting "", "NA",
    #   "NULL", "None" etc. into float NaN — we want those values preserved
    #   exactly as they appear in the CSV.
    df = pd.read_csv(
        csv_path,
        dtype=str,
        keep_default_na=False,
    )
    logger.info("CSV rows read: %d", len(df))

    # ── Step 3: Validate ──────────────────────────────────────────────────────
    validate_dataframe(df, settings)
    logger.info("Validation passed. Columns and row count are correct.")

    # ── Step 4: Compute SHA-256 ───────────────────────────────────────────────
    file_sha256 = _sha256_of_file(csv_path)
    logger.info("File SHA-256: %s", file_sha256)

    # ── Step 5: Write to database (truncate-and-reload) ───────────────────────
    rows_written = repository.write_raw(engine, df, file_sha256)
    logger.info("Rows written to DB: %d", rows_written)

    # ── Step 6: Post-write verification ───────────────────────────────────────
    db_count = repository.count_rows(engine)
    if db_count != rows_written:
        raise IngestionError(
            f"Post-write verification failed!\n"
            f"  Rows written : {rows_written}\n"
            f"  Rows in DB   : {db_count}\n"
            f"  → Something went wrong during the write. Do not proceed."
        )
    logger.info("Verification passed. DB row count: %d", db_count)

    # ── Step 7: Write ingestion_log ───────────────────────────────────────────
    repository.log_ingestion(
        engine=engine,
        file_name=csv_path.name,
        file_sha256=file_sha256,
        rows_read=len(df),
        rows_written=rows_written,
        status="success",
    )

    # ── Step 8: Write human-readable report ───────────────────────────────────
    report_path = _write_report(
        settings=settings,
        file_path=csv_path,
        file_sha256=file_sha256,
        df=df,
        rows_written=rows_written,
        timestamp=timestamp,
    )

    print(f"[OK] Ingest complete: {rows_written} rows in `{settings.raw_table}`.")
    print(f"[OK] Report: {report_path}")
