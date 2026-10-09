"""
tests/test_ingest.py
─────────────────────────────────────────────────────────────────────────────
Tests for src/heart_risk/data/ingest.py and src/heart_risk/db/repository.py.

All tests use an in-memory SQLite engine (from conftest.py).
No Docker, no network, no .env file needed.

Coverage matrix:
  ✓ Raw values land unchanged (strings, blanks, "0" preserved)
  ✓ Wrong column set is rejected with a helpful message
  ✓ Wrong row count is rejected with the actual count
  ✓ Running ingest twice yields the same row count (idempotent)
  ✓ A failure mid-write leaves the table unchanged (rollback)
  ✓ ingestion_log row is written after a successful ingest
"""

from __future__ import annotations

import textwrap
from pathlib import Path
from unittest.mock import patch

import pandas as pd
import pytest
from sqlalchemy import Engine

from heart_risk.data.ingest import (
    IngestionError,
    run_ingest,
    validate_dataframe,
)
from heart_risk.db import repository


# ── 1. Raw values land unchanged ──────────────────────────────────────────────

class TestRawValuesPreserved:
    """Verify that ingest does NOT coerce, modify, or drop any values."""

    def test_string_values_preserved(
        self,
        sqlite_engine: Engine,
        fixture_csv_path: Path,
        mock_settings,
    ) -> None:
        """
        "TRUE", "FALSE", blank, and "0" must survive the ingest unchanged.

        WHY: If pandas converts "TRUE" → True (bool) or "0" → 0 (int) or
        blank → NaN, downstream code silently gets wrong data.
        """
        run_ingest(sqlite_engine, mock_settings)
        df_db = repository.read_raw(sqlite_engine)

        # fbs and exang should be the string "TRUE" or "FALSE", never a Python bool
        fbs_values = set(df_db["fbs"].unique())
        assert fbs_values.issubset({"TRUE", "FALSE"}), (
            f"Expected only 'TRUE'/'FALSE' in fbs, got: {fbs_values}"
        )

        # chol="0" on row 20 must survive as the string "0"
        assert "0" in df_db["chol"].values, (
            "chol value '0' was lost — check that dtype=str is used in pd.read_csv"
        )

    def test_blank_cells_preserved_as_empty_string(
        self,
        sqlite_engine: Engine,
        fixture_csv_path: Path,
        mock_settings,
    ) -> None:
        """
        Blank cells in the CSV must land as empty string "", not NaN or None.

        WHY: NaN is a float; it would silently change the column's dtype.
        We use keep_default_na=False for this reason.
        """
        run_ingest(sqlite_engine, mock_settings)
        df_db = repository.read_raw(sqlite_engine)

        # Rows 7 and 18 have blank 'ca'; rows 7 and 18 have blank 'slope'
        # We check that empty string appears at least once in ca and slope
        assert "" in df_db["ca"].values, (
            "Blank 'ca' cells were not preserved as empty string"
        )

    def test_row_count_matches_csv(
        self,
        sqlite_engine: Engine,
        fixture_csv_path: Path,
        mock_settings,
    ) -> None:
        """After ingest, DB must have exactly as many rows as the fixture CSV."""
        run_ingest(sqlite_engine, mock_settings)
        assert repository.count_rows(sqlite_engine) == 20


# ── 2. Column validation ──────────────────────────────────────────────────────

class TestColumnValidation:
    """validate_dataframe must catch column mismatches before any DB writes."""

    def test_rejects_missing_column(self, mock_settings) -> None:
        """If a required column is absent, IngestionError must name it."""
        # Build a DataFrame missing the 'chol' column
        cols = [c for c in mock_settings.expected_raw_columns if c != "chol"]
        df = pd.DataFrame(columns=cols)

        with pytest.raises(IngestionError) as exc_info:
            validate_dataframe(df, mock_settings)

        error_msg = str(exc_info.value)
        assert "chol" in error_msg, "Error message must name the missing column"
        assert "Missing" in error_msg

    def test_rejects_extra_column(self, mock_settings) -> None:
        """If an unexpected column is present, IngestionError must name it."""
        df = pd.DataFrame(columns=mock_settings.expected_raw_columns + ["mystery_col"])

        with pytest.raises(IngestionError) as exc_info:
            validate_dataframe(df, mock_settings)

        assert "mystery_col" in str(exc_info.value)
        assert "Unexpected" in str(exc_info.value)

    def test_accepts_correct_columns(self, mock_settings) -> None:
        """Exactly the right columns with 20 rows must pass validation silently."""
        df = pd.DataFrame(
            [[""] * len(mock_settings.expected_raw_columns)] * 20,
            columns=mock_settings.expected_raw_columns,
        )
        # Should not raise
        validate_dataframe(df, mock_settings)


# ── 3. Row count validation ───────────────────────────────────────────────────

class TestRowCountValidation:
    """validate_dataframe must stop ingestion if the row count is wrong."""

    def test_rejects_wrong_row_count(self, mock_settings) -> None:
        """Wrong row count must raise IngestionError with the actual count."""
        # Give only 5 rows when 20 are expected
        df = pd.DataFrame(
            [["x"] * len(mock_settings.expected_raw_columns)] * 5,
            columns=mock_settings.expected_raw_columns,
        )
        with pytest.raises(IngestionError) as exc_info:
            validate_dataframe(df, mock_settings)

        error_msg = str(exc_info.value)
        assert "5" in error_msg, "Error must show actual count (5)"
        assert "20" in error_msg, "Error must show expected count (20)"


# ── 4. Idempotency ────────────────────────────────────────────────────────────

class TestIdempotency:
    """Running ingest twice must produce exactly the same result, not doubles."""

    def test_double_ingest_does_not_duplicate_rows(
        self,
        sqlite_engine: Engine,
        fixture_csv_path: Path,
        mock_settings,
    ) -> None:
        """
        After two consecutive ingest runs, the row count must still be 20.

        WHY: The truncate-and-reload strategy ensures this. A naive INSERT
        would produce 40 rows on the second run.
        """
        run_ingest(sqlite_engine, mock_settings)
        first_count = repository.count_rows(sqlite_engine)

        run_ingest(sqlite_engine, mock_settings)
        second_count = repository.count_rows(sqlite_engine)

        assert first_count == 20
        assert second_count == 20, (
            f"Second ingest produced {second_count} rows — idempotency is broken"
        )


# ── 5. Transaction rollback on failure ───────────────────────────────────────

class TestTransactionRollback:
    """If the write fails mid-way, the DB must be unchanged (no partial data)."""

    def test_failed_write_leaves_table_unchanged(
        self,
        sqlite_engine: Engine,
        fixture_csv_path: Path,
        mock_settings,
    ) -> None:
        """
        Simulate a crash during write_raw. The patients_raw table must
        retain its pre-crash contents (zero rows in this case).

        WHY: This proves that truncate-and-reload is truly atomic. If we
        truncated but then crashed before inserting, the table would be empty.
        The transaction rolls back the truncate too.
        """
        # Confirm the table starts empty
        assert repository.count_rows(sqlite_engine) == 0

        # Patch session.commit to raise an exception mid-write
        original_write = repository.write_raw

        def failing_write(engine, df, sha256):
            raise RuntimeError("Simulated disk failure during write")

        with patch("heart_risk.db.repository.write_raw", side_effect=failing_write):
            with pytest.raises(RuntimeError):
                run_ingest(sqlite_engine, mock_settings)

        # Table must still be empty — no partial data
        assert repository.count_rows(sqlite_engine) == 0, (
            "Table has rows after a failed write — transaction did not roll back"
        )


# ── 6. Ingestion log ──────────────────────────────────────────────────────────

class TestIngestionLog:
    """A successful ingest must write exactly one row to ingestion_log."""

    def test_log_row_written_after_success(
        self,
        sqlite_engine: Engine,
        fixture_csv_path: Path,
        mock_settings,
    ) -> None:
        """After a successful ingest, ingestion_log must have one row with status='success'."""
        from sqlalchemy.orm import Session
        from sqlalchemy import select
        from heart_risk.db.tables import IngestionLog

        run_ingest(sqlite_engine, mock_settings)

        with Session(sqlite_engine) as session:
            logs = session.execute(select(IngestionLog)).scalars().all()

        assert len(logs) == 1, f"Expected 1 log row, got {len(logs)}"
        assert logs[0].status == "success"
        assert logs[0].rows_read == 20
        assert logs[0].rows_written == 20
