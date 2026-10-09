"""
src/heart_risk/db/repository.py
─────────────────────────────────────────────────────────────────────────────
All database reads and writes for this project live here. No other module
may issue SQL or ORM calls directly.

WHY a repository layer?
  • It creates one stable interface between "business logic" (ingest.py)
    and "storage" (Postgres or SQLite).
  • Swapping Postgres for SQLite in tests requires changing only the engine
    passed in — no logic changes anywhere else.
  • If we later migrate to a different ORM or a REST API, only this file
    needs to change.
  • It also prevents SQL strings from being scattered across the codebase,
    which makes security audits and refactors much easier.

All functions accept an Engine as their first argument so the caller
controls which database is used. This is called *dependency injection*.
"""

from __future__ import annotations

import datetime
from typing import List

import pandas as pd
from sqlalchemy import Engine, func, select, text
from sqlalchemy.orm import Session

from heart_risk.db.tables import IngestionLog, PatientRaw, PatientClean


# ── Schema creation ───────────────────────────────────────────────────────────

def create_tables(engine: Engine) -> None:
    """
    Create all tables if they do not already exist.

    WHY: When running tests with SQLite, there is no Docker container to
    initialise the schema. This function lets us create tables on-the-fly
    from the ORM definitions in tables.py.

    Args:
        engine: SQLAlchemy Engine (Postgres or SQLite).
    """
    from heart_risk.db.tables import Base
    Base.metadata.create_all(bind=engine)


# ── Raw patient data ──────────────────────────────────────────────────────────

def write_raw(
    engine: Engine,
    df: pd.DataFrame,
    sha256: str,
) -> int:
    """
    Truncate `patients_raw` and reload it from a DataFrame in one transaction.

    WHY truncate-and-reload?
      Re-running ingest should always produce the same final state (idempotent).
      Truncating first prevents duplicate rows from accumulating. Wrapping
      both the truncate and the insert in one transaction means that if
      anything fails mid-way, the database rolls back to its previous state —
      no partial data lands.

    Args:
        engine: SQLAlchemy Engine.
        df:     DataFrame with columns matching patients_raw (CSV column names).
        sha256: SHA-256 hex digest of the source CSV file.

    Returns:
        Number of rows written.

    Raises:
        Exception: Any DB error triggers a rollback; the table is unchanged.
    """
    now = datetime.datetime.now(tz=datetime.timezone.utc)

    # Add metadata columns that are not in the CSV
    df = df.copy()
    df["ingested_at"] = now
    df["source_file_sha256"] = sha256

    with Session(engine) as session:
        # TRUNCATE clears all rows instantly; safer than DELETE for bulk reload
        # WHY inside the same transaction: if the INSERT fails, the TRUNCATE
        # also rolls back, leaving the table intact.
        session.execute(text(f"DELETE FROM {PatientRaw.__tablename__}"))

        # Build ORM objects from each DataFrame row
        rows = [
            PatientRaw(**row.to_dict())
            for _, row in df.iterrows()
        ]
        session.add_all(rows)
        session.commit()

    return len(rows)


def read_raw(engine: Engine) -> pd.DataFrame:
    """
    Read all rows from `patients_raw` into a DataFrame.

    Args:
        engine: SQLAlchemy Engine.

    Returns:
        DataFrame with all columns including metadata columns.
    """
    with Session(engine) as session:
        results = session.execute(select(PatientRaw)).scalars().all()
        if not results:
            return pd.DataFrame()

        # Convert ORM objects to dicts, then to DataFrame
        data = [
            {
                col.name: getattr(row, col.name)
                for col in PatientRaw.__table__.columns
            }
            for row in results
        ]
    return pd.DataFrame(data)


def count_rows(engine: Engine) -> int:
    """
    Return the number of rows in `patients_raw`.

    Args:
        engine: SQLAlchemy Engine.

    Returns:
        Row count as an integer.
    """
    with Session(engine) as session:
        result = session.execute(
            select(func.count()).select_from(PatientRaw)
        )
        return result.scalar_one()


# ── Ingestion audit log ───────────────────────────────────────────────────────

def log_ingestion(
    engine: Engine,
    file_name: str,
    file_sha256: str,
    rows_read: int,
    rows_written: int,
    status: str,
) -> None:
    """
    Write one row to `ingestion_log` recording the outcome of an ingest run.

    Args:
        engine:       SQLAlchemy Engine.
        file_name:    Base name of the CSV file ingested.
        file_sha256:  SHA-256 hex digest of the file.
        rows_read:    Number of rows read from the CSV.
        rows_written: Number of rows successfully written to the DB.
        status:       "success" or "failure".
    """
    now = datetime.datetime.now(tz=datetime.timezone.utc)
    with Session(engine) as session:
        entry = IngestionLog(
            run_at=now,
            file_name=file_name,
            file_sha256=file_sha256,
            rows_read=rows_read,
            rows_written=rows_written,
            status=status,
        )
        session.add(entry)
        session.commit()

# ── Clean patient data ────────────────────────────────────────────────────────

def write_clean(engine: Engine, df: pd.DataFrame) -> None:
    """
    Writes the cleaned dataframe to the `patients_clean` table.
    Overwrites the table entirely since this is a reproducible pipeline.
    """
    with Session(engine) as session:
        try:
            # Idempotency: truncate before writing
            session.execute(text(f"DELETE FROM {PatientClean.__tablename__}"))
            
            # Write records
            records = df.to_dict(orient="records")
            # Replace NaNs with None for SQLAlchemy
            for r in records:
                for k, v in r.items():
                    if pd.isna(v):
                        r[k] = None
                        
            session.bulk_insert_mappings(PatientClean, records)
            
            session.commit()
        except Exception as e:
            session.rollback()
            raise RuntimeError(f"Failed to write clean data: {e}") from e
