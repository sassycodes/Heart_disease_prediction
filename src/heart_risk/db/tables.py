"""
src/heart_risk/db/tables.py
─────────────────────────────────────────────────────────────────────────────
SQLAlchemy table definitions that mirror sql/schema.sql.

WHY define tables in Python AND in SQL?
  • sql/schema.sql is used by Docker to initialise the real database.
  • These Python definitions let SQLAlchemy understand the table structure
    so it can build INSERT statements, type-check columns, and support
    SQLite (for tests) without raw SQL strings scattered around the codebase.
  • The two definitions MUST stay in sync. If you add a column to one,
    add it to the other.

WHY all String (TEXT) for CSV columns?
  See AGENTS.md rule 1 and docs/decisions.md. Short answer: raw data
  contains mixed types, booleans as text, and blank strings. TEXT is the
  only type that accepts everything without silent coercion.
"""

from __future__ import annotations

from sqlalchemy import Column, Integer, String, Text, DateTime
from sqlalchemy.orm import DeclarativeBase


class Base(DeclarativeBase):
    """Base class for all SQLAlchemy ORM models in this project."""
    pass


class PatientRaw(Base):
    """
    ORM model for the `patients_raw` table.

    Each row is one record from the raw CSV, landed exactly as-is.
    The only additions are metadata columns added by the ingest step.
    """

    __tablename__ = "patients_raw"

    # ── Surrogate primary key (not from the CSV) ──────────────────────────────
    row_id = Column(Integer, primary_key=True, autoincrement=True)

    # ── Original CSV columns (all Text — no type coercion) ───────────────────
    id       = Column(Text)   # patient identifier from the source
    age      = Column(Text)
    sex      = Column(Text)
    dataset  = Column(Text)   # hospital source
    cp       = Column(Text)   # chest pain type
    trestbps = Column(Text)   # resting blood pressure
    chol     = Column(Text)   # serum cholesterol
    fbs      = Column(Text)   # fasting blood sugar ("TRUE"/"FALSE")
    restecg  = Column(Text)   # resting ECG
    thalch   = Column(Text)   # max heart rate
    exang    = Column(Text)   # exercise-induced angina ("TRUE"/"FALSE")
    oldpeak  = Column(Text)   # ST depression
    slope    = Column(Text)
    ca       = Column(Text)   # major vessels coloured by fluoroscopy
    thal     = Column(Text)   # thalassemia
    num      = Column(Text)   # diagnosis target (0 = no disease)

    # ── Ingest metadata ───────────────────────────────────────────────────────
    ingested_at        = Column(DateTime(timezone=True))
    source_file_sha256 = Column(Text, nullable=False)

    def __repr__(self) -> str:
        return f"<PatientRaw row_id={self.row_id} id={self.id}>"


class IngestionLog(Base):
    """
    ORM model for the `ingestion_log` table.

    One row per ingest run. Lets us audit: when was data loaded,
    from which file, and did it succeed?
    """

    __tablename__ = "ingestion_log"

    log_id       = Column(Integer, primary_key=True, autoincrement=True)
    run_at       = Column(DateTime(timezone=True))
    file_name    = Column(Text, nullable=False)
    file_sha256  = Column(Text, nullable=False)
    rows_read    = Column(Integer, nullable=False)
    rows_written = Column(Integer, nullable=False)
    status       = Column(String(32), nullable=False)  # "success" or "failure"

    def __repr__(self) -> str:
        return f"<IngestionLog log_id={self.log_id} status={self.status}>"
