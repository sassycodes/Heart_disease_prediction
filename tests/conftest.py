"""
tests/conftest.py
─────────────────────────────────────────────────────────────────────────────
Shared pytest fixtures used by all tests.

Key design choices:
  • An in-memory SQLite engine (not Postgres) is used for tests.
    WHY: Tests must run without Docker. SQLite is built into Python and
    requires no installation. Our repository layer is swappable, so the
    tests exercise the exact same logic.

  • A tiny 20-row CSV fixture covers corner cases:
    blanks, "TRUE"/"FALSE" booleans, "0" for cholesterol.
    WHY 20 rows and not the full 920?
    Tests should be fast and not depend on external files.
"""

from __future__ import annotations

import io
import os
import textwrap
from pathlib import Path

import pandas as pd
import pytest
from sqlalchemy import create_engine, Engine

from heart_risk.db.repository import create_tables


# ── Fixture CSV content ───────────────────────────────────────────────────────
# 20 rows with:
#   - blanks in several columns (to test empty-string preservation)
#   - "TRUE"/"FALSE" in fbs and exang columns (not Python booleans)
#   - "0" in chol (a real quirk of this dataset — must stay as "0", not 0)
#   - varied num values (0, 1, 2, 3, 4) for completeness

FIXTURE_CSV = textwrap.dedent("""\
    id,age,sex,dataset,cp,trestbps,chol,fbs,restecg,thalch,exang,oldpeak,slope,ca,thal,num
    1,63,Male,Cleveland,typical angina,145,233,TRUE,lv hypertrophy,150,FALSE,2.3,downsloping,0,fixed defect,0
    2,37,Male,Cleveland,non-anginal,130,250,FALSE,normal,187,FALSE,3.5,downsloping,0,normal,0
    3,41,Female,Cleveland,atypical angina,130,204,FALSE,lv hypertrophy,172,FALSE,1.4,upsloping,0,normal,0
    4,56,Male,Cleveland,atypical angina,120,236,FALSE,normal,178,FALSE,0.8,upsloping,0,normal,0
    5,57,Female,Cleveland,typical angina,120,354,FALSE,normal,163,TRUE,0.6,upsloping,0,normal,0
    6,57,Male,Cleveland,atypical angina,140,192,FALSE,normal,148,FALSE,0.4,flat,0,fixed defect,0
    7,56,Female,Cleveland,atypical angina,140,294,FALSE,lv hypertrophy,153,FALSE,1.3,flat,,reversable defect,0
    8,44,Male,Cleveland,atypical angina,120,263,FALSE,normal,173,FALSE,0,upsloping,,normal,0
    9,52,Male,Cleveland,non-anginal,172,199,TRUE,normal,162,FALSE,0.5,upsloping,0,normal,0
    10,57,Male,Cleveland,non-anginal,150,168,FALSE,normal,174,FALSE,1.6,flat,0,normal,0
    11,54,Male,Cleveland,atypical angina,140,239,FALSE,normal,160,FALSE,1.2,upsloping,0,normal,0
    12,48,Female,Cleveland,non-anginal,130,275,FALSE,normal,139,FALSE,0.2,upsloping,0,normal,0
    13,49,Male,Cleveland,atypical angina,130,266,FALSE,normal,171,FALSE,0.6,upsloping,0,normal,0
    14,64,Male,Cleveland,typical angina,110,211,FALSE,lv hypertrophy,144,TRUE,1.8,flat,0,normal,0
    15,58,Female,Cleveland,typical angina,150,283,TRUE,lv hypertrophy,162,FALSE,1,upsloping,0,normal,0
    16,58,Male,Cleveland,atypical angina,132,224,FALSE,lv hypertrophy,173,FALSE,3.2,upsloping,2,reversable defect,3
    17,60,Male,Cleveland,asymptomatic,130,206,FALSE,lv hypertrophy,132,TRUE,2.4,flat,2,reversable defect,2
    18,50,Female,Cleveland,non-anginal,120,219,FALSE,normal,158,FALSE,1.6,flat,,reversable defect,1
    19,58,Male,Cleveland,asymptomatic,112,230,FALSE,lv hypertrophy,160,FALSE,0,upsloping,1,reversable defect,1
    20,65,Female,Cleveland,atypical angina,150,0,FALSE,lv hypertrophy,187,FALSE,0,upsloping,0,normal,4
""")
# Note: row 8 and rows 7/18 have intentional blanks in 'ca' and 'slope'
# Note: row 20 has chol="0" — must survive as the string "0", not integer 0 or NaN


@pytest.fixture
def fixture_csv_path(tmp_path: Path) -> Path:
    """
    Write FIXTURE_CSV to a temp file and return its Path.

    tmp_path is a pytest built-in that gives each test a unique temp directory.
    """
    csv_file = tmp_path / "heart_disease_uci.csv"
    csv_file.write_text(FIXTURE_CSV, encoding="utf-8")
    return csv_file


@pytest.fixture
def sqlite_engine() -> Engine:
    """
    Create a fresh in-memory SQLite engine with the project schema applied.

    WHY in-memory?
      It is created and destroyed for each test run — no cleanup needed.
      No files written to disk. No Docker required.
    """
    engine = create_engine("sqlite:///:memory:")
    create_tables(engine)
    return engine


@pytest.fixture
def mock_settings(fixture_csv_path: Path, tmp_path: Path):
    """
    Return a Settings-like object configured for testing.

    WHY not call get_settings()?
      get_settings() requires .env variables for Postgres. Tests must run
      without any Postgres credentials. We build a minimal mock instead.

    Returns:
        A SimpleNamespace that has the same attributes as Settings.
    """
    from types import SimpleNamespace

    expected_columns = [
        "id", "age", "sex", "dataset", "cp", "trestbps", "chol", "fbs",
        "restecg", "thalch", "exang", "oldpeak", "slope", "ca", "thal", "num",
    ]
    return SimpleNamespace(
        raw_csv_path=fixture_csv_path,
        raw_table="patients_raw",
        log_table="ingestion_log",
        expected_raw_columns=expected_columns,
        expected_row_count=20,   # our fixture has 20 rows
        reports_dir=tmp_path / "reports",
        random_seed=42,
    )
