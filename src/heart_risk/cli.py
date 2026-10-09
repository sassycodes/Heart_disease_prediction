"""
src/heart_risk/cli.py
─────────────────────────────────────────────────────────────────────────────
Command-line interface for the heart-risk project.

Usage:
  heart-risk db-check    → verify the database connection
  heart-risk ingest      → run the full ingest pipeline

WHY a CLI?
  Running `make ingest` is more readable and repeatable than typing a long
  Python command. Click provides argument parsing, error messages, and
  --help documentation for free.
"""

from __future__ import annotations

import logging
import sys

import click

from heart_risk.config import get_settings
from heart_risk.db.engine import build_engine, check_connection
from heart_risk.db.repository import create_tables
from heart_risk.data.ingest import run_ingest, IngestionError

# Configure basic logging so INFO messages appear in the terminal
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%S",
)


@click.group()
def cli() -> None:
    """Heart Disease Risk Prediction — Phase 1 CLI."""
    pass


@cli.command("db-check")
def db_check() -> None:
    """Verify that the database is reachable (runs SELECT 1)."""
    try:
        settings = get_settings()
        engine = build_engine(settings)
        ok = check_connection(engine)
    except Exception as exc:
        click.echo(f"[FAIL] db-check FAILED: {exc}", err=True)
        sys.exit(1)

    if ok:
        # Print safe URL (password masked) so user can confirm they hit the right DB
        click.echo(f"[OK] db-check OK - connected to {settings.database.safe_url}")
    else:
        click.echo("[FAIL] db-check FAILED - could not execute SELECT 1.", err=True)
        sys.exit(1)


@cli.command("ingest")
def ingest() -> None:
    """Read the raw CSV and load it into patients_raw."""
    try:
        settings = get_settings()
        engine = build_engine(settings)
        create_tables(engine)   # Ensure schema exists (no-op if already created)
        run_ingest(engine, settings)
    except FileNotFoundError as exc:
        click.echo(f"[FAIL] ingest FAILED - file not found:\n  {exc}", err=True)
        sys.exit(1)
    except IngestionError as exc:
        click.echo(f"[FAIL] ingest FAILED - validation error:\n  {exc}", err=True)
        sys.exit(1)
    except Exception as exc:
        click.echo(f"[FAIL] ingest FAILED - unexpected error:\n  {exc}", err=True)
        sys.exit(1)

@cli.command("eda")
def eda() -> None:
    """Run exploratory data analysis (EDA) profiling on patients_raw."""
    from heart_risk.data.profile import run_profiling
    try:
        settings = get_settings()
        engine = build_engine(settings)
        run_profiling(engine, settings)
        click.echo(f"[OK] EDA complete. Reports generated in {settings.reports_dir}")
    except Exception as exc:
        click.echo(f"[FAIL] eda FAILED - {exc}", err=True)
        sys.exit(1)

@cli.command("clean")
def clean() -> None:
    """Run Phase 3 data cleaning."""
    from heart_risk.data.clean import run_cleaning
    try:
        settings = get_settings()
        engine = build_engine(settings)
        create_tables(engine) # Ensure tables exist
        run_cleaning(settings)
    except Exception as exc:
        click.echo(f"[FAIL] clean FAILED - {exc}", err=True)
        sys.exit(1)

@cli.command("split")
def split() -> None:
    """Run Phase 4 data split."""
    from heart_risk.data.split import run_split
    try:
        settings = get_settings()
        run_split(settings)
    except Exception as exc:
        click.echo(f"[FAIL] split FAILED - {exc}", err=True)
        sys.exit(1)

@cli.command("train-cv")
def train_cv() -> None:
    """Run Phase 4 CV model comparison."""
    from heart_risk.models.train import run_training_cv
    try:
        settings = get_settings()
        run_training_cv(settings)
    except Exception as exc:
        click.echo(f"[FAIL] train-cv FAILED - {exc}", err=True)
        sys.exit(1)
