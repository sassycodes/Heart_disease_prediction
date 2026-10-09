"""
src/heart_risk/config.py
─────────────────────────────────────────────────────────────────────────────
Single source of truth for all project configuration.

WHY a dedicated config module?
  If every file loads its own settings, they can drift out of sync.
  Centralising here means one fix propagates everywhere.

HOW it works:
  1. python-dotenv loads `.env` into os.environ (if the file exists).
  2. PyYAML parses `configs/project.yaml`.
  3. We combine both into a typed dataclass: `Settings`.
  4. Any missing required value raises a clear error immediately —
     so you find out at startup, not mid-pipeline.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import List

import yaml
from dotenv import load_dotenv


# ── Locate the project root reliably ─────────────────────────────────────────
# __file__ = src/heart_risk/config.py
# .parent.parent.parent = project root (three levels up)
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent


@dataclass(frozen=True)
class DatabaseSettings:
    """Connection details for PostgreSQL, loaded from the .env file."""

    user: str
    password: str
    db: str
    host: str
    port: int

    @property
    def url(self) -> str:
        """
        Build a SQLAlchemy connection URL.

        WHY we never log this string: it contains the password.
        Format: postgresql+psycopg2://user:password@host:port/db
        """
        return (
            f"postgresql+psycopg2://{self.user}:{self.password}"
            f"@{self.host}:{self.port}/{self.db}"
        )

    @property
    def safe_url(self) -> str:
        """A password-masked URL safe to print in logs."""
        return (
            f"postgresql+psycopg2://{self.user}:***"
            f"@{self.host}:{self.port}/{self.db}"
        )


@dataclass(frozen=True)
class Settings:
    """
    All project settings in one place.

    Usage:
        from heart_risk.config import get_settings
        cfg = get_settings()
        print(cfg.raw_csv_path)
    """

    # Database connection
    database: DatabaseSettings

    # Reproducibility
    random_seed: int

    # File paths
    raw_csv_path: Path
    reports_dir: Path

    # Table names
    raw_table: str
    log_table: str

    # Validation
    expected_raw_columns: List[str]
    expected_row_count: int


def _require_env(key: str) -> str:
    """
    Read an environment variable and raise a clear error if it is missing.

    WHY: Silent defaults for credentials are dangerous. We prefer to crash
    loudly with an actionable message rather than connect to the wrong DB.
    """
    value = os.environ.get(key)
    if value is None:
        raise EnvironmentError(
            f"Required environment variable '{key}' is not set.\n"
            f"  → Copy .env.example to .env and fill in the value."
        )
    return value


def get_settings(
    env_file: Path | None = None,
    yaml_file: Path | None = None,
) -> Settings:
    """
    Load and return all project settings.

    Args:
        env_file:  Path to the .env file. Defaults to PROJECT_ROOT/.env.
        yaml_file: Path to the YAML config. Defaults to PROJECT_ROOT/configs/project.yaml.

    Returns:
        A frozen Settings dataclass.

    Raises:
        EnvironmentError: If a required environment variable is missing.
        FileNotFoundError: If the YAML config file does not exist.
    """
    # Step 1 — Load .env into os.environ (does nothing if file is absent,
    # which is fine in CI where env vars are set directly).
    env_path = env_file or (PROJECT_ROOT / ".env")
    load_dotenv(dotenv_path=env_path, override=False)

    # Step 2 — Parse the YAML config.
    yaml_path = yaml_file or (PROJECT_ROOT / "configs" / "project.yaml")
    if not yaml_path.exists():
        raise FileNotFoundError(
            f"Project config not found at: {yaml_path}\n"
            f"  → Ensure configs/project.yaml exists in the project root."
        )
    with yaml_path.open("r", encoding="utf-8") as fh:
        raw_yaml = yaml.safe_load(fh)

    # Step 3 — Pull database credentials from env (secrets never in YAML).
    db = DatabaseSettings(
        user=_require_env("POSTGRES_USER"),
        password=_require_env("POSTGRES_PASSWORD"),
        db=_require_env("POSTGRES_DB"),
        host=_require_env("POSTGRES_HOST"),
        port=int(_require_env("POSTGRES_PORT")),
    )

    # Step 4 — Extract YAML values and build the Settings object.
    paths_cfg = raw_yaml["paths"]
    db_cfg = raw_yaml["database"]
    val_cfg = raw_yaml["validation"]
    repro_cfg = raw_yaml["reproducibility"]

    return Settings(
        database=db,
        random_seed=int(repro_cfg["random_seed"]),
        raw_csv_path=PROJECT_ROOT / paths_cfg["raw_csv"],
        reports_dir=PROJECT_ROOT / paths_cfg["reports_dir"],
        raw_table=db_cfg["raw_table"],
        log_table=db_cfg["log_table"],
        expected_raw_columns=list(val_cfg["expected_raw_columns"]),
        expected_row_count=int(val_cfg["expected_row_count"]),
    )
