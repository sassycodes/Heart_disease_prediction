"""
tests/test_config.py
─────────────────────────────────────────────────────────────────────────────
Tests for src/heart_risk/config.py.

Focus: the config module must fail loudly with a clear error when required
environment variables are missing. This prevents silent misconfigurations.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from heart_risk.config import get_settings, _require_env


class TestRequireEnv:
    """Tests for the _require_env helper."""

    def test_returns_value_when_set(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """If the env var is set, _require_env returns its value."""
        monkeypatch.setenv("TEST_VAR_XYZ", "hello")
        assert _require_env("TEST_VAR_XYZ") == "hello"

    def test_raises_when_missing(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """If the env var is not set, _require_env must raise EnvironmentError."""
        monkeypatch.delenv("TEST_VAR_XYZ", raising=False)
        with pytest.raises(EnvironmentError) as exc_info:
            _require_env("TEST_VAR_XYZ")
        # The error message must name the missing variable so the user knows what to fix
        assert "TEST_VAR_XYZ" in str(exc_info.value)
        assert ".env" in str(exc_info.value)  # points user to the fix


class TestGetSettings:
    """Tests for the get_settings() function."""

    def test_raises_when_postgres_user_missing(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """
        get_settings() must raise EnvironmentError if POSTGRES_USER is absent.

        WHY: A missing credential should never silently default to anything.
        The function must stop and tell the user exactly what is missing.
        """
        # Ensure POSTGRES_USER is NOT in the environment
        monkeypatch.delenv("POSTGRES_USER", raising=False)

        # Write a minimal valid YAML to a temp path
        yaml_content = """
project:
  name: test
  version: "0.0.0"
reproducibility:
  random_seed: 42
paths:
  raw_csv: data/raw/test.csv
  reports_dir: artifacts/reports
database:
  raw_table: patients_raw
  log_table: ingestion_log
validation:
  expected_raw_columns: [id]
  expected_row_count: 1
"""
        yaml_file = tmp_path / "project.yaml"
        yaml_file.write_text(yaml_content, encoding="utf-8")

        # Non-existent .env so dotenv doesn't accidentally pick up real creds
        env_file = tmp_path / ".env"

        with pytest.raises(EnvironmentError) as exc_info:
            get_settings(env_file=env_file, yaml_file=yaml_file)

        # Error message must name the missing variable
        assert "POSTGRES_USER" in str(exc_info.value)

    def test_raises_when_yaml_missing(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """get_settings() must raise FileNotFoundError if the YAML config is absent."""
        nonexistent = tmp_path / "no_such_file.yaml"
        env_file = tmp_path / ".env"
        with pytest.raises(FileNotFoundError):
            get_settings(env_file=env_file, yaml_file=nonexistent)
