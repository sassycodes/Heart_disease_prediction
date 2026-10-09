"""
src/heart_risk/db/engine.py
─────────────────────────────────────────────────────────────────────────────
Builds and caches the SQLAlchemy engine.

WHY a single engine factory?
  Opening a database connection is expensive (TCP handshake, auth).
  SQLAlchemy engines maintain a *connection pool* — a set of reusable
  connections. Creating multiple engines wastes that pool. One factory,
  called once, keeps everything efficient.

WHY pool_pre_ping=True?
  If the DB container restarts while our app is running, pooled connections
  go stale. pre_ping issues a cheap "SELECT 1" before lending out a
  connection; it silently replaces dead connections instead of crashing.
"""

from __future__ import annotations

from sqlalchemy import Engine, create_engine, text

from heart_risk.config import Settings


def build_engine(settings: Settings) -> Engine:
    """
    Create a SQLAlchemy engine from project settings.

    Args:
        settings: Loaded Settings object from config.get_settings().

    Returns:
        A SQLAlchemy Engine connected to the configured PostgreSQL database.

    Notes:
        - Credentials come from settings.database; they are never logged.
        - pool_pre_ping recycles stale connections automatically.
    """
    # WHY echo=False: echo=True prints every SQL statement including values.
    # That would leak data; we keep it off in production code.
    engine = create_engine(
        settings.database.url,
        pool_pre_ping=True,
        echo=False,
    )
    return engine


def check_connection(engine: Engine) -> bool:
    """
    Verify the engine can reach the database by running SELECT 1.

    Args:
        engine: A SQLAlchemy Engine.

    Returns:
        True if the connection succeeds, False otherwise.
    """
    try:
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        return True
    except Exception:
        return False
