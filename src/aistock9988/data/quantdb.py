from __future__ import annotations

import os
from contextlib import contextmanager


def connection_kwargs() -> dict[str, object]:
    """Read connection settings without ever persisting the password."""
    required = {"host": "AISTOCK_DB_HOST", "port": "AISTOCK_DB_PORT", "user": "AISTOCK_DB_USER",
                "database": "AISTOCK_DB_NAME"}
    missing = [env for env in required.values() if not os.getenv(env)]
    if missing:
        raise RuntimeError(f"missing database environment variables: {missing}")
    return {key: os.environ[env] for key, env in required.items()} | {
        "port": int(os.environ[required["port"]]), "password": os.getenv("AISTOCK_DB_PASSWORD", "")
    }


@contextmanager
def readonly_connection():
    """Create a read-only MySQL connection; no schema changes are permitted."""
    try:
        import pymysql
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("install pymysql to connect to quant_db") from exc
    kwargs = connection_kwargs()
    conn = pymysql.connect(**kwargs, autocommit=False, read_timeout=120, write_timeout=120)
    try:
        with conn.cursor() as cursor:
            cursor.execute("SET SESSION TRANSACTION READ ONLY")
        yield conn
    finally:
        conn.rollback()
        conn.close()
