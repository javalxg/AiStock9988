from __future__ import annotations

from contextlib import contextmanager

from aistock9988.config import get_runtime_config


def connection_kwargs() -> dict[str, object]:
    """Read connection settings without ever persisting the password."""
    mysql = get_runtime_config().mysql
    required = {"AISTOCK_DB_HOST": mysql.host, "AISTOCK_DB_USER": mysql.user,
                "AISTOCK_DB_NAME": mysql.database}
    missing = [name for name, value in required.items() if not value]
    if missing:
        raise RuntimeError(f"missing database configuration: {missing}")
    return {"host": mysql.host, "port": mysql.port, "user": mysql.user,
            "database": mysql.database, "password": mysql.password}


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
