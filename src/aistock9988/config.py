"""Centralized runtime configuration.

Secrets are supplied by environment variables.  This module deliberately
does not write, log, or serialize credentials.
"""
from __future__ import annotations

import os
from dataclasses import dataclass


def _value(name: str, default: str | None = None) -> str | None:
    value = os.getenv(name)
    return default if value is None or value == "" else value


@dataclass(frozen=True)
class MySQLConfig:
    host: str | None
    port: int
    user: str | None
    password: str
    database: str | None


@dataclass(frozen=True)
class RedisConfig:
    host: str | None
    port: int
    database: int
    password: str


@dataclass(frozen=True)
class TushareConfig:
    token: str
    base_url: str


class RuntimeConfig:
    """One immutable snapshot of process configuration."""

    def __init__(self) -> None:
        self.mysql = MySQLConfig(
            host=_value("AISTOCK_DB_HOST"),
            port=int(_value("AISTOCK_DB_PORT", "3306")),
            user=_value("AISTOCK_DB_USER"),
            password=_value("AISTOCK_DB_PASSWORD", "") or "",
            database=_value("AISTOCK_DB_NAME"),
        )
        self.redis = RedisConfig(
            host=_value("AISTOCK_REDIS_HOST"),
            port=int(_value("AISTOCK_REDIS_PORT", "6379")),
            database=int(_value("AISTOCK_REDIS_DB", "0")),
            password=_value("AISTOCK_REDIS_PASSWORD", "") or "",
        )
        self.tushare = TushareConfig(
            token=_value("TUSHARE_TOKEN", "") or "",
            base_url=_value("TUSHARE_BASE_URL", "") or "",
        )


_runtime: RuntimeConfig | None = None


def get_runtime_config() -> RuntimeConfig:
    """Return the process-wide immutable configuration snapshot."""
    global _runtime
    if _runtime is None:
        _runtime = RuntimeConfig()
    return _runtime


__all__ = ["MySQLConfig", "RedisConfig", "TushareConfig", "RuntimeConfig", "get_runtime_config"]
