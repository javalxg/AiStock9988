import os

import pytest

from aistock9988.data.quantdb import connection_kwargs


def test_quantdb_factory_does_not_require_or_expose_password(monkeypatch):
    for key, value in {"AISTOCK_DB_HOST": "127.0.0.1", "AISTOCK_DB_PORT": "3306",
                       "AISTOCK_DB_USER": "readonly", "AISTOCK_DB_NAME": "quant_db"}.items():
        monkeypatch.setenv(key, value)
    monkeypatch.delenv("AISTOCK_DB_PASSWORD", raising=False)
    args = connection_kwargs()
    assert args["host"] == "127.0.0.1" and args["port"] == 3306
    assert args["password"] == ""


def test_quantdb_factory_rejects_partial_configuration(monkeypatch):
    monkeypatch.delenv("AISTOCK_DB_HOST", raising=False)
    with pytest.raises(RuntimeError, match="AISTOCK_DB_HOST"):
        connection_kwargs()
