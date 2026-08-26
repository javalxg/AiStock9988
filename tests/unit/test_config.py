from aistock9988.config import RuntimeConfig


def test_runtime_config_reads_mysql_env(monkeypatch):
    monkeypatch.setenv("AISTOCK_DB_HOST", "db.example")
    monkeypatch.setenv("AISTOCK_DB_PORT", "3307")
    monkeypatch.setenv("AISTOCK_DB_USER", "alice")
    monkeypatch.setenv("AISTOCK_DB_PASSWORD", "secret")
    monkeypatch.setenv("AISTOCK_DB_NAME", "quant_test")

    cfg = RuntimeConfig()
    assert cfg.mysql.host == "db.example"
    assert cfg.mysql.port == 3307
    assert cfg.mysql.user == "alice"
    assert cfg.mysql.database == "quant_test"
    assert cfg.mysql.password == "secret"
