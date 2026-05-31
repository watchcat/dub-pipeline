import importlib
from src import config


def test_orchestrator_config_reads_env(monkeypatch):
    monkeypatch.setenv("DATABASE_URL", "postgres://x")
    monkeypatch.setenv("ORCH_CALLBACK_SECRET", "s3cr3t")
    monkeypatch.setenv("ORCH_BASE_URL", "https://orch.local")
    monkeypatch.setenv("BUZZBOT_RESULT_URL", "https://app/internal/dub_result")
    monkeypatch.setenv("BUZZBOT_TRANSCRIPT_URL", "https://app/internal/transcript_result")
    monkeypatch.setenv("CPU_TEXT_URL", "http://cpu-text/run")
    monkeypatch.setenv("CPU_MUX_URL", "http://cpu-mux/run")
    importlib.reload(config)
    assert config.DATABASE_URL == "postgres://x"
    assert config.ORCH_CALLBACK_SECRET == "s3cr3t"
    assert config.ORCH_BASE_URL == "https://orch.local"
    assert config.BUZZBOT_RESULT_URL.endswith("/dub_result")
    assert config.BUZZBOT_TRANSCRIPT_URL.endswith("/transcript_result")
    assert config.CPU_TEXT_URL == "http://cpu-text/run"
    assert config.CPU_MUX_URL == "http://cpu-mux/run"
    assert config.MAX_STEP_ATTEMPTS == 3
