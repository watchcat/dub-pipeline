# tests/test_config_cloud.py
import importlib
from src import config

def test_cloud_config_reads_env(monkeypatch):
    for k in ["PROGRESS_URL","R2_ENDPOINT","R2_ACCESS_KEY_ID","R2_SECRET_ACCESS_KEY",
              "R2_BUCKET","R2_PUBLIC_URL","GEMINI_API_KEY","HF_TOKEN"]:
        monkeypatch.setenv(k, "x")
    monkeypatch.setenv("NEBIUS_API_KEY", "nk")
    monkeypatch.setenv("NEBIUS_PROJECT_ID", "proj")
    monkeypatch.setenv("GPU_IMAGE", "registry/gpu:1")
    monkeypatch.setenv("NEBIUS_PREP_PRESET", "1gpu-16vcpu-200gb")
    monkeypatch.setenv("NEBIUS_SYNTH_PRESET", "1gpu-8vcpu-100gb")
    monkeypatch.setenv("STEP_TIMEOUT_PREP", "1800")
    monkeypatch.setenv("STEP_TIMEOUT_SYNTH", "1200")
    monkeypatch.setenv("RECONCILER_INTERVAL_SEC", "60")
    importlib.reload(config)
    assert config.NEBIUS_API_KEY == "nk"
    assert config.NEBIUS_PROJECT_ID == "proj"
    assert config.GPU_IMAGE == "registry/gpu:1"
    assert config.NEBIUS_PRESET["prep"] == "1gpu-16vcpu-200gb"
    assert config.NEBIUS_PRESET["synth"] == "1gpu-8vcpu-100gb"
    assert config.STEP_TIMEOUT["prep"] == 1800
    assert config.STEP_TIMEOUT["synth"] == 1200
    assert config.RECONCILER_INTERVAL_SEC == 60
