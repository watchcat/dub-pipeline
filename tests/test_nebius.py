# tests/test_nebius.py
from unittest.mock import MagicMock, patch
from src.orchestrator import nebius

def test_create_job_posts_spec_and_returns_id(monkeypatch):
    monkeypatch.setattr(nebius.config, "NEBIUS_API_BASE", "https://api.neb")
    monkeypatch.setattr(nebius.config, "NEBIUS_API_KEY", "nk")
    monkeypatch.setattr(nebius.config, "NEBIUS_PROJECT_ID", "proj")
    resp = MagicMock(status_code=200); resp.json.return_value = {"id": "job-9"}
    with patch("src.orchestrator.nebius.requests.post", return_value=resp) as post:
        jid = nebius.create_job("img:1", "1gpu", {"INPUT_JSON": "{}"}, 1800)
    assert jid == "job-9"
    args, kwargs = post.call_args
    assert args[0] == "https://api.neb/jobs"
    assert kwargs["json"]["image"] == "img:1"
    assert kwargs["json"]["preset"] == "1gpu"
    assert kwargs["json"]["timeout_seconds"] == 1800
    assert kwargs["json"]["environment"] == {"INPUT_JSON": "{}"}
    assert kwargs["headers"]["Authorization"] == "Bearer nk"

def test_get_status_maps_states(monkeypatch):
    monkeypatch.setattr(nebius.config, "NEBIUS_API_BASE", "https://api.neb")
    def resp_for(state, code=200):
        r = MagicMock(status_code=code); r.json.return_value = {"status": state}; return r
    with patch("src.orchestrator.nebius.requests.get", return_value=resp_for("RUNNING")):
        assert nebius.get_status("j") == "running"
    with patch("src.orchestrator.nebius.requests.get", return_value=resp_for("SUCCEEDED")):
        assert nebius.get_status("j") == "succeeded"
    with patch("src.orchestrator.nebius.requests.get", return_value=resp_for("FAILED")):
        assert nebius.get_status("j") == "failed"
    with patch("src.orchestrator.nebius.requests.get", return_value=resp_for("", 404)):
        assert nebius.get_status("j") == "gone"
