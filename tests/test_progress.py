# tests/test_progress.py
from fastapi.testclient import TestClient
from src.orchestrator import app as appmod, auth
from src.orchestrator.runs import Run, InMemoryRunStore
from tests.fakes import FakeDispatcher, FakeReporter

def _setup():
    store = InMemoryRunStore()
    store.create(Run(id="r1", workflow_type="dub", episode_id=1,
                     callback_url="cb", current_step="synth"))
    rep = FakeReporter()
    appmod.configure(store, {"gpu": FakeDispatcher("gpu"), "cpu": FakeDispatcher("cpu")}, rep)
    return TestClient(appmod.app), rep

def test_progress_relays_synth_pct(monkeypatch):
    monkeypatch.setattr(auth.config, "ORCH_CALLBACK_SECRET", "k")
    client, rep = _setup()
    tok = auth.make_token("r1", "synth")
    r = client.post(f"/progress?run_id=r1&step=synth&token={tok}", json={"pct": 42})
    assert r.status_code == 200
    assert rep.progress_calls == [("r1", "synthesizing", 42)]

def test_progress_rejects_bad_token(monkeypatch):
    monkeypatch.setattr(auth.config, "ORCH_CALLBACK_SECRET", "k")
    client, _ = _setup()
    r = client.post("/progress?run_id=r1&step=synth&token=bad", json={"pct": 42})
    assert r.status_code == 401
