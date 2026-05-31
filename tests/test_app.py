from unittest.mock import patch
from fastapi.testclient import TestClient
from src.orchestrator import app as appmod
from src.orchestrator import auth

def client_with_fakes():
    from src.orchestrator.runs import InMemoryRunStore
    from tests.fakes import FakeDispatcher, FakeReporter
    store = InMemoryRunStore()
    gpu, cpu = FakeDispatcher("gpu"), FakeDispatcher("cpu")
    rep = FakeReporter()
    appmod.configure(store, {"gpu": gpu, "cpu": cpu}, rep)
    return TestClient(appmod.app), store, gpu, cpu, rep

def test_dispatch_creates_run_and_starts_pipeline():
    client, store, gpu, cpu, rep = client_with_fakes()
    resp = client.post("/dispatch", json={
        "workflow_type": "dub", "run_id": "r1", "dub_id": 123, "episode_id": 456,
        "audio_url": "https://a.mp3", "language": "es", "bg_volume": 0.15,
        "callback_url": "https://app/internal/dub_result"})
    assert resp.status_code == 202
    assert store.get("r1").current_step == "prep"
    assert gpu.calls[0][0] == "prep"

def test_callback_with_valid_token_advances():
    client, store, gpu, cpu, rep = client_with_fakes()
    client.post("/dispatch", json={
        "workflow_type": "dub", "run_id": "r1", "dub_id": 123, "episode_id": 456,
        "audio_url": "https://a.mp3", "language": "es",
        "callback_url": "https://app/internal/dub_result"})
    with patch.object(auth.config, "ORCH_CALLBACK_SECRET", "dev-secret"):
        tok = auth.make_token("r1", "prep")
        resp = client.post(f"/callback?run_id=r1&step=prep&token={tok}",
                           json={"ok": True, "source_lang": "en",
                                 "speaker_keys": {}, "segments_key": "k1"})
    assert resp.status_code == 200
    assert store.get("r1").current_step == "text"

def test_callback_with_bad_token_rejected():
    client, store, gpu, cpu, rep = client_with_fakes()
    client.post("/dispatch", json={
        "workflow_type": "dub", "run_id": "r1", "dub_id": 123, "episode_id": 456,
        "audio_url": "https://a.mp3", "language": "es",
        "callback_url": "https://app/internal/dub_result"})
    resp = client.post("/callback?run_id=r1&step=prep&token=bad",
                       json={"ok": True, "segments_key": "k1"})
    assert resp.status_code == 401
    assert store.get("r1").current_step == "prep"  # unchanged
