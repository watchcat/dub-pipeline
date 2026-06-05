# tests/test_dispatch_nebius.py
import json
from datetime import datetime, timezone
from src.orchestrator.dispatch import NebiusDispatcher
from src.orchestrator.runs import Run
from src.orchestrator.workflows import PREP

class FakeNebius:
    def __init__(self): self.calls = []
    def create_job(self, image, preset, env, timeout_sec):
        self.calls.append((image, preset, env, timeout_sec)); return "job-x"

class FakeStore:
    def __init__(self): self.updates = []
    def update(self, run_id, **f): self.updates.append((run_id, f))

def test_nebius_dispatch_creates_job_and_records_id_and_deadline(monkeypatch):
    from src.orchestrator import dispatch as d
    monkeypatch.setattr(d.config, "GPU_IMAGE", "img:7")
    monkeypatch.setattr(d.config, "NEBIUS_PRESET", {"prep": "1gpu-prep"})
    monkeypatch.setattr(d.config, "STEP_TIMEOUT", {"prep": 1800})
    neb, store = FakeNebius(), FakeStore()
    fixed = datetime(2026, 6, 5, 12, 0, 0, tzinfo=timezone.utc)
    disp = NebiusDispatcher(store, nebius_client=neb, now=lambda: fixed)
    run = Run(id="r1", workflow_type="dub", episode_id=456, callback_url="cb")
    disp.dispatch(PREP, run, {"run_id": "r1", "episode_id": 456}, "https://orch/callback?x=1")
    image, preset, env, timeout = neb.calls[0]
    assert image == "img:7" and preset == "1gpu-prep" and timeout == 1800
    assert env["CALLBACK_URL"] == "https://orch/callback?x=1"
    assert env["STEP"] == "prep"
    assert json.loads(env["INPUT_JSON"]) == {"run_id": "r1", "episode_id": 456}
    rid, fields = store.updates[0]
    assert rid == "r1" and fields["nebius_job_id"] == "job-x"
    assert fields["step_deadline"].isoformat() == "2026-06-05T12:30:00+00:00"
