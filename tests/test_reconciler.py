# tests/test_reconciler.py
from datetime import datetime, timezone
from src.orchestrator.reconciler import Reconciler
from src.orchestrator.runs import Run

NOW = datetime(2026, 6, 5, 12, 0, 0, tzinfo=timezone.utc)

class FakeStore:
    def __init__(self, due): self._due = due; self.updates = []
    def due_for_reconcile(self, now): return self._due
    def update(self, run_id, **f): self.updates.append((run_id, f)); return None

class FakeSM:
    def __init__(self): self.failures = []
    def _on_failure(self, run, step, error): self.failures.append((run.id, step, error))

class FakeNebius:
    def __init__(self, status): self._status = status
    def get_status(self, job_id): return self._status

def _run(): return Run(id="r1", workflow_type="dub", episode_id=1, callback_url="cb",
                       status="running", current_step="prep", nebius_job_id="job-1")

def test_dead_job_routes_through_failure():
    store, sm = FakeStore([_run()]), FakeSM()
    Reconciler(store, sm, FakeNebius("failed"), now=lambda: NOW).tick()
    assert sm.failures == [("r1", "prep", "nebius job failed without callback")]

def test_gone_job_routes_through_failure():
    store, sm = FakeStore([_run()]), FakeSM()
    Reconciler(store, sm, FakeNebius("gone"), now=lambda: NOW).tick()
    assert sm.failures and sm.failures[0][2].endswith("gone without callback")

def test_running_job_extends_deadline_no_failure():
    store, sm = FakeStore([_run()]), FakeSM()
    Reconciler(store, sm, FakeNebius("running"), now=lambda: NOW).tick()
    assert sm.failures == []
    rid, fields = store.updates[0]
    assert rid == "r1" and "step_deadline" in fields
