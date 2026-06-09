import pytest
from src.orchestrator.runs import Run, InMemoryRunStore

def make_run(**kw):
    base = dict(id="r1", workflow_type="dub", episode_id=456,
                callback_url="https://app/internal/dub_result")
    base.update(kw)
    return Run(**base)

def test_create_get_update():
    store = InMemoryRunStore()
    store.create(make_run())
    assert store.get("r1").status == "running"
    updated = store.update("r1", current_step="prep", attempts=2)
    assert updated.current_step == "prep"
    assert updated.attempts == 2
    assert store.get("r1").attempts == 2

def test_get_missing_raises():
    with pytest.raises(KeyError):
        InMemoryRunStore().get("nope")


from datetime import datetime, timezone, timedelta

def _dt(s): return datetime(2026, 6, 5, 12, 0, 0, tzinfo=timezone.utc) + timedelta(seconds=s)

def test_run_has_reconcile_fields_defaulting_none():
    r = Run(id="r1", workflow_type="dub", episode_id=1, callback_url="cb")
    assert r.nebius_job_id is None
    assert r.step_deadline is None

def test_due_for_reconcile_returns_overdue_gpu_runs():
    store = InMemoryRunStore()
    store.create(Run(id="a", workflow_type="dub", episode_id=1, callback_url="cb"))
    store.update("a", status="running", nebius_job_id="job-a", step_deadline=_dt(0))
    store.create(Run(id="b", workflow_type="dub", episode_id=2, callback_url="cb"))
    store.update("b", status="running", nebius_job_id="job-b", step_deadline=_dt(120))
    store.create(Run(id="c", workflow_type="dub", episode_id=3, callback_url="cb"))
    store.update("c", status="done", nebius_job_id="job-c", step_deadline=_dt(0))
    store.create(Run(id="d", workflow_type="dub", episode_id=4, callback_url="cb"))
    store.update("d", status="running", step_deadline=_dt(0))
    due = {r.id for r in store.due_for_reconcile(_dt(60))}
    assert due == {"a"}   # b not overdue, c done, d has no nebius job
