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
