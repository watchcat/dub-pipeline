import os
import uuid
import pytest

pytestmark = pytest.mark.skipif(not os.environ.get("DATABASE_URL"),
                                reason="DATABASE_URL not set")

def test_pg_roundtrip():
    from src.orchestrator.runs import Run, PgRunStore
    store = PgRunStore(os.environ["DATABASE_URL"])
    store.init_schema()
    rid = f"test-{uuid.uuid4().hex[:8]}"
    store.create(Run(id=rid, workflow_type="dub", episode_id=456,
                     callback_url="cb", dub_id=123, language="es"))
    assert store.get(rid).language == "es"
    store.update(rid, current_step="prep", attempts=2,
                 speaker_keys={"S0": "k"})
    got = store.get(rid)
    assert got.current_step == "prep" and got.attempts == 2
    assert got.speaker_keys == {"S0": "k"}
