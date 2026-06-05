"""Run record + storage abstraction."""
import json
import os
from dataclasses import asdict, dataclass, fields as dataclass_fields, replace
from datetime import datetime
from typing import Optional, Protocol


@dataclass(frozen=True)
class Run:
    id: str
    workflow_type: str          # "dub" | "transcribe"
    episode_id: int
    callback_url: str
    dub_id: Optional[int] = None
    language: Optional[str] = None
    audio_url: Optional[str] = None
    bg_volume: float = 0.15
    status: str = "running"     # running | done | failed
    current_step: str = ""
    attempts: int = 0
    segments_key: Optional[str] = None
    source_lang: Optional[str] = None
    speaker_keys: Optional[dict] = None
    r2_url: Optional[str] = None
    duration_sec: Optional[float] = None
    segment_count: Optional[int] = None
    nebius_job_id: Optional[str] = None
    step_deadline: Optional["datetime"] = None


class RunStore(Protocol):
    def create(self, run: Run) -> None: ...
    def get(self, run_id: str) -> Run: ...
    def update(self, run_id: str, **fields) -> Run: ...
    def due_for_reconcile(self, now: "datetime") -> list[Run]: ...


class InMemoryRunStore:
    def __init__(self) -> None:
        self._runs: dict[str, Run] = {}

    def create(self, run: Run) -> None:
        self._runs[run.id] = run

    def get(self, run_id: str) -> Run:
        return self._runs[run_id]

    def update(self, run_id: str, **fields) -> Run:
        self._runs[run_id] = replace(self._runs[run_id], **fields)
        return self._runs[run_id]

    def due_for_reconcile(self, now):
        return [r for r in self._runs.values()
                if r.status == "running" and r.nebius_job_id is not None
                and r.step_deadline is not None and r.step_deadline < now]


import psycopg  # noqa: E402  (stdlib imports above, third-party below)

_PERSISTED = [f.name for f in dataclass_fields(Run)]  # all Run columns


class PgRunStore:
    def __init__(self, dsn: str):
        self.dsn = dsn

    def _conn(self):
        return psycopg.connect(self.dsn, autocommit=True)

    def init_schema(self) -> None:
        with open(os.path.join(os.path.dirname(__file__), "schema.sql")) as f:
            ddl = f.read()
        with self._conn() as c:
            c.execute(ddl)

    def create(self, run: Run) -> None:
        d = asdict(run)
        d["speaker_keys"] = json.dumps(d["speaker_keys"]) if d["speaker_keys"] is not None else None
        cols = ", ".join(_PERSISTED)
        ph = ", ".join(f"%({k})s" for k in _PERSISTED)
        with self._conn() as c:
            c.execute(f"INSERT INTO orch_run ({cols}) VALUES ({ph})", d)

    def get(self, run_id: str) -> Run:
        with self._conn() as c:
            row = c.execute(
                f"SELECT {', '.join(_PERSISTED)} FROM orch_run WHERE id = %s",
                (run_id,)).fetchone()
        if row is None:
            raise KeyError(run_id)
        data = dict(zip(_PERSISTED, row))
        return Run(**data)

    def update(self, run_id: str, **fields) -> Run:
        if not fields:
            return self.get(run_id)
        unknown = set(fields) - set(_PERSISTED)
        if unknown:
            raise ValueError(f"unknown run fields: {unknown}")
        if "speaker_keys" in fields and fields["speaker_keys"] is not None:
            fields = {**fields, "speaker_keys": json.dumps(fields["speaker_keys"])}
        sets = ", ".join(f"{k} = %({k})s" for k in fields)
        params = {**fields, "rid": run_id}
        with self._conn() as c:
            c.execute(f"UPDATE orch_run SET {sets}, updated_at = now() WHERE id = %(rid)s", params)
        return self.get(run_id)

    def due_for_reconcile(self, now):
        with self._conn() as c:
            rows = c.execute(
                f"SELECT {', '.join(_PERSISTED)} FROM orch_run "
                "WHERE status = 'running' AND nebius_job_id IS NOT NULL "
                "AND step_deadline IS NOT NULL AND step_deadline < %s",
                (now,)).fetchall()
        return [Run(**dict(zip(_PERSISTED, row))) for row in rows]
