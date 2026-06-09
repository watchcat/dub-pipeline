# Pipeline Decomposition — Cloud Deployment & Cutover (Plan 2) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the Plan-1 orchestrator deployable and live — dispatch GPU work to Nebius scale-to-zero Jobs, deploy orchestrator + CPU workers to k3s, cut buzz-bot's dub path over behind a feature flag, and ship the free transcribe-without-dubbing feature end to end.

**Architecture:** A `NebiusDispatcher` slots into the existing `dispatchers["gpu"]` slot and launches Nebius Jobs; an in-process reconciler thread backstops jobs that die before calling back. Orchestrator + two CPU workers run in k3s (one CPU image, command-selected entrypoint); GPU runs as one image with two entrypoints on Nebius. buzz-bot routes dubs to `{orch}/dispatch` behind a `dub_orchestrator` flag and gains a free transcribe button that reuses existing `DubSegment` storage and the subtitle UI.

**Tech Stack:** Python 3.11 (FastAPI, psycopg 3, requests, pytest), Crystal/Kemal + crystal-pg, ClojureScript/re-frame (shadow-cljs), Docker, k3s (Traefik + cert-manager), Nebius Jobs API.

**Spec:** `docs/superpowers/specs/2026-06-05-pipeline-decomposition-cloud-design.md`

**Repos:** Tasks 1–14 + 22 are in `dub-pipeline`; Tasks 15–21 are in `buzz-bot` (`/Users/watchcat/work/crystal/buzz-bot`). Commit in each repo's own working tree.

---

## File Structure

**dub-pipeline — new:**
- `src/orchestrator/nebius.py` — thin Nebius Jobs client (`create_job`, `get_status`).
- `src/orchestrator/reconciler.py` — `Reconciler` (deadline scan → state-machine failure).
- `src/orchestrator/main.py` — production wiring (store + dispatchers + reporter + reconciler + uvicorn).
- `Dockerfile.gpu`, `Dockerfile.cpu` — GPU (prep+synth) and CPU (orch+text+mux) images.
- `k8s/orchestrator.yaml`, `k8s/cpu-text.yaml`, `k8s/cpu-mux.yaml`, `k8s/secret.example.yaml`.
- `DEPLOY.md` — deploy + smoke-test runbook.
- Tests: `tests/test_nebius.py`, `tests/test_dispatch_nebius.py`, `tests/test_reconciler.py`, `tests/test_progress.py`, `tests/test_main.py`, plus additions to `tests/test_runs.py`, `tests/test_gpu_synth.py`.

**dub-pipeline — modified:**
- `src/config.py` — Nebius + reconciler accessors.
- `src/orchestrator/runs.py` — `Run` columns `nebius_job_id`, `step_deadline`; `due_for_reconcile` on stores.
- `src/orchestrator/schema.sql` — matching columns.
- `src/orchestrator/dispatch.py` — `NebiusDispatcher`.
- `src/orchestrator/app.py` — `POST /progress`.
- `src/steps/synthesize.py` — optional `on_progress` callback.
- `src/workers/gpu_synth.py` — post incremental pct.

**buzz-bot — new:**
- `src/models/transcript_job.cr` — coalescing status table + model.
- `src/web/routes/transcribe.cr` — `POST /episodes/:id/transcribe`.
- `src/web/routes/transcript_result.cr` — `POST /internal/transcript_result`.
- Specs under `spec/` mirroring the project's existing spec layout.

**buzz-bot — modified:**
- `src/feature_flags.cr` — `dub_orchestrator` default.
- `src/config.cr` — `orch_base_url`, `orch_dispatch_secret`.
- `src/web/routes/dub.cr` — flag branch.
- `src/cljs/buzz_bot/events.cljs`, `subs.cljs`, `views/player.cljs` — transcribe button + poll.

---

## Phase 0 — dub-pipeline config + Run/schema extensions

### Task 1: Nebius + reconciler config accessors

**Files:**
- Modify: `src/config.py`
- Test: `tests/test_config_cloud.py`

- [ ] **Step 1: Write the failing test**

```python
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_config_cloud.py -v`
Expected: FAIL — `AttributeError: module 'src.config' has no attribute 'NEBIUS_API_KEY'`.

- [ ] **Step 3: Implement**

Append to `src/config.py`:
```python
# ── Nebius GPU jobs + reconciler ─────────────────────────────────────────────
NEBIUS_API_KEY    = os.environ.get("NEBIUS_API_KEY", "")
NEBIUS_PROJECT_ID = os.environ.get("NEBIUS_PROJECT_ID", "")
NEBIUS_API_BASE   = os.environ.get("NEBIUS_API_BASE", "https://api.nebius.cloud")
GPU_IMAGE         = os.environ.get("GPU_IMAGE", "")
NEBIUS_PRESET = {
    "prep":  os.environ.get("NEBIUS_PREP_PRESET", ""),
    "synth": os.environ.get("NEBIUS_SYNTH_PRESET", ""),
}
STEP_TIMEOUT = {
    "prep":  int(os.environ.get("STEP_TIMEOUT_PREP", "1800")),
    "synth": int(os.environ.get("STEP_TIMEOUT_SYNTH", "1200")),
}
RECONCILER_INTERVAL_SEC = int(os.environ.get("RECONCILER_INTERVAL_SEC", "60"))
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_config_cloud.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/config.py tests/test_config_cloud.py
git commit -m "feat: Nebius + reconciler config accessors"
```

### Task 2: `Run` reconciler columns + `due_for_reconcile` store method

**Files:**
- Modify: `src/orchestrator/runs.py`, `src/orchestrator/schema.sql`
- Test: `tests/test_runs.py` (additions)

- [ ] **Step 1: Write the failing test**

Append to `tests/test_runs.py`:
```python
from datetime import datetime, timezone, timedelta
from src.orchestrator.runs import Run, InMemoryRunStore

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
    # No nebius_job_id (a CPU step) → never reconciled.
    store.create(Run(id="d", workflow_type="dub", episode_id=4, callback_url="cb"))
    store.update("d", status="running", step_deadline=_dt(0))
    due = {r.id for r in store.due_for_reconcile(_dt(60))}
    assert due == {"a"}   # b not overdue, c done, d has no nebius job
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_runs.py -v -k reconcile`
Expected: FAIL — `TypeError: __init__() got an unexpected keyword 'nebius_job_id'` / `AttributeError: 'InMemoryRunStore' object has no attribute 'due_for_reconcile'`.

- [ ] **Step 3: Implement**

In `src/orchestrator/runs.py` add two fields at the end of the `Run` dataclass (after `segment_count`):
```python
    nebius_job_id: Optional[str] = None
    step_deadline: Optional["datetime"] = None
```
Add `from datetime import datetime` to the top imports. Add to the `RunStore` Protocol:
```python
    def due_for_reconcile(self, now: "datetime") -> list[Run]: ...
```
Add to `InMemoryRunStore`:
```python
    def due_for_reconcile(self, now):
        return [r for r in self._runs.values()
                if r.status == "running" and r.nebius_job_id is not None
                and r.step_deadline is not None and r.step_deadline < now]
```
Add to `PgRunStore` (psycopg returns timezone-aware datetimes for TIMESTAMPTZ):
```python
    def due_for_reconcile(self, now):
        with self._conn() as c:
            rows = c.execute(
                f"SELECT {', '.join(_PERSISTED)} FROM orch_run "
                "WHERE status = 'running' AND nebius_job_id IS NOT NULL "
                "AND step_deadline IS NOT NULL AND step_deadline < %s",
                (now,)).fetchall()
        return [Run(**dict(zip(_PERSISTED, row))) for row in rows]
```

In `src/orchestrator/schema.sql`, add two columns before `created_at`:
```sql
    nebius_job_id TEXT,
    step_deadline TIMESTAMPTZ,
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_runs.py -v`
Expected: PASS (existing + new).

- [ ] **Step 5: Commit**

```bash
git add src/orchestrator/runs.py src/orchestrator/schema.sql tests/test_runs.py
git commit -m "feat: Run reconcile columns + due_for_reconcile store query"
```

---

## Phase 1 — Nebius client + dispatcher

### Task 3: `nebius.py` Jobs client

**Files:**
- Create: `src/orchestrator/nebius.py`
- Test: `tests/test_nebius.py`

> The exact Nebius REST shape is verified against Nebius docs at deploy time; this
> client isolates every HTTP call so the rest of the system is testable without it.

- [ ] **Step 1: Write the failing test**

```python
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_nebius.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'src.orchestrator.nebius'`.

- [ ] **Step 3: Implement**

```python
# src/orchestrator/nebius.py
"""Thin Nebius Jobs REST client. Endpoint shapes verified against Nebius docs at deploy."""
import logging
import requests
from src import config

log = logging.getLogger(__name__)
_TIMEOUT = 30

_TERMINAL = {"succeeded": "succeeded", "failed": "failed",
             "cancelled": "failed", "error": "failed"}


def _headers() -> dict:
    return {"Authorization": f"Bearer {config.NEBIUS_API_KEY}",
            "Content-Type": "application/json"}


def create_job(image: str, preset: str, env: dict, timeout_sec: int) -> str:
    """Create a Nebius GPU job; return its job id."""
    spec = {
        "project_id": config.NEBIUS_PROJECT_ID,
        "image": image,
        "preset": preset,
        "timeout_seconds": timeout_sec,
        "environment": env,
    }
    resp = requests.post(f"{config.NEBIUS_API_BASE}/jobs", json=spec,
                         headers=_headers(), timeout=_TIMEOUT)
    resp.raise_for_status()
    return resp.json()["id"]


def get_status(job_id: str) -> str:
    """Return one of: running | succeeded | failed | gone."""
    resp = requests.get(f"{config.NEBIUS_API_BASE}/jobs/{job_id}",
                        headers=_headers(), timeout=_TIMEOUT)
    if resp.status_code == 404:
        return "gone"
    resp.raise_for_status()
    state = (resp.json().get("status") or "").lower()
    return _TERMINAL.get(state, "running")
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_nebius.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/orchestrator/nebius.py tests/test_nebius.py
git commit -m "feat: Nebius Jobs REST client (create_job, get_status)"
```

### Task 4: `NebiusDispatcher`

**Files:**
- Modify: `src/orchestrator/dispatch.py`
- Test: `tests/test_dispatch_nebius.py`

- [ ] **Step 1: Write the failing test**

```python
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_dispatch_nebius.py -v`
Expected: FAIL — `ImportError: cannot import name 'NebiusDispatcher'`.

- [ ] **Step 3: Implement**

Append to `src/orchestrator/dispatch.py` (add imports at top: `import json as _json`, `from datetime import datetime, timedelta, timezone`, `from src import config`, `from src.orchestrator import nebius`):
```python
class NebiusDispatcher:
    """Dispatch a GPU step by launching a Nebius job; record its id + deadline."""
    def __init__(self, store, nebius_client=nebius,
                 now=lambda: datetime.now(timezone.utc)):
        self.store = store
        self.nebius = nebius_client
        self.now = now

    def dispatch(self, step: Step, run: Run, payload: dict, callback_url: str) -> None:
        timeout = config.STEP_TIMEOUT[step.name]
        job_id = self.nebius.create_job(
            image=config.GPU_IMAGE,
            preset=config.NEBIUS_PRESET[step.name],
            env={"INPUT_JSON": _json.dumps(payload), "CALLBACK_URL": callback_url,
                 "STEP": step.name},
            timeout_sec=timeout)
        self.store.update(run.id, nebius_job_id=job_id,
                          step_deadline=self.now() + timedelta(seconds=timeout))
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_dispatch_nebius.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/orchestrator/dispatch.py tests/test_dispatch_nebius.py
git commit -m "feat: NebiusDispatcher — launch GPU job, record id + deadline"
```

---

## Phase 2 — Reconciler (failure backstop)

### Task 5: `Reconciler` deadline scan

**Files:**
- Create: `src/orchestrator/reconciler.py`
- Test: `tests/test_reconciler.py`

- [ ] **Step 1: Write the failing test**

```python
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_reconciler.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'src.orchestrator.reconciler'`.

- [ ] **Step 3: Implement**

```python
# src/orchestrator/reconciler.py
"""Backstop for GPU jobs that die (OOM/preempt/crash) before posting /callback.

Scans runs whose GPU step deadline has passed with no callback, asks Nebius for
the job's real state, and either re-dispatches via the state machine's failure
path (which reuses R2 artifacts) or extends the deadline for a still-running job.
"""
import logging
import time
from datetime import datetime, timedelta, timezone
from src import config

log = logging.getLogger(__name__)


class Reconciler:
    def __init__(self, store, state_machine, nebius_client,
                 now=lambda: datetime.now(timezone.utc)):
        self.store = store
        self.sm = state_machine
        self.nebius = nebius_client
        self.now = now

    def tick(self) -> int:
        """One reconcile pass; returns how many runs it acted on."""
        acted = 0
        for run in self.store.due_for_reconcile(self.now()):
            status = self.nebius.get_status(run.nebius_job_id)
            if status == "running":
                self.store.update(run.id, step_deadline=self._next_deadline(run))
            else:
                # failed | gone | succeeded-without-callback: re-dispatch (R2 reuse
                # makes the retry cheap and idempotent).
                self.sm._on_failure(run, run.current_step,
                                    f"nebius job {status} without callback")
            acted += 1
        return acted

    def _next_deadline(self, run):
        secs = config.STEP_TIMEOUT.get(run.current_step, config.RECONCILER_INTERVAL_SEC)
        return self.now() + timedelta(seconds=secs)

    def run_forever(self, interval: int | None = None) -> None:
        interval = interval or config.RECONCILER_INTERVAL_SEC
        while True:
            try:
                self.tick()
            except Exception:  # noqa: BLE001 — the loop must never die
                log.exception("reconciler tick failed")
            time.sleep(interval)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_reconciler.py -v`
Expected: PASS (3).

- [ ] **Step 5: Commit**

```bash
git add src/orchestrator/reconciler.py tests/test_reconciler.py
git commit -m "feat: reconciler backstop for dead GPU jobs"
```

---

## Phase 3 — Synth progress relay

### Task 6: orchestrator `POST /progress` + `StateMachine.relay_progress`

**Files:**
- Modify: `src/orchestrator/app.py`, `src/orchestrator/state_machine.py`
- Test: `tests/test_progress.py`

- [ ] **Step 1: Write the failing test**

```python
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_progress.py -v`
Expected: FAIL — 404 (no `/progress` route) / `AttributeError: relay_progress`.

- [ ] **Step 3: Implement**

Add to `src/orchestrator/state_machine.py` (uses the existing `workflows` import):
```python
    def relay_progress(self, run_id: str, step_name: str, pct) -> None:
        run = self.store.get(run_id)
        label = workflows.step_by_name(run.workflow_type, step_name).progress
        self.reporter.progress(run, label, pct)
```

Add to `src/orchestrator/app.py` (after the `callback` handler):
```python
@app.post("/progress")
async def progress(run_id: str, step: str, token: str, request: Request):
    if not auth.verify_token(run_id, step, token):
        return Response(status_code=401)
    body = await request.json()
    _sm.relay_progress(run_id, step, body.get("pct"))
    return {"ok": True}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_progress.py -v`
Expected: PASS (2).

- [ ] **Step 5: Commit**

```bash
git add src/orchestrator/app.py src/orchestrator/state_machine.py tests/test_progress.py
git commit -m "feat: orchestrator /progress relay for synth pct"
```

### Task 7: `synthesize` optional `on_progress` callback

**Files:**
- Modify: `src/steps/synthesize.py`
- Test: `tests/test_synthesize_progress.py`

> `synthesize` loads VoxCPM2, so the unit test exercises only the loop's progress
> calls with the model fully mocked.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_synthesize_progress.py
from unittest.mock import patch, MagicMock
from src.steps import synthesize

def test_on_progress_called_per_segment(monkeypatch):
    model = MagicMock()
    model.tts_model.sample_rate = 16000
    model.generate.return_value = [0.0] * 1600
    segs = [{"idx": 0, "speaker": "S", "translated_text": "hola"},
            {"idx": 1, "speaker": "S", "translated_text": "adios"}]
    calls = []
    with patch("src.steps.synthesize._load_model", return_value=model), \
         patch("src.steps.synthesize.soundfile.write"):
        synthesize.synthesize(segs, {"S": "/w/s.wav"}, "es", "/w",
                              on_progress=lambda d, t: calls.append((d, t)))
    assert calls[0] == (0, 2)
    assert calls[-1] == (2, 2)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_synthesize_progress.py -v`
Expected: FAIL — `TypeError: synthesize() got an unexpected keyword argument 'on_progress'`.

- [ ] **Step 3: Implement**

In `src/steps/synthesize.py`, add `from typing import Callable, Optional` if not present, and change the signature + loop:
```python
def synthesize(
    segments: list[dict],
    speaker_samples: dict[str, str],
    target_lang: str,
    out_dir: str,
    on_progress: Optional[Callable[[int, int], None]] = None,
) -> list[dict]:
```
Replace `for seg in segments:` with progress-aware iteration:
```python
    total = len(segments)
    for i, seg in enumerate(segments):
        if on_progress:
            on_progress(i, total)
```
(indent the existing loop body one level deeper under the new `for`). After the loop, before `return out`, add:
```python
    if on_progress:
        on_progress(total, total)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_synthesize_progress.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/steps/synthesize.py tests/test_synthesize_progress.py
git commit -m "feat: synthesize on_progress callback (per-segment)"
```

### Task 8: `gpu_synth` posts incremental pct

**Files:**
- Modify: `src/workers/gpu_synth.py`
- Test: `tests/test_gpu_synth.py` (additions)

- [ ] **Step 1: Write the failing test**

Append to `tests/test_gpu_synth.py`:
```python
import json as _json
from unittest.mock import patch

def test_main_posts_incremental_progress(monkeypatch):
    posts = []
    def fake_synth(segs, samples, lang, wd, on_progress=None):
        on_progress(0, 2); on_progress(1, 2); on_progress(2, 2)
        return [{"idx": 0, "synth_wav": None, "synth_duration": None}]
    monkeypatch.setenv("INPUT_JSON", _json.dumps(
        {"run_id": "r1", "episode_id": 456, "language": "es",
         "segments_key": "k", "speaker_keys": {}}))
    monkeypatch.setenv("CALLBACK_URL", "https://orch/callback?run_id=r1&step=synth&token=t")
    with patch("src.workers.gpu_synth.artifacts.read_segments", return_value=[{"idx": 0}]), \
         patch("src.workers.gpu_synth.synthesize.synthesize", side_effect=fake_synth), \
         patch("src.workers.gpu_synth.storage.upload"), \
         patch("src.workers.gpu_synth.artifacts.write_segments", return_value="k2"), \
         patch("src.workers.gpu_synth.common.run_in_tempdir", side_effect=lambda body: body("/w")), \
         patch("src.workers.gpu_synth.common.post_callback",
               side_effect=lambda url, body: posts.append((url, body))):
        gpu_synth.main()
    prog = [b["pct"] for (u, b) in posts if "/progress?" in u]
    assert set(prog) == {0, 50, 100}
    assert any("/callback?" in u and b.get("ok") for (u, b) in posts)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_gpu_synth.py -v -k progress`
Expected: FAIL — no `/progress` posts (run ignores `on_progress`).

- [ ] **Step 3: Implement**

In `src/workers/gpu_synth.py`, change `run` to accept and forward the callback, and build the poster in `main`:
```python
def run(inp: dict, on_progress=None) -> dict:
    run_id       = inp["run_id"]
    episode_id   = inp["episode_id"]
    language     = inp["language"]
    segments_key = inp["segments_key"]
    speaker_keys = inp.get("speaker_keys") or {}

    segments = artifacts.read_segments(segments_key)

    def body(work_dir: str) -> dict:
        speaker_samples: dict[str, str] = {}
        for speaker, key in speaker_keys.items():
            local = os.path.join(work_dir, f"speaker_{speaker}.wav")
            storage.download(key, local)
            speaker_samples[speaker] = local

        synthed = synthesize.synthesize(segments, speaker_samples, language, work_dir,
                                        on_progress=on_progress)

        out_segments = []
        for seg in synthed:
            seg = {**seg}
            wav = seg.pop("synth_wav", None)
            if wav:
                key = artifacts.stem_key(episode_id, f"synth_{language}_{seg['idx']:04d}.wav")
                storage.upload(wav, key, "audio/wav")
                seg["synth_r2_key"] = key
            out_segments.append(seg)

        return {"segments_key": artifacts.write_segments(run_id, out_segments)}

    return common.run_in_tempdir(body)


def main() -> None:
    inp = json.loads(os.environ["INPUT_JSON"])
    callback_url = os.environ["CALLBACK_URL"]
    progress_url = callback_url.replace("/callback", "/progress")
    last = {"pct": -1}

    def on_progress(done: int, total: int) -> None:
        pct = int(100 * done / total) if total else 0
        if pct != last["pct"]:
            last["pct"] = pct
            try:
                common.post_callback(progress_url, {"pct": pct})
            except Exception:  # noqa: BLE001 — progress is best-effort
                log.warning("synth progress post failed")

    try:
        common.post_callback(callback_url, {"ok": True, **run(inp, on_progress=on_progress)})
    except Exception as e:  # noqa: BLE001
        log.exception("gpu_synth failed")
        common.post_callback(callback_url, {"ok": False, "error": str(e)})
        sys.exit(1)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_gpu_synth.py -v`
Expected: PASS (existing + new).

- [ ] **Step 5: Commit**

```bash
git add src/workers/gpu_synth.py tests/test_gpu_synth.py
git commit -m "feat: gpu-synth posts incremental synth progress to orchestrator"
```

---

## Phase 4 — Production wiring

### Task 9: `main.py` entrypoint

**Files:**
- Create: `src/orchestrator/main.py`
- Modify: `src/orchestrator/app.py` (make `configure` return the state machine)
- Test: `tests/test_main.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_main.py
from unittest.mock import patch
from src.orchestrator import main as m

def test_build_wires_app_and_reconciler(monkeypatch):
    monkeypatch.setattr(m.config, "DATABASE_URL", "postgres://x")
    monkeypatch.setattr(m.config, "CPU_TEXT_URL", "http://t/run")
    monkeypatch.setattr(m.config, "CPU_MUX_URL", "http://mx/run")
    with patch("src.orchestrator.main.PgRunStore") as Store, \
         patch("src.orchestrator.main.appmod.configure") as cfg:
        store, reconciler = m.build()
    Store.assert_called_once_with("postgres://x")
    _, dispatchers, _ = cfg.call_args.args
    assert set(dispatchers) == {"gpu", "cpu"}
    assert reconciler.store is store
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_main.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'src.orchestrator.main'`.

- [ ] **Step 3: Implement**

First, in `src/orchestrator/app.py`, make `configure` return the state machine (change its last line):
```python
def configure(store, dispatchers, reporter) -> "StateMachine":
    global _sm
    _sm = StateMachine(store, dispatchers, reporter)
    return _sm
```

Then create `src/orchestrator/main.py`:
```python
"""Production entrypoint: wire collaborators, start the reconciler, serve uvicorn."""
import threading
import uvicorn
from src import config
from src.orchestrator import app as appmod, nebius
from src.orchestrator.dispatch import HttpDispatcher, NebiusDispatcher
from src.orchestrator.reconciler import Reconciler
from src.orchestrator.reporting import Reporter
from src.orchestrator.runs import PgRunStore


def build():
    store = PgRunStore(config.DATABASE_URL)
    dispatchers = {
        "gpu": NebiusDispatcher(store, nebius_client=nebius),
        "cpu": HttpDispatcher({"text": config.CPU_TEXT_URL, "mux": config.CPU_MUX_URL}),
    }
    sm = appmod.configure(store, dispatchers, Reporter())
    reconciler = Reconciler(store, sm, nebius)
    return store, reconciler


def main():
    store, reconciler = build()
    store.init_schema()
    threading.Thread(target=reconciler.run_forever, daemon=True).start()
    uvicorn.run(appmod.app, host="0.0.0.0", port=8080)


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_main.py -v && pytest -q`
Expected: PASS (new test, and the full suite still green).

- [ ] **Step 5: Commit**

```bash
git add src/orchestrator/main.py src/orchestrator/app.py tests/test_main.py
git commit -m "feat: orchestrator production entrypoint (wiring + reconciler thread)"
```

---

## Phase 5 — Images

### Task 10: GPU entrypoint dispatcher + `Dockerfile.gpu`

**Files:**
- Create: `src/workers/gpu_entry.py`, `Dockerfile.gpu`
- Test: `tests/test_gpu_entry.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_gpu_entry.py
from unittest.mock import patch
import pytest
from src.workers import gpu_entry

def test_entry_runs_prep(monkeypatch):
    monkeypatch.setenv("STEP", "prep")
    with patch("src.workers.gpu_entry.gpu_prep.main") as prep, \
         patch("src.workers.gpu_entry.gpu_synth.main") as synth:
        gpu_entry.main()
    prep.assert_called_once(); synth.assert_not_called()

def test_entry_runs_synth(monkeypatch):
    monkeypatch.setenv("STEP", "synth")
    with patch("src.workers.gpu_entry.gpu_prep.main") as prep, \
         patch("src.workers.gpu_entry.gpu_synth.main") as synth:
        gpu_entry.main()
    synth.assert_called_once(); prep.assert_not_called()

def test_entry_unknown_step_exits(monkeypatch):
    monkeypatch.setenv("STEP", "bogus")
    with pytest.raises(SystemExit) as e:
        gpu_entry.main()
    assert e.value.code == 2
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_gpu_entry.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'src.workers.gpu_entry'`.

- [ ] **Step 3: Implement**

```python
# src/workers/gpu_entry.py
"""GPU container entrypoint: run prep or synth based on the STEP env var."""
import os
import sys
from src.workers import gpu_prep, gpu_synth

_ENTRY = {"prep": gpu_prep.main, "synth": gpu_synth.main}


def main() -> None:
    step = os.environ.get("STEP", "")
    fn = _ENTRY.get(step)
    if fn is None:
        sys.stderr.write(f"gpu_entry: unknown STEP={step!r}\n")
        raise SystemExit(2)
    fn()


if __name__ == "__main__":
    main()
```

```dockerfile
# Dockerfile.gpu — one GPU image, prep + synth entrypoints (Nebius Jobs)
# Models live on a mounted volume (HF_HOME); not baked in.
FROM runpod/pytorch:1.0.3-cu1281-torch271-ubuntu2204

WORKDIR /app
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg libsndfile1 && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY src/ src/

ENV TORCHDYNAMO_DISABLE=1
ENV TEMP_DIR=/tmp/dub-pipeline
ENV WHISPER_DEVICE=cuda
ENV WHISPER_COMPUTE=float16

# STEP (prep|synth), INPUT_JSON, CALLBACK_URL are supplied by the orchestrator's
# NebiusDispatcher at job-create time.
CMD ["python", "-m", "src.workers.gpu_entry"]
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_gpu_entry.py -v`
Expected: PASS (3).

> The GPU image is **not** built locally (multi-GB CUDA base). Its build is in
> `DEPLOY.md` (Task 22). Verification here is the unit test + a syntax read.

- [ ] **Step 5: Commit**

```bash
git add src/workers/gpu_entry.py Dockerfile.gpu tests/test_gpu_entry.py
git commit -m "feat: GPU entrypoint dispatcher + Dockerfile.gpu"
```

### Task 11: CPU image (`Dockerfile.cpu` + `requirements-cpu.txt`)

**Files:**
- Create: `Dockerfile.cpu`, `requirements-cpu.txt`

> The CPU image serves the orchestrator + cpu-text + cpu-mux. It must **not** pull
> torch/demucs/voxcpm/whisper — those are GPU-only. Hence a slim requirements file.

- [ ] **Step 1: Create `requirements-cpu.txt`**

```
boto3>=1.34.0
requests>=2.31.0
numpy<2.0
google-genai>=1.0.0
pydub>=0.25.1
soundfile
fastapi>=0.110.0
uvicorn>=0.29.0
psycopg[binary]>=3.1.0
httpx>=0.27.0
```

- [ ] **Step 2: Create `Dockerfile.cpu`**

```dockerfile
# Dockerfile.cpu — orchestrator + cpu-text + cpu-mux (k3s). Slim, no GPU deps.
FROM python:3.11-slim

WORKDIR /app
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg libsndfile1 && rm -rf /var/lib/apt/lists/*

COPY requirements-cpu.txt .
RUN pip install --no-cache-dir -r requirements-cpu.txt
COPY src/ src/

ENV TEMP_DIR=/tmp/dub-pipeline
# Default command = orchestrator; cpu-text/cpu-mux Deployments override it.
CMD ["python", "-m", "src.orchestrator.main"]
```

- [ ] **Step 3: Build to verify (light image)**

Run: `docker build -f Dockerfile.cpu -t dub-orch-cpu:test .`
Expected: builds successfully (no torch download). If Docker is unavailable in the
authoring environment, defer to `DEPLOY.md` and verify by review.

- [ ] **Step 4: Commit**

```bash
git add Dockerfile.cpu requirements-cpu.txt
git commit -m "feat: slim CPU image for orchestrator + cpu workers"
```

---

## Phase 6 — k8s manifests (buzz-bot cluster)

> Follow the buzz-bot cluster conventions: namespace `buzz-bot`, Traefik ingress,
> cert-manager (`cluster-issuer: letsencrypt-prod`), `imagePullPolicy: IfNotPresent`
> with locally-imported images (see project k3s image-import notes). Validate each
> manifest offline with `kubectl apply --dry-run=client -f <file>` (or `kubeconform`).

### Task 12: Orchestrator Deployment + Service + Ingress

**Files:**
- Create: `k8s/orchestrator.yaml`

- [ ] **Step 1: Create the manifest**

```yaml
# k8s/orchestrator.yaml — orchestrator brain (CPU image), public for Nebius callbacks.
apiVersion: apps/v1
kind: Deployment
metadata:
  name: dub-orchestrator
  namespace: buzz-bot
spec:
  replicas: 1
  selector:
    matchLabels: {app: dub-orchestrator}
  template:
    metadata:
      labels: {app: dub-orchestrator}
    spec:
      containers:
        - name: orchestrator
          image: dub-orch-cpu:latest
          imagePullPolicy: IfNotPresent
          command: ["python", "-m", "src.orchestrator.main"]
          ports: [{containerPort: 8080}]
          envFrom:
            - secretRef: {name: orch-secret}
          readinessProbe:
            httpGet: {path: /healthz, port: 8080}
            initialDelaySeconds: 5
            periodSeconds: 10
          livenessProbe:
            httpGet: {path: /healthz, port: 8080}
            initialDelaySeconds: 10
            periodSeconds: 30
          resources:
            requests: {cpu: 50m, memory: 128Mi}
            limits: {cpu: 500m, memory: 512Mi}
---
apiVersion: v1
kind: Service
metadata:
  name: dub-orchestrator
  namespace: buzz-bot
spec:
  selector: {app: dub-orchestrator}
  ports: [{port: 80, targetPort: 8080}]
---
apiVersion: networking.k8s.io/v1
kind: Ingress
metadata:
  name: dub-orchestrator
  namespace: buzz-bot
  annotations:
    cert-manager.io/cluster-issuer: letsencrypt-prod
spec:
  ingressClassName: traefik
  rules:
    - host: orch.buzz-bot.top
      http:
        paths:
          - path: /
            pathType: Prefix
            backend:
              service:
                name: dub-orchestrator
                port: {number: 80}
  tls:
    - hosts: [orch.buzz-bot.top]
      secretName: dub-orchestrator-tls
```

- [ ] **Step 2: Validate**

Run: `kubectl apply --dry-run=client -f k8s/orchestrator.yaml`
Expected: `... created (dry run)` for Deployment, Service, Ingress (no schema errors).

- [ ] **Step 3: Commit**

```bash
git add k8s/orchestrator.yaml
git commit -m "feat: k8s orchestrator Deployment + Service + Ingress"
```

### Task 13: cpu-text + cpu-mux Deployments + Services

**Files:**
- Create: `k8s/cpu-text.yaml`, `k8s/cpu-mux.yaml`

- [ ] **Step 1: Create `k8s/cpu-text.yaml`**

```yaml
# k8s/cpu-text.yaml — split+translate HTTP worker (in-cluster only).
apiVersion: apps/v1
kind: Deployment
metadata:
  name: cpu-text
  namespace: buzz-bot
spec:
  replicas: 1
  selector:
    matchLabels: {app: cpu-text}
  template:
    metadata:
      labels: {app: cpu-text}
    spec:
      containers:
        - name: cpu-text
          image: dub-orch-cpu:latest
          imagePullPolicy: IfNotPresent
          command: ["uvicorn", "src.workers.cpu_text:app", "--host", "0.0.0.0", "--port", "8000"]
          ports: [{containerPort: 8000}]
          envFrom:
            - secretRef: {name: orch-secret}
          readinessProbe:
            httpGet: {path: /healthz, port: 8000}
            initialDelaySeconds: 5
            periodSeconds: 10
          resources:
            requests: {cpu: 50m, memory: 128Mi}
            limits: {cpu: 500m, memory: 512Mi}
---
apiVersion: v1
kind: Service
metadata:
  name: cpu-text
  namespace: buzz-bot
spec:
  selector: {app: cpu-text}
  ports: [{port: 80, targetPort: 8000}]
```

- [ ] **Step 2: Create `k8s/cpu-mux.yaml`**

Identical shape with `cpu-text` → `cpu-mux` and the command targeting `src.workers.cpu_mux:app`:
```yaml
# k8s/cpu-mux.yaml — assemble+mix HTTP worker (in-cluster only).
apiVersion: apps/v1
kind: Deployment
metadata:
  name: cpu-mux
  namespace: buzz-bot
spec:
  replicas: 1
  selector:
    matchLabels: {app: cpu-mux}
  template:
    metadata:
      labels: {app: cpu-mux}
    spec:
      containers:
        - name: cpu-mux
          image: dub-orch-cpu:latest
          imagePullPolicy: IfNotPresent
          command: ["uvicorn", "src.workers.cpu_mux:app", "--host", "0.0.0.0", "--port", "8000"]
          ports: [{containerPort: 8000}]
          envFrom:
            - secretRef: {name: orch-secret}
          readinessProbe:
            httpGet: {path: /healthz, port: 8000}
            initialDelaySeconds: 5
            periodSeconds: 10
          resources:
            requests: {cpu: 50m, memory: 128Mi}
            limits: {cpu: 1000m, memory: 1Gi}
---
apiVersion: v1
kind: Service
metadata:
  name: cpu-mux
  namespace: buzz-bot
spec:
  selector: {app: cpu-mux}
  ports: [{port: 80, targetPort: 8000}]
```

- [ ] **Step 3: Validate**

Run: `kubectl apply --dry-run=client -f k8s/cpu-text.yaml -f k8s/cpu-mux.yaml`
Expected: dry-run success for both Deployments + Services.

- [ ] **Step 4: Commit**

```bash
git add k8s/cpu-text.yaml k8s/cpu-mux.yaml
git commit -m "feat: k8s cpu-text + cpu-mux worker Deployments + Services"
```

### Task 14: Secret template

**Files:**
- Create: `k8s/secret.example.yaml`

> The real `k8s/secret.yaml` is never committed (gitignore it, mirroring buzz-bot).
> `CPU_TEXT_URL`/`CPU_MUX_URL` are the full `/run` URLs the `HttpDispatcher` POSTs to.

- [ ] **Step 1: Create the template**

```yaml
# k8s/secret.example.yaml — copy to secret.yaml, fill values, never commit secret.yaml.
apiVersion: v1
kind: Secret
metadata:
  name: orch-secret
  namespace: buzz-bot
type: Opaque
stringData:
  DATABASE_URL: "postgres://...neon..."
  ORCH_CALLBACK_SECRET: "<random-32-bytes>"
  ORCH_BASE_URL: "https://orch.buzz-bot.top"
  NEBIUS_API_KEY: "<nebius-api-key>"
  NEBIUS_PROJECT_ID: "<nebius-project-id>"
  GPU_IMAGE: "<registry>/dub-gpu:latest"
  NEBIUS_PREP_PRESET: "<gpu-preset-for-prep>"
  NEBIUS_SYNTH_PRESET: "<gpu-preset-for-synth>"
  PROGRESS_URL: "https://app.buzz-bot.top/internal/dub_progress"
  BUZZBOT_RESULT_URL: "https://app.buzz-bot.top/internal/dub_result"
  BUZZBOT_TRANSCRIPT_URL: "https://app.buzz-bot.top/internal/transcript_result"
  CPU_TEXT_URL: "http://cpu-text.buzz-bot.svc.cluster.local/run"
  CPU_MUX_URL: "http://cpu-mux.buzz-bot.svc.cluster.local/run"
  R2_ENDPOINT: "<r2-endpoint>"
  R2_ACCESS_KEY_ID: "<r2-key>"
  R2_SECRET_ACCESS_KEY: "<r2-secret>"
  R2_BUCKET: "<r2-bucket>"
  R2_PUBLIC_URL: "<r2-public-url>"
  GEMINI_API_KEY: "<gemini-key>"
  HF_TOKEN: "<hf-token>"
```

- [ ] **Step 2: Gitignore the real secret**

Add to `.gitignore`: `k8s/secret.yaml`.

- [ ] **Step 3: Commit**

```bash
git add k8s/secret.example.yaml .gitignore
git commit -m "feat: orch-secret template + gitignore real secret"
```

---

# buzz-bot tasks

> Tasks 15–21 are in the **buzz-bot** repo (`/Users/watchcat/work/crystal/buzz-bot`).
> Commit there. Crystal specs cover **pure** units (see `spec/web/proxy_helpers_spec.cr`);
> DB/route code is gated by a compile check: `crystal build --no-codegen src/buzz_bot.cr`.

## Phase 7 — Dub cutover (flagged)

### Task 15: `dub_orchestrator` flag + orchestrator config

**Files:**
- Modify: `src/feature_flags.cr`, `src/config.cr`

- [ ] **Step 1: Add the flag default**

In `src/feature_flags.cr`, add to `DEFAULTS`:
```crystal
    "dub_orchestrator" => false,
```

- [ ] **Step 2: Add config accessors**

In `src/config.cr`, add:
```crystal
  def self.orch_base_url : String
    ENV["ORCH_BASE_URL"]? || raise "ORCH_BASE_URL not set"
  end

  def self.orch_dispatch_secret : String
    ENV["ORCH_DISPATCH_SECRET"]? || raise "ORCH_DISPATCH_SECRET not set"
  end
```

- [ ] **Step 3: Compile to verify**

Run: `crystal build --no-codegen src/buzz_bot.cr`
Expected: compiles with no errors.

- [ ] **Step 4: Commit**

```bash
git add src/feature_flags.cr src/config.cr
git commit -m "feat: dub_orchestrator flag + orchestrator config accessors"
```

### Task 16: Pure dispatch-payload builder + dub.cr branch

**Files:**
- Create: `src/web/dub_dispatch.cr`
- Test: `spec/web/dub_dispatch_spec.cr`
- Modify: `src/web/routes/dub.cr`, `src/web/server.cr` (require)

- [ ] **Step 1: Write the failing test**

```crystal
# spec/web/dub_dispatch_spec.cr
require "../spec_helper"
require "../../src/web/dub_dispatch"

describe Web::DubDispatch do
  it "builds a dub dispatch payload" do
    json = JSON.parse(Web::DubDispatch.dub_payload(
      "run1", 42_i64, 456_i64, "https://a.mp3", "es", 0.15, "https://cb/internal/dub_result"))
    json["run_id"].should eq("run1")
    json["workflow_type"].should eq("dub")
    json["dub_id"].should eq(42)
    json["episode_id"].should eq(456)
    json["audio_url"].should eq("https://a.mp3")
    json["language"].should eq("es")
    json["bg_volume"].should eq(0.15)
    json["callback_url"].should eq("https://cb/internal/dub_result")
  end

  it "builds a transcribe dispatch payload (no dub_id/language)" do
    json = JSON.parse(Web::DubDispatch.transcribe_payload(
      "run2", 456_i64, "https://a.mp3", "https://cb/internal/transcript_result"))
    json["workflow_type"].should eq("transcribe")
    json["run_id"].should eq("run2")
    json["episode_id"].should eq(456)
    json["audio_url"].should eq("https://a.mp3")
    json["callback_url"].should eq("https://cb/internal/transcript_result")
    json.as_h.has_key?("dub_id").should be_false
  end
end
```

- [ ] **Step 2: Run test to verify it fails**

Run: `crystal spec spec/web/dub_dispatch_spec.cr`
Expected: FAIL — can't require `dub_dispatch` (file missing).

- [ ] **Step 3: Implement the builder**

```crystal
# src/web/dub_dispatch.cr
require "json"

# Pure builders for the orchestrator POST /dispatch request bodies. No I/O —
# unit-tested in isolation; the routes (dub.cr, transcribe.cr) post the result.
module Web::DubDispatch
  def self.dub_payload(run_id : String, dub_id : Int64, episode_id : Int64,
                       audio_url : String, language : String, bg_volume : Float64,
                       callback_url : String) : String
    {
      run_id:        run_id,
      workflow_type: "dub",
      dub_id:        dub_id,
      episode_id:    episode_id,
      audio_url:     audio_url,
      language:      language,
      bg_volume:     bg_volume,
      callback_url:  callback_url,
    }.to_json
  end

  def self.transcribe_payload(run_id : String, episode_id : Int64,
                              audio_url : String, callback_url : String) : String
    {
      run_id:        run_id,
      workflow_type: "transcribe",
      episode_id:    episode_id,
      audio_url:     audio_url,
      callback_url:  callback_url,
    }.to_json
  end
end
```

- [ ] **Step 4: Run test to verify it passes**

Run: `crystal spec spec/web/dub_dispatch_spec.cr`
Expected: PASS (2).

- [ ] **Step 5: Branch the dispatch in `dub.cr`**

In `src/web/routes/dub.cr`, add near the top (after the existing requires):
```crystal
require "../dub_dispatch"
```
Replace the block from `bg_volume = ...` through the RunPod `Log.info { ... submitted to RunPod ... }` (currently lines ~60–85) with:
```crystal
        bg_volume     = data["bg_volume"]?.try(&.as_f?) || 0.15
        job_id        = Random::Secure.hex(16)
        callback_base = Config.dub_callback_base

        if FeatureFlags.enabled?("dub_orchestrator")
          dispatch_body = Web::DubDispatch.dub_payload(
            job_id, dub_id, episode_id, episode.audio_url, language, bg_volume,
            "#{callback_base}/internal/dub_result")
          orch = HTTP::Client.new(URI.parse(Config.orch_base_url))
          orch.connect_timeout = 5.seconds
          orch.read_timeout = 10.seconds
          response = orch.post("/dispatch",
            headers: HTTP::Headers{
              "X-Dispatch-Token" => Config.orch_dispatch_secret,
              "Content-Type"     => "application/json"},
            body: dispatch_body)
          raise "Orchestrator dispatch error: #{response.status_code} #{response.body}" unless response.success?
          Log.info { "Dub[#{dub_id}]: dispatched to orchestrator (episode #{episode_id} → #{language})" }
        else
          payload = {
            job_id:       job_id,
            dub_id:       dub_id,
            episode_id:   episode_id,
            audio_url:    episode.audio_url,
            language:     language,
            bg_volume:    bg_volume,
            callback_url: "#{callback_base}/internal/dub_result",
          }.to_json
          runpod_payload = {input: JSON.parse(payload)}.to_json
          runpod_client = HTTP::Client.new(URI.parse("https://api.runpod.ai"))
          runpod_client.connect_timeout = 5.seconds
          runpod_client.read_timeout = 10.seconds
          response = runpod_client.post(
            "/v2/#{Config.runpod_endpoint_id}/run",
            headers: HTTP::Headers{
              "Authorization" => "Bearer #{Config.runpod_api_key}",
              "Content-Type"  => "application/json"
            },
            body: runpod_payload
          )
          raise "RunPod API error: #{response.status_code} #{response.body}" unless response.success?
          Log.info { "Dub[#{dub_id}]: job #{job_id} submitted to RunPod (episode #{episode_id} → #{language})" }
        end
```

- [ ] **Step 6: Compile to verify the branch**

Run: `crystal build --no-codegen src/buzz_bot.cr`
Expected: compiles with no errors.

- [ ] **Step 7: Commit**

```bash
git add src/web/dub_dispatch.cr spec/web/dub_dispatch_spec.cr src/web/routes/dub.cr
git commit -m "feat: route dubs to orchestrator behind dub_orchestrator flag"
```

---

## Phase 8 — Transcribe backend (free)

### Task 17: `transcript_jobs` table + `TranscriptJob` model

**Files:**
- Create: `migrations/021_transcript_jobs.sql`, `src/models/transcript_job.cr`

- [ ] **Step 1: Create the migration**

```sql
-- migrations/021_transcript_jobs.sql
CREATE TABLE IF NOT EXISTS transcript_jobs (
    episode_id BIGINT PRIMARY KEY,
    status     TEXT NOT NULL DEFAULT 'pending',  -- pending | done | failed
    run_id     TEXT,
    updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
);
```

- [ ] **Step 2: Create the model**

```crystal
# src/models/transcript_job.cr
require "../db"

# Coalescing status for transcribe-only runs, keyed by episode (transcripts have
# no target language). `claim` returns true only when the caller should dispatch.
module TranscriptJob
  def self.find(episode_id : Int64) : {String, String?}?
    AppDB.pool.query_one?(
      "SELECT status, run_id FROM transcript_jobs WHERE episode_id = $1",
      episode_id, as: {String, String?})
  end

  # Insert a pending job, or revive a previously-failed one. Returns true when a
  # row was created/revived (→ dispatch), false when one is already pending/done.
  def self.claim(episode_id : Int64, run_id : String) : Bool
    AppDB.pool.exec(
      "INSERT INTO transcript_jobs (episode_id, status, run_id)
       VALUES ($1, 'pending', $2)
       ON CONFLICT (episode_id) DO UPDATE
         SET status = 'pending', run_id = $2, updated_at = now()
         WHERE transcript_jobs.status = 'failed'",
      episode_id, run_id).rows_affected > 0
  end

  def self.set_done(episode_id : Int64)
    AppDB.pool.exec(
      "UPDATE transcript_jobs SET status = 'done', updated_at = now() WHERE episode_id = $1",
      episode_id)
  end

  def self.set_failed(episode_id : Int64)
    AppDB.pool.exec(
      "UPDATE transcript_jobs SET status = 'failed', updated_at = now() WHERE episode_id = $1",
      episode_id)
  end
end
```

- [ ] **Step 3: Compile to verify**

Run: `crystal build --no-codegen src/buzz_bot.cr` (after Task 18/19 wire it in; for now `crystal build --no-codegen src/models/transcript_job.cr` to type-check the model).
Expected: compiles.

- [ ] **Step 4: Commit**

```bash
git add migrations/021_transcript_jobs.sql src/models/transcript_job.cr
git commit -m "feat: transcript_jobs table + TranscriptJob coalescing model"
```

### Task 18: `POST /episodes/:id/transcribe` route

**Files:**
- Create: `src/web/routes/transcribe.cr`
- Modify: `src/buzz_bot.cr` (require), `src/web/server.cr` (require + register)

- [ ] **Step 1: Create the route**

```crystal
# src/web/routes/transcribe.cr
require "../../models/episode"
require "../../models/transcript_job"
require "../dub_dispatch"

module Web::Routes::Transcribe
  def self.register
    # Free for all users: generate the source-language transcript for an episode.
    post "/episodes/:id/transcribe" do |env|
      user = Auth.current_user(env)
      halt env, status_code: 401, response: "Unauthorized" unless user

      episode_id = env.params.url["id"].to_i64
      episode = Episode.find(episode_id)
      halt env, status_code: 404, response: %({"error":"not_found"}) unless episode

      if (existing = TranscriptJob.find(episode_id))
        status, _ = existing
        if status == "done"
          env.response.content_type = "application/json"
          next %({"status":"done"})
        elsif status == "pending"
          env.response.content_type = "application/json"
          next %({"status":"pending"})
        end
      end

      unless FeatureFlags.enabled?("dub_orchestrator")
        env.response.content_type = "application/json"
        env.response.status_code = 503
        next %({"error":"transcribe_unavailable"})
      end

      run_id = Random::Secure.hex(16)
      unless TranscriptJob.claim(episode_id, run_id)
        env.response.content_type = "application/json"
        next %({"status":"pending"})
      end

      begin
        body = Web::DubDispatch.transcribe_payload(
          run_id, episode_id, episode.audio_url,
          "#{Config.dub_callback_base}/internal/transcript_result")
        orch = HTTP::Client.new(URI.parse(Config.orch_base_url))
        orch.connect_timeout = 5.seconds
        orch.read_timeout = 10.seconds
        response = orch.post("/dispatch",
          headers: HTTP::Headers{
            "X-Dispatch-Token" => Config.orch_dispatch_secret,
            "Content-Type"     => "application/json"},
          body: body)
        raise "Orchestrator dispatch error: #{response.status_code} #{response.body}" unless response.success?
        Log.info { "Transcribe[ep=#{episode_id}]: dispatched run #{run_id}" }
      rescue ex
        Log.error { "Transcribe[ep=#{episode_id}]: enqueue failed — #{ex.message}" }
        TranscriptJob.set_failed(episode_id)
        env.response.content_type = "application/json"
        env.response.status_code = 500
        next %({"error":"enqueue_failed"})
      end

      env.response.content_type = "application/json"
      env.response.status_code = 202
      %({"status":"pending"})
    end
  end
end
```

- [ ] **Step 2: Wire it up**

In `src/buzz_bot.cr`, add after the other route requires:
```crystal
require "./web/routes/transcribe"
```
In `src/web/server.cr`, add `Web::Routes::Transcribe.register` alongside the other `.register` calls.

- [ ] **Step 3: Compile to verify**

Run: `crystal build --no-codegen src/buzz_bot.cr`
Expected: compiles with no errors.

- [ ] **Step 4: Commit**

```bash
git add src/web/routes/transcribe.cr src/buzz_bot.cr src/web/server.cr
git commit -m "feat: POST /episodes/:id/transcribe — free transcribe dispatch"
```

### Task 19: `POST /internal/transcript_result` consumer

**Files:**
- Create: `src/web/routes/transcript_result.cr`
- Modify: `src/web/server.cr` (require + register)

> Matches the existing `/internal/dub_result` convention: no auth, reachable only
> from inside the cluster (the orchestrator posts it).

- [ ] **Step 1: Create the route**

```crystal
# src/web/routes/transcript_result.cr
require "json"
require "../../models/dub_segment"
require "../../models/episode"
require "../../models/transcript_job"

module Web::Routes::TranscriptResult
  private struct Result
    include JSON::Serializable
    getter episode_id  : Int64
    getter source_lang : String?
    getter segments    : Array(JSON::Any)?
  end

  def self.register
    # Internal endpoint — called by the orchestrator, not the Mini App.
    post "/internal/transcript_result" do |env|
      body   = env.request.body.try(&.gets_to_end) || ""
      result = Result.from_json(body)
      ep_id  = result.episode_id

      if (src = result.source_lang.presence)
        Episode.save_original_language(ep_id, src)
        if (segs = result.segments) && !segs.empty?
          begin
            DubSegment.bulk_upsert(ep_id, src, segs)
            Log.info { "TranscriptResult[ep=#{ep_id}]: persisted #{segs.size} segments (#{src})" }
          rescue ex
            Log.warn { "TranscriptResult[ep=#{ep_id}]: persist failed — #{ex.message}" }
          end
        end
      end

      TranscriptJob.set_done(ep_id)
      env.response.content_type = "application/json"
      {ok: true}.to_json
    end
  end
end
```

- [ ] **Step 2: Wire it up**

In `src/web/server.cr`, add the require `require "./routes/transcript_result"` near the other internal-route requires, and `Web::Routes::TranscriptResult.register` alongside `DubResult.register`.

- [ ] **Step 3: Compile to verify**

Run: `crystal build --no-codegen src/buzz_bot.cr`
Expected: compiles with no errors.

- [ ] **Step 4: Commit**

```bash
git add src/web/routes/transcript_result.cr src/web/server.cr
git commit -m "feat: /internal/transcript_result — store source-lang transcript"
```

---

## Phase 9 — Transcribe UI (free button)

> Frontend builds use the project's JDK on PATH (see buzz-bot build notes):
> `export JAVA_HOME=/nix/store/<zulu-jdk-21>; export PATH="$JAVA_HOME/bin:$PATH"`.

### Task 20: Transcribe events, sub, db state, poll fx

**Files:**
- Modify: `src/cljs/buzz_bot/db.cljs`, `src/cljs/buzz_bot/events.cljs`,
  `src/cljs/buzz_bot/subs.cljs`, `src/cljs/buzz_bot/fx.cljs`
- Test: `test/buzz_bot/transcribe_test.cljs`

- [ ] **Step 1: Write the failing test**

```clojure
;; test/buzz_bot/transcribe_test.cljs
(ns buzz-bot.transcribe-test
  (:require [cljs.test :refer [deftest is testing]]
            [buzz-bot.events :as e]))

(deftest transcribe-url-test
  (testing "builds the transcribe endpoint for an episode"
    (is (= "/episodes/456/transcribe" (e/transcribe-url 456)))))
```

- [ ] **Step 2: Run test to verify it fails**

Run: `npx shadow-cljs compile test && node out/node-tests.js`
Expected: FAIL — `transcribe-url` undefined.

- [ ] **Step 3: Implement**

In `src/cljs/buzz_bot/db.cljs`, extend the `:subtitles` default map:
```clojure
   :subtitles {:ep-id nil
               :cues  []
               :lang  :off
               :transcribe-pending? false}
```

In `src/cljs/buzz_bot/events.cljs`, add the helper + events:
```clojure
(defn transcribe-url [ep-id] (str "/episodes/" ep-id "/transcribe"))

(rf/reg-event-fx
 ::request-transcript
 (fn [{:keys [db]} [_ episode-id]]
   {:db (assoc-in db [:subtitles :transcribe-pending?] true)
    ::buzz-bot.fx/http-fetch
    {:method :post
     :url    (transcribe-url episode-id)
     :on-ok  [::transcript-poll-tick episode-id 0]
     :on-err [::transcript-failed]}}))

(rf/reg-event-fx
 ::transcript-poll-tick
 (fn [{:keys [db]} [_ episode-id attempt]]
   (let [cues (get-in db [:subtitles :cues])]
     (cond
       (pos? (count cues))
       {:db (assoc-in db [:subtitles :transcribe-pending?] false)}
       (>= attempt 40)                          ; ~2 min at 3s; give up quietly
       {:db (assoc-in db [:subtitles :transcribe-pending?] false)}
       :else
       {:dispatch [::fetch-subtitles episode-id nil]
        ::buzz-bot.fx/schedule-poll
        {:ms 3000 :event [::transcript-poll-tick episode-id (inc attempt)]}}))))

(rf/reg-event-db
 ::transcript-failed
 (fn [db _] (assoc-in db [:subtitles :transcribe-pending?] false)))
```

In `src/cljs/buzz_bot/subs.cljs`, add:
```clojure
(rf/reg-sub ::transcribe-pending? :<- [::subtitles] (fn [s _] (:transcribe-pending? s)))
```

In `src/cljs/buzz_bot/fx.cljs`, add the scheduling effect:
```clojure
(rf/reg-fx
 ::schedule-poll
 (fn [{:keys [ms event]}]
   (js/setTimeout #(rf/dispatch event) ms)))
```

- [ ] **Step 4: Run test to verify it passes**

Run: `npx shadow-cljs compile test && node out/node-tests.js`
Expected: PASS (transcribe-url test + existing suite green).

- [ ] **Step 5: Commit**

```bash
git add src/cljs/buzz_bot/db.cljs src/cljs/buzz_bot/events.cljs \
        src/cljs/buzz_bot/subs.cljs src/cljs/buzz_bot/fx.cljs \
        test/buzz_bot/transcribe_test.cljs
git commit -m "feat(cljs): transcribe request + poll events/sub/fx"
```

### Task 21: "Generate transcript" button in the subtitle panel

**Files:**
- Modify: `src/cljs/buzz_bot/views/player.cljs`
- Build: `public/js/main.js` (regenerated)

- [ ] **Step 1: Add the pending sub to `subtitle-panel`**

In `src/cljs/buzz_bot/views/player.cljs`, inside the `subtitle-panel` `let`, add:
```clojure
        pending?    @(rf/subscribe [::subs/transcribe-pending?])
```

- [ ] **Step 2: Show the button / pending state when there are no cues**

Replace the cues area's empty branch:
```clojure
      [:div.subtitle-panel__cues
       (if (seq window)
         (for [{:keys [cue role]} window]
           ^{:key (:idx cue)}
           [:div.subtitle-cue-line {:class (name role)} (cue-text cue lang)])
         [:div.subtitle-cue-line.no-cue "…"])]
```
with:
```clojure
      [:div.subtitle-panel__cues
       (cond
         (seq window)
         (for [{:keys [cue role]} window]
           ^{:key (:idx cue)}
           [:div.subtitle-cue-line {:class (name role)} (cue-text cue lang)])
         pending?
         [:div.subtitle-cue-line.no-cue "Generating transcript…"]
         :else
         [:button.sub-generate-transcript
          {:on-click #(rf/dispatch [::events/request-transcript episode-id])}
          "Generate transcript"])]
```

- [ ] **Step 3: Build the bundle and verify**

Run: `npx shadow-cljs release app`
Expected: `Build completed. (… 0 warnings …)`.
Run: `grep -c "request-transcript" public/js/main.js`
Expected: ≥ 1 (the new event is in the bundle).

- [ ] **Step 4: Commit (force-add the tracked bundle)**

```bash
git add src/cljs/buzz_bot/views/player.cljs
git add -f public/js/main.js
git commit -m "feat(cljs): Generate transcript button in the subtitle panel"
```

---

## Phase 10 — Deploy runbook

### Task 22: `DEPLOY.md`

**Files:**
- Create: `DEPLOY.md` (dub-pipeline repo)

- [ ] **Step 1: Write the runbook**

````markdown
# Deploying the Orchestrated Pipeline (Plan 2)

Prerequisites: Nebius account + project, GPU presets chosen for prep/synth, k3s
access (`k8s/kubeconfig`), R2 + Gemini + HF creds, the buzz-bot Neon `DATABASE_URL`.

## 1. Build & import images
```bash
# CPU image (orchestrator + cpu-text + cpu-mux)
docker build -f Dockerfile.cpu -t dub-orch-cpu:latest .
# GPU image (prep + synth) — pushed to the registry Nebius pulls from
docker build -f Dockerfile.gpu -t <registry>/dub-gpu:latest .
docker push <registry>/dub-gpu:latest
```
Import the CPU image into k3s containerd (per buzz-bot's image-import convention),
since the Deployments use `imagePullPolicy: IfNotPresent`.

## 2. Database
Apply the orchestrator schema to Neon, and the buzz-bot transcript table:
```bash
psql "$DATABASE_URL" -f src/orchestrator/schema.sql
psql "$BUZZBOT_DATABASE_URL" -f ../buzz-bot/migrations/021_transcript_jobs.sql
```
(The orchestrator also calls `store.init_schema()` on startup as a safety net.)

## 3. Secrets + manifests
```bash
cp k8s/secret.example.yaml k8s/secret.yaml   # fill in real values (gitignored)
kubectl apply -f k8s/secret.yaml
kubectl apply -f k8s/orchestrator.yaml -f k8s/cpu-text.yaml -f k8s/cpu-mux.yaml
kubectl -n buzz-bot rollout status deploy/dub-orchestrator
```
Add `ORCH_BASE_URL=https://orch.buzz-bot.top` and `ORCH_DISPATCH_SECRET=<same as
ORCH_CALLBACK_SECRET>` to the buzz-bot secret, then redeploy buzz-bot (flag still off).

## 4. Nebius smoke test (paid — run once)
Dispatch a short clip and watch one GPU job complete:
```bash
curl -X POST https://orch.buzz-bot.top/dispatch \
  -H "X-Dispatch-Token: $ORCH_CALLBACK_SECRET" -H "Content-Type: application/json" \
  -d '{"run_id":"smoke1","workflow_type":"transcribe","episode_id":<short-ep>,
       "audio_url":"<short-clip-url>","callback_url":"https://app.buzz-bot.top/internal/transcript_result"}'
# verify: a Nebius gpu-prep job runs, /callback advances the run, transcript appears.
```

## 5. Cut over + verify
```bash
# In Telegram, as an admin:
/flag dub_orchestrator on
```
- Trigger a dub from the Mini App → confirm it runs via the orchestrator (Nebius
  prep → cpu-text → Nebius synth → cpu-mux) and the episode plays dubbed.
- Open an un-transcribed episode's subtitle panel → tap **Generate transcript** →
  confirm cues appear.
- Roll back instantly if needed: `/flag dub_orchestrator off` (reverts to RunPod).

## 6. Decommission (later)
Once the flag has been on in production without issue, remove `src/worker.py` and
the RunPod path in a follow-up cleanup.
````

- [ ] **Step 2: Commit**

```bash
git add DEPLOY.md
git commit -m "docs: Plan 2 deploy + smoke-test runbook"
```

---

## Self-Review

**Spec coverage (design A–G → tasks):**
- A (NebiusDispatcher + reconciler): Tasks 1–5. ✓
- B (images): Tasks 10 (GPU + entry), 11 (CPU). ✓
- C (k8s): Tasks 12–14. ✓ + schema/migration apply in Task 22.
- D (dub cutover, flagged): Tasks 15–16. ✓
- E (transcribe feature): Tasks 17–21 (table/model, route, result consumer, UI). ✓
- F (synth progress relay): Tasks 6–8. ✓
- G (testing + runbook): unit tests per task; Task 22 runbook. ✓
- Production wiring (`main.py`, reconciler thread, `configure` returns SM): Task 9. ✓

**Placeholder scan:** No "TBD/TODO". Deploy-time blanks (GPU presets, registry,
Nebius creds, short-clip URL) are intentional `<...>` placeholders confined to
`secret.example.yaml` and `DEPLOY.md`, flagged as fill-at-deploy — not code gaps.

**Type/contract consistency:**
- `/dispatch` body (Task 16/18 `dub_payload`/`transcribe_payload`) matches
  `app.py`'s fields (`run_id, workflow_type, episode_id, callback_url, dub_id?,
  language?, audio_url?, bg_volume?`). ✓
- `X-Dispatch-Token` (buzz-bot) is compared against `ORCH_CALLBACK_SECRET`
  (orchestrator `app.py`); the secret template sets `ORCH_DISPATCH_SECRET ==
  ORCH_CALLBACK_SECRET`. ✓
- `NebiusDispatcher` env keys (`INPUT_JSON, CALLBACK_URL, STEP`) are consumed by
  `gpu_entry` (`STEP`) and `gpu_prep/gpu_synth.main` (`INPUT_JSON, CALLBACK_URL`). ✓
- `Run` new fields (`nebius_job_id`, `step_deadline`) flow through the dataclass,
  `schema.sql`, `PgRunStore` (auto via `_PERSISTED`), `due_for_reconcile`, and the
  reconciler. ✓
- `/progress` query params (`run_id, step, token`) + `{pct}` body match
  `gpu_synth.main`'s `progress_url` (derived by `/callback`→`/progress`) and
  `StateMachine.relay_progress` → `Reporter.progress(run, "synthesizing", pct)`. ✓
- `transcript_result` consumes `{episode_id, source_lang, segments}` exactly as
  `reporting.py:transcript_result` posts. ✓
- `TranscriptJob.claim` coalescing matches the route's dispatch guard (Task 18). ✓

**Cross-repo note:** dub-pipeline Tasks 1–14, 22 and buzz-bot Tasks 15–21 each
commit in their own repo; there is no shared build step — only the runtime
`/dispatch` + callback contracts couple them, verified above.

