# Pipeline Decomposition — Core Engine Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the in-process RunPod monolith with a central orchestrator that drives four task-runner entrypoints (gpu-prep, cpu-text, gpu-synth, cpu-mux) through a Postgres-backed state machine, runnable end-to-end locally without any cloud.

**Architecture:** A pure `StateMachine` advances a `Run` through a declarative workflow (`dub` or `transcribe`), dispatching each step via a pluggable `Dispatcher` (HTTP for CPU workers, local-subprocess for everything in dev). Workers are thin adapters over the existing pure functions in `src/steps/`; large artifacts pass through R2 by key, small scalars through callbacks. The orchestrator posts buzz-bot's existing `/internal/dub_progress` + `/internal/dub_result` callbacks unchanged.

**Tech Stack:** Python 3.11, FastAPI + uvicorn (orchestrator + CPU worker HTTP), psycopg 3 (Neon Postgres), boto3 (R2, already present), pytest + httpx TestClient. Nebius Jobs client + k8s deploy + buzz-bot cutover are **Plan 2** (this plan stops at a local end-to-end run).

**Scope note:** This plan does NOT touch `src/worker.py` (the RunPod monolith stays as-is for coexistence), does NOT call any cloud GPU API, and does NOT modify buzz-bot. It produces a working orchestrator + workers exercised by a `LocalDispatcher` on a short clip.

---

## File Structure

**New:**
- `src/artifacts.py` — R2 key conventions + segments.json read/write.
- `src/orchestrator/__init__.py`
- `src/orchestrator/auth.py` — per-run HMAC callback token.
- `src/orchestrator/runs.py` — `Run` dataclass, `RunStore` protocol, `InMemoryRunStore`, `PgRunStore`.
- `src/orchestrator/schema.sql` — `run` table DDL.
- `src/orchestrator/workflows.py` — declarative `Step`s + workflow definitions + per-step input builder.
- `src/orchestrator/reporting.py` — `Reporter` (progress/result/failed → buzz-bot).
- `src/orchestrator/state_machine.py` — `StateMachine` (the core).
- `src/orchestrator/dispatch.py` — `Dispatcher` protocol, `HttpDispatcher`, `LocalDispatcher`.
- `src/orchestrator/app.py` — FastAPI app: `POST /dispatch`, `POST /callback`, `GET /healthz`.
- `src/workers/__init__.py`
- `src/workers/common.py` — input parsing + callback POST helper.
- `src/workers/gpu_prep.py` — separate→transcribe→(extract).
- `src/workers/gpu_synth.py` — synthesize.
- `src/workers/cpu_text.py` — split→translate (+ FastAPI app for HTTP dispatch).
- `src/workers/cpu_mux.py` — assemble→mix (+ FastAPI app for HTTP dispatch).
- `tests/conftest.py`, `tests/test_*.py`

**Modified:**
- `src/storage.py` — add `download_bytes`.
- `src/config.py` — add orchestrator config accessors.
- `requirements.txt` — add fastapi, uvicorn, psycopg[binary], httpx, pytest.

---

## Phase 0 — Scaffolding

### Task 0: Dependencies, package dirs, pytest

**Files:**
- Modify: `requirements.txt`
- Create: `src/orchestrator/__init__.py`, `src/workers/__init__.py`, `tests/__init__.py`, `pytest.ini`

- [ ] **Step 1: Add dependencies**

Append to `requirements.txt`:
```
fastapi>=0.110.0
uvicorn>=0.29.0
psycopg[binary]>=3.1.0
httpx>=0.27.0
pytest>=8.0.0
```

- [ ] **Step 2: Create package markers**

Create empty files: `src/orchestrator/__init__.py`, `src/workers/__init__.py`, `tests/__init__.py`.

- [ ] **Step 3: Create `pytest.ini`**

```ini
[pytest]
testpaths = tests
addopts = -q
```

- [ ] **Step 4: Install and verify**

Run: `pip install -r requirements.txt && pytest -q`
Expected: `no tests ran` (exit 5) — confirms pytest is wired and imports resolve.

- [ ] **Step 5: Commit**

```bash
git add requirements.txt src/orchestrator/__init__.py src/workers/__init__.py tests/__init__.py pytest.ini
git commit -m "chore: scaffolding for orchestrator + workers (deps, packages, pytest)"
```

---

## Phase 1 — R2 artifact layer

### Task 1: `storage.download_bytes`

**Files:**
- Modify: `src/storage.py`
- Test: `tests/test_storage.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_storage.py
from unittest.mock import MagicMock, patch
from src import storage

def test_download_bytes_reads_object_body():
    fake_body = MagicMock()
    fake_body.read.return_value = b'{"hello": 1}'
    fake_s3 = MagicMock()
    fake_s3.get_object.return_value = {"Body": fake_body}
    with patch.object(storage, "_s3", return_value=fake_s3):
        data = storage.download_bytes("some/key.json")
    assert data == b'{"hello": 1}'
    fake_s3.get_object.assert_called_once()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_storage.py -v`
Expected: FAIL — `AttributeError: module 'src.storage' has no attribute 'download_bytes'`.

- [ ] **Step 3: Implement**

Add to `src/storage.py`:
```python
def download_bytes(r2_key: str) -> bytes:
    """Read an R2 object fully into memory."""
    obj = _s3().get_object(Bucket=config.R2_BUCKET, Key=r2_key)
    return obj["Body"].read()
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_storage.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/storage.py tests/test_storage.py
git commit -m "feat: storage.download_bytes for reading R2 artifacts"
```

### Task 2: `src/artifacts.py` keys + segments JSON

**Files:**
- Create: `src/artifacts.py`
- Test: `tests/test_artifacts.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_artifacts.py
import json
from unittest.mock import patch
from src import artifacts

def test_key_conventions():
    assert artifacts.stem_key(456, "vocals.wav") == "dub-stems/456/vocals.wav"
    assert artifacts.dub_key(456, "es") == "dubbed/456/es.mp3"
    assert artifacts.segments_key("abc") == "dub-runs/abc/segments.json"

def test_write_then_read_segments_roundtrip():
    segs = [{"idx": 0, "text": "hi", "words": [{"w": "hi"}]}]
    captured = {}
    def fake_upload_bytes(data, key, ctype):
        captured["data"] = data; captured["key"] = key; return "url"
    with patch("src.artifacts.storage.upload_bytes", side_effect=fake_upload_bytes):
        key = artifacts.write_segments("abc", segs)
    assert key == "dub-runs/abc/segments.json"
    with patch("src.artifacts.storage.download_bytes", return_value=captured["data"]):
        assert artifacts.read_segments(key) == segs
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_artifacts.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'src.artifacts'`.

- [ ] **Step 3: Implement**

```python
# src/artifacts.py
"""R2 key conventions and segments.json artifact I/O for pipeline runs."""
import json
from src import storage


def stem_key(episode_id: int, filename: str) -> str:
    return f"dub-stems/{episode_id}/{filename}"


def dub_key(episode_id: int, language: str) -> str:
    return f"dubbed/{episode_id}/{language}.mp3"


def segments_key(run_id: str) -> str:
    return f"dub-runs/{run_id}/segments.json"


def write_segments(run_id: str, segments: list[dict]) -> str:
    key = segments_key(run_id)
    storage.upload_bytes(json.dumps(segments).encode("utf-8"), key, "application/json")
    return key


def read_segments(key: str) -> list[dict]:
    return json.loads(storage.download_bytes(key).decode("utf-8"))
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_artifacts.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/artifacts.py tests/test_artifacts.py
git commit -m "feat: artifacts module — R2 key conventions + segments.json I/O"
```

---

## Phase 2 — Orchestrator config + auth

### Task 3: Orchestrator config accessors

**Files:**
- Modify: `src/config.py`
- Test: `tests/test_config_orch.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_config_orch.py
import importlib, os
from src import config

def test_orchestrator_config_reads_env(monkeypatch):
    monkeypatch.setenv("DATABASE_URL", "postgres://x")
    monkeypatch.setenv("ORCH_CALLBACK_SECRET", "s3cr3t")
    monkeypatch.setenv("ORCH_BASE_URL", "https://orch.local")
    monkeypatch.setenv("BUZZBOT_RESULT_URL", "https://app/internal/dub_result")
    monkeypatch.setenv("BUZZBOT_TRANSCRIPT_URL", "https://app/internal/transcript_result")
    monkeypatch.setenv("CPU_TEXT_URL", "http://cpu-text/run")
    monkeypatch.setenv("CPU_MUX_URL", "http://cpu-mux/run")
    importlib.reload(config)
    assert config.DATABASE_URL == "postgres://x"
    assert config.ORCH_CALLBACK_SECRET == "s3cr3t"
    assert config.ORCH_BASE_URL == "https://orch.local"
    assert config.BUZZBOT_RESULT_URL.endswith("/dub_result")
    assert config.BUZZBOT_TRANSCRIPT_URL.endswith("/transcript_result")
    assert config.CPU_TEXT_URL == "http://cpu-text/run"
    assert config.CPU_MUX_URL == "http://cpu-mux/run"
```

> Note: existing `config.py` reads several vars at import with `os.environ[...]`. The test sets only the new vars; ensure the conftest in Step 0 of Phase 3 sets the pre-existing required ones. For this task, also set them here:
> add at the top of the test, before `importlib.reload`:
> ```python
>     for k in ["PROGRESS_URL","R2_ENDPOINT","R2_ACCESS_KEY_ID","R2_SECRET_ACCESS_KEY",
>               "R2_BUCKET","R2_PUBLIC_URL","GEMINI_API_KEY","HF_TOKEN"]:
>         monkeypatch.setenv(k, "x")
> ```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_config_orch.py -v`
Expected: FAIL — `AttributeError: module 'src.config' has no attribute 'DATABASE_URL'`.

- [ ] **Step 3: Implement**

Append to `src/config.py`:
```python
# ── Orchestrator ────────────────────────────────────────────────────────────
DATABASE_URL          = os.environ.get("DATABASE_URL", "")
ORCH_CALLBACK_SECRET  = os.environ.get("ORCH_CALLBACK_SECRET", "dev-secret")
ORCH_BASE_URL         = os.environ.get("ORCH_BASE_URL", "http://localhost:8080")
BUZZBOT_RESULT_URL     = os.environ.get("BUZZBOT_RESULT_URL", "")
BUZZBOT_TRANSCRIPT_URL = os.environ.get("BUZZBOT_TRANSCRIPT_URL", "")
CPU_TEXT_URL          = os.environ.get("CPU_TEXT_URL", "")
CPU_MUX_URL           = os.environ.get("CPU_MUX_URL", "")
MAX_STEP_ATTEMPTS     = int(os.environ.get("MAX_STEP_ATTEMPTS", "3"))
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_config_orch.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/config.py tests/test_config_orch.py
git commit -m "feat: orchestrator config accessors"
```

### Task 4: Per-run callback token (`auth.py`)

**Files:**
- Create: `src/orchestrator/auth.py`
- Test: `tests/test_auth.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_auth.py
from src.orchestrator import auth

def test_token_roundtrip_and_tamper(monkeypatch):
    monkeypatch.setattr(auth.config, "ORCH_CALLBACK_SECRET", "k")
    t = auth.make_token("run1", "prep")
    assert auth.verify_token("run1", "prep", t) is True
    assert auth.verify_token("run1", "synth", t) is False
    assert auth.verify_token("run2", "prep", t) is False
    assert auth.verify_token("run1", "prep", "deadbeef") is False
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_auth.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'src.orchestrator.auth'`.

- [ ] **Step 3: Implement**

```python
# src/orchestrator/auth.py
"""HMAC token scoping a callback to a single (run_id, step)."""
import hashlib
import hmac
from src import config


def make_token(run_id: str, step: str) -> str:
    msg = f"{run_id}:{step}".encode("utf-8")
    return hmac.new(config.ORCH_CALLBACK_SECRET.encode("utf-8"), msg, hashlib.sha256).hexdigest()


def verify_token(run_id: str, step: str, token: str) -> bool:
    return hmac.compare_digest(make_token(run_id, step), token or "")
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_auth.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/orchestrator/auth.py tests/test_auth.py
git commit -m "feat: per-run HMAC callback token"
```

---

## Phase 3 — Run model + store

### Task 5: `Run` + `InMemoryRunStore`

**Files:**
- Create: `src/orchestrator/runs.py`
- Test: `tests/test_runs.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_runs.py
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_runs.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'src.orchestrator.runs'`.

- [ ] **Step 3: Implement**

```python
# src/orchestrator/runs.py
"""Run record + storage abstraction."""
from dataclasses import dataclass, replace
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


class RunStore(Protocol):
    def create(self, run: Run) -> None: ...
    def get(self, run_id: str) -> Run: ...
    def update(self, run_id: str, **fields) -> Run: ...


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
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_runs.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/orchestrator/runs.py tests/test_runs.py
git commit -m "feat: Run record + InMemoryRunStore"
```

---

## Phase 4 — Declarative workflows

### Task 6: `workflows.py`

**Files:**
- Create: `src/orchestrator/workflows.py`
- Test: `tests/test_workflows.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_workflows.py
from src.orchestrator import workflows as wf
from src.orchestrator.runs import Run

def make_run(**kw):
    base = dict(id="r1", workflow_type="dub", episode_id=456,
                callback_url="cb", audio_url="https://a.mp3", language="es",
                segments_key="dub-runs/r1/segments.json", source_lang="en",
                speaker_keys={"SPEAKER_00": "dub-stems/456/speaker_SPEAKER_00.wav"})
    base.update(kw); return Run(**base)

def test_dub_step_order_and_next():
    assert [s.name for s in wf.steps_for("dub")] == ["prep", "text", "synth", "mux"]
    assert wf.first_step("dub").name == "prep"
    assert wf.next_step("dub", "prep").name == "text"
    assert wf.next_step("dub", "mux") is None

def test_transcribe_is_prep_only():
    assert [s.name for s in wf.steps_for("transcribe")] == ["prep"]
    assert wf.next_step("transcribe", "prep") is None

def test_step_tiers_and_progress_labels():
    by = {s.name: s for s in wf.steps_for("dub")}
    assert by["prep"].tier == "gpu" and by["synth"].tier == "gpu"
    assert by["text"].tier == "cpu" and by["mux"].tier == "cpu"
    assert by["prep"].progress == "separating"

def test_build_input_prep_for_dub():
    r = make_run()
    assert wf.build_input(wf.first_step("dub"), r) == {
        "run_id": "r1", "episode_id": 456, "audio_url": "https://a.mp3", "extract": True}

def test_build_input_prep_extract_false_for_transcribe():
    r = make_run(workflow_type="transcribe")
    assert wf.build_input(wf.first_step("transcribe"), r)["extract"] is False

def test_build_input_text_synth_mux():
    r = make_run()
    assert wf.build_input(wf.next_step("dub", "prep"), r) == {
        "run_id": "r1", "episode_id": 456,
        "segments_key": "dub-runs/r1/segments.json", "source_lang": "en", "language": "es"}
    assert wf.build_input(wf.next_step("dub", "text"), r) == {
        "run_id": "r1", "episode_id": 456,
        "segments_key": "dub-runs/r1/segments.json", "language": "es",
        "speaker_keys": {"SPEAKER_00": "dub-stems/456/speaker_SPEAKER_00.wav"}}
    assert wf.build_input(wf.next_step("dub", "synth"), r) == {
        "run_id": "r1", "episode_id": 456,
        "segments_key": "dub-runs/r1/segments.json", "language": "es", "bg_volume": 0.15}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_workflows.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'src.orchestrator.workflows'`.

- [ ] **Step 3: Implement**

```python
# src/orchestrator/workflows.py
"""Declarative pipeline workflows composed from step descriptors."""
from dataclasses import dataclass
from src.orchestrator.runs import Run


@dataclass(frozen=True)
class Step:
    name: str         # prep | text | synth | mux
    tier: str         # gpu | cpu
    progress: str     # buzz-bot progress label posted when this step starts


PREP  = Step("prep",  "gpu", "separating")
TEXT  = Step("text",  "cpu", "translating")
SYNTH = Step("synth", "gpu", "synthesizing")
MUX   = Step("mux",   "cpu", "assembling")

WORKFLOWS: dict[str, list[Step]] = {
    "dub":        [PREP, TEXT, SYNTH, MUX],
    "transcribe": [PREP],
}


def steps_for(workflow_type: str) -> list[Step]:
    return WORKFLOWS[workflow_type]


def first_step(workflow_type: str) -> Step:
    return steps_for(workflow_type)[0]


def step_by_name(workflow_type: str, name: str) -> Step:
    for s in steps_for(workflow_type):
        if s.name == name:
            return s
    raise KeyError(name)


def next_step(workflow_type: str, completed: str) -> Step | None:
    steps = steps_for(workflow_type)
    names = [s.name for s in steps]
    i = names.index(completed)
    return steps[i + 1] if i + 1 < len(steps) else None


def build_input(step: Step, run: Run) -> dict:
    base = {"run_id": run.id, "episode_id": run.episode_id}
    if step.name == "prep":
        return {**base, "audio_url": run.audio_url, "extract": run.workflow_type == "dub"}
    if step.name == "text":
        return {**base, "segments_key": run.segments_key,
                "source_lang": run.source_lang, "language": run.language}
    if step.name == "synth":
        return {**base, "segments_key": run.segments_key,
                "language": run.language, "speaker_keys": run.speaker_keys}
    if step.name == "mux":
        return {**base, "segments_key": run.segments_key,
                "language": run.language, "bg_volume": run.bg_volume}
    raise ValueError(step.name)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_workflows.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/orchestrator/workflows.py tests/test_workflows.py
git commit -m "feat: declarative dub/transcribe workflows + per-step input builder"
```

---

## Phase 5 — State machine (core)

### Task 7: `StateMachine.start` and happy-path advance

**Files:**
- Create: `src/orchestrator/state_machine.py`
- Test: `tests/test_state_machine.py`
- Test helper: `tests/fakes.py`

- [ ] **Step 1: Write the test fakes**

```python
# tests/fakes.py
class FakeDispatcher:
    def __init__(self, tier): self.tier = tier; self.calls = []
    def dispatch(self, step, run, payload, callback_url):
        self.calls.append((step.name, run.id, payload, callback_url))

class FakeReporter:
    def __init__(self):
        self.progress_calls = []; self.dub_results = []
        self.transcript_results = []; self.failures = []
    def progress(self, run, label, pct=None): self.progress_calls.append((run.id, label, pct))
    def dub_result(self, run): self.dub_results.append(run.id)
    def transcript_result(self, run): self.transcript_results.append(run.id)
    def failed(self, run, step, error): self.failures.append((run.id, step, error))
```

- [ ] **Step 2: Write the failing test**

```python
# tests/test_state_machine.py
from src.orchestrator.runs import Run, InMemoryRunStore
from src.orchestrator.state_machine import StateMachine
from tests.fakes import FakeDispatcher, FakeReporter

def build_sm():
    store = InMemoryRunStore()
    gpu = FakeDispatcher("gpu"); cpu = FakeDispatcher("cpu")
    rep = FakeReporter()
    sm = StateMachine(store, {"gpu": gpu, "cpu": cpu}, rep, max_attempts=3)
    return sm, store, gpu, cpu, rep

def make_run(**kw):
    base = dict(id="r1", workflow_type="dub", episode_id=456, callback_url="cb",
                audio_url="https://a.mp3", language="es")
    base.update(kw); return Run(**base)

def test_start_dispatches_prep_on_gpu_and_reports_progress():
    sm, store, gpu, cpu, rep = build_sm()
    sm.start(make_run())
    assert store.get("r1").current_step == "prep"
    assert gpu.calls[0][0] == "prep"
    assert gpu.calls[0][2] == {"run_id": "r1", "episode_id": 456,
                               "audio_url": "https://a.mp3", "extract": True}
    assert "token=" in gpu.calls[0][3]
    assert rep.progress_calls[0] == ("r1", "separating", None)

def test_prep_callback_advances_to_text_on_cpu():
    sm, store, gpu, cpu, rep = build_sm()
    sm.start(make_run())
    sm.handle_callback("r1", "prep", True, {
        "source_lang": "en", "speaker_keys": {"SPEAKER_00": "k"},
        "segments_key": "dub-runs/r1/segments.json"})
    run = store.get("r1")
    assert run.current_step == "text"
    assert run.source_lang == "en"
    assert run.speaker_keys == {"SPEAKER_00": "k"}
    assert run.segments_key == "dub-runs/r1/segments.json"
    assert cpu.calls[0][0] == "text"

def test_full_dub_chain_finalizes_with_dub_result():
    sm, store, gpu, cpu, rep = build_sm()
    sm.start(make_run())
    sm.handle_callback("r1", "prep", True, {"source_lang": "en",
        "speaker_keys": {"S": "k"}, "segments_key": "k1"})
    sm.handle_callback("r1", "text", True, {"segments_key": "k2"})
    sm.handle_callback("r1", "synth", True, {"segments_key": "k3"})
    sm.handle_callback("r1", "mux", True, {"segments_key": "k4",
        "r2_url": "https://r2/es.mp3", "duration_sec": 100.0, "segment_count": 3})
    run = store.get("r1")
    assert run.status == "done"
    assert run.r2_url == "https://r2/es.mp3"
    assert rep.dub_results == ["r1"]
    assert [c[1] for c in rep.progress_calls][-1] == "complete"

def test_transcribe_finalizes_after_prep_with_transcript_result():
    sm, store, gpu, cpu, rep = build_sm()
    sm.start(make_run(workflow_type="transcribe"))
    assert gpu.calls[0][2]["extract"] is False
    sm.handle_callback("r1", "prep", True, {"source_lang": "en",
        "speaker_keys": {}, "segments_key": "k1"})
    assert store.get("r1").status == "done"
    assert rep.transcript_results == ["r1"]

def test_failure_retries_then_gives_up():
    sm, store, gpu, cpu, rep = build_sm()
    sm.start(make_run())
    sm.handle_callback("r1", "prep", False, {"error": "OOM"})   # attempt 1 -> retry
    assert store.get("r1").attempts == 1
    assert len([c for c in gpu.calls if c[0] == "prep"]) == 2
    sm.handle_callback("r1", "prep", False, {"error": "OOM"})   # attempt 2 -> retry
    assert len([c for c in gpu.calls if c[0] == "prep"]) == 3
    sm.handle_callback("r1", "prep", False, {"error": "OOM"})   # attempt 3 -> fail
    assert store.get("r1").status == "failed"
    assert rep.failures == [("r1", "prep", "OOM")]

def test_successful_step_resets_attempts():
    sm, store, gpu, cpu, rep = build_sm()
    sm.start(make_run())
    sm.handle_callback("r1", "prep", False, {"error": "x"})     # attempts=1
    sm.handle_callback("r1", "prep", True, {"segments_key": "k1"})
    assert store.get("r1").attempts == 0
```

- [ ] **Step 3: Run test to verify it fails**

Run: `pytest tests/test_state_machine.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'src.orchestrator.state_machine'`.

- [ ] **Step 4: Implement**

```python
# src/orchestrator/state_machine.py
"""Advances a Run through its workflow, dispatching steps and reporting progress."""
from src import config
from src.orchestrator import auth, workflows
from src.orchestrator.runs import Run, RunStore

# Maps a step-completion callback's result fields onto Run columns.
_RESULT_FIELDS = ("segments_key", "source_lang", "speaker_keys",
                  "r2_url", "duration_sec", "segment_count")


class StateMachine:
    def __init__(self, store: RunStore, dispatchers: dict, reporter,
                 max_attempts: int = config.MAX_STEP_ATTEMPTS):
        self.store = store
        self.dispatchers = dispatchers
        self.reporter = reporter
        self.max_attempts = max_attempts

    def start(self, run: Run) -> None:
        self.store.create(run)
        self._dispatch(run, workflows.first_step(run.workflow_type))

    def handle_callback(self, run_id: str, step_name: str, ok: bool, result: dict) -> None:
        run = self.store.get(run_id)
        if not ok:
            self._on_failure(run, step_name, result.get("error"))
            return
        updates = {k: result[k] for k in _RESULT_FIELDS if k in result}
        run = self.store.update(run_id, attempts=0, **updates)
        nxt = workflows.next_step(run.workflow_type, step_name)
        if nxt is None:
            self._finalize(run)
        else:
            self._dispatch(run, nxt)

    # ── internals ──
    def _dispatch(self, run: Run, step) -> None:
        run = self.store.update(run.id, current_step=step.name)
        self.reporter.progress(run, step.progress)
        payload = workflows.build_input(step, run)
        token = auth.make_token(run.id, step.name)
        cb = (f"{config.ORCH_BASE_URL}/callback"
              f"?run_id={run.id}&step={step.name}&token={token}")
        self.dispatchers[step.tier].dispatch(step, run, payload, cb)

    def _on_failure(self, run: Run, step_name: str, error) -> None:
        if run.attempts + 1 < self.max_attempts:
            run = self.store.update(run.id, attempts=run.attempts + 1)
            self._dispatch(run, workflows.step_by_name(run.workflow_type, step_name))
        else:
            run = self.store.update(run.id, status="failed")
            self.reporter.failed(run, step_name, error)

    def _finalize(self, run: Run) -> None:
        run = self.store.update(run.id, status="done")
        if run.workflow_type == "transcribe":
            self.reporter.transcript_result(run)
        else:
            self.reporter.dub_result(run)
        self.reporter.progress(run, "complete", 100)
```

- [ ] **Step 5: Run test to verify it passes**

Run: `pytest tests/test_state_machine.py -v`
Expected: PASS (all 7).

- [ ] **Step 6: Commit**

```bash
git add src/orchestrator/state_machine.py tests/test_state_machine.py tests/fakes.py
git commit -m "feat: orchestrator state machine (dub+transcribe, retries, finalize)"
```

---

## Phase 6 — Reporter (buzz-bot callbacks)

### Task 8: `reporting.py` — progress/failed/dub_result/transcript_result

**Files:**
- Create: `src/orchestrator/reporting.py`
- Test: `tests/test_reporting.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_reporting.py
from unittest.mock import patch
from src.orchestrator.reporting import Reporter
from src.orchestrator.runs import Run

def make_run(**kw):
    base = dict(id="r1", workflow_type="dub", episode_id=456,
                callback_url="https://app/internal/dub_result", dub_id=123,
                language="es", source_lang="en", segments_key="k4",
                speaker_keys={"S0": "k", "S1": "k"}, r2_url="https://r2/es.mp3",
                duration_sec=100.0, segment_count=3)
    base.update(kw); return Run(**base)

SEGS = [{"idx": 0, "start_sec": 1.0, "end_sec": 2.0, "speaker": "S0",
         "text": "hi", "words": [{"w": "hi"}], "translated_text": "hola",
         "synth_r2_key": "dub-stems/456/synth_es_0000.wav",
         "synth_duration": 1.1, "synth_start_sec": 1.0}]

def test_progress_posts_to_progress_url():
    with patch("src.orchestrator.reporting.requests.post") as post, \
         patch("src.orchestrator.reporting.config.PROGRESS_URL", "https://app/internal/dub_progress"):
        Reporter().progress(make_run(), "synthesizing", 40)
    args, kwargs = post.call_args
    assert args[0] == "https://app/internal/dub_progress"
    assert kwargs["json"] == {"dub_id": 123, "step": "synthesizing", "pct": 40}

def test_dub_result_builds_buzzbot_payload():
    with patch("src.orchestrator.reporting.artifacts.read_segments", return_value=SEGS), \
         patch("src.orchestrator.reporting.requests.post") as post:
        Reporter().dub_result(make_run())
    body = post.call_args.kwargs["json"]
    assert body["dub_id"] == 123 and body["success"] is True
    assert body["r2_url"] == "https://r2/es.mp3"
    assert body["source_lang"] == "en" and body["language"] == "es"
    assert body["segment_count"] == 3 and body["speaker_count"] == 2
    seg = body["segments"][0]
    assert seg["idx"] == 0 and seg["translated_text"] == "hola"
    assert seg["synth_r2_key"] == "dub-stems/456/synth_es_0000.wav"
    assert seg["synth_start_sec"] == 1.0
    assert post.call_args.args[0] == "https://app/internal/dub_result"

def test_failed_posts_failure_shape():
    with patch("src.orchestrator.reporting.requests.post") as post:
        Reporter().failed(make_run(), "separating", "Demucs OOM")
    body = post.call_args.kwargs["json"]
    assert body == {"dub_id": 123, "success": False,
                    "step": "separating", "error": "Demucs OOM"}

def test_transcript_result_posts_to_transcript_url():
    with patch("src.orchestrator.reporting.artifacts.read_segments", return_value=SEGS), \
         patch("src.orchestrator.reporting.requests.post") as post, \
         patch("src.orchestrator.reporting.config.BUZZBOT_TRANSCRIPT_URL",
               "https://app/internal/transcript_result"):
        Reporter().transcript_result(make_run(workflow_type="transcribe"))
    assert post.call_args.args[0] == "https://app/internal/transcript_result"
    body = post.call_args.kwargs["json"]
    assert body["episode_id"] == 456 and body["source_lang"] == "en"
    assert body["segments"][0]["text"] == "hi"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_reporting.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'src.orchestrator.reporting'`.

- [ ] **Step 3: Implement**

```python
# src/orchestrator/reporting.py
"""Posts progress + final results back to buzz-bot (matches existing contract)."""
import logging
import requests
from src import artifacts, config
from src.orchestrator.runs import Run

log = logging.getLogger(__name__)
_TIMEOUT = 30


def _post(url: str, body: dict) -> None:
    try:
        requests.post(url, json=body, timeout=_TIMEOUT)
    except Exception as e:  # noqa: BLE001 — best-effort callback
        log.warning("callback failed (%s): %s", url, e)


class Reporter:
    def progress(self, run: Run, label: str, pct: int | None = None) -> None:
        body = {"dub_id": run.dub_id, "step": label}
        if pct is not None:
            body["pct"] = pct
        _post(config.PROGRESS_URL, body)

    def failed(self, run: Run, step_label: str, error) -> None:
        _post(config.BUZZBOT_RESULT_URL, {
            "dub_id": run.dub_id,
            "success": False,
            "step": step_label,
            "error": str(error) if error is not None else None,
        })

    def dub_result(self, run: Run) -> None:
        segments = artifacts.read_segments(run.segments_key)
        _post(config.BUZZBOT_RESULT_URL, {
            "dub_id": run.dub_id,
            "episode_id": run.episode_id,
            "language": run.language,
            "source_lang": run.source_lang,
            "success": True,
            "r2_url": run.r2_url,
            "duration_sec": run.duration_sec,
            "segment_count": run.segment_count,
            "speaker_count": len(run.speaker_keys or {}),
            "segments": [_segment_payload(s) for s in segments],
        })

    def transcript_result(self, run: Run) -> None:
        segments = artifacts.read_segments(run.segments_key)
        _post(config.BUZZBOT_TRANSCRIPT_URL, {
            "episode_id": run.episode_id,
            "source_lang": run.source_lang,
            "segments": [_segment_payload(s) for s in segments],
        })


def _segment_payload(s: dict) -> dict:
    return {
        "idx": s["idx"],
        "start_sec": s.get("start_sec"),
        "end_sec": s.get("end_sec"),
        "speaker_id": s.get("speaker"),
        "text": s.get("text", ""),
        "words": s.get("words"),
        "translated_text": s.get("translated_text"),
        "synth_r2_key": s.get("synth_r2_key"),
        "synth_duration": s.get("synth_duration"),
        "synth_start_sec": s.get("synth_start_sec"),
    }
```

> The failure test (`test_failed_posts_failure_shape`) asserts the body shape; `failed` posts it to `config.BUZZBOT_RESULT_URL` (buzz-bot's `/internal/dub_result` handles both success and failure payloads today).

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_reporting.py -v`
Expected: PASS (4).

- [ ] **Step 5: Commit**

```bash
git add src/orchestrator/reporting.py tests/test_reporting.py
git commit -m "feat: Reporter — progress/failure/dub_result/transcript_result callbacks"
```

---

## Phase 7 — Worker adapters

> Workers are thin: parse input → call existing `src/steps` functions → upload artifacts → return a result dict. The `run(input) -> dict` function is pure-ish and unit-tested with mocked steps/storage; `main()` wraps it with callback posting.

### Task 9: `workers/common.py`

**Files:**
- Create: `src/workers/common.py`
- Test: `tests/test_worker_common.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_worker_common.py
from unittest.mock import patch
from src.workers import common

def test_post_callback_ok():
    with patch("src.workers.common.requests.post") as post:
        common.post_callback("http://cb", {"x": 1})
    assert post.call_args.args[0] == "http://cb"
    assert post.call_args.kwargs["json"] == {"x": 1}

def test_run_in_tempdir_cleans_up(tmp_path):
    seen = {}
    def body(d):
        seen["dir"] = d
        assert d  # exists during call
        return "result"
    out = common.run_in_tempdir(body)
    assert out == "result"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_worker_common.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'src.workers.common'`.

- [ ] **Step 3: Implement**

```python
# src/workers/common.py
"""Shared worker helpers: callback POST + scratch-dir lifecycle."""
import shutil
import tempfile
import requests
from src import config


def post_callback(callback_url: str, body: dict) -> None:
    requests.post(callback_url, json=body, timeout=30)


def run_in_tempdir(body):
    """Call body(work_dir) in a fresh temp dir, always cleaning it up."""
    import os
    os.makedirs(config.TEMP_DIR, exist_ok=True)
    work_dir = tempfile.mkdtemp(dir=config.TEMP_DIR, prefix="dubstep_")
    try:
        return body(work_dir)
    finally:
        shutil.rmtree(work_dir, ignore_errors=True)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_worker_common.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/workers/common.py tests/test_worker_common.py
git commit -m "feat: worker common helpers (callback post + tempdir)"
```

### Task 10: `workers/gpu_prep.py`

**Files:**
- Create: `src/workers/gpu_prep.py`
- Test: `tests/test_gpu_prep.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_gpu_prep.py
from unittest.mock import patch
from src.workers import gpu_prep

SEGS = [{"idx": 0, "start_sec": 0.0, "end_sec": 2.0, "speaker": "SPEAKER_00",
         "text": "hi", "words": []}]

def base_mocks(extract=True):
    return {
        "separate": patch("src.workers.gpu_prep.separate.separate",
                          return_value=("/w/vocals.wav", "/w/background.wav")),
        "transcribe": patch("src.workers.gpu_prep.transcribe.transcribe",
                            return_value=(SEGS, "en")),
        "extract": patch("src.workers.gpu_prep.extract_samples.extract_samples",
                         return_value={"SPEAKER_00": "/w/speaker_SPEAKER_00.wav"}),
        "upload": patch("src.workers.gpu_prep.storage.upload", return_value="url"),
        "write": patch("src.workers.gpu_prep.artifacts.write_segments",
                      return_value="dub-runs/r1/segments.json"),
        "tmp": patch("src.workers.gpu_prep.common.run_in_tempdir",
                    side_effect=lambda body: body("/w")),
    }

def test_prep_dub_uploads_stems_speakers_and_returns_keys():
    m = base_mocks()
    with m["separate"], m["transcribe"], m["extract"], m["upload"] as up, \
         m["write"], m["tmp"]:
        out = gpu_prep.run({"run_id": "r1", "episode_id": 456,
                            "audio_url": "https://a.mp3", "extract": True})
    assert out["source_lang"] == "en"
    assert out["segments_key"] == "dub-runs/r1/segments.json"
    assert out["speaker_keys"] == {"SPEAKER_00": "dub-stems/456/speaker_SPEAKER_00.wav"}
    uploaded_keys = [c.args[1] for c in up.call_args_list]
    assert "dub-stems/456/vocals.wav" in uploaded_keys
    assert "dub-stems/456/background.wav" in uploaded_keys
    assert "dub-stems/456/speaker_SPEAKER_00.wav" in uploaded_keys

def test_prep_transcribe_skips_extract_and_returns_empty_speakers():
    m = base_mocks()
    with m["separate"], m["transcribe"], \
         patch("src.workers.gpu_prep.extract_samples.extract_samples") as ex, \
         m["upload"], m["write"], m["tmp"]:
        out = gpu_prep.run({"run_id": "r1", "episode_id": 456,
                            "audio_url": "https://a.mp3", "extract": False})
    ex.assert_not_called()
    assert out["speaker_keys"] == {}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_gpu_prep.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'src.workers.gpu_prep'`.

- [ ] **Step 3: Implement**

```python
# src/workers/gpu_prep.py
"""GPU step group: separate -> transcribe -> (extract). Nebius job entrypoint."""
import json
import logging
import os
import sys

from src import artifacts, storage
from src.steps import separate, transcribe, extract_samples
from src.workers import common

log = logging.getLogger(__name__)


def run(inp: dict) -> dict:
    run_id     = inp["run_id"]
    episode_id = inp["episode_id"]
    audio_url  = inp["audio_url"]
    do_extract = inp.get("extract", True)

    def body(work_dir: str) -> dict:
        vocals, background = separate.separate(audio_url, work_dir)
        storage.upload(vocals,     artifacts.stem_key(episode_id, "vocals.wav"),     "audio/wav")
        storage.upload(background, artifacts.stem_key(episode_id, "background.wav"), "audio/wav")

        segments, source_lang = transcribe.transcribe(vocals)

        speaker_keys: dict[str, str] = {}
        if do_extract:
            samples = extract_samples.extract_samples(segments, vocals, work_dir)
            for speaker, local_path in samples.items():
                key = artifacts.stem_key(episode_id, f"speaker_{speaker}.wav")
                storage.upload(local_path, key, "audio/wav")
                speaker_keys[speaker] = key

        segments_key = artifacts.write_segments(run_id, segments)
        return {"source_lang": source_lang, "speaker_keys": speaker_keys,
                "segments_key": segments_key}

    return common.run_in_tempdir(body)


def main() -> None:
    inp = json.loads(os.environ["INPUT_JSON"])
    callback_url = os.environ["CALLBACK_URL"]
    try:
        result = run(inp)
        common.post_callback(callback_url, {"ok": True, **result})
    except Exception as e:  # noqa: BLE001
        log.exception("gpu_prep failed")
        common.post_callback(callback_url, {"ok": False, "error": str(e)})
        sys.exit(1)


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_gpu_prep.py -v`
Expected: PASS (2).

- [ ] **Step 5: Commit**

```bash
git add src/workers/gpu_prep.py tests/test_gpu_prep.py
git commit -m "feat: gpu-prep worker (separate+transcribe+extract, extract flag)"
```

### Task 11: `workers/gpu_synth.py`

**Files:**
- Create: `src/workers/gpu_synth.py`
- Test: `tests/test_gpu_synth.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_gpu_synth.py
from unittest.mock import patch
from src.workers import gpu_synth

IN_SEGS = [
    {"idx": 0, "speaker": "SPEAKER_00", "translated_text": "hola", "start_sec": 0, "end_sec": 2},
    {"idx": 1, "speaker": "SPEAKER_00", "translated_text": "", "start_sec": 2, "end_sec": 3},
]
SYNTHED = [
    {**IN_SEGS[0], "synth_wav": "/w/synth_0000.wav", "synth_duration": 1.1},
    {**IN_SEGS[1], "synth_wav": None, "synth_duration": None},
]

def test_synth_uploads_wavs_and_records_keys():
    with patch("src.workers.gpu_synth.artifacts.read_segments", return_value=IN_SEGS), \
         patch("src.workers.gpu_synth.storage.download"), \
         patch("src.workers.gpu_synth.synthesize.synthesize", return_value=SYNTHED), \
         patch("src.workers.gpu_synth.storage.upload", return_value="url") as up, \
         patch("src.workers.gpu_synth.artifacts.write_segments",
               return_value="dub-runs/r1/segments.json") as wr, \
         patch("src.workers.gpu_synth.common.run_in_tempdir",
               side_effect=lambda body: body("/w")):
        out = gpu_synth.run({"run_id": "r1", "episode_id": 456, "language": "es",
                             "segments_key": "k", "speaker_keys": {"SPEAKER_00": "spk-key"}})
    assert out["segments_key"] == "dub-runs/r1/segments.json"
    saved = wr.call_args.args[1]
    assert saved[0]["synth_r2_key"] == "dub-stems/456/synth_es_0000.wav"
    assert saved[0]["synth_duration"] == 1.1
    assert "synth_r2_key" not in saved[1] or saved[1]["synth_r2_key"] is None
    assert "dub-stems/456/synth_es_0000.wav" in [c.args[1] for c in up.call_args_list]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_gpu_synth.py -v`
Expected: FAIL — `ModuleNotFoundError`.

- [ ] **Step 3: Implement**

```python
# src/workers/gpu_synth.py
"""GPU step: synthesize translated segments with VoxCPM2. Nebius job entrypoint."""
import json
import logging
import os
import sys

from src import artifacts, storage
from src.steps import synthesize
from src.workers import common

log = logging.getLogger(__name__)


def run(inp: dict) -> dict:
    run_id       = inp["run_id"]
    episode_id   = inp["episode_id"]
    language     = inp["language"]
    segments_key = inp["segments_key"]
    speaker_keys = inp.get("speaker_keys") or {}

    segments = artifacts.read_segments(segments_key)

    def body(work_dir: str) -> dict:
        # Download speaker samples referenced by key to local paths.
        speaker_samples: dict[str, str] = {}
        for speaker, key in speaker_keys.items():
            local = os.path.join(work_dir, f"speaker_{speaker}.wav")
            storage.download(key, local)
            speaker_samples[speaker] = local

        synthed = synthesize.synthesize(segments, speaker_samples, language, work_dir)

        out_segments = []
        for seg in synthed:
            wav = seg.get("synth_wav")
            if wav:
                key = artifacts.stem_key(episode_id, f"synth_{language}_{seg['idx']:04d}.wav")
                storage.upload(wav, key, "audio/wav")
                seg = {**seg, "synth_r2_key": key}
            seg.pop("synth_wav", None)  # local path not useful downstream
            out_segments.append(seg)

        segments_key_out = artifacts.write_segments(run_id, out_segments)
        return {"segments_key": segments_key_out}

    return common.run_in_tempdir(body)


def main() -> None:
    inp = json.loads(os.environ["INPUT_JSON"])
    callback_url = os.environ["CALLBACK_URL"]
    try:
        common.post_callback(callback_url, {"ok": True, **run(inp)})
    except Exception as e:  # noqa: BLE001
        log.exception("gpu_synth failed")
        common.post_callback(callback_url, {"ok": False, "error": str(e)})
        sys.exit(1)


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_gpu_synth.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/workers/gpu_synth.py tests/test_gpu_synth.py
git commit -m "feat: gpu-synth worker (VoxCPM2 synth, uploads per-segment wavs)"
```

### Task 12: `workers/cpu_text.py`

**Files:**
- Create: `src/workers/cpu_text.py`
- Test: `tests/test_cpu_text.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_cpu_text.py
from unittest.mock import patch
from src.workers import cpu_text

SEGS = [{"idx": 0, "text": "hi", "start_sec": 0, "end_sec": 40}]
SPLIT = [{"idx": 0, "text": "hi", "start_sec": 0, "end_sec": 20},
         {"idx": 1, "text": "there", "start_sec": 20, "end_sec": 40}]
TRANSLATED = [{**SPLIT[0], "translated_text": "hola"},
              {**SPLIT[1], "translated_text": "ahi"}]

def test_text_splits_then_translates_and_writes_artifact():
    with patch("src.workers.cpu_text.artifacts.read_segments", return_value=SEGS), \
         patch("src.workers.cpu_text.split_segments.split_long_segments", return_value=SPLIT) as sp, \
         patch("src.workers.cpu_text.translate.translate", return_value=TRANSLATED) as tr, \
         patch("src.workers.cpu_text.artifacts.write_segments",
               return_value="dub-runs/r1/segments.json") as wr:
        out = cpu_text.run({"run_id": "r1", "episode_id": 456,
                            "segments_key": "k", "source_lang": "en", "language": "es"})
    sp.assert_called_once_with(SEGS)
    tr.assert_called_once_with(SPLIT, "en", "es")
    assert wr.call_args.args == ("r1", TRANSLATED)
    assert out == {"segments_key": "dub-runs/r1/segments.json"}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_cpu_text.py -v`
Expected: FAIL — `ModuleNotFoundError`.

- [ ] **Step 3: Implement**

```python
# src/workers/cpu_text.py
"""CPU step group: split -> translate. Runs as an HTTP worker (FastAPI)."""
import logging

from fastapi import FastAPI, Request
from src import artifacts
from src.steps import split_segments, translate
from src.workers import common

log = logging.getLogger(__name__)


def run(inp: dict) -> dict:
    segments = artifacts.read_segments(inp["segments_key"])
    segments = split_segments.split_long_segments(segments)
    segments = translate.translate(segments, inp["source_lang"], inp["language"])
    return {"segments_key": artifacts.write_segments(inp["run_id"], segments)}


app = FastAPI()


@app.post("/run")
async def run_endpoint(request: Request):
    payload = await request.json()
    inp = payload["input"]
    callback_url = payload["callback_url"]
    try:
        common.post_callback(callback_url, {"ok": True, **run(inp)})
    except Exception as e:  # noqa: BLE001
        log.exception("cpu_text failed")
        common.post_callback(callback_url, {"ok": False, "error": str(e)})
    return {"accepted": True}


@app.get("/healthz")
async def healthz():
    return {"ok": True}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_cpu_text.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/workers/cpu_text.py tests/test_cpu_text.py
git commit -m "feat: cpu-text worker (split+translate, HTTP)"
```

### Task 13: `workers/cpu_mux.py`

**Files:**
- Create: `src/workers/cpu_mux.py`
- Test: `tests/test_cpu_mux.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_cpu_mux.py
from unittest.mock import patch
from src.workers import cpu_mux

SEGS = [
    {"idx": 0, "speaker": "S0", "synth_r2_key": "dub-stems/456/synth_es_0000.wav",
     "start_sec": 0, "end_sec": 2, "translated_text": "hola"},
    {"idx": 1, "speaker": "S0", "synth_r2_key": None,
     "start_sec": 2, "end_sec": 3, "translated_text": ""},
]
ASSEMBLED = [
    {**SEGS[0], "synth_start_sec": 0.0},
    {**SEGS[1], "synth_start_sec": 2.0},
]

def test_mux_downloads_synth_assembles_mixes_uploads():
    with patch("src.workers.cpu_mux.artifacts.read_segments", return_value=SEGS), \
         patch("src.workers.cpu_mux.storage.download") as dl, \
         patch("src.workers.cpu_mux.assemble.assemble",
               return_value=("/w/dubbed_vocals.wav", ASSEMBLED)), \
         patch("src.workers.cpu_mux.mix.mix", return_value="/w/final.mp3"), \
         patch("src.workers.cpu_mux.storage.upload", return_value="https://r2/es.mp3") as up, \
         patch("src.workers.cpu_mux._ffprobe_duration", return_value=100.0), \
         patch("src.workers.cpu_mux.artifacts.write_segments",
               return_value="dub-runs/r1/segments.json") as wr, \
         patch("src.workers.cpu_mux.common.run_in_tempdir",
               side_effect=lambda body: body("/w")):
        out = cpu_mux.run({"run_id": "r1", "episode_id": 456,
                           "segments_key": "k", "language": "es", "bg_volume": 0.15})
    # Background + the one real synth segment are downloaded.
    dl_keys = [c.args[0] for c in dl.call_args_list]
    assert "dub-stems/456/background.wav" in dl_keys
    assert "dub-stems/456/synth_es_0000.wav" in dl_keys
    assert out["r2_url"] == "https://r2/es.mp3"
    assert out["duration_sec"] == 100.0
    assert out["segment_count"] == 1     # one segment actually synthesized
    assert up.call_args.args[1] == "dubbed/456/es.mp3"
    assert out["segments_key"] == "dub-runs/r1/segments.json"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_cpu_mux.py -v`
Expected: FAIL — `ModuleNotFoundError`.

- [ ] **Step 3: Implement**

```python
# src/workers/cpu_mux.py
"""CPU step group: assemble -> mix -> upload final mp3. Runs as an HTTP worker."""
import logging
import os
import subprocess

from fastapi import FastAPI, Request
from src import artifacts, storage
from src.steps import assemble, mix
from src.workers import common

log = logging.getLogger(__name__)


def _ffprobe_duration(path: str) -> float:
    result = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration",
         "-of", "default=noprint_wrappers=1:nokey=1", path],
        capture_output=True, text=True, check=True)
    return float(result.stdout.strip())


def run(inp: dict) -> dict:
    run_id     = inp["run_id"]
    episode_id = inp["episode_id"]
    language   = inp["language"]
    bg_volume  = inp.get("bg_volume", 0.15)

    segments = artifacts.read_segments(inp["segments_key"])

    def body(work_dir: str) -> dict:
        background = os.path.join(work_dir, "background.wav")
        storage.download(artifacts.stem_key(episode_id, "background.wav"), background)

        # Pull each synthesized segment back to a local path for assembly.
        for seg in segments:
            key = seg.get("synth_r2_key")
            if key:
                local = os.path.join(work_dir, f"synth_{seg['idx']:04d}.wav")
                storage.download(key, local)
                seg["synth_wav"] = local
            else:
                seg["synth_wav"] = None

        dubbed_vocals, assembled = assemble.assemble(segments, work_dir)
        final_mp3 = mix.mix(dubbed_vocals, background, work_dir, bg_volume=bg_volume)

        r2_url = storage.upload(final_mp3, artifacts.dub_key(episode_id, language), "audio/mpeg")
        duration = _ffprobe_duration(final_mp3)
        count = len([s for s in assembled if s.get("synth_r2_key")])

        for seg in assembled:
            seg.pop("synth_wav", None)
        segments_key_out = artifacts.write_segments(run_id, assembled)
        return {"r2_url": r2_url, "duration_sec": duration,
                "segment_count": count, "segments_key": segments_key_out}

    return common.run_in_tempdir(body)


app = FastAPI()


@app.post("/run")
async def run_endpoint(request: Request):
    payload = await request.json()
    try:
        common.post_callback(payload["callback_url"], {"ok": True, **run(payload["input"])})
    except Exception as e:  # noqa: BLE001
        log.exception("cpu_mux failed")
        common.post_callback(payload["callback_url"], {"ok": False, "error": str(e)})
    return {"accepted": True}


@app.get("/healthz")
async def healthz():
    return {"ok": True}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_cpu_mux.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/workers/cpu_mux.py tests/test_cpu_mux.py
git commit -m "feat: cpu-mux worker (assemble+mix+upload, HTTP)"
```

---

## Phase 8 — Dispatchers

### Task 14: `Dispatcher` protocol + `HttpDispatcher` + `LocalDispatcher`

**Files:**
- Create: `src/orchestrator/dispatch.py`
- Test: `tests/test_dispatch.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_dispatch.py
from unittest.mock import patch, MagicMock
from src.orchestrator.dispatch import HttpDispatcher, LocalDispatcher
from src.orchestrator.workflows import TEXT, PREP
from src.orchestrator.runs import Run

def make_run():
    return Run(id="r1", workflow_type="dub", episode_id=456, callback_url="cb",
               audio_url="https://a.mp3", language="es")

def test_http_dispatcher_posts_input_and_callback_to_worker_url():
    d = HttpDispatcher({"text": "http://cpu-text/run"})
    with patch("src.orchestrator.dispatch.requests.post") as post:
        d.dispatch(TEXT, make_run(), {"run_id": "r1"}, "http://orch/callback?x=1")
    assert post.call_args.args[0] == "http://cpu-text/run"
    assert post.call_args.kwargs["json"] == {
        "input": {"run_id": "r1"}, "callback_url": "http://orch/callback?x=1"}

def test_local_dispatcher_runs_worker_and_posts_callback():
    worker = MagicMock(return_value={"segments_key": "k"})
    d = LocalDispatcher({"prep": worker})
    with patch("src.orchestrator.dispatch.requests.post") as post:
        d.dispatch(PREP, make_run(), {"run_id": "r1"}, "http://orch/callback")
    worker.assert_called_once_with({"run_id": "r1"})
    assert post.call_args.args[0] == "http://orch/callback"
    assert post.call_args.kwargs["json"] == {"ok": True, "segments_key": "k"}

def test_local_dispatcher_reports_failure():
    worker = MagicMock(side_effect=RuntimeError("boom"))
    d = LocalDispatcher({"prep": worker})
    with patch("src.orchestrator.dispatch.requests.post") as post:
        d.dispatch(PREP, make_run(), {"run_id": "r1"}, "http://orch/callback")
    assert post.call_args.kwargs["json"] == {"ok": False, "error": "boom"}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_dispatch.py -v`
Expected: FAIL — `ModuleNotFoundError`.

- [ ] **Step 3: Implement**

```python
# src/orchestrator/dispatch.py
"""Dispatchers turn a (step, run, payload, callback) into an actual invocation.

- HttpDispatcher  : POST to a long-running CPU worker's /run endpoint.
- LocalDispatcher : run the worker function in-process (dev / local e2e).
- NebiusDispatcher: launches a Nebius GPU job — added in Plan 2.
"""
import logging
import threading
from typing import Callable, Protocol

import requests
from src.orchestrator.runs import Run
from src.orchestrator.workflows import Step

log = logging.getLogger(__name__)


class Dispatcher(Protocol):
    def dispatch(self, step: Step, run: Run, payload: dict, callback_url: str) -> None: ...


class HttpDispatcher:
    """Dispatch a step to a CPU worker reachable at a per-step URL."""
    def __init__(self, urls: dict[str, str]):
        self.urls = urls

    def dispatch(self, step: Step, run: Run, payload: dict, callback_url: str) -> None:
        requests.post(self.urls[step.name],
                      json={"input": payload, "callback_url": callback_url}, timeout=30)


class LocalDispatcher:
    """Run a worker's `run(input)->dict` in a background thread, then POST the callback."""
    def __init__(self, workers: dict[str, Callable[[dict], dict]]):
        self.workers = workers

    def dispatch(self, step: Step, run: Run, payload: dict, callback_url: str) -> None:
        worker = self.workers[step.name]

        def _go():
            try:
                result = worker(payload)
                requests.post(callback_url, json={"ok": True, **result}, timeout=30)
            except Exception as e:  # noqa: BLE001
                log.exception("local worker %s failed", step.name)
                requests.post(callback_url, json={"ok": False, "error": str(e)}, timeout=30)

        _go()  # synchronous in tests; see note for local e2e
```

> For the local end-to-end run (Task 17) the synchronous `_go()` would recurse through the state machine on the same stack. That is acceptable for a short clip. If depth becomes an issue, swap the last line to:
> `threading.Thread(target=_go, daemon=True).start()`
> The unit tests above assert synchronous behavior, so keep `_go()` direct for now.

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_dispatch.py -v`
Expected: PASS (3).

- [ ] **Step 5: Commit**

```bash
git add src/orchestrator/dispatch.py tests/test_dispatch.py
git commit -m "feat: HttpDispatcher + LocalDispatcher (Nebius dispatcher deferred to Plan 2)"
```

---

## Phase 9 — Orchestrator HTTP app

### Task 15: FastAPI `/dispatch` + `/callback`

**Files:**
- Create: `src/orchestrator/app.py`
- Test: `tests/test_app.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_app.py
from unittest.mock import patch
from fastapi.testclient import TestClient
from src.orchestrator import app as appmod
from src.orchestrator import auth

def client_with_fakes():
    # Rebuild the app wiring against an in-memory store + fake dispatchers/reporter.
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_app.py -v`
Expected: FAIL — `AttributeError: module 'src.orchestrator.app' has no attribute 'configure'`.

- [ ] **Step 3: Implement**

```python
# src/orchestrator/app.py
"""Orchestrator HTTP surface: POST /dispatch (start) and POST /callback (advance)."""
import logging

from fastapi import FastAPI, Request, Response
from src.orchestrator import auth
from src.orchestrator.runs import Run
from src.orchestrator.state_machine import StateMachine

log = logging.getLogger(__name__)
app = FastAPI()

_sm: StateMachine | None = None


def configure(store, dispatchers, reporter) -> None:
    """Wire the app to its collaborators (called at startup and from tests)."""
    global _sm
    _sm = StateMachine(store, dispatchers, reporter)


@app.post("/dispatch", status_code=202)
async def dispatch(request: Request):
    b = await request.json()
    run = Run(
        id=b["run_id"], workflow_type=b["workflow_type"], episode_id=b["episode_id"],
        callback_url=b["callback_url"], dub_id=b.get("dub_id"),
        language=b.get("language"), audio_url=b.get("audio_url"),
        bg_volume=b.get("bg_volume", 0.15))
    _sm.start(run)
    return {"run_id": run.id, "status": "started"}


@app.post("/callback")
async def callback(run_id: str, step: str, token: str, request: Request):
    if not auth.verify_token(run_id, step, token):
        return Response(status_code=401)
    body = await request.json()
    ok = body.pop("ok", False)
    _sm.handle_callback(run_id, step, ok, body)
    return {"ok": True}


@app.get("/healthz")
async def healthz():
    return {"ok": True}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_app.py -v`
Expected: PASS (3).

- [ ] **Step 5: Commit**

```bash
git add src/orchestrator/app.py tests/test_app.py
git commit -m "feat: orchestrator FastAPI app (/dispatch, /callback with token auth)"
```

---

## Phase 10 — Postgres-backed store

### Task 16: `PgRunStore` + schema

**Files:**
- Create: `src/orchestrator/schema.sql`
- Modify: `src/orchestrator/runs.py`
- Test: `tests/test_pg_runs.py` (gated on `DATABASE_URL`)

- [ ] **Step 1: Write the schema**

```sql
-- src/orchestrator/schema.sql
CREATE TABLE IF NOT EXISTS orch_run (
    id            TEXT PRIMARY KEY,
    workflow_type TEXT NOT NULL,
    episode_id    BIGINT NOT NULL,
    callback_url  TEXT NOT NULL,
    dub_id        BIGINT,
    language      TEXT,
    audio_url     TEXT,
    bg_volume     DOUBLE PRECISION NOT NULL DEFAULT 0.15,
    status        TEXT NOT NULL DEFAULT 'running',
    current_step  TEXT NOT NULL DEFAULT '',
    attempts      INTEGER NOT NULL DEFAULT 0,
    segments_key  TEXT,
    source_lang   TEXT,
    speaker_keys  JSONB,
    r2_url        TEXT,
    duration_sec  DOUBLE PRECISION,
    segment_count INTEGER,
    created_at    TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at    TIMESTAMPTZ NOT NULL DEFAULT now()
);
```

- [ ] **Step 2: Write the failing test**

```python
# tests/test_pg_runs.py
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
```

- [ ] **Step 3: Run test to verify it fails (or skips without DB)**

Run: `pytest tests/test_pg_runs.py -v`
Expected: FAIL `ImportError: cannot import name 'PgRunStore'` (or SKIP if `DATABASE_URL` unset — set a throwaway Neon/local URL to actually exercise it).

- [ ] **Step 4: Implement `PgRunStore`**

Append to `src/orchestrator/runs.py`:
```python
import json
import os
from dataclasses import asdict, fields as dataclass_fields

import psycopg

_PERSISTED = [f.name for f in dataclass_fields(Run)]  # all Run columns


class PgRunStore:
    def __init__(self, dsn: str):
        self.dsn = dsn

    def _conn(self):
        return psycopg.connect(self.dsn, autocommit=True)

    def init_schema(self) -> None:
        ddl = open(os.path.join(os.path.dirname(__file__), "schema.sql")).read()
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
        if "speaker_keys" in fields and fields["speaker_keys"] is not None:
            fields = {**fields, "speaker_keys": json.dumps(fields["speaker_keys"])}
        sets = ", ".join(f"{k} = %({k})s" for k in fields)
        params = {**fields, "rid": run_id}
        with self._conn() as c:
            c.execute(f"UPDATE orch_run SET {sets}, updated_at = now() WHERE id = %(rid)s", params)
        return self.get(run_id)
```

> `psycopg` returns `JSONB` columns already decoded to Python dicts, so `get` needs no manual `json.loads` for `speaker_keys`. Verify in Step 5; if your psycopg returns a string, add `data["speaker_keys"] = json.loads(...)` guarding for `None`.

- [ ] **Step 5: Run test to verify it passes**

Run: `DATABASE_URL=<your test pg url> pytest tests/test_pg_runs.py -v`
Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add src/orchestrator/schema.sql src/orchestrator/runs.py tests/test_pg_runs.py
git commit -m "feat: PgRunStore + orch_run schema"
```

---

## Phase 11 — Local end-to-end wiring

### Task 17: `run_local.py` — drive the whole pipeline with `LocalDispatcher`

**Files:**
- Create: `src/orchestrator/run_local.py`
- Test: `tests/test_run_local.py`

- [ ] **Step 1: Write the failing test (mocked steps, real wiring)**

```python
# tests/test_run_local.py
from unittest.mock import patch
from src.orchestrator import run_local

# End-to-end through StateMachine + LocalDispatcher + real workers,
# with the heavy step functions and storage mocked.
def test_local_dub_runs_all_four_steps_and_reports_result():
    seg = [{"idx": 0, "start_sec": 0, "end_sec": 2, "speaker": "S0",
            "text": "hi", "words": [], "translated_text": "hola",
            "synth_r2_key": "dub-stems/456/synth_es_0000.wav",
            "synth_duration": 1.1, "synth_start_sec": 0.0}]
    posted = {}
    with patch("src.workers.gpu_prep.separate.separate", return_value=("/w/v.wav", "/w/b.wav")), \
         patch("src.workers.gpu_prep.transcribe.transcribe", return_value=(seg, "en")), \
         patch("src.workers.gpu_prep.extract_samples.extract_samples",
               return_value={"S0": "/w/s.wav"}), \
         patch("src.workers.gpu_synth.synthesize.synthesize",
               return_value=[{**seg[0], "synth_wav": "/w/synth_0000.wav", "synth_duration": 1.1}]), \
         patch("src.workers.cpu_text.split_segments.split_long_segments", side_effect=lambda s: s), \
         patch("src.workers.cpu_text.translate.translate", side_effect=lambda s, a, b: s), \
         patch("src.workers.cpu_mux.assemble.assemble", return_value=("/w/dv.wav", seg)), \
         patch("src.workers.cpu_mux.mix.mix", return_value="/w/final.mp3"), \
         patch("src.workers.cpu_mux._ffprobe_duration", return_value=100.0), \
         patch("src.storage.upload", return_value="https://r2/es.mp3"), \
         patch("src.storage.download"), \
         patch("src.artifacts.write_segments", side_effect=lambda rid, s: "segkey"), \
         patch("src.artifacts.read_segments", return_value=seg), \
         patch("src.orchestrator.reporting.requests.post",
               side_effect=lambda url, **kw: posted.setdefault("last", (url, kw["json"]))), \
         patch("src.config.BUZZBOT_RESULT_URL", "https://app/internal/dub_result"), \
         patch("src.config.PROGRESS_URL", "https://app/internal/dub_progress"):
        store = run_local.run_dub_locally(
            run_id="r1", dub_id=123, episode_id=456,
            audio_url="https://a.mp3", language="es")
    assert store.get("r1").status == "done"
    assert posted["last"][0] == "https://app/internal/dub_result"
    assert posted["last"][1]["success"] is True
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_run_local.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'src.orchestrator.run_local'`.

- [ ] **Step 3: Implement**

```python
# src/orchestrator/run_local.py
"""Drive the full pipeline in-process using LocalDispatcher — no cloud, no GPU API.

Used for local end-to-end testing on a short clip. The orchestrator's /callback
HTTP hop is short-circuited: LocalDispatcher posts to ORCH_BASE_URL/callback, so
point ORCH_BASE_URL at a locally-running app, OR use the in-process variant here
which calls the state machine directly via a loopback reporter.
"""
from src import config
from src.orchestrator.dispatch import LocalDispatcher
from src.orchestrator.reporting import Reporter
from src.orchestrator.runs import InMemoryRunStore, Run
from src.orchestrator.state_machine import StateMachine
from src.workers import gpu_prep, gpu_synth, cpu_text, cpu_mux


class _LoopbackDispatcher:
    """LocalDispatcher variant that feeds results straight back into the state
    machine instead of doing an HTTP callback (keeps the e2e test cloud-free)."""
    def __init__(self, workers, sm_ref):
        self.workers = workers
        self.sm_ref = sm_ref

    def dispatch(self, step, run, payload, callback_url):
        try:
            result = self.workers[step.name](payload)
            self.sm_ref[0].handle_callback(run.id, step.name, True, result)
        except Exception as e:  # noqa: BLE001
            self.sm_ref[0].handle_callback(run.id, step.name, False, {"error": str(e)})


def run_dub_locally(run_id, dub_id, episode_id, audio_url, language, bg_volume=0.15):
    store = InMemoryRunStore()
    sm_ref: list = [None]
    workers = {"prep": gpu_prep.run, "synth": gpu_synth.run,
               "text": cpu_text.run, "mux": cpu_mux.run}
    disp = _LoopbackDispatcher(workers, sm_ref)
    sm = StateMachine(store, {"gpu": disp, "cpu": disp}, Reporter())
    sm_ref[0] = sm
    sm.start(Run(id=run_id, workflow_type="dub", episode_id=episode_id,
                 callback_url=config.BUZZBOT_RESULT_URL, dub_id=dub_id,
                 language=language, audio_url=audio_url, bg_volume=bg_volume))
    return store
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_run_local.py -v`
Expected: PASS.

- [ ] **Step 5: Run the full suite**

Run: `pytest -q`
Expected: all tests pass.

- [ ] **Step 6: Commit**

```bash
git add src/orchestrator/run_local.py tests/test_run_local.py
git commit -m "feat: local end-to-end driver (LoopbackDispatcher, no cloud)"
```

---

## Phase 12 — Contract snapshot

### Task 18: Lock the buzz-bot result schema with a snapshot test

**Files:**
- Create: `tests/test_contract_result.py`

- [ ] **Step 1: Write the test asserting the exact buzz-bot payload keys**

```python
# tests/test_contract_result.py
from unittest.mock import patch
from src.orchestrator.reporting import Reporter
from src.orchestrator.runs import Run

# These keys are buzz-bot's contract (README "Result" schema). Changing them
# is a breaking change to buzz-bot and must be intentional.
RESULT_KEYS = {"dub_id", "episode_id", "language", "source_lang", "success",
               "r2_url", "duration_sec", "segment_count", "speaker_count", "segments"}
SEGMENT_KEYS = {"idx", "start_sec", "end_sec", "speaker_id", "text", "words",
                "translated_text", "synth_r2_key", "synth_duration", "synth_start_sec"}

def test_dub_result_payload_matches_buzzbot_contract():
    run = Run(id="r1", workflow_type="dub", episode_id=456, callback_url="cb",
              dub_id=123, language="es", source_lang="en", segments_key="k",
              speaker_keys={"S0": "k"}, r2_url="https://r2/es.mp3",
              duration_sec=100.0, segment_count=1)
    segs = [{"idx": 0, "start_sec": 0.0, "end_sec": 2.0, "speaker": "S0",
             "text": "hi", "words": [], "translated_text": "hola",
             "synth_r2_key": "dub-stems/456/synth_es_0000.wav",
             "synth_duration": 1.1, "synth_start_sec": 0.0}]
    captured = {}
    with patch("src.orchestrator.reporting.artifacts.read_segments", return_value=segs), \
         patch("src.orchestrator.reporting.requests.post",
               side_effect=lambda url, **kw: captured.update(body=kw["json"])):
        Reporter().dub_result(run)
    assert set(captured["body"].keys()) == RESULT_KEYS
    assert set(captured["body"]["segments"][0].keys()) == SEGMENT_KEYS
```

- [ ] **Step 2: Run test to verify it passes**

Run: `pytest tests/test_contract_result.py -v`
Expected: PASS (Reporter already built in Task 8).

- [ ] **Step 3: Commit**

```bash
git add tests/test_contract_result.py
git commit -m "test: lock buzz-bot dub_result contract with snapshot"
```

---

## Phase 13 — Docs

### Task 19: Update README with the decomposed architecture

**Files:**
- Modify: `README.md`

- [ ] **Step 1: Add an "Architecture (decomposed)" section**

After the existing `## Pipeline` section, add:
```markdown
## Architecture (decomposed)

The pipeline runs as a central **orchestrator** (k3s) driving four task-runners:

| Runner    | Tier | Steps                         | Host           |
|-----------|------|-------------------------------|----------------|
| gpu-prep  | GPU  | separate, transcribe, extract | Nebius job     |
| cpu-text  | CPU  | split, translate              | k3s worker     |
| gpu-synth | GPU  | synthesize                    | Nebius job     |
| cpu-mux   | CPU  | assemble, mix                 | k3s worker     |

The orchestrator advances a `Run` (Postgres `orch_run`) through a declarative
workflow (`dub` = all four; `transcribe` = gpu-prep only). Large artifacts pass
through R2 by key (`dub-runs/{run_id}/segments.json`); small scalars via signed
`/callback` posts. buzz-bot's `/internal/dub_progress` + `/internal/dub_result`
contract is unchanged.

`src/worker.py` (the original RunPod monolith) remains for coexistence until
cutover. Nebius dispatch + k8s deploy + buzz-bot routing are tracked in
`docs/superpowers/plans/2026-05-31-pipeline-decomposition-cloud.md` (Plan 2).
```

- [ ] **Step 2: Commit**

```bash
git add README.md
git commit -m "docs: document decomposed orchestrator architecture"
```

---

## Out of Scope (Plan 2 — cloud + cutover)

Tracked separately; not implemented here:

1. **NebiusDispatcher** — implements `Dispatcher` by launching a Nebius GPU job
   (`nebius ai job create ... --image ... --preset 1gpu-... --timeout ...` with
   `INPUT_JSON` + `CALLBACK_URL` env), plus a status-poll backstop for jobs that
   die before calling back. Slots into `dispatchers["gpu"]`.
2. **Dockerfiles** — one GPU image (prep+synth entrypoints) reusing the current
   model/Network-Volume setup; one CPU image (cpu-text/cpu-mux + orchestrator).
3. **k8s manifests** — orchestrator Deployment + Service + Ingress, cpu-text and
   cpu-mux Deployments, secrets (`DATABASE_URL`, `ORCH_CALLBACK_SECRET`,
   Nebius creds), in the buzz-bot cluster.
4. **buzz-bot cutover** — replace the RunPod dispatch call with `POST {orch}/dispatch`,
   behind a feature flag (coexist vs hard cutover — decided at that time).
5. **Transcribe consumer in buzz-bot** — `/internal/transcript_result` handler,
   storage, trigger, UI (separate feature, per the design's non-goals).
6. **Synth progress relay** — gpu-synth posting incremental pct to the
   orchestrator for finer `synthesizing` progress (cosmetic; deferred).

---

## Self-Review Notes

- **Spec coverage:** orchestrator state machine (Tasks 7), Postgres state (16),
  R2-by-key artifacts (2, 11), HMAC callback auth (4, 15), two-tier workers
  (10–13), dub+transcribe workflows (6), progress/result contract (8, 18),
  retries + max-attempts (7), local e2e (17). Idempotent R2 reuse / per-segment
  resume and the Nebius status backstop are intentionally Plan 2 (need the real
  dispatcher) — noted in Out of Scope.
- **Known plan caveats to honor during execution:** keep `LocalDispatcher._go()`
  synchronous (Task 14) so the unit tests' synchronous assertions hold — only
  switch to a background thread for the local e2e run if stack depth becomes an
  issue. `PgRunStore.get` assumes psycopg decodes `JSONB` to a dict (Task 16
  Step 4 note); verify on first real-DB run.
- **Placeholder scan:** no TBD/TODO/"add error handling" left; every code step
  carries complete code. Deliberate teaching-stubs removed.
- **Type consistency:** `Run` fields, `Step` (name/tier/progress), `build_input`
  keys, callback result fields (`segments_key`/`source_lang`/`speaker_keys`/
  `r2_url`/`duration_sec`/`segment_count`), and the `Dispatcher.dispatch`
  signature are consistent across Tasks 5–17.
