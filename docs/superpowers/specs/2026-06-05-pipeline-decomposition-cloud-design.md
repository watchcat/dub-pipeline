# Pipeline Decomposition — Cloud Deployment & Cutover (Plan 2) Design

**Date:** 2026-06-05
**Status:** Approved (design)
**Repos:** `dub-pipeline` (primary), `buzz-bot` (cutover + transcribe consumer)
**Branch:** `feat/pipeline-decomposition`
**Builds on:** Plan 1 — `docs/superpowers/plans/2026-05-31-pipeline-decomposition-core.md`
(core orchestrator + four workers + local end-to-end, already implemented) and the
parent design `docs/superpowers/specs/2026-05-31-pipeline-decomposition-design.md`.

## Problem & Goal

Plan 1 produced a working orchestrator + two-tier workers exercised only by a
`LocalDispatcher` (no cloud). Plan 2 makes it **real**: dispatch GPU work to
Nebius scale-to-zero Jobs, deploy the orchestrator + CPU workers into the existing
k3s cluster, cut buzz-bot's dub path over to the orchestrator behind a feature
flag, and ship the transcribe-without-dubbing feature end to end.

This is **one combined spec and one big implementation plan** covering all six
items the Plan 1 doc deferred to "Out of Scope (Plan 2)".

## End-state for this work

**Author + unit-test, deploy later.** All code, manifests, and tests are written
and reviewable; every external call (Nebius, k8s, real DB) is mocked in tests.
No live Nebius/k8s calls happen during implementation. A `DEPLOY.md` runbook
captures the deploy + smoke-test steps for when Nebius credentials and the GPU
presets are available. The single paid **Nebius smoke test** is CI-gated and run
manually.

## Decisions (resolved during brainstorming)

| # | Decision | Choice |
|---|----------|--------|
| 1 | End-state | Author + unit-test now; deploy later via runbook |
| 2 | Scope | All six deferred items (A–G below) |
| 3 | Structure | One combined spec, one big implementation plan |
| 4 | Dub cutover | Coexist behind `dub_orchestrator` FeatureFlag (instant rollback) |
| 5 | Transcribe trigger/access | Explicit "Generate transcript" button, free for all users, prep-only workflow, status-tracked to coalesce |
| 6 | GPU-job failure backstop | **In-process reconciler thread** in the orchestrator (not a separate CronJob `/reconcile`) |
| 7 | Transcribe progress | **Poll-based** (transcribe has no `dub_id`, so no new progress channel) |
| 8 | Image packaging | **One CPU image** serving orchestrator + cpu-text + cpu-mux (entrypoint chosen by k8s `command`); one GPU image with both GPU entrypoints |

The architecture itself (two tiers, central orchestration, Postgres state, R2-by-key
artifacts, HMAC callbacks) is unchanged from the parent design and is not re-litigated
here.

---

## A. dub-pipeline — Nebius GPU dispatch

**New: `src/orchestrator/nebius.py`** — a thin, mockable Nebius Jobs client.
- `create_job(image, preset, env: dict, timeout) -> job_id`
- `get_status(job_id) -> "running" | "succeeded" | "failed" | "gone"`
- Auth via env (`NEBIUS_API_KEY` / service-account creds). All HTTP/SDK calls
  isolated here so the dispatcher and reconciler are unit-tested against a fake.

**`NebiusDispatcher`** (added to `src/orchestrator/dispatch.py`, implements the
existing `Dispatcher` protocol):
```
dispatch(step, run, payload, callback_url):
    job_id = nebius.create_job(
        image=GPU_IMAGE,
        preset=PRESET_FOR[step.name],          # NEBIUS_PREP_PRESET | NEBIUS_SYNTH_PRESET
        env={"INPUT_JSON": json(payload), "CALLBACK_URL": callback_url},
        timeout=STEP_TIMEOUT[step.name])
    store.update(run.id, nebius_job_id=job_id, step_deadline=now()+timeout)
```
Slots into `dispatchers["gpu"]`. CPU steps keep using `HttpDispatcher`.

**Failure backstop — in-process reconciler.** A background thread started by the
orchestrator process scans, on an interval, runs whose `current_step` is a GPU
step and whose `step_deadline` has passed without a callback. For each, it calls
`nebius.get_status(nebius_job_id)`; if `failed`/`gone`, it routes the run through
the existing `StateMachine._on_failure(run, step, error)` (retry up to
`MAX_STEP_ATTEMPTS`, then terminal-fail). A job that is still `running` gets its
deadline extended. This keeps all run-advancement logic in the state machine.

**`Run` / schema additions** (`runs.py` dataclass + `schema.sql` + `PgRunStore`):
`nebius_job_id: Optional[str]`, `step_deadline: Optional[datetime]`. These are the
only new columns; everything else Plan 1 already persists.

GPU presets and step timeouts are config-driven (`NEBIUS_PREP_PRESET`,
`NEBIUS_SYNTH_PRESET`, `STEP_TIMEOUT_*`); exact values are filled at deploy time.

## B. dub-pipeline — Images

**GPU image** (`Dockerfile.gpu`): reuses the current model / Network-Volume setup
(Demucs, WhisperX, pyannote, VoxCPM2). A single image carries both
`gpu_prep.main` and `gpu_synth.main`; the Nebius job's `INPUT_JSON`/entry selects
which runs. Synth-only jobs still pay only synth wall-clock because the job exits
after its step.

**CPU image** (`Dockerfile.cpu`): installs ffmpeg and serves three entrypoints
selected by k8s `command` — the orchestrator (`src/orchestrator/main.py`),
`cpu_text` (FastAPI `/run`), and `cpu_mux` (FastAPI `/run`).

**New: `src/orchestrator/main.py`** — production wiring Plan 1 left unbuilt.
Constructs `PgRunStore`, `{"gpu": NebiusDispatcher(...), "cpu": HttpDispatcher({"text": CPU_TEXT_URL, "mux": CPU_MUX_URL})}`, and `Reporter()`, calls
`app.configure(store, dispatchers, reporter)`, starts the reconciler thread, and
runs uvicorn. (Plan 1's `app.configure()` is currently only called from tests.)

## C. dub-pipeline — k8s manifests (buzz-bot cluster)

New `k8s/` directory in dub-pipeline, following buzz-bot's existing Traefik /
cert-manager / k3s-image-import conventions:
- **orchestrator**: Deployment + Service + **Ingress** (public host, e.g.
  `orch.buzz-bot.top`). Public because Nebius jobs POST `/callback` over the
  internet — guarded by the per-run HMAC token already implemented in `auth.py`.
- **cpu-text**, **cpu-mux**: Deployment + Service each, **in-cluster only**
  (reached by `HttpDispatcher` via service DNS, e.g.
  `http://cpu-text.buzz-bot.svc.cluster.local/run`).
- **Secret** `orch-secret`: `DATABASE_URL`, `ORCH_CALLBACK_SECRET`, Nebius creds,
  R2 creds, `GEMINI_API_KEY`, `HF_TOKEN`, `PROGRESS_URL`, `BUZZBOT_RESULT_URL`,
  `BUZZBOT_TRANSCRIPT_URL`, `ORCH_BASE_URL`, `CPU_TEXT_URL`, `CPU_MUX_URL`.
- Runbook step: apply `src/orchestrator/schema.sql` (the `run` table + new
  columns) to Neon.

## D. buzz-bot — Dub cutover (flagged)

- `FeatureFlags::DEFAULTS` gains `"dub_orchestrator" => false`.
- `src/web/routes/dub.cr`: branch on `FeatureFlags.enabled?("dub_orchestrator")`.
  - **on** → POST `{Config.orch_base_url}/dispatch`, header
    `X-Dispatch-Token: {Config.orch_dispatch_secret}`, body
    `{run_id: job_id, workflow_type: "dub", dub_id, episode_id, audio_url,
    language, bg_volume, callback_url: "{base}/internal/dub_result"}`
    (exactly the contract `app.py` validates today).
  - **off** → the current RunPod path, unchanged.
  - Same `DubbedEpisode.upsert_pending` + `202 {"id":…,"status":"pending"}`
    response on both branches; same failure handling.
- New config accessors: `Config.orch_base_url` (`ORCH_BASE_URL`),
  `Config.orch_dispatch_secret` (`ORCH_DISPATCH_SECRET`).
- buzz-bot's `/internal/dub_progress` and `/internal/dub_result` handlers are
  **unchanged** — from buzz-bot's side the only difference is the dispatch URL.

## E. buzz-bot — Transcribe feature (free, button)

**Backend**
- **`POST /episodes/:id/transcribe`** (new route): no premium gate. Dispatches
  `workflow_type: "transcribe"` to the orchestrator with `{run_id, episode_id,
  audio_url, callback_url: "{base}/internal/transcript_result"}`. Requires the
  orchestrator, so it is gated on the `dub_orchestrator` flag (the transcribe
  path only exists post-cutover). Status-tracked via a lightweight
  `transcript_jobs(episode_id PK, status, run_id, updated_at)` table (parallels
  `DubbedEpisode` but keyed by episode only — transcripts have no target
  language) so repeated taps while one is in flight coalesce to the existing run;
  a `done` transcript short-circuits.
- **`POST /internal/transcript_result`** (new internal route): validates the
  shared secret, then `DubSegment.bulk_upsert(episode_id, source_lang, segments)`
  + `Episode.save_original_language(episode_id, source_lang)`. This reuses the
  exact storage the dub path already writes; no new segment tables. Marks the
  `transcript_jobs` row `done`.

**Frontend** (`src/cljs/buzz_bot/`)
- In the subtitle panel, when `GET /episodes/:id/subtitles` returns no cues, show
  a **"Generate transcript"** button. On tap → POST `/transcribe`, show a pending
  state, then **poll** `/episodes/:id/subtitles` (decision #7) until cues appear,
  then render them through the existing subtitle UI. No SSE / dub_id channel.

## F. Synth progress relay (cosmetic)

- `gpu_synth` periodically POSTs `{run_id, step: "synth", pct}` to a new
  orchestrator endpoint **`POST /progress`** (authenticated with the per-run HMAC
  token, same scheme as `/callback`).
- The orchestrator relays via `Reporter.progress(run, "synthesizing", pct)` to
  buzz-bot's existing `/internal/dub_progress`. Purely cosmetic finer-grained
  progress; the run does not advance on these posts.

---

## Contracts (summary)

| Endpoint | Auth | Body |
|----------|------|------|
| `POST {orch}/dispatch` | `X-Dispatch-Token: ORCH_CALLBACK_SECRET` | `{run_id, workflow_type, episode_id, callback_url, dub_id?, language?, audio_url?, bg_volume?}` |
| `POST {orch}/callback?run_id&step&token` | per-run HMAC | `{ok, …step result fields}` |
| `POST {orch}/progress?run_id&step&token` | per-run HMAC | `{pct}` |
| `POST {buzz}/internal/dub_result` | shared secret (existing) | unchanged dub result schema |
| `POST {buzz}/internal/transcript_result` | shared secret | `{episode_id, source_lang, segments:[…]}` |

## New configuration

**dub-pipeline** (extends `src/config.py`): `NEBIUS_API_KEY`, Nebius
project/region creds, `NEBIUS_PREP_PRESET`, `NEBIUS_SYNTH_PRESET`, `GPU_IMAGE`,
`STEP_TIMEOUT_PREP`, `STEP_TIMEOUT_SYNTH`, `RECONCILER_INTERVAL_SEC`.
(`DATABASE_URL`, `ORCH_*`, `BUZZBOT_*`, `CPU_*` already added in Plan 1.)

**buzz-bot** (extends `src/config.cr`): `ORCH_BASE_URL`, `ORCH_DISPATCH_SECRET`.

## Testing strategy (all mocked)

- `nebius.py` client — unit-tested against a stubbed HTTP/SDK layer.
- `NebiusDispatcher` — asserts job created with right image/preset/env and
  `nebius_job_id`/`step_deadline` persisted (fake nebius + fake store).
- Reconciler — fake clock + fake `get_status`: past-deadline failed job →
  `_on_failure`; still-running → deadline extended; healthy callback path
  untouched.
- `main.py` wiring — constructs collaborators without real connections (mocked).
- Synth progress relay — `/progress` posts call `Reporter.progress`, run unchanged.
- buzz-bot cutover — Crystal spec: flag **on** → orchestrator POST with correct
  header/body; flag **off** → RunPod path; response identical.
- buzz-bot transcribe — route dispatches `transcribe`; `transcript_result`
  persists segments via `bulk_upsert`; coalescing logic.
- Frontend — the subtitle-panel button appears only with no cues; poll loads cues.

## Deploy runbook (`DEPLOY.md`, authored not executed)

1. Build + push GPU and CPU images; import into k3s containerd (per buzz-bot's
   image-import convention).
2. Apply `schema.sql` to Neon.
3. `kubectl apply` orchestrator + cpu-text + cpu-mux Deployments/Services/Ingress;
   create `orch-secret`.
4. Set the buzz-bot secret keys `ORCH_BASE_URL`, `ORCH_DISPATCH_SECRET`; deploy
   buzz-bot with the new code (flag still off).
5. **Nebius smoke test** — one short clip through `gpu-prep` (paid, manual).
6. Flip `dub_orchestrator` on via the `/flag` bot command; verify a dub; verify a
   transcribe; roll back by flipping off if needed.

## Out of scope / deferred

- Throughput / parallel GPU scaling, Redis/broker — unchanged from parent design.
- Retiring `src/worker.py` (RunPod monolith) — stays for coexistence; removal is a
  later cleanup once the flag has been on in production for a while.
- Transcribe subscription metering/tiers (the `ideas.md` Free/Basic/Pro limits) —
  the feature ships free-for-all now; metering is a separate product decision.

## Self-review checklist

- No TBD/placeholder: presets and the smoke test are intentionally deploy-time
  values, flagged as such, not gaps in the design.
- Consistency: the `/dispatch` body matches `app.py` exactly; `transcript_result`
  reuses `DubSegment.bulk_upsert` + `Episode.save_original_language` verified in
  buzz-bot today; cutover preserves the existing 202 response.
- Scope: large but single-threaded by dependency (A → C deploy → D/E/F), suitable
  for one phased implementation plan.
