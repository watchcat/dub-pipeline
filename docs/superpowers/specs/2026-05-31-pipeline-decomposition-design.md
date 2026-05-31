# Pipeline Decomposition — Orchestrated Two-Tier Design

**Date:** 2026-05-31
**Status:** Approved (design); migration mechanics deferred to the implementation plan
**Repo:** `dub-pipeline`
**Branch:** `feat/pipeline-decomposition`

## Problem

Today the entire dubbing pipeline runs as a single RunPod Serverless handler
(`src/worker.py`): one GPU container executes all eight steps in-process. The
GPU-heavy steps (Demucs separate, WhisperX transcribe, VoxCPM2 synthesize) are
interleaved with CPU-only / no-compute steps (split, **Gemini translate**,
ffmpeg assemble, ffmpeg mix). The GPU is billed for the whole wall-clock,
including the stretches where it sits idle doing API calls and ffmpeg work.

**Primary goal: cut cost** by getting CPU-only work off the GPU. Secondary
benefit unlocked by the same change: a **transcribe-without-dubbing** feature
that reuses a shorter path through the same steps.

## Goals

- Bill GPU time only for actual GPU work (`separate`, `transcribe`, `extract`,
  `synthesize`); run CPU work on compute that is already paid for.
- Keep the per-episode R2 artifact reuse (stems, speaker samples, segments) that
  already makes re-dubbing cheap.
- Add a `transcribe` workflow (pipeline + orchestrator side only) that runs
  `separate → transcribe` and stops.
- Preserve buzz-bot's existing contract: the same `/internal/dub_progress` and
  `/internal/dub_result` callbacks, same step vocabulary, same result schema.

## Non-Goals

- buzz-bot UI / storage / trigger for the transcribe feature (separate
  follow-up; this spec only makes the pipeline + orchestrator support it).
- Migration mechanics (coexist-behind-flag vs hard cutover) — decided when the
  implementation plan is written. `src/worker.py` stays in place meanwhile.
- Increasing throughput / parallel GPU scaling. Volume is low/bursty, so GPU
  runs scale-to-zero and cold starts are acceptable.
- Introducing Redis or a message broker. Orchestration + Postgres is sufficient
  at this volume.

## Key Decisions (and why)

| Decision | Choice | Rationale |
|---|---|---|
| Decomposition granularity | Two tiers: GPU vs CPU | Biggest cost win, least overhead. Splitting GPU steps further would reload multi-GB models per hop. |
| GPU hosting | Nebius **scale-to-zero Jobs** | Low/bursty volume — a warm GPU would mostly idle. Pay per use; cold start is fine for non-realtime dubbing. |
| Two GPU jobs, not one | `gpu-prep` + `gpu-synth` | Prep needs Demucs+WhisperX+pyannote; synth needs only VoxCPM2. Separate jobs each cold-start a *smaller* model set. |
| Coordination pattern | Central **orchestration** (not choreography) | Workflow visible in one place; conditional branches (stem reuse, same-language skip, transcribe short-circuit) are natural; single source of truth for run state. Choreography's payoff needs many equal steps + high parallel throughput, which we don't have. |
| Orchestrator | **Separate service** | Clean brain/execution separation; hosted free in existing k3s. |
| CPU tier | **Separate CPU workers** | Brain decides, workers execute; cleaner separation than inlining CPU steps into the orchestrator. |
| Hosting (orchestrator + CPU workers) | Existing **k3s** (Hetzner cpx32) | Near-zero marginal cost; co-located with buzz-bot. Only GPU work leaves to Nebius. |
| State / queue tech | **Postgres** (existing Neon), no Redis | Orchestration needs a state store, not a broker. Avoids an always-on stateful box for a mostly-empty queue. |
| Transport | **HTTP dispatch + completion callbacks** + Nebius Jobs API | Mirrors today's RunPod→buzz-bot callback pattern; no broker. |

## Architecture

The monolith splits into a **brain + four task-runners**, all reusing the
existing pure-function step modules in `src/steps/` (unchanged).

```
buzz-bot ──dispatch(job)──►  ORCHESTRATOR  ──progress/result callbacks──►  buzz-bot
                            (k3s pod, Postgres state machine)
                                  │
        ┌─────────────────┬───────┴────────┬──────────────────┐
        ▼                 ▼                 ▼                  ▼
   gpu-prep          cpu-text          gpu-synth           cpu-mux
  (Nebius job)     (k3s worker)       (Nebius job)       (k3s worker)
  separate         split              synthesize          assemble
  transcribe       translate          (VoxCPM2)           mix
  extract          (Gemini)
   [Demucs+        [no GPU]           [VoxCPM2 only]       [ffmpeg]
    Whisper+
    pyannote]
```

### Components

1. **Orchestrator** (`src/orchestrator/`, Python, k3s pod) — the workflow brain.
   Holds per-run state in Postgres, dispatches each step, advances on completion
   callbacks, relays progress and posts the final result to buzz-bot. No GPU, no
   heavy lifting. The only stateful coordinator — no Redis.
   - Exposes `POST /dispatch` (buzz-bot calls it to start a run) and
     `POST /callback` (workers/jobs report step completion).
   - A small **multi-workflow engine**: `dub` and `transcribe` workflows, both
     composed from the same step units.

2. **gpu-prep** (Nebius scale-to-zero Job, `src/workers/gpu_prep.py`) —
   `separate → transcribe → extract`. Loads Demucs + WhisperX + pyannote.
   `extract` stays here because `vocals.wav` is already local (no extra R2 hop).
   Takes an `extract` flag — set `false` for the transcribe workflow.
   Writes stems + speaker samples + `segments.json` to R2.

3. **cpu-text** (k3s worker, `src/workers/cpu_text.py`) — `split → translate`.
   Pure CPU: sentence-boundary splitting + Gemini batch translate. The headline
   cost win — never touches a GPU.

4. **gpu-synth** (Nebius scale-to-zero Job, `src/workers/gpu_synth.py`) —
   `synthesize`. Loads **only** VoxCPM2. Writes synth segment wavs to R2.

5. **cpu-mux** (k3s worker, `src/workers/cpu_mux.py`) — `assemble → mix`.
   ffmpeg timeline assembly + stem mixing; uploads the final mp3.

### Repo layout

```
src/steps/            ← unchanged (pure functions)
src/orchestrator/     ← state machine, Nebius API client, dispatch, callback HTTP, Postgres
src/workers/
    gpu_prep.py       ← Nebius job entrypoint  (separate+transcribe+extract)
    gpu_synth.py      ← Nebius job entrypoint  (synthesize)
    cpu_text.py       ← k3s worker            (split+translate)
    cpu_mux.py        ← k3s worker            (assemble+mix)
src/worker.py         ← kept (RunPod monolith) until cutover
```

## Data & State Flow

**State store:** orchestrator-owned tables in the existing Neon Postgres. One
`run` row per execution:

```
run(id, workflow_type[dub|transcribe], dub_id?, episode_id, language?,
    status, current_step, attempts, segments_key?, source_lang?,
    speaker_keys?, callback_url, created_at, updated_at)
```

**Data-passing rule:** big artifacts go through **R2 by key**, never through the
orchestrator. Segment data (large with per-word timings) is written to R2 as a
JSON artifact `dub-runs/{run_id}/segments.json` and passed by key; only small
scalars live in Postgres. Each worker reads the input key, writes an updated
artifact, and returns the new key.

### Dub workflow (happy path)

```
buzz-bot ─POST /dispatch{dub,dub_id,episode_id,audio_url,language,bg_volume,callback}→ orchestrator
  └ creates run(step=prep), 202

orchestrator → launch gpu-prep (Nebius job, env=input+callback)     ▸ progress: separating/transcribing
  gpu-prep → R2: vocals, background, speaker_*, segments.json
           → POST /callback{run_id, prep, source_lang, speaker_keys, segments_key}
orchestrator → dispatch cpu-text (HTTP, in-cluster)                  ▸ progress: translating
  cpu-text → split+translate → R2: segments.json(+translated)
           → POST /callback{run_id, text, segments_key}
orchestrator → launch gpu-synth (Nebius job)                         ▸ progress: synthesizing (pct relayed)
  gpu-synth → R2: synth_{lang}_NNNN.wav, segments.json(+synth keys/dur)
            → POST /callback{run_id, synth, segments_key}
orchestrator → dispatch cpu-mux (HTTP, in-cluster)                   ▸ progress: assembling/mixing/uploading
  cpu-mux → assemble+mix → R2: dubbed/{episode}/{lang}.mp3
          → POST /callback{run_id, mux, r2_url, duration, counts, segments_key}
orchestrator → POST /internal/dub_result to buzz-bot (same shape as today) ▸ progress: complete
```

### Transcribe workflow (short-circuit after prep)

```
orchestrator → launch gpu-prep(extract=false)  → callback
orchestrator → POST /internal/transcript_result to buzz-bot (transcript + segments)  ▸ complete
```

`extract` is skipped because speaker voice-clips are only needed for cloning
(i.e. dubbing). A transcribe run still leaves `vocals.wav` / `background.wav` and
`segments.json` in R2, so a **later dub of the same episode reuses them** and
skips Demucs/transcribe entirely.

### Progress

The orchestrator is the **single** place that posts `/internal/dub_progress`,
using today's exact step vocabulary (`separating → transcribing → translating →
synthesizing` (with pct) `→ assembling → mixing → uploading → complete`).
gpu-synth reports incremental pct to the orchestrator, which relays it.
buzz-bot's progress/result handlers are unchanged — from its side the only
change is the dispatch URL (RunPod → orchestrator).

### Idempotency & resume (replaces the RunPod Network Volume checkpoint)

- Run state is persisted after every step; on orchestrator restart it
  re-dispatches the `current_step`. Steps are safe to re-run because they reuse
  R2 artifacts.
- Episode-scoped R2 reuse preserved — Demucs/transcribe skipped if stems /
  segments already exist (this is what makes transcribe-then-dub cheap).
- Per-segment synth resume preserved via R2-existence check (today's behavior).
- A run carries `attempts`; the orchestrator also polls Nebius job status as a
  backstop, so a GPU job that dies before calling back is detected and retried
  rather than hanging.

### Nebius job mechanics

The orchestrator uses the Nebius API to create a Job (container image, GPU
preset, `--timeout`, env vars carrying the step input + per-run callback URL).
The job runs its entrypoint, posts its completion callback, and exits — releasing
the GPU (scale to zero). Job status polling is the failure backstop.

## Error Handling & Retries

- **Failure detection, two signals:** (a) explicit `POST /callback {ok:false,
  step, error}` from a worker/job; (b) backstop — per-step deadline + Nebius job
  status polling, catching jobs that die before calling back (OOM, preemption,
  crash).
- **Retry policy:** per-step `attempts` counter with a cap (default 3) and
  backoff. Re-dispatch is safe because every step reuses its R2 artifacts (a
  retried `gpu-synth` skips segments already in R2; a retried `gpu-prep` reuses
  existing stems). The cap also bounds GPU spend on a poison run.
- **Terminal failure:** orchestrator posts the same failure callback buzz-bot
  already handles (`{success:false, step, error}`) and progress `failed`.
  Completed artifacts remain in R2 so a later retry resumes cheaply.
- **Callback auth:** workers/jobs post to `/callback` with a per-run signed token
  (shared-secret header); buzz-bot → `/dispatch` likewise. Prevents a stray
  caller from advancing a run.

## Observability (serves the cost goal)

The orchestrator timestamps every step transition, so per-dub **GPU-seconds**
(prep + synth wall-clock) are queryable. That is the metric used to verify the
refactor actually cut cost versus the monolith.

## Testing Strategy

- **Step functions** — existing unit tests (e.g. `test_translate`) kept; pure
  functions stay testable in isolation.
- **Worker entrypoints** — thin adapters (R2 in → step call → R2 out →
  callback). Test with mocked `storage` + a stub callback; assert artifact keys
  and callback payload shapes.
- **Orchestrator state machine** (the core) — unit-tested with a *fake
  dispatcher* (no real Nebius/HTTP): feed callbacks, assert the next dispatch +
  progress posts. Covers both workflows, retries, the timeout backstop,
  idempotent re-dispatch, and the episode-level R2-reuse short-circuit.
- **Contract tests** — snapshot `/internal/dub_result` + `/internal/dub_progress`
  payloads against the README schema so the cutover can't silently break buzz-bot.
- **Local end-to-end** — a "local dispatcher" mode that runs the workers as local
  subprocesses on a short clip instead of Nebius, exercising the whole DAG
  without GPU/cloud (extends today's `test_job.py`).
- **Nebius smoke test** — one CI-gated test that launches a real `gpu-prep` on a
  short clip (costs money, run rarely).

## Open Questions (resolved at planning time)

- Migration: coexist-behind-flag vs hard cutover (buzz-bot routes dubs to old
  RunPod path or new orchestrator).
- Exact Nebius GPU preset(s) for prep vs synth.
- Whether the orchestrator and CPU workers are one deployable or separate pods.
- Transcript result payload/endpoint contract with buzz-bot (the follow-up
  feature consumes it).
