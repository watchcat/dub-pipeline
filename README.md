# dub-pipeline

Python service that produces dubbed podcast audio. Runs on **RunPod Serverless** (GPU cloud). Receives jobs via the RunPod handler API, runs the full dubbing pipeline, and reports progress and results back to buzz-bot via HTTP callbacks.

## Pipeline

```
Audio URL
    │
    ▼
1. Separate (Demucs htdemucs_ft)
    │  vocals.wav       — 16 kHz mono, uploaded to R2 as dub-stems/{episode_id}/vocals.wav
    │  background.wav   — 44.1 kHz stereo, uploaded to R2 as dub-stems/{episode_id}/background.wav
    ▼
2. Transcribe (WhisperX large-v3 + pyannote diarization)
    │  segments: [{idx, start, end, speaker, text, words}, ...]
    ▼
3. Extract speaker samples
    │  Best 15–30 s clip per speaker from vocals stem (high word-confidence)
    │  Uploaded to R2 as dub-stems/{episode_id}/speaker_{id}.wav
    ▼
3b. Split long segments
    │  Segments > 30 s split at sentence/pause boundaries (VoxCPM2 context limit)
    ▼
4. Translate (Gemini Flash, batch with context)
    │  segments: [{..., translated_text}, ...]
    │  Same-language dubs copy text verbatim — no API call
    ▼
5. Synthesize (VoxCPM2, per segment)
    │  Speaker voice cloned from sample; VoxCPM2 auto-detects target language
    │  Segments < 0.5 s skipped (silence placeholder)
    │  Output: 48 kHz mono WAV per segment
    ▼
6. Assemble
    │  Cursor-based timeline placement:
    │    - Segment ran long → consume following gap (no time-stretch)
    │    - Segment ran short → insert 50% of original gap as pacing silence
    │    - actual_cursor tracks real ffmpeg concat position for accurate synth_start_sec
    │  Total duration capped at 150% of original
    │  dubbed_vocals.wav — 48 kHz mono
    ▼
7. Mix
    │  ffmpeg: dubbed_vocals + background at configurable volume (default 15%)
    ▼
dubbed/{episode_id}/{language}.mp3 → R2
    │
    ▼
POST /internal/dub_result  (buzz-bot callback)
    includes segment data with synth_start_sec for subtitle sync
```

Stems (vocals, background, speaker samples) are **episode-scoped** — reused across languages. If `dub-stems/{episode_id}/vocals.wav` already exists in R2, Demucs is skipped.

### Checkpointing

The worker saves pipeline state to the RunPod Network Volume after each step. If a worker is interrupted (spot instance preempted, OOM), the next worker picks up from the last completed step rather than starting over.

## Requirements

- Python 3.11+
- ffmpeg in PATH
- CUDA GPU (RunPod; `WHISPER_DEVICE=cuda`)

## Deployment (RunPod Serverless)

### 1. Build and push the Docker image

```bash
docker buildx build --platform linux/amd64 \
  -t watchcat/dub-pipeline:latest --push .
```

The image is based on `runpod/pytorch:1.0.3-cu1281-torch271-ubuntu2204`. Models are **not** baked in — they live on a RunPod Network Volume at `/runpod-volume/models`.

### 2. Create a RunPod Network Volume

Create a volume (≥ 50 GB) in the RunPod console. The worker mounts it at `/runpod-volume/`. The first job on a fresh volume will download all models (~15 GB); subsequent jobs reuse the cache.

### 3. Configure the serverless endpoint

In the RunPod console:
- **Container image**: your pushed image
- **Network Volume**: attach the volume above (mount at `/runpod-volume`)
- **Environment variables**: see Configuration below
- **GPU**: RTX 4090 or similar recommended

### 4. Link buzz-bot

Set `RUNPOD_API_KEY` and `RUNPOD_ENDPOINT_ID` in the buzz-bot k8s secret.

## Configuration

| Variable | Required | Default | Description |
|---|---|---|---|
| `PROGRESS_URL` | yes | — | `https://app.buzz-bot.top/internal/dub_progress` |
| `TEMP_DIR` | no | `/tmp/dub-pipeline` | Working directory for in-progress jobs |
| `MODEL_DIR` | no | `/runpod-volume/models` | Model cache directory (Network Volume) |
| `CHECKPOINT_DIR` | no | `/runpod-volume/checkpoints` | Pipeline checkpoint directory (Network Volume) |
| `R2_ENDPOINT` | yes | — | `https://{account_id}.r2.cloudflarestorage.com` |
| `R2_ACCESS_KEY_ID` | yes | — | R2 API token ID |
| `R2_SECRET_ACCESS_KEY` | yes | — | R2 API token secret |
| `R2_BUCKET` | yes | — | R2 bucket name |
| `R2_PUBLIC_URL` | yes | — | Public base URL for R2 objects |
| `GEMINI_API_KEY` | yes | — | Google Gemini API key |
| `GEMINI_MODEL` | no | `gemini-2.5-flash` | Gemini model for translation |
| `HF_TOKEN` | yes | — | HuggingFace token (pyannote diarization models) |
| `DEMUCS_MODEL` | no | `htdemucs_ft` | Demucs model name |
| `WHISPER_MODEL` | no | `large-v3` | WhisperX model name |
| `WHISPER_DEVICE` | no | `cuda` | `cuda` / `cpu` |
| `WHISPER_COMPUTE` | no | `float16` | `float16` / `int8` |
| `MAX_DURATION_RATIO` | no | `1.5` | Max output duration as ratio of original |
| `BG_VOLUME_DEFAULT` | no | `0.15` | Background stem volume (0.0–0.5) |

### HuggingFace setup (one-time)

The pyannote diarization models require accepting their licence before the token works:

1. Create a token at https://huggingface.co/settings/tokens
2. Accept licence at https://huggingface.co/pyannote/speaker-diarization-3.1
3. Accept licence at https://huggingface.co/pyannote/segmentation-3.0

## Job format

buzz-bot dispatches jobs via the RunPod Serverless API (`POST /v2/{endpoint_id}/run`). The `input` field:

```json
{
  "job_id":       "a3f8c2...",
  "dub_id":       123,
  "episode_id":   456,
  "audio_url":    "https://...",
  "language":     "es",
  "bg_volume":    0.15,
  "callback_url": "https://app.buzz-bot.top/internal/dub_result"
}
```

## Callbacks

**Progress** (posted after each step to `PROGRESS_URL`):

```json
{"dub_id": 123, "step": "transcribing", "pct": 40}
```

Step values: `separating` → `transcribing` → `translating` → `synthesizing` (with `pct`) → `assembling` → `mixing` → `uploading` → `complete`

**Result** (posted to `callback_url` on completion):

```json
{
  "job_id":        "a3f8c2...",
  "dub_id":        123,
  "episode_id":    456,
  "language":      "es",
  "success":       true,
  "r2_url":        "https://pub-xxx.r2.dev/dubbed/456/es.mp3",
  "duration_sec":  2847.3,
  "segment_count": 142,
  "speaker_count": 2,
  "source_lang":   "en",
  "segments": [
    {
      "idx":             0,
      "start_sec":       1.2,
      "end_sec":         4.5,
      "text":            "Hello, welcome to the show.",
      "translated_text": "Hola, bienvenido al programa.",
      "synth_start_sec": 1.2,
      "synth_duration":  3.1,
      "synth_r2_key":    "dub-stems/456/synth_es_0000.wav",
      "speaker_id":      "SPEAKER_00",
      "words":           [...]
    }
  ]
}
```

The `segments` array is stored in buzz-bot's `dub_segments` / `dub_segment_translations` tables and powers the karaoke subtitle panel.

On failure:

```json
{
  "job_id": "a3f8c2...", "dub_id": 123,
  "success": false, "step": "separating", "error": "Demucs OOM"
}
```

## Local testing

```bash
python test_job.py [audio_url] [language]
# Submits a local job (dub_id=999999) with callback to localhost:9999 (will 404 gracefully)
# Output uploaded to R2: dubbed/999999/{language}.mp3
```

## R2 layout

```
dub-stems/{episode_id}/
    vocals.wav                      — 16 kHz mono vocals stem
    background.wav                  — 44.1 kHz stereo background stem
    speaker_SPEAKER_00.wav          — voice clone sample per speaker
    synth_{language}_{idx:04d}.wav  — synthesised segment audio

dubbed/{episode_id}/{language}.mp3  — final dubbed episode
```

## Source layout

```
src/
    config.py               — ENV var accessors
    storage.py              — R2 upload/download (boto3)
    progress.py             — POST step updates to buzz-bot
    worker.py               — RunPod handler + pipeline orchestration + checkpointing
    steps/
        separate.py         — Demucs source separation
        transcribe.py       — WhisperX transcription + pyannote diarization
        extract_samples.py  — voice clip extraction per speaker
        split_segments.py   — split long segments at sentence boundaries
        translate.py        — Gemini batch translation
        synthesize.py       — VoxCPM2 voice cloning + synthesis
        assemble.py         — cursor-based timeline assembly with actual_cursor tracking
        mix.py              — ffmpeg stem mixing
```
