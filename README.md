# dub-pipeline

Python service that produces dubbed podcast audio. Runs locally on the Mac Mini (Apple M4, MPS GPU). Receives jobs from a Redis queue, runs the full dubbing pipeline, and posts results back to buzz-bot via HTTP callbacks.

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
    │  segments: [{start, end, speaker, text, words}, ...]
    ▼
3. Extract speaker samples
    │  Best 15–30 s clip per speaker from vocals stem (high word-confidence)
    │  Uploaded to R2 as dub-stems/{episode_id}/speaker_{id}.wav
    ▼
3b. Split long segments
    │  Segments > 30 s split at sentence/pause boundaries (XTTS-v2 context limit)
    ▼
4. Translate (DeepL Pro, batch with context)
    │  segments: [{..., translated_text}, ...]
    │  Same-language dubs skip DeepL entirely
    ▼
5. Synthesize (Coqui XTTS-v2, per segment)
    │  Speaker voice cloned from sample; synthesised at target language
    │  Segments < 0.5 s skipped (silence placeholder)
    ▼
6. Assemble
    │  Cursor-based timeline placement:
    │    - Segment ran long → consume following gap (no time-stretch)
    │    - Segment ran short → insert 50% of original gap as pacing silence
    │  Total duration capped at 110% of original
    │  dubbed_vocals.wav — 24 kHz mono
    ▼
7. Mix
    │  ffmpeg: dubbed_vocals + background at 15% volume
    ▼
dubbed/{episode_id}/{language}.mp3 → R2 → callback to buzz-bot
```

Stems (vocals, background, speaker samples) are **episode-scoped** — reused across languages. If `dub-stems/{episode_id}/vocals.wav` already exists in R2, Demucs is skipped.

## Requirements

- Python 3.11+
- ffmpeg in PATH
- Apple Silicon Mac (MPS used for Demucs, WhisperX, XTTS-v2); change `WHISPER_DEVICE`/`TTS_DEVICE` to `cuda` or `cpu` for other hardware

## Setup

```bash
# 1. Install Python 3.11 (if not already present)
brew install python@3.11

# 2. Create virtualenv
python3.11 -m venv .venv
source .venv/bin/activate

# 3. Install dependencies
pip install --upgrade pip
pip install torch torchaudio
pip install -r requirements.txt

# 4. Configure environment
cp .env.example .env
# Edit .env — fill in HF_TOKEN, verify Redis/R2/DeepL credentials
```

## Configuration

Copy `.env.example` to `.env` and fill in the required values.

| Variable | Required | Default | Description |
|---|---|---|---|
| `REDIS_URL` | yes | — | Redis connection URL (`redis://default:pass@host:port`) |
| `QUEUE_KEY` | no | `dub:jobs` | Redis list key to BRPOP |
| `PROGRESS_URL` | yes | — | `https://app.buzz-bot.top/internal/dub_progress` |
| `TEMP_DIR` | no | `/tmp/dub-pipeline` | Working directory for in-progress jobs |
| `MODEL_DIR` | no | `./models` | Model cache directory |
| `R2_ENDPOINT` | yes | — | `https://{account_id}.r2.cloudflarestorage.com` |
| `R2_ACCESS_KEY_ID` | yes | — | R2 API token ID |
| `R2_SECRET_ACCESS_KEY` | yes | — | R2 API token secret |
| `R2_BUCKET` | yes | — | R2 bucket name |
| `R2_PUBLIC_URL` | yes | — | Public base URL for R2 objects |
| `DEEPL_API_KEY` | yes | — | DeepL Pro API key |
| `HF_TOKEN` | yes | — | HuggingFace token (pyannote diarization models) |
| `DEMUCS_MODEL` | no | `htdemucs_ft` | Demucs model name |
| `WHISPER_MODEL` | no | `large-v3` | WhisperX model name |
| `WHISPER_DEVICE` | no | `mps` | `mps` / `cuda` / `cpu` |
| `WHISPER_COMPUTE` | no | `float16` | `float16` / `int8` |
| `TTS_DEVICE` | no | `mps` | `mps` / `cuda` / `cpu` |
| `BG_VOLUME_DEFAULT` | no | `0.15` | Background stem volume (0.0–0.5) |

### HuggingFace setup (one-time)

The pyannote diarization models require accepting their licence on HuggingFace before the token works:

1. Create a token at https://huggingface.co/settings/tokens
2. Accept licence at https://huggingface.co/pyannote/speaker-diarization-3.1
3. Accept licence at https://huggingface.co/pyannote/segmentation-3.0

## Running

```bash
./run-worker.sh
```

First run downloads all models (~10 GB: Demucs htdemucs_ft, Whisper large-v3, pyannote diarization, XTTS-v2). Subsequent starts are fast.

To run in the background:

```bash
nohup ./run-worker.sh >> /tmp/dub-pipeline.log 2>&1 &
```

## Job format

buzz-bot pushes jobs via `RPUSH dub:jobs`:

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
  "job_id": "a3f8c2...", "dub_id": 123, "episode_id": 456, "language": "es",
  "success": true,
  "r2_url": "https://pub-xxx.r2.dev/dubbed/456/es.mp3",
  "duration_sec": 2847.3, "segment_count": 142, "speaker_count": 2
}
```

On failure:

```json
{
  "job_id": "a3f8c2...", "dub_id": 123,
  "success": false, "step": "separating", "error": "Demucs OOM"
}
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
    worker.py               — BRPOP loop; orchestrates all steps
    steps/
        separate.py         — Demucs source separation
        transcribe.py       — WhisperX transcription + diarization
        extract_samples.py  — voice clip extraction per speaker
        split_segments.py   — split long segments at sentence boundaries
        translate.py        — DeepL Pro batch translation
        synthesize.py       — Coqui XTTS-v2 local synthesis
        assemble.py         — cursor-based timeline assembly
        mix.py              — ffmpeg stem mixing
```
