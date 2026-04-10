"""Main worker — BRPOP loop for dub:jobs queue.

Job payload (JSON, per spec):
{
  "job_id":       "hex32",
  "dub_id":       123,
  "episode_id":   456,
  "audio_url":    "https://...",
  "language":     "es",
  "bg_volume":    0.15,
  "callback_url": "https://app.buzz-bot.top/internal/dub_result"
}

Success callback:
{
  "job_id": "hex32", "dub_id": 123, "episode_id": 456, "language": "es",
  "success": true, "r2_url": "https://...",
  "duration_sec": 2847.3, "segment_count": 142, "speaker_count": 2
}

Failure callback:
{
  "job_id": "hex32", "dub_id": 123,
  "success": false, "step": "separating", "error": "..."
}
"""
import os
# Set before any torch/TTS import so MPS fallback is active from the start
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

import json
import logging
import shutil
import tempfile
import time

import redis
import requests

from src import config, progress, storage
from src.steps import (
    assemble,
    extract_samples,
    mix,
    separate,
    split_segments,
    transcribe,
    translate,
)
from src.steps.synthesize import _load_tts, _xtts_lang_code, _wav_duration as _synth_wav_duration

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s — %(message)s",
)
log = logging.getLogger(__name__)

# Segment duration constraints (spec §Constraints)
MIN_SEGMENT_SEC = 0.5
MAX_SEGMENT_SEC = 30.0


# ── R2 key helpers ────────────────────────────────────────────────────────────
# Stems and speaker samples are episode-scoped (reused across languages).
# Final mix is episode+language-scoped.

def _stem_key(episode_id: int, filename: str) -> str:
    return f"dub-stems/{episode_id}/{filename}"


def _dub_key(episode_id: int, language: str) -> str:
    return f"dubbed/{episode_id}/{language}.mp3"


# ── Main job processor ────────────────────────────────────────────────────────

def process_job(job: dict):
    job_id      = job["job_id"]
    dub_id      = job["dub_id"]
    episode_id  = job["episode_id"]
    audio_url   = job["audio_url"]
    language    = job["language"]
    bg_volume   = job.get("bg_volume", config.BG_VOLUME_DEFAULT)
    callback_url = job["callback_url"]

    work_dir = tempfile.mkdtemp(dir=config.TEMP_DIR, prefix=f"dub_{dub_id}_")
    log.info(f"dub {dub_id}: starting in {work_dir}")

    current_step = "queued"

    try:
        # ── Step 1: Separate ──────────────────────────────────────────────────
        current_step = "separating"
        progress.report(dub_id, current_step)

        vocals_r2_key     = _stem_key(episode_id, "vocals.wav")
        background_r2_key = _stem_key(episode_id, "background.wav")
        vocals_wav     = os.path.join(work_dir, "vocals.wav")
        background_wav = os.path.join(work_dir, "background.wav")

        if _r2_exists(vocals_r2_key) and _r2_exists(background_r2_key):
            log.info(f"dub {dub_id}: reusing existing stems from R2")
            storage.download(vocals_r2_key, vocals_wav)
            storage.download(background_r2_key, background_wav)
        else:
            vocals_wav, background_wav = separate.separate(audio_url, work_dir)
            storage.upload(vocals_wav,     vocals_r2_key,     "audio/wav")
            storage.upload(background_wav, background_r2_key, "audio/wav")

        # ── Step 2: Transcribe ────────────────────────────────────────────────
        current_step = "transcribing"
        progress.report(dub_id, current_step)
        segments, source_lang = transcribe.transcribe(vocals_wav)
        log.info(f"dub {dub_id}: {len(segments)} segments, lang={source_lang}")

        # ── Step 3: Extract speaker samples ───────────────────────────────────
        # (no separate progress step — part of transcribing phase)
        speaker_samples = extract_samples.extract_samples(segments, vocals_wav, work_dir)
        log.info(f"dub {dub_id}: {len(speaker_samples)} speakers sampled")

        speaker_samples_r2: dict[str, str] = {}
        for speaker, local_path in speaker_samples.items():
            r2_key = _stem_key(episode_id, f"speaker_{speaker}.wav")
            storage.upload(local_path, r2_key, "audio/wav")
            speaker_samples_r2[speaker] = r2_key

        # ── Step 3b: Split long segments ──────────────────────────────────────
        # Must run before translation — keeps each segment within XTTS-v2 limits.
        segments = split_segments.split_long_segments(segments)
        log.info(f"dub {dub_id}: {len(segments)} segments after split")

        # ── Step 4: Translate ─────────────────────────────────────────────────
        current_step = "translating"
        progress.report(dub_id, current_step)
        segments = translate.translate(segments, source_lang, language)

        # ── Step 5: Synthesize ────────────────────────────────────────────────
        current_step = "synthesizing"
        synth_dir = os.path.join(work_dir, "synth")
        os.makedirs(synth_dir, exist_ok=True)
        segments = _synthesize_with_progress(
            segments, speaker_samples, language, synth_dir, dub_id
        )

        # Upload synthesized segments to R2
        for seg in segments:
            if seg.get("synth_wav"):
                r2_key = _stem_key(episode_id, f"synth_{language}_{seg['idx']:04d}.wav")
                storage.upload(seg["synth_wav"], r2_key, "audio/wav")
                seg["synth_r2_key"] = r2_key

        # ── Step 6: Assemble ──────────────────────────────────────────────────
        current_step = "assembling"
        progress.report(dub_id, current_step)
        total_duration = _wav_duration(vocals_wav)
        dubbed_vocals = assemble.assemble(segments, total_duration, work_dir)

        # ── Step 7: Mix ───────────────────────────────────────────────────────
        current_step = "mixing"
        progress.report(dub_id, current_step)
        final_mp3 = mix.mix(dubbed_vocals, background_wav, work_dir, bg_volume=bg_volume)

        # ── Upload final output ───────────────────────────────────────────────
        current_step = "uploading"
        progress.report(dub_id, current_step)
        r2_key = _dub_key(episode_id, language)
        r2_url = storage.upload(final_mp3, r2_key, "audio/mpeg")
        log.info(f"dub {dub_id}: uploaded → {r2_url}")

        duration_sec   = _wav_duration(dubbed_vocals)
        segment_count  = len([s for s in segments if s.get("synth_wav")])
        speaker_count  = len(speaker_samples)

        _callback(callback_url, {
            "job_id":        job_id,
            "dub_id":        dub_id,
            "episode_id":    episode_id,
            "language":      language,
            "success":       True,
            "r2_url":        r2_url,
            "duration_sec":  round(duration_sec, 1),
            "segment_count": segment_count,
            "speaker_count": speaker_count,
        })
        progress.report(dub_id, "complete", 100)

    except Exception as e:
        log.exception(f"dub {dub_id}: failed at step {current_step}")
        _callback(callback_url, {
            "job_id":  job_id,
            "dub_id":  dub_id,
            "success": False,
            "step":    current_step,
            "error":   str(e),
        })
        progress.report(dub_id, "failed")

    finally:
        shutil.rmtree(work_dir, ignore_errors=True)


# ── Synthesis with per-segment progress + duration enforcement ────────────────

def _synthesize_with_progress(
    segments: list[dict],
    speaker_samples: dict[str, str],
    language: str,
    synth_dir: str,
    dub_id: int,
) -> list[dict]:
    """
    Synthesize each segment with XTTS-v2.
    - Skips segments shorter than MIN_SEGMENT_SEC (inserts silence placeholder).
    - Skips segments longer than MAX_SEGMENT_SEC (too long for XTTS-v2 context).
    - Reports progress after each segment.
    """
    tts = _load_tts()
    xtts_lang = _xtts_lang_code(language)

    out = []
    total = len(segments)
    for i, seg in enumerate(segments):
        speaker   = seg.get("speaker", "SPEAKER_00")
        sample    = speaker_samples.get(speaker)
        text      = seg.get("translated_text", "").strip()
        orig_dur  = seg["end_sec"] - seg["start_sec"]

        synth_wav  = None
        synth_dur  = None

        if orig_dur < MIN_SEGMENT_SEC:
            log.debug(f"dub {dub_id}: seg {seg['idx']} too short ({orig_dur:.2f}s) — skipping")
        elif not sample or not text:
            log.warning(f"dub {dub_id}: seg {seg['idx']} — no sample or text")
        else:
            synth_path = os.path.join(synth_dir, f"synth_{seg['idx']:04d}.wav")
            try:
                tts.tts_to_file(
                    text=text,
                    speaker_wav=sample,
                    language=xtts_lang,
                    file_path=synth_path,
                )
                synth_dur = _synth_wav_duration(synth_path)
                synth_wav = synth_path
                log.info(f"dub {dub_id}: seg {seg['idx']} ({speaker}) → {synth_dur:.2f}s")
            except Exception as e:
                log.error(f"dub {dub_id}: seg {seg['idx']} synth failed: {e}")

        pct = int((i + 1) / total * 100)
        progress.report(dub_id, "synthesizing", pct)
        out.append({**seg, "synth_wav": synth_wav, "synth_duration": synth_dur})

    return out


# ── Helpers ───────────────────────────────────────────────────────────────────

def _wav_duration(path: str) -> float:
    import wave
    with wave.open(path, "rb") as wf:
        return wf.getnframes() / wf.getframerate()


def _r2_exists(key: str) -> bool:
    """Return True if the R2 object exists (HEAD request via boto3)."""
    try:
        storage._s3().head_object(Bucket=config.R2_BUCKET, Key=key)
        return True
    except Exception:
        return False


def _callback(url: str, payload: dict):
    try:
        requests.post(url, json=payload, timeout=30)
    except Exception as e:
        log.warning(f"callback failed ({url}): {e}")


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    os.makedirs(config.TEMP_DIR, exist_ok=True)

    r = redis.from_url(config.REDIS_URL)
    log.info(f"dub-worker: listening on {config.QUEUE_KEY}")

    while True:
        try:
            item = r.brpop(config.QUEUE_KEY, timeout=5)
            if item is None:
                continue
            _, raw = item
            job = json.loads(raw)
            log.info(f"dub-worker: got job dub_id={job.get('dub_id')}")
            process_job(job)
        except redis.exceptions.ConnectionError as e:
            log.error(f"Redis connection lost: {e} — retrying in 5s")
            time.sleep(5)
        except Exception:
            log.exception("dub-worker: unexpected error")
            time.sleep(1)


if __name__ == "__main__":
    main()
