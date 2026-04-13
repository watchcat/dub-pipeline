"""RunPod Serverless worker — processes dub jobs dispatched by buzz-bot.

Job payload (JSON, received via RunPod handler input):
{
  "job_id":       "hex32",
  "dub_id":       123,
  "episode_id":   456,
  "audio_url":    "https://...",
  "language":     "es",
  "bg_volume":    0.15,
  "callback_url": "https://app.buzz-bot.top/internal/dub_result"
}

Progress and result are reported back to buzz-bot via HTTP callbacks
(/internal/dub_progress and /internal/dub_result).
"""
import os
import warnings
# Set before any torch/TTS import so MPS fallback is active from the start
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
# Suppress noisy HuggingFace tokenizer warning (pad_token == eos_token)
warnings.filterwarnings("ignore", message=".*attention_mask.*")

import json
import logging
logging.getLogger("google_genai.models").setLevel(logging.WARNING)
import shutil
import tempfile

import requests
import runpod

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
from src.steps.synthesize import _load_tts, _xtts_lang_code, _wav_duration as _synth_wav_duration, _clean_for_tts

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

        segment_data = [
            {
                "idx":             seg["idx"],
                "start_sec":       seg["start_sec"],
                "end_sec":         seg["end_sec"],
                "speaker_id":      seg.get("speaker"),
                "text":            seg.get("text", ""),
                "words":           seg.get("words"),
                "translated_text": seg.get("translated_text"),
                "synth_r2_key":    seg.get("synth_r2_key"),
                "synth_duration":  seg.get("synth_duration"),
            }
            for seg in segments
        ]

        _callback(callback_url, {
            "job_id":          job_id,
            "dub_id":          dub_id,
            "episode_id":      episode_id,
            "language":        language,
            "success":         True,
            "r2_url":          r2_url,
            "duration_sec":    round(duration_sec, 1),
            "segment_count":   segment_count,
            "speaker_count":   speaker_count,
            "speaker_samples": json.dumps(speaker_samples_r2),
            "segments":        segment_data,
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
    import torch
    import scipy.io.wavfile

    tts = _load_tts()
    model = tts.synthesizer.tts_model
    xtts_lang = _xtts_lang_code(language)

    # Pre-compute speaker embeddings once per speaker.
    speaker_latents: dict[str, tuple] = {}
    for speaker_id, wav_path in speaker_samples.items():
        if wav_path and os.path.exists(wav_path):
            log.info(f"dub {dub_id}: computing embeddings for {speaker_id}")
            gpt_cond_latent, speaker_embedding = model.get_conditioning_latents(
                audio_path=[wav_path]
            )
            speaker_latents[speaker_id] = (gpt_cond_latent, speaker_embedding)

    out = []
    total = len(segments)
    for i, seg in enumerate(segments):
        speaker   = seg.get("speaker", "SPEAKER_00")
        latents   = speaker_latents.get(speaker)
        text      = _clean_for_tts(seg.get("translated_text", ""))
        orig_dur  = seg["end_sec"] - seg["start_sec"]

        synth_wav  = None
        synth_dur  = None

        if orig_dur < MIN_SEGMENT_SEC:
            log.debug(f"dub {dub_id}: seg {seg['idx']} too short ({orig_dur:.2f}s) — skipping")
        elif not latents or not text:
            log.warning(f"dub {dub_id}: seg {seg['idx']} — no latents or text")
        else:
            synth_path = os.path.join(synth_dir, f"synth_{seg['idx']:04d}.wav")
            try:
                gpt_cond_latent, speaker_embedding = latents
                result = model.inference(
                    text=text,
                    language=xtts_lang,
                    gpt_cond_latent=gpt_cond_latent,
                    speaker_embedding=speaker_embedding,
                )
                wav = result["wav"]
                if isinstance(wav, torch.Tensor):
                    wav = wav.cpu().numpy()
                wav_int16 = (wav * 32767).clip(-32768, 32767).astype("int16")
                scipy.io.wavfile.write(synth_path, 24000, wav_int16)
                synth_dur = len(wav_int16) / 24000
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

def handler(job: dict) -> dict:
    """RunPod Serverless handler — called once per job by the RunPod runtime."""
    try:
        process_job(job["input"])
        return {"ok": True}
    except Exception as exc:
        log.exception(f"handler: unhandled exception: {exc}")
        return {"ok": False, "error": str(exc)}


# Ensure TEMP_DIR exists regardless of how this module is loaded.
os.makedirs(config.TEMP_DIR, exist_ok=True)

if __name__ == "__main__":
    runpod.serverless.start({"handler": handler})
