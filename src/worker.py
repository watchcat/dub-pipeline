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
# Enable MPS fallback for PyTorch ops not natively supported on Apple Silicon
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
# Suppress noisy HuggingFace tokenizer warning (pad_token == eos_token)
warnings.filterwarnings("ignore", message=".*attention_mask.*")

import json
import logging
logging.getLogger("google_genai.models").setLevel(logging.WARNING)
import shutil
import subprocess
import tempfile

import requests
import runpod
import soundfile as sf

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
from src.steps.synthesize import _load_model, _wav_duration as _synth_wav_duration, _clean_for_tts

logging.basicConfig(
    level=logging.INFO,
    format="%(filename)-20s:%-5d %(asctime)s %(message)s",
)
log = logging.getLogger(__name__)

# Segment duration constraints (spec §Constraints)
MIN_SEGMENT_SEC = 0.5
MAX_SEGMENT_SEC = 30.0


# ── Checkpoint ────────────────────────────────────────────────────────────────

class _Checkpoint:
    """Persists pipeline state to the RunPod Network Volume so jobs can resume
    after a container restart without re-running completed steps."""

    def __init__(self, dub_id: int):
        self.dir = os.path.join(config.CHECKPOINT_DIR, str(dub_id))
        self._state_path = os.path.join(self.dir, "state.json")
        os.makedirs(self.dir, exist_ok=True)
        self._state = self._load()

    def _load(self) -> dict:
        if os.path.exists(self._state_path):
            try:
                with open(self._state_path) as f:
                    state = json.load(f)
                log.info(f"checkpoint: resuming from {self._state_path}, steps done: {list(state.keys())}")
                return state
            except Exception as e:
                log.warning(f"checkpoint: failed to load state: {e} — starting fresh")
        return {}

    def _flush(self):
        with open(self._state_path, "w") as f:
            json.dump(self._state, f)

    def done(self, step: str) -> bool:
        return self._state.get(step, {}).get("done", False)

    def get(self, step: str, key: str, default=None):
        return self._state.get(step, {}).get(key, default)

    def save(self, step: str, **kwargs):
        self._state[step] = {"done": True, **kwargs}
        self._flush()
        log.info(f"checkpoint: saved step '{step}'")

    def synth_path(self, idx: int) -> str:
        synth_dir = os.path.join(self.dir, "synth")
        os.makedirs(synth_dir, exist_ok=True)
        return os.path.join(synth_dir, f"{idx:04d}.wav")

    def synth_done(self, idx: int) -> bool:
        return os.path.exists(self.synth_path(idx))

    def cleanup(self):
        shutil.rmtree(self.dir, ignore_errors=True)
        log.info(f"checkpoint: cleaned up {self.dir}")


# ── R2 key helpers ────────────────────────────────────────────────────────────

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

    ckpt = _Checkpoint(dub_id)
    work_dir = tempfile.mkdtemp(dir=config.TEMP_DIR, prefix=f"dub_{dub_id}_")
    log.info(f"dub {dub_id}: starting in {work_dir}")

    current_step = "queued"

    try:
        vocals_wav     = os.path.join(work_dir, "vocals.wav")
        background_wav = os.path.join(work_dir, "background.wav")

        # ── Step 1: Separate ──────────────────────────────────────────────────
        current_step = "separating"
        progress.report(dub_id, current_step)

        vocals_r2_key     = _stem_key(episode_id, "vocals.wav")
        background_r2_key = _stem_key(episode_id, "background.wav")

        if ckpt.done("separate") or (_r2_exists(vocals_r2_key) and _r2_exists(background_r2_key)):
            log.info(f"dub {dub_id}: reusing existing stems from R2")
            storage.download(vocals_r2_key, vocals_wav)
            storage.download(background_r2_key, background_wav)
            if not ckpt.done("separate"):
                ckpt.save("separate", vocals_r2_key=vocals_r2_key, background_r2_key=background_r2_key)
        else:
            vocals_wav, background_wav = separate.separate(audio_url, work_dir)
            storage.upload(vocals_wav,     vocals_r2_key,     "audio/wav")
            storage.upload(background_wav, background_r2_key, "audio/wav")
            ckpt.save("separate", vocals_r2_key=vocals_r2_key, background_r2_key=background_r2_key)

        # ── Step 2: Transcribe ────────────────────────────────────────────────
        current_step = "transcribing"
        progress.report(dub_id, current_step)

        if ckpt.done("transcribe"):
            segments    = ckpt.get("transcribe", "segments")
            source_lang = ckpt.get("transcribe", "source_lang")
            log.info(f"dub {dub_id}: resumed transcription — {len(segments)} segments, lang={source_lang}")
        else:
            segments, source_lang = transcribe.transcribe(vocals_wav)
            log.info(f"dub {dub_id}: {len(segments)} segments, lang={source_lang}")
            ckpt.save("transcribe", segments=segments, source_lang=source_lang)

        # ── Step 3: Extract speaker samples ───────────────────────────────────
        if ckpt.done("speakers"):
            speaker_samples_r2 = ckpt.get("speakers", "speaker_samples_r2")
            # Download speaker wavs to work_dir for TTS embedding computation
            speaker_samples = {}
            for speaker, r2_key in speaker_samples_r2.items():
                local_path = os.path.join(work_dir, f"speaker_{speaker}.wav")
                storage.download(r2_key, local_path)
                speaker_samples[speaker] = local_path
            log.info(f"dub {dub_id}: resumed {len(speaker_samples)} speakers")
        else:
            speaker_samples = extract_samples.extract_samples(segments, vocals_wav, work_dir)
            log.info(f"dub {dub_id}: {len(speaker_samples)} speakers sampled")
            speaker_samples_r2: dict[str, str] = {}
            for speaker, local_path in speaker_samples.items():
                r2_key = _stem_key(episode_id, f"speaker_{speaker}.wav")
                storage.upload(local_path, r2_key, "audio/wav")
                speaker_samples_r2[speaker] = r2_key
            ckpt.save("speakers", speaker_samples_r2=speaker_samples_r2)

        # ── Step 3b: Split long segments ──────────────────────────────────────
        if ckpt.done("split"):
            segments = ckpt.get("split", "segments")
            log.info(f"dub {dub_id}: resumed {len(segments)} segments after split")
        else:
            segments = split_segments.split_long_segments(segments)
            log.info(f"dub {dub_id}: {len(segments)} segments after split")
            ckpt.save("split", segments=segments)

        # ── Step 4: Translate ─────────────────────────────────────────────────
        current_step = "translating"
        progress.report(dub_id, current_step)

        if ckpt.done("translate"):
            segments = ckpt.get("translate", "segments")
            log.info(f"dub {dub_id}: resumed translation")
        else:
            segments = translate.translate(segments, source_lang, language)
            ckpt.save("translate", segments=segments)

        # ── Step 5: Synthesize ────────────────────────────────────────────────
        current_step = "synthesizing"
        synth_dir = os.path.join(work_dir, "synth")
        os.makedirs(synth_dir, exist_ok=True)
        segments, synth_r2_keys = _synthesize_with_progress(
            segments, speaker_samples, language, episode_id, ckpt, dub_id
        )

        # ── Step 6: Assemble ──────────────────────────────────────────────────
        current_step = "assembling"
        progress.report(dub_id, current_step)
        dubbed_vocals, segments = assemble.assemble(segments, work_dir)

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

        duration_sec   = _ffprobe_duration(final_mp3)
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
                "synth_r2_key":    synth_r2_keys.get(seg["idx"]),
                "synth_duration":  seg.get("synth_duration"),
                "synth_start_sec": seg.get("synth_start_sec"),
            }
            for seg in segments
        ]

        _callback(callback_url, {
            "job_id":          job_id,
            "dub_id":          dub_id,
            "episode_id":      episode_id,
            "language":        language,
            "source_lang":     source_lang,
            "success":         True,
            "r2_url":          r2_url,
            "duration_sec":    round(duration_sec, 1),
            "segment_count":   segment_count,
            "speaker_count":   speaker_count,
            "speaker_samples": json.dumps(speaker_samples_r2),
            "segments":        segment_data,
        })
        progress.report(dub_id, "complete", 100)
        ckpt.cleanup()

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


# ── Synthesis with per-segment checkpointing ──────────────────────────────────

def _synthesize_with_progress(
    segments: list[dict],
    speaker_samples: dict[str, str],
    language: str,
    episode_id: int,
    ckpt: _Checkpoint,
    dub_id: int,
) -> tuple[list[dict], dict[int, str]]:
    """
    Synthesize each segment with VoxCPM2.
    Saves each completed segment wav to the checkpoint dir immediately.
    On resume, skips already-synthesized segments.
    Returns (segments_with_synth, {idx: r2_key}).
    """
    model = _load_model()
    sr = model.tts_model.sample_rate
    log.info(f"dub {dub_id}: synthesizing {len(segments)} segments → {language} at {sr} Hz")

    out = []
    synth_r2_keys: dict[int, str] = {}
    total = len(segments)

    for i, seg in enumerate(segments):
        speaker  = seg.get("speaker", "SPEAKER_00")
        text     = _clean_for_tts(seg.get("translated_text", ""))
        orig_dur = seg["end_sec"] - seg["start_sec"]
        idx      = seg["idx"]

        synth_wav = None
        synth_dur = None
        ckpt_path = ckpt.synth_path(idx)
        r2_key    = _stem_key(episode_id, f"synth_{language}_{idx:04d}.wav")

        speaker_wav = speaker_samples.get(speaker)

        if orig_dur < MIN_SEGMENT_SEC:
            log.debug(f"dub {dub_id}: seg {idx} too short ({orig_dur:.2f}s) — skipping")

        elif not speaker_wav or not text:
            log.warning(f"dub {dub_id}: seg {idx} — no speaker sample or text")

        elif ckpt.synth_done(idx):
            # Resume: segment already synthesized in a previous run
            log.info(f"dub {dub_id}: seg {idx} — resumed from checkpoint")
            synth_dur = _synth_wav_duration(ckpt_path)
            synth_wav = ckpt_path
            if not _r2_exists(r2_key):
                storage.upload(ckpt_path, r2_key, "audio/wav")
            synth_r2_keys[idx] = r2_key

        else:
            try:
                if i < 3:
                    log.info(f"dub {dub_id}: seg {idx} sample text='{text[:80]}'")
                wav = model.generate(
                    text=text,
                    reference_wav_path=speaker_wav,
                    cfg_value=2.0,
                    inference_timesteps=10,
                )
                if wav is None:
                    raise ValueError("model.generate returned None")
                sf.write(ckpt_path, wav, sr)
                synth_dur = len(wav) / sr
                synth_wav = ckpt_path
                log.info(f"dub {dub_id}: seg {idx} ({speaker}) → {synth_dur:.2f}s")
            except Exception as e:
                log.error(f"dub {dub_id}: seg {idx} synth failed: {e}")

            # Upload to R2 separately — failure here doesn't discard the synthesized wav
            if synth_wav:
                try:
                    storage.upload(ckpt_path, r2_key, "audio/wav")
                    synth_r2_keys[idx] = r2_key
                except Exception as e:
                    log.warning(f"dub {dub_id}: seg {idx} R2 upload failed, will retry: {e}")

        pct = int((i + 1) / total * 100)
        progress.report(dub_id, "synthesizing", pct)
        out.append({**seg, "synth_wav": synth_wav, "synth_duration": synth_dur})

    # Retry any R2 uploads that failed during synthesis
    for seg in out:
        idx = seg["idx"]
        if seg.get("synth_wav") and idx not in synth_r2_keys:
            r2_key = _stem_key(episode_id, f"synth_{language}_{idx:04d}.wav")
            try:
                storage.upload(ckpt.synth_path(idx), r2_key, "audio/wav")
                synth_r2_keys[idx] = r2_key
                log.info(f"dub {dub_id}: seg {idx} R2 upload retry succeeded")
            except Exception as e:
                log.error(f"dub {dub_id}: seg {idx} R2 upload retry failed: {e}")

    return out, synth_r2_keys


# ── Helpers ───────────────────────────────────────────────────────────────────

def _ffprobe_duration(path: str) -> float:
    result = subprocess.run(
        ["ffprobe", "-v", "error",
         "-show_entries", "format=duration",
         "-of", "default=noprint_wrappers=1:nokey=1",
         path],
        capture_output=True, text=True, check=True,
    )
    return float(result.stdout.strip())


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
