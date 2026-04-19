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

import concurrent.futures
import json
import logging
logging.getLogger("google_genai.models").setLevel(logging.WARNING)
import shutil
import subprocess
import tempfile
import threading
from collections import defaultdict

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
    synthesize,
    transcribe,
    translate,
)

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

    def synth_path_v2(self, hash_hex: str) -> str:
        synth_dir = os.path.join(self.dir, "synth_v2")
        os.makedirs(synth_dir, exist_ok=True)
        return os.path.join(synth_dir, f"{hash_hex[:32]}.wav")

    def synth_done_v2(self, hash_hex: str) -> bool:
        return os.path.exists(self.synth_path_v2(hash_hex))

    def migrate_synth_checkpoints(self):
        """Delete legacy synth/ dir when synth_v2/ doesn't exist yet.
        Ensures in-flight jobs with old-style checkpoints re-synthesize cleanly
        rather than partially resuming with mismatched hash chains."""
        synth_v2_dir = os.path.join(self.dir, "synth_v2")
        synth_v1_dir = os.path.join(self.dir, "synth")
        if not os.path.isdir(synth_v2_dir) and os.path.isdir(synth_v1_dir):
            log.info("checkpoint: deleting legacy synth/ dir (first run with hash-based checkpointing)")
            shutil.rmtree(synth_v1_dir, ignore_errors=True)

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
            segments, speaker_samples, language, episode_id, ckpt, dub_id, work_dir
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


# ── Synthesis with per-segment checkpointing and prosodic context ─────────────

def _synthesize_with_progress(
    segments: list[dict],
    speaker_samples: dict[str, str],
    language: str,
    episode_id: int,
    ckpt: _Checkpoint,
    dub_id: int,
    work_dir: str,
) -> tuple[list[dict], dict[int, str]]:
    """
    Synthesize all segments using VoxCPM2 with prosodic context.

    Segments are grouped by speaker. Each speaker's segments are processed in
    original order so that prior synthesized output can be used as a prosodic
    prompt. Speakers are processed in parallel when SYNTH_PARALLEL_SPEAKERS=true.

    Returns (segments_with_synth_fields, {idx: r2_key}).
    """
    model = synthesize._load_model()
    sr = model.tts_model.sample_rate
    log.info(
        f"dub {dub_id}: synthesizing {len(segments)} segments → {language} "
        f"at {sr} Hz, parallel_speakers={config.SYNTH_PARALLEL_SPEAKERS}"
    )

    ckpt.migrate_synth_checkpoints()

    debug_dir = os.path.join(config.TEMP_DIR, "synth_debug")
    os.makedirs(debug_dir, exist_ok=True)

    # Compute speaker sample sha256 once per speaker (feeds into checkpoint hash)
    speaker_sha = {
        spk: synthesize._file_sha256(path) if path and os.path.exists(path) else ""
        for spk, path in speaker_samples.items()
    }

    # Group segments into per-speaker queues, preserving original order
    speaker_queues: dict[str, list[dict]] = defaultdict(list)
    for seg in segments:
        speaker_queues[seg.get("speaker", "SPEAKER_00")].append(seg)

    out_by_idx: dict[int, dict] = {}
    synth_r2_keys: dict[int, str] = {}
    completed = [0]
    total = len(segments)
    collect_lock = threading.Lock()

    def on_done(result_seg: dict, r2_key: str | None):
        with collect_lock:
            out_by_idx[result_seg["idx"]] = result_seg
            if r2_key:
                synth_r2_keys[result_seg["idx"]] = r2_key
            completed[0] += 1
            pct = int(completed[0] / total * 100)
        progress.report(dub_id, "synthesizing", pct)

    def process_speaker(speaker_id: str, spk_segs: list[dict]):
        sample_path = speaker_samples.get(speaker_id)
        sha = speaker_sha.get(speaker_id, "")
        done_segs: list[dict] = []  # successfully synthesized segs for this speaker
        for seg in spk_segs:
            result, r2_key = _synth_one_segment(
                seg=seg,
                speaker_id=speaker_id,
                sample_path=sample_path,
                speaker_sha=sha,
                done_segs=done_segs,
                model=model,
                sr=sr,
                language=language,
                episode_id=episode_id,
                ckpt=ckpt,
                work_dir=work_dir,
                debug_dir=debug_dir,
                dub_id=dub_id,
            )
            if result.get("synth_wav"):
                done_segs.append(result)
            on_done(result, r2_key)

    if config.SYNTH_PARALLEL_SPEAKERS and len(speaker_queues) > 1:
        with concurrent.futures.ThreadPoolExecutor(max_workers=len(speaker_queues)) as executor:
            futures = {
                executor.submit(process_speaker, spk, segs): spk
                for spk, segs in speaker_queues.items()
            }
            for f in concurrent.futures.as_completed(futures):
                exc = f.exception()
                if exc:
                    raise exc
    else:
        for spk, segs in speaker_queues.items():
            process_speaker(spk, segs)

    # Retry any R2 uploads that failed during synthesis
    for seg_out in out_by_idx.values():
        idx = seg_out["idx"]
        if seg_out.get("synth_wav") and idx not in synth_r2_keys:
            r2_key = _stem_key(episode_id, f"synth_{language}_{idx:04d}.wav")
            try:
                storage.upload(seg_out["synth_wav"], r2_key, "audio/wav")
                synth_r2_keys[idx] = r2_key
                log.info(f"dub {dub_id}: seg {idx} R2 upload retry succeeded")
            except Exception as e:
                log.error(f"dub {dub_id}: seg {idx} R2 upload retry failed: {e}")

    # Reassemble in original segment order
    out = [
        out_by_idx.get(seg["idx"], {**seg, "synth_wav": None, "synth_duration": None})
        for seg in segments
    ]

    synth_count = sum(1 for s in out if s.get("synth_wav"))
    skip_count  = total - synth_count
    log.info(
        f"dub {dub_id}: synthesis complete — "
        f"{synth_count}/{total} synthesized ({synth_count * 100 // total}%), "
        f"{skip_count} skipped"
    )
    return out, synth_r2_keys


def _synth_one_segment(
    seg: dict,
    speaker_id: str,
    sample_path: str | None,
    speaker_sha: str,
    done_segs: list[dict],
    model,
    sr: int,
    language: str,
    episode_id: int,
    ckpt: _Checkpoint,
    work_dir: str,
    debug_dir: str,
    dub_id: int,
) -> tuple[dict, str | None]:
    """
    Synthesize one segment with prosodic context from prior same-speaker segments.
    Returns (result_seg, r2_key_or_None).
    """
    idx      = seg["idx"]
    orig_dur = seg["end_sec"] - seg["start_sec"]
    text     = synthesize._clean_for_tts(seg.get("translated_text", ""))
    r2_key   = _stem_key(episode_id, f"synth_{language}_{idx:04d}.wav")

    debug: dict = {
        "idx":                   idx,
        "speaker_id":            speaker_id,
        "mode":                  None,
        "prompt_segment_indices": [],
        "prompt_duration_sec":   0.0,
        "prompt_text":           None,
        "target_text":           text,
        "synth_duration_sec":    None,
        "fallback_reason":       None,
    }

    def flush_debug():
        try:
            with open(os.path.join(debug_dir, f"{idx:04d}.json"), "w") as f:
                json.dump(debug, f, indent=2)
        except Exception:
            pass

    # ── Early skips ────────────────────────────────────────────────────────────

    if orig_dur < MIN_SEGMENT_SEC:
        debug["mode"] = "skipped_silence"
        flush_debug()
        return {**seg, "synth_wav": None, "synth_duration": None}, None

    if not sample_path or not text:
        debug["mode"] = "skipped_no_input"
        debug["fallback_reason"] = "no_speaker_sample" if not sample_path else "no_text"
        flush_debug()
        log.warning(f"dub {dub_id}: seg {idx} — no speaker sample or text")
        return {**seg, "synth_wav": None, "synth_duration": None}, None

    # ── Context selection and checkpoint hash ──────────────────────────────────

    ctx_segs, fallback_reason = synthesize._select_context(done_segs, seg)
    ctx_hashes = [synthesize._file_sha256(s["synth_wav"]) for s in ctx_segs]
    hash_key  = synthesize._synth_hash(
        text, speaker_id, speaker_sha,
        config.SYNTH_CFG_VALUE, config.SYNTH_INFERENCE_TIMESTEPS, ctx_hashes,
    )
    ckpt_path = ckpt.synth_path_v2(hash_key)

    # ── Resume: hash-based checkpoint ─────────────────────────────────────────

    if ckpt.synth_done_v2(hash_key):
        synth_dur = synthesize._wav_duration(ckpt_path)
        debug.update({
            "mode":                   f"context_n={len(ctx_segs)}" if ctx_segs else "isolated",
            "prompt_segment_indices": [s["idx"] for s in ctx_segs],
            "fallback_reason":        fallback_reason,
            "synth_duration_sec":     synth_dur,
        })
        flush_debug()
        log.info(f"dub {dub_id}: seg {idx} — resumed from checkpoint")
        if not _r2_exists(r2_key):
            try:
                storage.upload(ckpt_path, r2_key, "audio/wav")
            except Exception as e:
                log.warning(f"dub {dub_id}: seg {idx} R2 upload failed: {e}")
                r2_key = None
        return {**seg, "synth_wav": ckpt_path, "synth_duration": synth_dur}, r2_key

    # ── Build prosodic prompt ──────────────────────────────────────────────────

    prompt_wav  = None
    prompt_text = None
    prompt_is_temp = False

    if ctx_segs:
        try:
            prompt_wav, prompt_text, prompt_is_temp = synthesize._build_prompt(
                ctx_segs, work_dir, idx
            )
        except Exception as e:
            log.warning(f"dub {dub_id}: seg {idx} prompt build failed ({e}) — isolated mode")
            ctx_segs = []
            fallback_reason = "prompt_concat_failed"

    mode = f"context_n={len(ctx_segs)}" if ctx_segs else "isolated"
    debug.update({
        "mode":                   mode,
        "prompt_segment_indices": [s["idx"] for s in ctx_segs],
        "prompt_duration_sec":    sum(s.get("synth_duration") or 0.0 for s in ctx_segs),
        "prompt_text":            prompt_text,
        "fallback_reason":        fallback_reason,
    })

    # ── Synthesize ─────────────────────────────────────────────────────────────

    synth_wav = None
    synth_dur = None

    try:
        kwargs: dict = dict(
            text=text,
            reference_wav_path=sample_path,
            cfg_value=config.SYNTH_CFG_VALUE,
            inference_timesteps=config.SYNTH_INFERENCE_TIMESTEPS,
            normalize=True,   # expand digits/dates/currency to words in target language
            denoise=True,     # speaker sample is extracted from real podcast audio; denoise it
                              # (prompt_wav is already clean synth output — model applies
                              #  denoising only to reference_wav when both are present)
        )
        if prompt_wav and prompt_text:
            kwargs["prompt_wav_path"] = prompt_wav
            kwargs["prompt_text"]     = prompt_text

        wav = model.generate(**kwargs)
        if wav is None:
            raise ValueError("model.generate returned None")

        # Atomic write: temp + rename so partial files never look like valid checkpoints
        tmp_path = ckpt_path + ".tmp"
        sf.write(tmp_path, wav, sr, format="WAV")
        os.replace(tmp_path, ckpt_path)
        synth_dur = len(wav) / sr
        synth_wav = ckpt_path
        log.info(f"dub {dub_id}: seg {idx} ({speaker_id}) [{mode}] → {synth_dur:.2f}s")

    except Exception as e:
        log.error(f"dub {dub_id}: seg {idx} synth failed: {e}")
    finally:
        if prompt_is_temp and prompt_wav and os.path.exists(prompt_wav):
            try:
                os.unlink(prompt_wav)
            except Exception:
                pass

    debug["synth_duration_sec"] = synth_dur
    flush_debug()

    # ── Upload to R2 ──────────────────────────────────────────────────────────

    if synth_wav:
        try:
            storage.upload(ckpt_path, r2_key, "audio/wav")
        except Exception as e:
            log.warning(f"dub {dub_id}: seg {idx} R2 upload failed, will retry: {e}")
            r2_key = None
    else:
        r2_key = None

    return {**seg, "synth_wav": synth_wav, "synth_duration": synth_dur}, r2_key


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
