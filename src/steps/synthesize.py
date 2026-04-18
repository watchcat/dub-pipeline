"""Step 5 — Synthesize translated segments with VoxCPM2.

Context-aware synthesis: each segment is synthesized as a prosodic continuation
of the previous synthesized segments of the same speaker, while keeping the
speaker sample for timbre cloning. This reduces prosody resets at segment
boundaries and improves intonation coherence on long monologues.

Context selection (per segment, walking backwards within same speaker):
- Collect up to SYNTH_CONTEXT_MAX_SEGMENTS prior synthesized segments
- Stop if total prompt audio duration would exceed SYNTH_CONTEXT_MAX_DURATION_SEC
- Stop if original-timeline gap to the candidate exceeds SYNTH_CONTEXT_MAX_GAP_SEC
- Skip silence-placeholder segments (those without synth_wav)

The main synthesis loop lives in worker.py (_synthesize_with_progress).
This module provides helpers called from there.
"""
import hashlib
import logging
import os
import re
import subprocess
import tempfile
import threading
import unicodedata

from src import config

log = logging.getLogger(__name__)

_voxcpm_model = None

_sha256_cache: dict[str, str] = {}
_sha256_lock = threading.Lock()


# ── Model loading ─────────────────────────────────────────────────────────────

def _load_model():
    global _voxcpm_model
    if _voxcpm_model is None:
        import torch
        # TorchDynamo can't trace set.symmetric_difference inside einops, which
        # is called during VoxCPM2's warmup generate() in from_pretrained.
        torch._dynamo.config.disable = True
        from voxcpm import VoxCPM
        log.info("synthesize: loading VoxCPM2")
        _voxcpm_model = VoxCPM.from_pretrained("openbmb/VoxCPM2", load_denoiser=False)
    return _voxcpm_model


# ── Context selection ─────────────────────────────────────────────────────────

def _select_context(
    done_segs: list[dict],
    current_seg: dict,
) -> tuple[list[dict], str | None]:
    """
    Walk backwards through done_segs (same speaker, successfully synthesized)
    and collect prior segments to use as the prosodic prompt.

    Returns (context_segs_oldest_first, fallback_reason_or_None).
    fallback_reason is set when the returned list is empty.
    """
    if config.SYNTH_CONTEXT_MAX_SEGMENTS == 0:
        return [], "context_disabled"
    if not done_segs:
        return [], "first_segment"

    selected: list[dict] = []
    total_dur = 0.0

    for seg in reversed(done_segs):
        if len(selected) >= config.SYNTH_CONTEXT_MAX_SEGMENTS:
            break

        dur = seg.get("synth_duration") or 0.0
        gap = current_seg["start_sec"] - seg["end_sec"]

        if gap > config.SYNTH_CONTEXT_MAX_GAP_SEC:
            # Remaining candidates are even further back — stop.
            break
        if total_dur + dur > config.SYNTH_CONTEXT_MAX_DURATION_SEC:
            break

        total_dur += dur
        selected.append(seg)

    if not selected:
        last = done_segs[-1]
        gap = current_seg["start_sec"] - last["end_sec"]
        reason = "prev_gap_too_large" if gap > config.SYNTH_CONTEXT_MAX_GAP_SEC else "prev_segments_skipped"
        return [], reason

    selected.reverse()  # oldest first
    return selected, None


# ── Prompt assembly ───────────────────────────────────────────────────────────

def _build_prompt(
    ctx_segs: list[dict],
    work_dir: str,
    idx: int,
) -> tuple[str | None, str | None, bool]:
    """
    Build the prosodic prompt from context segments.

    Returns (wav_path, prompt_text, is_temp_file).
    Caller must os.unlink(wav_path) if is_temp_file is True.
    Raises subprocess.CalledProcessError on ffmpeg failure.
    """
    if not ctx_segs:
        return None, None, False

    prompt_text = " ".join(s.get("translated_text", "") for s in ctx_segs)

    if len(ctx_segs) == 1:
        # Use the checkpoint file directly — no copy needed.
        return ctx_segs[0]["synth_wav"], prompt_text, False

    out_path = os.path.join(work_dir, f"prompt_{idx:04d}.wav")
    _concat_wavs([s["synth_wav"] for s in ctx_segs], out_path)
    return out_path, prompt_text, True


# ── Checkpoint hash ───────────────────────────────────────────────────────────

def _synth_hash(
    translated_text: str,
    speaker_id: str,
    speaker_sha: str,
    cfg_value: float,
    inference_timesteps: int,
    context_hashes: list[str],
) -> str:
    """
    Deterministic hash that identifies the exact synthesis inputs for a segment,
    including the chain of prior context segment outputs. Changing any input
    (text, speaker sample, config, or context) produces a different hash,
    causing the checkpoint to miss and triggering re-synthesis.
    """
    h = hashlib.sha256()
    h.update(translated_text.encode("utf-8"))
    h.update(b"\x00")
    h.update(speaker_id.encode("utf-8"))
    h.update(b"\x00")
    h.update(speaker_sha.encode("utf-8"))
    h.update(b"\x00VoxCPM2\x00")
    h.update(f"{cfg_value:.4f}".encode())
    h.update(b"\x00")
    h.update(str(inference_timesteps).encode())
    for ctx_hash in context_hashes:
        h.update(b"\x00")
        h.update(ctx_hash.encode("utf-8"))
    return h.hexdigest()


def _file_sha256(path: str) -> str:
    """SHA-256 of a file, cached to avoid re-hashing the same file twice."""
    with _sha256_lock:
        if path in _sha256_cache:
            return _sha256_cache[path]
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    result = h.hexdigest()
    with _sha256_lock:
        _sha256_cache[path] = result
    return result


# ── Audio helpers ─────────────────────────────────────────────────────────────

def _concat_wavs(wav_paths: list[str], out_path: str):
    """Concatenate WAV files via ffmpeg concat demuxer (stream copy, no re-encode)."""
    with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False) as f:
        for p in wav_paths:
            f.write(f"file '{p}'\n")
        list_path = f.name
    try:
        subprocess.run(
            ["ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
             "-f", "concat", "-safe", "0", "-i", list_path,
             "-c", "copy", out_path],
            check=True,
        )
    finally:
        os.unlink(list_path)


def _wav_duration(path: str) -> float:
    import wave
    with wave.open(path, "rb") as wf:
        return wf.getnframes() / wf.getframerate()


def _clean_for_tts(text: str) -> str:
    """Normalize Unicode punctuation that may confuse TTS models."""
    text = text.replace('\u201c', '"').replace('\u201d', '"')
    text = text.replace('\u2018', "'").replace('\u2019', "'")
    text = text.replace('\u2014', ', ').replace('\u2013', ', ')
    text = text.replace('\u2026', '...')
    text = ''.join(c for c in text if unicodedata.category(c)[0] != 'C')
    text = re.sub(r' {2,}', ' ', text)
    return text.strip()
