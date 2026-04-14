"""Step 6 — Assemble synthesized segments into a single dubbed vocals track.

Uses a cursor-based algorithm:
- Segment runs long → consume the following gap to avoid time-stretching
- Segment runs short → insert 50% of the original gap as pacing silence
- Segments never overlap
- Total duration capped at 110% of original

Output: dubbed_vocals.wav at 24 kHz mono.
"""
import logging
import os
import subprocess
import tempfile

log = logging.getLogger(__name__)

SAMPLE_RATE = 24000  # Hz — spec requirement
CHANNELS = 1
MAX_DURATION_RATIO = 1.10  # 110% of original total duration


def assemble(segments: list[dict], total_duration: float, out_dir: str) -> tuple[str, dict[int, float]]:
    """
    Build dubbed_vocals.wav by placing synthesized clips on a timeline.

    segments: list of dicts with keys: idx, start_sec, end_sec, synth_wav,
              synth_duration (may be None if synthesis failed).
    total_duration: original episode duration in seconds.
    Returns (path to assembled WAV, {segment_idx: actual_start_sec_in_dubbed_audio}).
    """
    valid = [s for s in segments if s.get("synth_wav") and s.get("synth_duration")]
    if not valid:
        raise RuntimeError("assemble: no synthesized segments to assemble")

    valid.sort(key=lambda s: s["start_sec"])

    max_output_duration = total_duration * MAX_DURATION_RATIO

    # ── Cursor-based placement ────────────────────────────────────────────────
    # Each entry: (place_at_sec, wav_path)
    timeline: list[tuple[float, str]] = []
    synth_timeline: dict[int, float] = {}  # idx → actual start sec in dubbed audio
    cursor = 0.0
    prev_end = 0.0  # original end time of previous segment

    for seg in valid:
        orig_start   = seg["start_sec"]
        orig_end     = seg["end_sec"]
        orig_dur     = orig_end - orig_start
        synth_dur    = seg["synth_duration"]
        original_gap = orig_start - prev_end   # silence before this segment in original

        place_at = cursor
        synth_timeline[seg["idx"]] = place_at

        if synth_dur > orig_dur:
            # Segment ran long — consume part of the following gap instead of overlapping
            overshoot    = synth_dur - orig_dur
            compression  = min(original_gap, overshoot)
            cursor      += synth_dur - compression
        else:
            # Segment ran short — preserve 50% of original pacing gap
            silence = original_gap * 0.5
            cursor += synth_dur + silence

        timeline.append((place_at, seg["synth_wav"]))
        prev_end = orig_end

    # ── Duration cap ─────────────────────────────────────────────────────────
    actual_end = cursor
    if actual_end > max_output_duration:
        log.warning(
            f"assemble: output {actual_end:.1f}s > 110% of original {total_duration:.1f}s "
            f"({max_output_duration:.1f}s) — truncating"
        )
        actual_end = max_output_duration

    out_path = os.path.join(out_dir, "dubbed_vocals.wav")
    _build_timeline(timeline, actual_end, out_path)
    return out_path, synth_timeline


def _build_timeline(
    timeline: list[tuple[float, str]],
    total_duration: float,
    dst: str,
):
    """Interleave silence + clips to match the computed placement positions."""
    items: list[tuple[str, float | str]] = []  # ("silence", secs) | ("clip", path)
    cursor = 0.0

    for place_at, wav_path in timeline:
        clip_dur = _wav_duration(wav_path)

        # Guard: skip clips that start beyond the output duration cap
        if place_at >= total_duration:
            break

        if place_at > cursor + 0.005:
            items.append(("silence", place_at - cursor))

        # Trim clip if it would exceed the cap
        clip_end = place_at + clip_dur
        if clip_end > total_duration:
            trimmed = os.path.join(
                os.path.dirname(dst),
                f"_trim_{len(items)}.wav",
            )
            _trim_clip(wav_path, total_duration - place_at, trimmed)
            items.append(("clip", trimmed))
            cursor = total_duration
            break
        else:
            items.append(("clip", wav_path))
            cursor = clip_end

    if cursor < total_duration - 0.005:
        items.append(("silence", total_duration - cursor))

    _ffmpeg_concat(items, dst)


def _ffmpeg_concat(items: list[tuple], dst: str):
    with tempfile.TemporaryDirectory() as tmpdir:
        inputs = []
        for i, (kind, val) in enumerate(items):
            if kind == "silence":
                sil = os.path.join(tmpdir, f"sil_{i}.wav")
                _generate_silence(float(val), sil)
                inputs.append(sil)
            else:
                inputs.append(str(val))

        list_path = os.path.join(tmpdir, "concat.txt")
        with open(list_path, "w") as f:
            for p in inputs:
                f.write(f"file '{p}'\n")

        subprocess.run(
            ["ffmpeg", "-y", "-f", "concat", "-safe", "0",
             "-i", list_path,
             "-ar", str(SAMPLE_RATE), "-ac", str(CHANNELS),
             "-sample_fmt", "s16", "-c:a", "pcm_s16le", dst],
            check=True, capture_output=True,
        )


def _generate_silence(duration: float, path: str):
    subprocess.run(
        ["ffmpeg", "-y",
         "-f", "lavfi", "-i", f"anullsrc=r={SAMPLE_RATE}:cl=mono",
         "-t", str(duration),
         "-sample_fmt", "s16", "-c:a", "pcm_s16le", path],
        check=True, capture_output=True,
    )


def _trim_clip(src: str, duration: float, dst: str):
    subprocess.run(
        ["ffmpeg", "-y", "-i", src, "-t", str(duration),
         "-ar", str(SAMPLE_RATE), "-ac", str(CHANNELS),
         "-sample_fmt", "s16", "-c:a", "pcm_s16le", dst],
        check=True, capture_output=True,
    )


def _wav_duration(path: str) -> float:
    import wave
    with wave.open(path, "rb") as wf:
        return wf.getnframes() / wf.getframerate()
