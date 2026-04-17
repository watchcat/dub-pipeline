"""Step 6 — Assemble synthesized segments into a single dubbed vocals track.

Natural-length approach (no time-stretching of the vocals):
- Synthesized segments are placed at their natural spoken length.
- Gaps between segments are compressed:
    gap_out = min(gap_in, max(0.2, gap_in * 0.6))
  Long deliberate pauses are preserved at 60% of original; incidental gaps
  collapse to a 200 ms minimum. Zero gap stays zero.
- Skipped segments (synthesis failed or too short) → silence for their original
  duration plus trailing gap, so the rhythm of skipped beats is preserved.
- synth_start_sec on each segment records the actual position in the output.
- Total duration is whatever falls out naturally — no cap.

Output: dubbed_vocals.wav at 48 kHz mono.
"""
import logging
import os
import subprocess
import tempfile

log = logging.getLogger(__name__)

SAMPLE_RATE = 48000
CHANNELS = 1
MIN_GAP = 0.2    # minimum compressed inter-segment gap (seconds)
GAP_RATIO = 0.6  # proportion of original gap to preserve


def assemble(segments: list[dict], out_dir: str) -> tuple[str, list[dict]]:
    """
    Build dubbed_vocals.wav by concatenating synthesized clips with compressed gaps.

    segments: list of dicts — must have start_sec, end_sec, and optionally
              synth_wav / synth_duration (set by the synthesize step).
    Returns (path to assembled WAV, updated segments with synth_start_sec populated).
    Segments without synth audio have synth_start_sec = None.
    """
    segs = sorted(segments, key=lambda s: s["start_sec"])
    if not any(s.get("synth_wav") for s in segs):
        raise RuntimeError("assemble: no synthesized segments to assemble")

    items: list[tuple] = []  # ("silence", dur_sec) | ("clip", path)
    cursor = 0.0        # running position in the output file
    prev_orig_end = 0.0  # end_sec of last processed original segment

    for i, seg in enumerate(segs):
        orig_start = seg["start_sec"]
        orig_end   = seg["end_sec"]
        orig_dur   = orig_end - orig_start
        gap_in     = max(0.0, orig_start - prev_orig_end)
        has_synth  = bool(seg.get("synth_wav") and seg.get("synth_duration"))

        if has_synth:
            gap_out = _compress_gap(gap_in)
            if gap_out > 0.005:
                items.append(("silence", gap_out))
                cursor += gap_out

            seg["synth_start_sec"] = cursor
            items.append(("clip", seg["synth_wav"]))
            cursor += float(seg["synth_duration"])
            prev_orig_end = orig_end

        else:
            # Skipped segment — preserve its time slot as silence.
            # Consume leading gap + original duration + trailing gap so the
            # next segment's gap_in is naturally 0.
            next_orig_start = segs[i + 1]["start_sec"] if i + 1 < len(segs) else orig_end
            trailing_gap    = max(0.0, next_orig_start - orig_end)
            silence_dur     = gap_in + orig_dur + trailing_gap
            if silence_dur > 0.005:
                items.append(("silence", silence_dur))
                cursor += silence_dur
            seg["synth_start_sec"] = None
            prev_orig_end = next_orig_start

    out_path = os.path.join(out_dir, "dubbed_vocals.wav")
    _ffmpeg_concat(items, out_path)

    n_clips = sum(1 for s in segs if s.get("synth_wav"))
    log.info(f"assemble: {out_path} ({cursor:.1f}s, {n_clips} clips)")
    return out_path, segs


def _compress_gap(gap_in: float) -> float:
    """Compress an inter-segment gap: 60% of original, min 200 ms, 0 if no gap."""
    if gap_in <= 0.005:
        return 0.0
    return min(gap_in, max(MIN_GAP, gap_in * GAP_RATIO))


def _ffmpeg_concat(items: list[tuple], dst: str):
    with tempfile.TemporaryDirectory() as tmpdir:
        inputs = []
        for i, item in enumerate(items):
            if item[0] == "silence":
                sil = os.path.join(tmpdir, f"sil_{i}.wav")
                _generate_silence(float(item[1]), sil)
                inputs.append(sil)
            else:
                inputs.append(str(item[1]))

        list_path = os.path.join(tmpdir, "concat.txt")
        with open(list_path, "w") as f:
            for p in inputs:
                f.write(f"file '{p}'\n")

        subprocess.run(
            ["ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
             "-f", "concat", "-safe", "0", "-i", list_path,
             "-ar", str(SAMPLE_RATE), "-ac", str(CHANNELS),
             "-sample_fmt", "s16", "-c:a", "pcm_s16le", dst],
            check=True,
        )


def _generate_silence(duration: float, path: str):
    subprocess.run(
        ["ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
         "-f", "lavfi", "-i", f"anullsrc=r={SAMPLE_RATE}:cl=mono",
         "-t", str(duration),
         "-sample_fmt", "s16", "-c:a", "pcm_s16le", path],
        check=True,
    )
