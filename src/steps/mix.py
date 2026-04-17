"""Step 7 — Mix dubbed vocals with original background stem.

The background stem is time-adjusted to match the dubbed vocals duration
before mixing:

  - |ratio - 1| ≤ 30%  → arubberband time-stretch (high quality, nearly
                          inaudible at 15% volume on ambient/music stems)
  - ratio > 1.30        → loop with 2-second fade-out at end
  - ratio < 0.75        → trim with 1-second fade-out

where ratio = vocals_duration / background_duration.

Output: final MP3 at 128 kbps stereo, 44.1 kHz.
Requires ffmpeg compiled with librubberband (apt: librubberband2).
"""
import logging
import math
import os
import subprocess

log = logging.getLogger(__name__)


def mix(dubbed_vocals_wav: str, background_wav: str, out_dir: str, bg_volume: float = 0.15) -> str:
    """
    Mix dubbed vocals + time-adjusted background into a final MP3.
    Returns path to output MP3.
    """
    vocals_dur = _ffprobe_duration(dubbed_vocals_wav)
    bg_dur     = _ffprobe_duration(background_wav)
    ratio      = vocals_dur / bg_dur
    log.info(
        f"mix: vocals={vocals_dur:.1f}s bg={bg_dur:.1f}s "
        f"ratio={ratio:.3f} bg_volume={bg_volume}"
    )

    adjusted_bg = os.path.join(out_dir, "background_adjusted.wav")
    _adjust_background(background_wav, adjusted_bg, vocals_dur, bg_dur, ratio)

    out_path = os.path.join(out_dir, "dubbed_final.mp3")
    filter_graph = (
        "[0:a]aresample=44100,aformat=channel_layouts=stereo[v];"
        f"[1:a]volume={bg_volume}[b];"
        "[v][b]amix=inputs=2:duration=first:dropout_transition=0[out]"
    )
    subprocess.run(
        ["ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
         "-i", dubbed_vocals_wav,
         "-i", adjusted_bg,
         "-filter_complex", filter_graph,
         "-map", "[out]",
         "-c:a", "libmp3lame", "-b:a", "128k",
         out_path],
        check=True,
    )
    log.info(f"mix: done → {out_path}")
    return out_path


def _adjust_background(
    bg_wav: str,
    out_path: str,
    target_dur: float,
    bg_dur: float,
    ratio: float,
):
    """Stretch, loop, or trim background to match target_dur."""
    if 0.75 <= ratio <= 1.30:
        # High-quality time-stretch with arubberband.
        # tempo < 1 slows down (longer output); tempo > 1 speeds up (shorter).
        tempo = bg_dur / target_dur
        subprocess.run(
            ["ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
             "-i", bg_wav,
             "-filter:a", f"arubberband=tempo={tempo:.6f}",
             "-t", str(target_dur),
             out_path],
            check=True,
        )
        log.info(f"mix: arubberband stretch ratio={ratio:.3f} tempo={tempo:.3f}")

    elif ratio > 1.30:
        # Background too short — loop it, then trim with a 2-second fade-out.
        n_loops = math.ceil(ratio) + 1
        fade_start = max(0.0, target_dur - 2.0)
        subprocess.run(
            ["ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
             "-stream_loop", str(n_loops), "-i", bg_wav,
             "-t", str(target_dur),
             "-filter:a", f"afade=t=out:st={fade_start:.3f}:d=2.0",
             out_path],
            check=True,
        )
        log.info(f"mix: looped background ×{n_loops}, fade from {fade_start:.1f}s")

    else:
        # ratio < 0.75 — background longer than vocals, trim with 1-second fade-out.
        fade_start = max(0.0, target_dur - 1.0)
        subprocess.run(
            ["ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
             "-i", bg_wav,
             "-t", str(target_dur),
             "-filter:a", f"afade=t=out:st={fade_start:.3f}:d=1.0",
             out_path],
            check=True,
        )
        log.info(f"mix: trimmed background to {target_dur:.1f}s")


def _ffprobe_duration(path: str) -> float:
    result = subprocess.run(
        ["ffprobe", "-v", "error",
         "-show_entries", "format=duration",
         "-of", "default=noprint_wrappers=1:nokey=1",
         path],
        capture_output=True, text=True, check=True,
    )
    return float(result.stdout.strip())
