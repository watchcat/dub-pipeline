"""Step 7 — Mix dubbed vocals with original background stem.

Combines:
  - dubbed_vocals.wav  (22050 Hz mono, from assemble step)
  - background.wav     (44100 Hz stereo, from separate step)

Output: final MP3 at 128 kbps stereo, suitable for streaming.
"""
import logging
import os
import subprocess
from src import config

log = logging.getLogger(__name__)


def mix(dubbed_vocals_wav: str, background_wav: str, out_dir: str, bg_volume: float = 0.15) -> str:
    """
    Mix dubbed vocals + background into a final MP3.
    bg_volume: background level relative to dubbed vocals (0.0–1.0).
    Returns path to output MP3.
    """
    out_path = os.path.join(out_dir, "dubbed_final.mp3")

    log.info(f"mix: bg_volume={bg_volume}, output={out_path}")

    # ffmpeg filter graph:
    #   [0] dubbed vocals → upsample to 44100 stereo
    #   [1] background    → volume scaled
    #   amix with weights
    filter_graph = (
        f"[0:a]aresample=44100,aformat=channel_layouts=stereo[vocals];"
        f"[1:a]volume={bg_volume}[bg];"
        f"[vocals][bg]amix=inputs=2:duration=first:dropout_transition=0[out]"
    )

    subprocess.run(
        [
            "ffmpeg", "-y",
            "-i", dubbed_vocals_wav,
            "-i", background_wav,
            "-filter_complex", filter_graph,
            "-map", "[out]",
            "-c:a", "libmp3lame", "-b:a", "128k",
            out_path,
        ],
        check=True, capture_output=True,
    )

    log.info(f"mix: done → {out_path}")
    return out_path
