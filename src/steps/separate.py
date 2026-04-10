"""Step 1 — Source separation with Demucs (htdemucs_ft).

Returns paths to (vocals_wav, background_wav) as 16 kHz mono WAV files.
"""
import logging
import os
import subprocess
import urllib.request
from pathlib import Path

from src import config

log = logging.getLogger(__name__)


def separate(audio_url: str, out_dir: str) -> tuple[str, str]:
    """
    Download audio_url to a temp file, then run Demucs.
    Returns (vocals_wav, background_wav).
    vocals_wav:     16 kHz mono WAV (required by WhisperX and XTTS-v2)
    background_wav: 44.1 kHz stereo WAV (full quality for mixing)
    """
    # Download to local file — Demucs does not accept URLs
    ext = Path(audio_url.split("?")[0]).suffix or ".mp3"
    local_audio = os.path.join(out_dir, f"source{ext}")
    log.info(f"separate: downloading {audio_url}")
    req = urllib.request.Request(
        audio_url,
        headers={"User-Agent": "Mozilla/5.0 (compatible; dub-pipeline/1.0)"},
    )
    with urllib.request.urlopen(req) as resp, open(local_audio, "wb") as f:
        while chunk := resp.read(1 << 20):
            f.write(chunk)
    log.info(f"separate: downloaded → {local_audio}")

    log.info(f"separate: running demucs {config.DEMUCS_MODEL}")
    subprocess.run(
        [
            "python", "-m", "demucs",
            "--two-stems", "vocals",      # produces vocals + no_vocals (=background)
            "-n", config.DEMUCS_MODEL,
            "--out", out_dir,
            "--mp3",
            local_audio,
        ],
        check=True,
    )

    stem = Path(local_audio).stem
    demucs_out = Path(out_dir) / config.DEMUCS_MODEL / stem

    vocals_raw     = str(demucs_out / "vocals.mp3")
    background_raw = str(demucs_out / "no_vocals.mp3")

    vocals_wav     = os.path.join(out_dir, "vocals.wav")
    background_wav = os.path.join(out_dir, "background.wav")

    # Normalise vocals to 16 kHz mono (required by WhisperX and XTTS-v2)
    _to_wav_16k(vocals_raw, vocals_wav)
    # Background kept at original sample rate/channels for quality mixing
    _to_wav_44k(background_raw, background_wav)

    log.info(f"separate: done → {vocals_wav}, {background_wav}")
    return vocals_wav, background_wav


def _to_wav_16k(src: str, dst: str):
    subprocess.run(
        ["ffmpeg", "-y", "-i", src, "-ar", "16000", "-ac", "1",
         "-sample_fmt", "s16", "-c:a", "pcm_s16le", dst],
        check=True, capture_output=True,
    )


def _to_wav_44k(src: str, dst: str):
    subprocess.run(
        ["ffmpeg", "-y", "-i", src, "-ar", "44100", "-ac", "2",
         "-sample_fmt", "s16", "-c:a", "pcm_s16le", dst],
        check=True, capture_output=True,
    )
