"""Step 3 — Extract voice samples per speaker from the clean vocals stem.

Returns a dict of {speaker_id: local_wav_path}.
Selects the longest contiguous high-confidence segment per speaker,
targeting 15–30 seconds.
"""
import logging
import os
import subprocess

log = logging.getLogger(__name__)

TARGET_MIN = 15.0
TARGET_MAX = 30.0


def extract_samples(segments: list[dict], vocals_wav: str, out_dir: str) -> dict[str, str]:
    """
    For each unique speaker, find the best clip and extract from vocals_wav.
    Returns {speaker_id: path_to_wav}.
    """
    by_speaker: dict[str, list[dict]] = {}
    for seg in segments:
        by_speaker.setdefault(seg["speaker"], []).append(seg)

    samples = {}
    for speaker, segs in by_speaker.items():
        clip_path = _best_clip(speaker, segs, vocals_wav, out_dir)
        if clip_path:
            samples[speaker] = clip_path
            log.info(f"extract_samples: {speaker} → {clip_path}")

    return samples


def _best_clip(speaker: str, segs: list[dict], vocals_wav: str, out_dir: str) -> str | None:
    # Score each segment by avg word confidence; prefer longer segments
    def score(s):
        words = s.get("words") or []
        conf = sum(w.get("score", 0.5) for w in words) / max(len(words), 1)
        return conf * (s["end_sec"] - s["start_sec"])

    segs_sorted = sorted(
        [s for s in segs if s["end_sec"] - s["start_sec"] > 0.1],
        key=score, reverse=True,
    )

    # Greedily accumulate segments until we hit TARGET_MAX seconds
    selected = []
    total = 0.0
    for seg in segs_sorted:
        dur = seg["end_sec"] - seg["start_sec"]
        if total + dur > TARGET_MAX:
            break
        selected.append(seg)
        total += dur
        if total >= TARGET_MIN:
            break

    if not selected or total < 1.0:
        return None

    # Sort selected by time order for natural-sounding clip
    selected.sort(key=lambda s: s["start_sec"])

    out_path = os.path.join(out_dir, f"speaker_{speaker}.wav")

    if len(selected) == 1:
        seg = selected[0]
        _extract_clip(vocals_wav, seg["start_sec"], seg["end_sec"], out_path)
    else:
        # Concatenate multiple segments with a short silence gap
        clips = []
        for i, seg in enumerate(selected):
            clip = os.path.join(out_dir, f"_clip_{speaker}_{i}.wav")
            _extract_clip(vocals_wav, seg["start_sec"], seg["end_sec"], clip)
            clips.append(clip)
        _concat_clips(clips, out_path)
        for c in clips:
            os.unlink(c)

    return out_path


def _extract_clip(src: str, start: float, end: float, dst: str):
    subprocess.run(
        ["ffmpeg", "-y",
         "-ss", str(start), "-to", str(end),
         "-i", src,
         "-ar", "22050", "-ac", "1",
         "-sample_fmt", "s16", "-c:a", "pcm_s16le",
         dst],
        check=True, capture_output=True,
    )


def _concat_clips(clips: list[str], dst: str):
    import tempfile
    with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False) as f:
        for c in clips:
            f.write(f"file '{c}'\n")
        list_path = f.name
    try:
        subprocess.run(
            ["ffmpeg", "-y", "-f", "concat", "-safe", "0",
             "-i", list_path, "-c", "copy", dst],
            check=True, capture_output=True,
        )
    finally:
        os.unlink(list_path)
