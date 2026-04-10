"""Step 2 — Transcribe with mlx-whisper (Metal, Apple Silicon) + pyannote diarization.

mlx-whisper runs natively on MPS via Apple MLX — no ctranslate2/CUDA needed.
Pyannote speaker diarization runs on MPS via torch.

Returns a list of segments:
  [{"idx": 0, "start": 1.2, "end": 4.5, "speaker": "SPEAKER_00",
    "text": "Hello world.", "words": [...]}]
"""
import logging
import torch
from src import config

log = logging.getLogger(__name__)

_diarize_pipeline = None


def _patch_hf_hub():
    """pyannote.audio 3.x calls hf_hub_download(use_auth_token=...) which was
    removed in huggingface_hub 0.22. Translate it to `token` transparently."""
    import huggingface_hub
    _orig = huggingface_hub.hf_hub_download
    def _patched(*args, **kwargs):
        if "use_auth_token" in kwargs:
            kwargs.setdefault("token", kwargs.pop("use_auth_token"))
        return _orig(*args, **kwargs)
    huggingface_hub.hf_hub_download = _patched

_patch_hf_hub()


def _load_diarize():
    global _diarize_pipeline
    if _diarize_pipeline is None:
        from pyannote.audio import Pipeline
        log.info("transcribe: loading pyannote diarization pipeline")
        # HF_TOKEN env var is picked up automatically by huggingface_hub.
        # _patch_hf_hub() above translates the internal use_auth_token kwarg.
        _diarize_pipeline = Pipeline.from_pretrained(
            "pyannote/speaker-diarization-3.1",
        )
        _diarize_pipeline = _diarize_pipeline.to(torch.device("mps"))
    return _diarize_pipeline


def transcribe(vocals_wav: str) -> tuple[list[dict], str]:
    """
    Transcribe vocals_wav with mlx-whisper + pyannote diarization.
    Returns (segments, detected_language).
    """
    import mlx_whisper

    log.info("transcribe: transcribing with mlx-whisper large-v3")
    result = mlx_whisper.transcribe(
        vocals_wav,
        path_or_hf_repo="mlx-community/whisper-large-v3-mlx",
        word_timestamps=True,
        verbose=False,
    )
    language = result.get("language", "en")
    raw_segments = result.get("segments", [])
    log.info(f"transcribe: detected language={language}, {len(raw_segments)} raw segments")

    # Speaker diarization
    diarize_pipeline = _load_diarize()
    log.info("transcribe: diarizing")
    diarization = diarize_pipeline(vocals_wav)

    # Build speaker lookup: list of (start, end, speaker_label)
    speaker_turns = [
        (turn.start, turn.end, label)
        for turn, _, label in diarization.itertracks(yield_label=True)
    ]
    log.info(f"transcribe: {len({s for _, _, s in speaker_turns})} speakers detected")

    segments = _normalise(raw_segments, speaker_turns)
    log.info(f"transcribe: done — {len(segments)} segments")
    return segments, language


def _assign_speaker(start: float, end: float, speaker_turns: list[tuple]) -> str:
    """Return the speaker with the most overlap in [start, end]."""
    best_speaker = "SPEAKER_00"
    best_overlap = 0.0
    for t_start, t_end, label in speaker_turns:
        overlap = min(end, t_end) - max(start, t_start)
        if overlap > best_overlap:
            best_overlap = overlap
            best_speaker = label
    return best_speaker


def _normalise(raw: list[dict], speaker_turns: list[tuple]) -> list[dict]:
    out = []
    for i, seg in enumerate(raw):
        start = round(float(seg.get("start", 0)), 3)
        end   = round(float(seg.get("end", 0)), 3)

        # mlx-whisper word format: {"word", "start", "end", "probability"}
        # normalise to {"word", "start", "end", "score"} for consistency
        words = [
            {
                "word":  w.get("word", ""),
                "start": round(float(w.get("start", start)), 3),
                "end":   round(float(w.get("end", end)), 3),
                "score": round(float(w.get("probability", 0.5)), 4),
            }
            for w in seg.get("words", [])
        ]

        speaker = _assign_speaker(start, end, speaker_turns)

        out.append({
            "idx":     i,
            "start_sec": start,
            "end_sec":   end,
            "speaker": speaker,
            "text":    seg.get("text", "").strip(),
            "words":   words,
        })
    return out
