"""Step 2 — Transcribe with faster-whisper (CUDA/CPU/MPS) + pyannote diarization.

faster-whisper uses ctranslate2 for efficient inference on CUDA and CPU.
Pyannote speaker diarization runs on the same device as configured.

Returns a list of segments:
  [{"idx": 0, "start": 1.2, "end": 4.5, "speaker": "SPEAKER_00",
    "text": "Hello world.", "words": [...]}]
"""
import logging
import torch
from src import config

log = logging.getLogger(__name__)

_diarize_pipeline = None
_whisper_model = None


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


def _load_whisper():
    global _whisper_model
    if _whisper_model is None:
        from faster_whisper import WhisperModel
        log.info(f"transcribe: loading faster-whisper {config.WHISPER_MODEL} "
                 f"device={config.WHISPER_DEVICE} compute={config.WHISPER_COMPUTE}")
        _whisper_model = WhisperModel(
            config.WHISPER_MODEL,
            device=config.WHISPER_DEVICE,
            compute_type=config.WHISPER_COMPUTE,
            download_root=config.MODEL_DIR,
        )
    return _whisper_model


def _load_diarize():
    global _diarize_pipeline
    if _diarize_pipeline is None:
        from pyannote.audio import Pipeline
        # PyTorch 2.6+ changed torch.load default to weights_only=True.
        # Pyannote checkpoints use custom classes not in the default allowlist.
        # Patch pyannote.audio.core.model.pl_load (the exact call site) to pass
        # weights_only=False. Pyannote checkpoints come from HF and are trusted.
        # PyTorch 2.6+ weights_only=True breaks pyannote checkpoints.
        # Both pyannote.audio.core.model AND pytorch_lightning.core.saving
        # import pl_load as a local reference and may pass weights_only=True.
        # Patch both to force weights_only=False.
        import pyannote.audio.core.model as _pam
        import pytorch_lightning.core.saving as _pl_saving

        def _pl_load_patched(path, map_location=None, **_kwargs):
            return torch.load(path, map_location=map_location, weights_only=False)

        _orig_pam = _pam.pl_load
        _orig_pls = _pl_saving.pl_load
        _pam.pl_load = _pl_load_patched
        _pl_saving.pl_load = _pl_load_patched
        log.info("transcribe: loading pyannote diarization pipeline")
        try:
            _diarize_pipeline = Pipeline.from_pretrained(
                "pyannote/speaker-diarization-3.1",
            )
        finally:
            _pam.pl_load = _orig_pam
            _pl_saving.pl_load = _orig_pls
        _diarize_pipeline = _diarize_pipeline.to(torch.device(config.WHISPER_DEVICE))
    return _diarize_pipeline


def transcribe(vocals_wav: str) -> tuple[list[dict], str]:
    """
    Transcribe vocals_wav with faster-whisper + pyannote diarization.
    Returns (segments, detected_language).
    """
    model = _load_whisper()

    log.info("transcribe: transcribing with faster-whisper")
    raw_segments_iter, info = model.transcribe(
        vocals_wav,
        word_timestamps=True,
        vad_filter=True,
        beam_size=10,
    )
    language = info.language
    raw_segments = list(raw_segments_iter)
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


def _normalise(raw: list, speaker_turns: list[tuple]) -> list[dict]:
    out = []
    for i, seg in enumerate(raw):
        start = round(float(seg.start), 3)
        end   = round(float(seg.end), 3)

        # faster-whisper word format: Word(start, end, word, probability)
        words = [
            {
                "word":  w.word,
                "start": round(float(w.start), 3),
                "end":   round(float(w.end), 3),
                "score": round(float(w.probability), 4),
            }
            for w in (seg.words or [])
        ]

        speaker = _assign_speaker(start, end, speaker_turns)

        out.append({
            "idx":       i,
            "start_sec": start,
            "end_sec":   end,
            "speaker":   speaker,
            "text":      seg.text.strip(),
            "words":     words,
        })
    return out
