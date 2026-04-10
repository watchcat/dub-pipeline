"""Step 5 — Synthesize translated segments with Coqui XTTS-v2 (local).

For each segment, clones the speaker's voice and generates dubbed audio.
Returns segments with "synth_wav" path added.
"""
import logging
import os

# Must be set before PyTorch attempts any MPS op — XTTS-v2 uses conv layers
# with >65536 output channels which are not natively supported on MPS.
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

from src import config

log = logging.getLogger(__name__)

_tts_model = None


def _load_tts():
    global _tts_model
    if _tts_model is None:
        from TTS.api import TTS
        log.info("synthesize: loading XTTS-v2")
        _tts_model = TTS("tts_models/multilingual/multi-dataset/xtts_v2")
        _tts_model.to(config.TTS_DEVICE)
    return _tts_model


def synthesize(
    segments: list[dict],
    speaker_samples: dict[str, str],
    target_lang: str,
    out_dir: str,
) -> list[dict]:
    """
    For each segment, synthesize translated_text using the speaker's voice sample.
    Returns segments with "synth_wav" path added (or None if synthesis failed).
    speaker_samples: {speaker_id: local_wav_path}
    """
    tts = _load_tts()
    xtts_lang = _xtts_lang_code(target_lang)

    out = []
    for seg in segments:
        speaker = seg.get("speaker", "SPEAKER_00")
        sample_wav = speaker_samples.get(speaker)
        translated = seg.get("translated_text", "").strip()

        if not sample_wav or not translated:
            log.warning(f"synthesize: skipping seg {seg['idx']} — no sample or text")
            out.append({**seg, "synth_wav": None})
            continue

        synth_path = os.path.join(out_dir, f"synth_{seg['idx']:04d}.wav")
        try:
            tts.tts_to_file(
                text=translated,
                speaker_wav=sample_wav,
                language=xtts_lang,
                file_path=synth_path,
            )
            synth_dur = _wav_duration(synth_path)
            log.info(f"synthesize: seg {seg['idx']} ({speaker}) → {synth_dur:.2f}s")
            out.append({**seg, "synth_wav": synth_path, "synth_duration": synth_dur})
        except Exception as e:
            log.error(f"synthesize: seg {seg['idx']} failed: {e}")
            out.append({**seg, "synth_wav": None, "synth_duration": None})

    return out


def _wav_duration(path: str) -> float:
    import wave
    with wave.open(path, "rb") as wf:
        return wf.getnframes() / wf.getframerate()


def _xtts_lang_code(lang: str) -> str:
    """Map ISO 639-1 to XTTS-v2 language codes (mostly the same, a few exceptions)."""
    mapping = {
        "zh": "zh-cn",
        "pt": "pt",
        "en": "en",
    }
    return mapping.get(lang.lower(), lang.lower())
