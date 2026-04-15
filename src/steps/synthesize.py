"""Step 5 — Synthesize translated segments with VoxCPM2.

For each segment, clones the speaker's voice and generates dubbed audio.
Returns segments with "synth_wav" path added.
VoxCPM2 auto-detects language from text — no explicit language code needed.
"""
import logging
import os
import re
import unicodedata

log = logging.getLogger(__name__)

_voxcpm_model = None


def _load_model():
    global _voxcpm_model
    if _voxcpm_model is None:
        from voxcpm import VoxCPM
        log.info("synthesize: loading VoxCPM2")
        _voxcpm_model = VoxCPM.from_pretrained("openbmb/VoxCPM2", load_denoiser=False)
    return _voxcpm_model


def synthesize(
    segments: list[dict],
    speaker_samples: dict[str, str],
    target_lang: str,
    out_dir: str,
) -> list[dict]:
    """
    For each segment, synthesize translated_text using the speaker's voice sample.
    Returns segments with "synth_wav" and "synth_duration" added (or None on failure).
    speaker_samples: {speaker_id: local_wav_path}
    VoxCPM2 handles voice cloning and language detection internally.
    """
    import soundfile

    model = _load_model()

    out = []
    for seg in segments:
        speaker = seg.get("speaker", "SPEAKER_00")
        speaker_wav_path = speaker_samples.get(speaker)
        translated = _clean_for_tts(seg.get("translated_text", ""))

        if not speaker_wav_path or not translated:
            log.warning(f"synthesize: skipping seg {seg['idx']} — no speaker sample or text")
            out.append({**seg, "synth_wav": None, "synth_duration": None})
            continue

        synth_path = os.path.join(out_dir, f"synth_{seg['idx']:04d}.wav")
        try:
            wav = model.generate(
                text=translated,
                reference_wav_path=speaker_wav_path,
                cfg_value=2.0,
                inference_timesteps=10,
            )
            soundfile.write(synth_path, wav, model.tts_model.sample_rate)
            synth_dur = len(wav) / model.tts_model.sample_rate
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


def _clean_for_tts(text: str) -> str:
    """Normalize Unicode punctuation that may confuse TTS models."""
    # Typographic quotes → straight quotes
    text = text.replace('\u201c', '"').replace('\u201d', '"')
    text = text.replace('\u2018', "'").replace('\u2019', "'")
    # Em/en dashes → comma+space
    text = text.replace('\u2014', ', ').replace('\u2013', ', ')
    # Ellipsis character → three dots
    text = text.replace('\u2026', '...')
    # Remove control characters and other non-printable Unicode
    text = ''.join(c for c in text if unicodedata.category(c)[0] != 'C')
    # Collapse multiple spaces
    text = re.sub(r' {2,}', ' ', text)
    return text.strip()
