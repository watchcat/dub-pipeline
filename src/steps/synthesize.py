"""Step 5 — Synthesize translated segments with Coqui XTTS-v2 (local).

For each segment, clones the speaker's voice and generates dubbed audio.
Returns segments with "synth_wav" path added.
"""
import logging
import os
import re
import unicodedata

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
    import torch
    import scipy.io.wavfile

    tts = _load_tts()
    model = tts.synthesizer.tts_model
    xtts_lang = _xtts_lang_code(target_lang)

    # Pre-compute speaker embeddings once per speaker (expensive — avoid per-segment).
    speaker_latents: dict[str, tuple] = {}
    for speaker_id, wav_path in speaker_samples.items():
        if wav_path and os.path.exists(wav_path):
            log.info(f"synthesize: computing embeddings for {speaker_id}")
            gpt_cond_latent, speaker_embedding = model.get_conditioning_latents(
                audio_path=[wav_path]
            )
            speaker_latents[speaker_id] = (gpt_cond_latent, speaker_embedding)

    out = []
    for seg in segments:
        speaker = seg.get("speaker", "SPEAKER_00")
        latents = speaker_latents.get(speaker)
        translated = _clean_for_tts(seg.get("translated_text", ""))

        if not latents or not translated:
            log.warning(f"synthesize: skipping seg {seg['idx']} — no latents or text")
            out.append({**seg, "synth_wav": None})
            continue

        synth_path = os.path.join(out_dir, f"synth_{seg['idx']:04d}.wav")
        try:
            gpt_cond_latent, speaker_embedding = latents
            result = model.inference(
                text=translated,
                language=xtts_lang,
                gpt_cond_latent=gpt_cond_latent,
                speaker_embedding=speaker_embedding,
            )
            wav = result["wav"]
            if isinstance(wav, torch.Tensor):
                wav = wav.cpu().numpy()
            wav_int16 = (wav * 32767).clip(-32768, 32767).astype("int16")
            scipy.io.wavfile.write(synth_path, 24000, wav_int16)
            synth_dur = len(wav_int16) / 24000
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
    """Normalize text for XTTS-v2: replace Unicode punctuation that confuses the model."""
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


def _xtts_lang_code(lang: str) -> str:
    """Map ISO 639-1 to XTTS-v2 language codes (mostly the same, a few exceptions)."""
    mapping = {
        "zh": "zh-cn",
        "pt": "pt",
        "en": "en",
    }
    return mapping.get(lang.lower(), lang.lower())
