# Replace XTTSv2 with VoxCPM — Implementation Plan

## Background

XTTSv2 (Coqui TTS) outputs 24 kHz audio and requires expensive per-speaker embedding
pre-computation (`get_conditioning_latents`). VoxCPM2 is a 2B-parameter tokenizer-free
TTS model that produces 48 kHz audio, clones voices from a 3–10 s reference wav, and
auto-detects language — no manual language code mapping needed.

Key differences that drive the implementation changes:

| | XTTSv2 | VoxCPM2 |
|---|---|---|
| **Install** | `TTS>=0.22.0` | `voxcpm` (PyPI) |
| **Model weights** | ~1.9 GB, HF | ~4–5 GB, `openbmb/VoxCPM2` |
| **VRAM** | ~4 GB | ≥8 GB (bfloat16) |
| **Sample rate out** | 24 kHz | 48 kHz |
| **Voice cloning** | pre-compute embeddings once; pass to each call | pass `reference_wav_path` directly to each call |
| **Language arg** | required, mapped (`zh-cn`, etc.) | not required (auto-detected) |
| **Output type** | `result["wav"]` numpy float32 | `wav` numpy float32 |
| **Write to file** | `scipy.io.wavfile.write(path, 24000, int16)` | `soundfile.write(path, wav, 48000)` |

---

## Files to Change

| File | Change |
|---|---|
| `requirements.txt` | Remove `TTS>=0.22.0`; add `voxcpm`, `soundfile` |
| `src/steps/synthesize.py` | Full rewrite — VoxCPM model loading, voice cloning API |
| `src/worker.py` | Update imports; remove embedding pre-computation; update per-segment synthesis call |
| `src/steps/assemble.py` | Change `SAMPLE_RATE = 24000` → `48000` |
| `Dockerfile` | Update model pre-download step |

---

## 1. `requirements.txt`

```diff
-TTS>=0.22.0
-transformers>=4.33.0,<4.41.0
+voxcpm
+soundfile
+transformers>=4.41.0
```

Remove the `transformers` upper-bound pin — it was required for XTTSv2 compatibility
only. `voxcpm` requires PyTorch ≥ 2.5.0 (already satisfied by the `torch>=2.3.0` range
— raise lower bound to `>=2.5.0`).

---

## 2. `src/steps/synthesize.py`

Full replacement. Key changes vs current file:

### Model loading

```python
_voxcpm_model = None

def _load_model():
    global _voxcpm_model
    if _voxcpm_model is None:
        from voxcpm import VoxCPM
        log.info("synthesize: loading VoxCPM2")
        _voxcpm_model = VoxCPM.from_pretrained("openbmb/VoxCPM2", load_denoiser=False)
    return _voxcpm_model
```

`load_denoiser=False` skips the optional denoiser model — reduces VRAM and load time;
quality is sufficient without it.

### Per-segment synthesis

```python
model = _load_model()
wav = model.generate(
    text=text,
    reference_wav_path=speaker_wav_path,   # 3–10 s speaker sample
    cfg_value=2.0,
    inference_timesteps=10,
)
import soundfile as sf
sf.write(synth_path, wav, model.tts_model.sample_rate)  # 48000 Hz, float32
synth_dur = len(wav) / model.tts_model.sample_rate
```

### No embedding pre-computation

XTTSv2 required:
```python
gpt_cond_latent, speaker_embedding = model.get_conditioning_latents(audio_path=[wav])
```
…once per speaker before the synthesis loop. VoxCPM does not expose this — voice
characteristics are extracted internally on each `generate()` call. Remove the
`speaker_latents` dict and pre-computation loop from worker.py entirely.

### Language code mapping

Remove `_xtts_lang_code()`. VoxCPM detects the language from the text automatically —
do not pass a `language=` argument.

### Helpers to keep / rename

- `_wav_duration(path)` — unchanged, reads WAV header
- `_clean_for_tts(text)` — keep as-is; Unicode normalization is still useful
- Export names change: `_load_tts` → `_load_model`

---

## 3. `src/worker.py`

### Import line

```diff
-from src.steps.synthesize import _load_tts, _xtts_lang_code, _wav_duration as _synth_wav_duration, _clean_for_tts
+from src.steps.synthesize import _load_model, _wav_duration as _synth_wav_duration, _clean_for_tts
```

### `_synthesize_with_progress` function

**Remove** the speaker embedding pre-computation block:
```python
# DELETE this entire block:
speaker_latents: dict[str, tuple] = {}
for speaker_id, wav_path in speaker_samples.items():
    ...
    gpt_cond_latent, speaker_embedding = model.get_conditioning_latents(...)
    speaker_latents[speaker_id] = (gpt_cond_latent, speaker_embedding)
```

**Replace** model loading:
```diff
-tts = _load_tts()
-model = tts.synthesizer.tts_model
-xtts_lang = _xtts_lang_code(language)
+model = _load_model()
```

**Replace** per-segment synthesis:
```diff
-gpt_cond_latent, speaker_embedding = latents
-result = model.inference(
-    text=text,
-    language=xtts_lang,
-    gpt_cond_latent=gpt_cond_latent,
-    speaker_embedding=speaker_embedding,
-)
-wav = result["wav"]
-if isinstance(wav, torch.Tensor):
-    wav = wav.cpu().numpy()
-wav_int16 = (wav * 32767).clip(-32768, 32767).astype("int16")
-scipy.io.wavfile.write(ckpt_path, 24000, wav_int16)
-synth_dur = len(wav_int16) / 24000
+import soundfile as sf
+wav = model.generate(
+    text=text,
+    reference_wav_path=speaker_samples[speaker],
+    cfg_value=2.0,
+    inference_timesteps=10,
+)
+sf.write(ckpt_path, wav, model.tts_model.sample_rate)
+synth_dur = len(wav) / model.tts_model.sample_rate
```

**Replace** the guard that previously checked `latents`:
```diff
-elif not latents or not text:
+elif not speaker_samples.get(speaker) or not text:
```

Remove `torch` and `scipy.io.wavfile` imports from worker.py (no longer needed there).

---

## 4. `src/steps/assemble.py`

```diff
-SAMPLE_RATE = 24000  # Hz — spec requirement
+SAMPLE_RATE = 48000  # Hz — VoxCPM2 native output
```

The rest of the file is format-agnostic (ffmpeg concat handles any sample rate).

---

## 5. `Dockerfile`

Current model pre-download step runs a Python snippet to cache XTTSv2 weights. Replace
with VoxCPM:

```dockerfile
# Pre-download VoxCPM2 weights into the image layer
RUN python -c "from voxcpm import VoxCPM; VoxCPM.from_pretrained('openbmb/VoxCPM2', load_denoiser=False)"
```

Remove any XTTSv2-specific `RUN python -c "from TTS.api import TTS; TTS('tts_models/...')"` lines.

---

## VRAM Budget Check

RunPod GPU tier used is likely an A40 (48 GB) or similar. VoxCPM2 requires ≥8 GB VRAM
(bfloat16). This is well within budget. No change needed to the RunPod endpoint config.

---

## What Does NOT Change

- `src/steps/transcribe.py` — WhisperX, unchanged
- `src/steps/translate.py` — Google Gemini, unchanged
- `src/steps/separate.py` — Demucs, unchanged
- `src/steps/extract_samples.py` — speaker diarisation, unchanged
- `src/steps/mix.py` — ffmpeg, unchanged; handles any input sample rate
- `src/steps/assemble.py` — only `SAMPLE_RATE` constant changes
- `buzz-bot` backend — no changes; `dub_segment.cr` uses WAV header duration, which is sample-rate-agnostic

---

## Verification Steps

1. Run `test_job.py` locally (CPU or GPU) — confirm synthesis produces a WAV file at 48 kHz.
2. Check `synth_dur` is plausible (similar to original segment duration ± 30%).
3. Confirm the assembled `dubbed_vocals.wav` plays correctly and has the right voice.
4. Check subtitle sync on a short test episode: cue timestamps should align with dubbed speech.
5. Build updated Docker image, push to RunPod, run a real job end-to-end.
