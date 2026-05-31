from unittest.mock import patch
from src.workers import gpu_prep

SEGS = [{"idx": 0, "start_sec": 0.0, "end_sec": 2.0, "speaker": "SPEAKER_00",
         "text": "hi", "words": []}]

def base_mocks(extract=True):
    return {
        "separate": patch("src.workers.gpu_prep.separate.separate",
                          return_value=("/w/vocals.wav", "/w/background.wav")),
        "transcribe": patch("src.workers.gpu_prep.transcribe.transcribe",
                            return_value=(SEGS, "en")),
        "extract": patch("src.workers.gpu_prep.extract_samples.extract_samples",
                         return_value={"SPEAKER_00": "/w/speaker_SPEAKER_00.wav"}),
        "upload": patch("src.workers.gpu_prep.storage.upload", return_value="url"),
        "write": patch("src.workers.gpu_prep.artifacts.write_segments",
                      return_value="dub-runs/r1/segments.json"),
        "tmp": patch("src.workers.gpu_prep.common.run_in_tempdir",
                    side_effect=lambda body: body("/w")),
    }

def test_prep_dub_uploads_stems_speakers_and_returns_keys():
    m = base_mocks()
    with m["separate"], m["transcribe"], m["extract"], m["upload"] as up, \
         m["write"], m["tmp"]:
        out = gpu_prep.run({"run_id": "r1", "episode_id": 456,
                            "audio_url": "https://a.mp3", "extract": True})
    assert out["source_lang"] == "en"
    assert out["segments_key"] == "dub-runs/r1/segments.json"
    assert out["speaker_keys"] == {"SPEAKER_00": "dub-stems/456/speaker_SPEAKER_00.wav"}
    uploaded_keys = [c.args[1] for c in up.call_args_list]
    assert "dub-stems/456/vocals.wav" in uploaded_keys
    assert "dub-stems/456/background.wav" in uploaded_keys
    assert "dub-stems/456/speaker_SPEAKER_00.wav" in uploaded_keys

def test_prep_transcribe_skips_extract_and_returns_empty_speakers():
    m = base_mocks()
    with m["separate"], m["transcribe"], \
         patch("src.workers.gpu_prep.extract_samples.extract_samples") as ex, \
         m["upload"], m["write"], m["tmp"]:
        out = gpu_prep.run({"run_id": "r1", "episode_id": 456,
                            "audio_url": "https://a.mp3", "extract": False})
    ex.assert_not_called()
    assert out["speaker_keys"] == {}
