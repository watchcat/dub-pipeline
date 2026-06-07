# tests/test_synthesize_progress.py
from unittest.mock import patch, MagicMock
from src.steps import synthesize

def test_on_progress_called_per_segment(monkeypatch):
    model = MagicMock()
    model.tts_model.sample_rate = 16000
    model.generate.return_value = [0.0] * 1600
    segs = [{"idx": 0, "speaker": "S", "translated_text": "hola"},
            {"idx": 1, "speaker": "S", "translated_text": "adios"}]
    calls = []
    with patch("src.steps.synthesize._load_model", return_value=model), \
         patch("src.steps.synthesize.soundfile.write"):
        synthesize.synthesize(segs, {"S": "/w/s.wav"}, "es", "/w",
                              on_progress=lambda d, t: calls.append((d, t)))
    assert calls[0] == (0, 2)
    assert calls[-1] == (2, 2)
