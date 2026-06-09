import json
from unittest.mock import patch
from src import artifacts


def test_key_conventions():
    assert artifacts.stem_key(456, "vocals.wav") == "dub-stems/456/vocals.wav"
    assert artifacts.dub_key(456, "es") == "dubbed/456/es.mp3"
    assert artifacts.segments_key("abc") == "dub-runs/abc/segments.json"


def test_write_then_read_segments_roundtrip():
    segs = [{"idx": 0, "text": "hi", "words": [{"w": "hi"}]}]
    captured = {}

    def fake_upload_bytes(data, key, ctype):
        captured["data"] = data
        captured["key"] = key
        return "url"

    with patch("src.artifacts.storage.upload_bytes", side_effect=fake_upload_bytes):
        key = artifacts.write_segments("abc", segs)

    assert key == "dub-runs/abc/segments.json"

    with patch("src.artifacts.storage.download_bytes", return_value=captured["data"]):
        assert artifacts.read_segments(key) == segs
