from unittest.mock import patch
from src.workers import cpu_mux

SEGS = [
    {"idx": 0, "speaker": "S0", "synth_r2_key": "dub-stems/456/synth_es_0000.wav",
     "start_sec": 0, "end_sec": 2, "translated_text": "hola"},
    {"idx": 1, "speaker": "S0", "synth_r2_key": None,
     "start_sec": 2, "end_sec": 3, "translated_text": ""},
]
ASSEMBLED = [
    {**SEGS[0], "synth_start_sec": 0.0},
    {**SEGS[1], "synth_start_sec": 2.0},
]

def test_mux_downloads_synth_assembles_mixes_uploads():
    with patch("src.workers.cpu_mux.artifacts.read_segments", return_value=SEGS), \
         patch("src.workers.cpu_mux.storage.download") as dl, \
         patch("src.workers.cpu_mux.assemble.assemble",
               return_value=("/w/dubbed_vocals.wav", ASSEMBLED)), \
         patch("src.workers.cpu_mux.mix.mix", return_value="/w/final.mp3"), \
         patch("src.workers.cpu_mux.storage.upload", return_value="https://r2/es.mp3") as up, \
         patch("src.workers.cpu_mux._ffprobe_duration", return_value=100.0), \
         patch("src.workers.cpu_mux.artifacts.write_segments",
               return_value="dub-runs/r1/segments.json") as wr, \
         patch("src.workers.cpu_mux.common.run_in_tempdir",
               side_effect=lambda body: body("/w")):
        out = cpu_mux.run({"run_id": "r1", "episode_id": 456,
                           "segments_key": "k", "language": "es", "bg_volume": 0.15})
    dl_keys = [c.args[0] for c in dl.call_args_list]
    assert "dub-stems/456/background.wav" in dl_keys
    assert "dub-stems/456/synth_es_0000.wav" in dl_keys
    assert out["r2_url"] == "https://r2/es.mp3"
    assert out["duration_sec"] == 100.0
    assert out["segment_count"] == 1
    assert up.call_args.args[1] == "dubbed/456/es.mp3"
    assert out["segments_key"] == "dub-runs/r1/segments.json"
