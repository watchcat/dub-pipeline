from unittest.mock import patch
from src.workers import gpu_synth

IN_SEGS = [
    {"idx": 0, "speaker": "SPEAKER_00", "translated_text": "hola", "start_sec": 0, "end_sec": 2},
    {"idx": 1, "speaker": "SPEAKER_00", "translated_text": "", "start_sec": 2, "end_sec": 3},
]
SYNTHED = [
    {**IN_SEGS[0], "synth_wav": "/w/synth_0000.wav", "synth_duration": 1.1},
    {**IN_SEGS[1], "synth_wav": None, "synth_duration": None},
]

def test_synth_uploads_wavs_and_records_keys():
    with patch("src.workers.gpu_synth.artifacts.read_segments", return_value=IN_SEGS), \
         patch("src.workers.gpu_synth.storage.download"), \
         patch("src.workers.gpu_synth.synthesize.synthesize", return_value=SYNTHED), \
         patch("src.workers.gpu_synth.storage.upload", return_value="url") as up, \
         patch("src.workers.gpu_synth.artifacts.write_segments",
               return_value="dub-runs/r1/segments.json") as wr, \
         patch("src.workers.gpu_synth.common.run_in_tempdir",
               side_effect=lambda body: body("/w")):
        out = gpu_synth.run({"run_id": "r1", "episode_id": 456, "language": "es",
                             "segments_key": "k", "speaker_keys": {"SPEAKER_00": "spk-key"}})
    assert out["segments_key"] == "dub-runs/r1/segments.json"
    saved = wr.call_args.args[1]
    assert saved[0]["synth_r2_key"] == "dub-stems/456/synth_es_0000.wav"
    assert saved[0]["synth_duration"] == 1.1
    assert "synth_r2_key" not in saved[1] or saved[1]["synth_r2_key"] is None
    assert "dub-stems/456/synth_es_0000.wav" in [c.args[1] for c in up.call_args_list]
