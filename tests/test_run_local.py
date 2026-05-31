from unittest.mock import patch
from src.orchestrator import run_local

# End-to-end through StateMachine + LocalDispatcher + real workers,
# with the heavy step functions and storage mocked.
def test_local_dub_runs_all_four_steps_and_reports_result():
    seg = [{"idx": 0, "start_sec": 0, "end_sec": 2, "speaker": "S0",
            "text": "hi", "words": [], "translated_text": "hola",
            "synth_r2_key": "dub-stems/456/synth_es_0000.wav",
            "synth_duration": 1.1, "synth_start_sec": 0.0}]
    posted = {}
    with patch("src.workers.gpu_prep.separate.separate", return_value=("/w/v.wav", "/w/b.wav")), \
         patch("src.workers.gpu_prep.transcribe.transcribe", return_value=(seg, "en")), \
         patch("src.workers.gpu_prep.extract_samples.extract_samples",
               return_value={"S0": "/w/s.wav"}), \
         patch("src.workers.gpu_synth.synthesize.synthesize",
               return_value=[{**seg[0], "synth_wav": "/w/synth_0000.wav", "synth_duration": 1.1}]), \
         patch("src.workers.cpu_text.split_segments.split_long_segments", side_effect=lambda s: s), \
         patch("src.workers.cpu_text.translate.translate", side_effect=lambda s, a, b: s), \
         patch("src.workers.cpu_mux.assemble.assemble", return_value=("/w/dv.wav", seg)), \
         patch("src.workers.cpu_mux.mix.mix", return_value="/w/final.mp3"), \
         patch("src.workers.cpu_mux._ffprobe_duration", return_value=100.0), \
         patch("src.storage.upload", return_value="https://r2/es.mp3"), \
         patch("src.storage.download"), \
         patch("src.artifacts.write_segments", side_effect=lambda rid, s: "segkey"), \
         patch("src.artifacts.read_segments", return_value=seg), \
         patch("src.orchestrator.reporting.requests.post",
               side_effect=lambda url, **kw: posted.setdefault("last", (url, kw["json"]))), \
         patch("src.config.BUZZBOT_RESULT_URL", "https://app/internal/dub_result"), \
         patch("src.config.PROGRESS_URL", "https://app/internal/dub_progress"):
        store = run_local.run_dub_locally(
            run_id="r1", dub_id=123, episode_id=456,
            audio_url="https://a.mp3", language="es")
    assert store.get("r1").status == "done"
    assert posted["last"][0] == "https://app/internal/dub_result"
    assert posted["last"][1]["success"] is True
