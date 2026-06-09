from unittest.mock import patch
from src.orchestrator.reporting import Reporter
from src.orchestrator.runs import Run

# These keys are buzz-bot's contract (README "Result" schema). Changing them
# is a breaking change to buzz-bot and must be intentional.
RESULT_KEYS = {"dub_id", "episode_id", "language", "source_lang", "success",
               "r2_url", "duration_sec", "segment_count", "speaker_count", "segments"}
SEGMENT_KEYS = {"idx", "start_sec", "end_sec", "speaker_id", "text", "words",
                "translated_text", "synth_r2_key", "synth_duration", "synth_start_sec"}

def test_dub_result_payload_matches_buzzbot_contract():
    run = Run(id="r1", workflow_type="dub", episode_id=456, callback_url="cb",
              dub_id=123, language="es", source_lang="en", segments_key="k",
              speaker_keys={"S0": "k"}, r2_url="https://r2/es.mp3",
              duration_sec=100.0, segment_count=1)
    segs = [{"idx": 0, "start_sec": 0.0, "end_sec": 2.0, "speaker": "S0",
             "text": "hi", "words": [], "translated_text": "hola",
             "synth_r2_key": "dub-stems/456/synth_es_0000.wav",
             "synth_duration": 1.1, "synth_start_sec": 0.0}]
    captured = {}
    with patch("src.orchestrator.reporting.artifacts.read_segments", return_value=segs), \
         patch("src.orchestrator.reporting.requests.post",
               side_effect=lambda url, **kw: captured.update(body=kw["json"])):
        Reporter().dub_result(run)
    assert set(captured["body"].keys()) == RESULT_KEYS
    assert set(captured["body"]["segments"][0].keys()) == SEGMENT_KEYS
