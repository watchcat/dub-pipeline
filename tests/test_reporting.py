from unittest.mock import patch
from src.orchestrator.reporting import Reporter
from src.orchestrator.runs import Run

def make_run(**kw):
    base = dict(id="r1", workflow_type="dub", episode_id=456,
                callback_url="https://app/internal/dub_result", dub_id=123,
                language="es", source_lang="en", segments_key="k4",
                speaker_keys={"S0": "k", "S1": "k"}, r2_url="https://r2/es.mp3",
                duration_sec=100.0, segment_count=3)
    base.update(kw); return Run(**base)

SEGS = [{"idx": 0, "start_sec": 1.0, "end_sec": 2.0, "speaker": "S0",
         "text": "hi", "words": [{"w": "hi"}], "translated_text": "hola",
         "synth_r2_key": "dub-stems/456/synth_es_0000.wav",
         "synth_duration": 1.1, "synth_start_sec": 1.0}]

def test_progress_posts_to_progress_url():
    with patch("src.orchestrator.reporting.requests.post") as post, \
         patch("src.orchestrator.reporting.config.PROGRESS_URL", "https://app/internal/dub_progress"):
        Reporter().progress(make_run(), "synthesizing", 40)
    args, kwargs = post.call_args
    assert args[0] == "https://app/internal/dub_progress"
    assert kwargs["json"] == {"dub_id": 123, "step": "synthesizing", "pct": 40}

def test_progress_skipped_for_transcribe_run_without_dub_id():
    # A transcribe run has no dub_id; buzz-bot's dub_progress is dub-keyed and
    # rejects a null dub_id, so the reporter must not post at all.
    with patch("src.orchestrator.reporting.requests.post") as post:
        Reporter().progress(make_run(workflow_type="transcribe", dub_id=None), "separating")
    post.assert_not_called()

def test_dub_result_builds_buzzbot_payload():
    with patch("src.orchestrator.reporting.artifacts.read_segments", return_value=SEGS), \
         patch("src.orchestrator.reporting.requests.post") as post:
        Reporter().dub_result(make_run())
    body = post.call_args.kwargs["json"]
    assert body["dub_id"] == 123 and body["success"] is True
    assert body["r2_url"] == "https://r2/es.mp3"
    assert body["source_lang"] == "en" and body["language"] == "es"
    assert body["segment_count"] == 3 and body["speaker_count"] == 2
    seg = body["segments"][0]
    assert seg["idx"] == 0 and seg["translated_text"] == "hola"
    assert seg["synth_r2_key"] == "dub-stems/456/synth_es_0000.wav"
    assert seg["synth_start_sec"] == 1.0
    assert post.call_args.args[0] == "https://app/internal/dub_result"

def test_failed_posts_failure_shape():
    with patch("src.orchestrator.reporting.requests.post") as post:
        Reporter().failed(make_run(), "separating", "Demucs OOM")
    body = post.call_args.kwargs["json"]
    assert body == {"dub_id": 123, "success": False,
                    "step": "separating", "error": "Demucs OOM"}

def test_transcript_result_posts_to_transcript_url():
    with patch("src.orchestrator.reporting.artifacts.read_segments", return_value=SEGS), \
         patch("src.orchestrator.reporting.requests.post") as post, \
         patch("src.orchestrator.reporting.config.BUZZBOT_TRANSCRIPT_URL",
               "https://app/internal/transcript_result"):
        Reporter().transcript_result(make_run(workflow_type="transcribe"))
    assert post.call_args.args[0] == "https://app/internal/transcript_result"
    body = post.call_args.kwargs["json"]
    assert body["episode_id"] == 456 and body["source_lang"] == "en"
    assert body["segments"][0]["text"] == "hi"
