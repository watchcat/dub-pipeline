from unittest.mock import patch, MagicMock
from src.orchestrator.dispatch import HttpDispatcher, LocalDispatcher
from src.orchestrator.workflows import TEXT, PREP
from src.orchestrator.runs import Run

def make_run():
    return Run(id="r1", workflow_type="dub", episode_id=456, callback_url="cb",
               audio_url="https://a.mp3", language="es")

def test_http_dispatcher_posts_input_and_callback_to_worker_url():
    d = HttpDispatcher({"text": "http://cpu-text/run"})
    with patch("src.orchestrator.dispatch.requests.post") as post:
        d.dispatch(TEXT, make_run(), {"run_id": "r1"}, "http://orch/callback?x=1")
    assert post.call_args.args[0] == "http://cpu-text/run"
    assert post.call_args.kwargs["json"] == {
        "input": {"run_id": "r1"}, "callback_url": "http://orch/callback?x=1"}

def test_local_dispatcher_runs_worker_and_posts_callback():
    worker = MagicMock(return_value={"segments_key": "k"})
    d = LocalDispatcher({"prep": worker})
    with patch("src.orchestrator.dispatch.requests.post") as post:
        d.dispatch(PREP, make_run(), {"run_id": "r1"}, "http://orch/callback")
    worker.assert_called_once_with({"run_id": "r1"})
    assert post.call_args.args[0] == "http://orch/callback"
    assert post.call_args.kwargs["json"] == {"ok": True, "segments_key": "k"}

def test_local_dispatcher_reports_failure():
    worker = MagicMock(side_effect=RuntimeError("boom"))
    d = LocalDispatcher({"prep": worker})
    with patch("src.orchestrator.dispatch.requests.post") as post:
        d.dispatch(PREP, make_run(), {"run_id": "r1"}, "http://orch/callback")
    assert post.call_args.kwargs["json"] == {"ok": False, "error": "boom"}
