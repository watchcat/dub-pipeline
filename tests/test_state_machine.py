from src.orchestrator.runs import Run, InMemoryRunStore
from src.orchestrator.state_machine import StateMachine
from tests.fakes import FakeDispatcher, FakeReporter

def build_sm():
    store = InMemoryRunStore()
    gpu = FakeDispatcher("gpu"); cpu = FakeDispatcher("cpu")
    rep = FakeReporter()
    sm = StateMachine(store, {"gpu": gpu, "cpu": cpu}, rep, max_attempts=3)
    return sm, store, gpu, cpu, rep

def make_run(**kw):
    base = dict(id="r1", workflow_type="dub", episode_id=456, callback_url="cb",
                audio_url="https://a.mp3", language="es")
    base.update(kw); return Run(**base)

def test_start_dispatches_prep_on_gpu_and_reports_progress():
    sm, store, gpu, cpu, rep = build_sm()
    sm.start(make_run())
    assert store.get("r1").current_step == "prep"
    assert gpu.calls[0][0] == "prep"
    assert gpu.calls[0][2] == {"run_id": "r1", "episode_id": 456,
                               "audio_url": "https://a.mp3", "extract": True}
    assert "token=" in gpu.calls[0][3]
    assert rep.progress_calls[0] == ("r1", "separating", None)

def test_prep_callback_advances_to_text_on_cpu():
    sm, store, gpu, cpu, rep = build_sm()
    sm.start(make_run())
    sm.handle_callback("r1", "prep", True, {
        "source_lang": "en", "speaker_keys": {"SPEAKER_00": "k"},
        "segments_key": "dub-runs/r1/segments.json"})
    run = store.get("r1")
    assert run.current_step == "text"
    assert run.source_lang == "en"
    assert run.speaker_keys == {"SPEAKER_00": "k"}
    assert run.segments_key == "dub-runs/r1/segments.json"
    assert cpu.calls[0][0] == "text"

def test_full_dub_chain_finalizes_with_dub_result():
    sm, store, gpu, cpu, rep = build_sm()
    sm.start(make_run())
    sm.handle_callback("r1", "prep", True, {"source_lang": "en",
        "speaker_keys": {"S": "k"}, "segments_key": "k1"})
    sm.handle_callback("r1", "text", True, {"segments_key": "k2"})
    sm.handle_callback("r1", "synth", True, {"segments_key": "k3"})
    sm.handle_callback("r1", "mux", True, {"segments_key": "k4",
        "r2_url": "https://r2/es.mp3", "duration_sec": 100.0, "segment_count": 3})
    run = store.get("r1")
    assert run.status == "done"
    assert run.r2_url == "https://r2/es.mp3"
    assert rep.dub_results == ["r1"]
    assert [c[1] for c in rep.progress_calls][-1] == "complete"

def test_transcribe_finalizes_after_prep_with_transcript_result():
    sm, store, gpu, cpu, rep = build_sm()
    sm.start(make_run(workflow_type="transcribe"))
    assert gpu.calls[0][2]["extract"] is False
    sm.handle_callback("r1", "prep", True, {"source_lang": "en",
        "speaker_keys": {}, "segments_key": "k1"})
    assert store.get("r1").status == "done"
    assert rep.transcript_results == ["r1"]

def test_failure_retries_then_gives_up():
    sm, store, gpu, cpu, rep = build_sm()
    sm.start(make_run())
    sm.handle_callback("r1", "prep", False, {"error": "OOM"})   # attempt 1 -> retry
    assert store.get("r1").attempts == 1
    assert len([c for c in gpu.calls if c[0] == "prep"]) == 2
    sm.handle_callback("r1", "prep", False, {"error": "OOM"})   # attempt 2 -> retry
    assert len([c for c in gpu.calls if c[0] == "prep"]) == 3
    sm.handle_callback("r1", "prep", False, {"error": "OOM"})   # attempt 3 -> fail
    assert store.get("r1").status == "failed"
    assert rep.failures == [("r1", "prep", "OOM")]

def test_successful_step_resets_attempts():
    sm, store, gpu, cpu, rep = build_sm()
    sm.start(make_run())
    sm.handle_callback("r1", "prep", False, {"error": "x"})     # attempts=1
    sm.handle_callback("r1", "prep", True, {"segments_key": "k1"})
    assert store.get("r1").attempts == 0
