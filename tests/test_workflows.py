from src.orchestrator import workflows as wf
from src.orchestrator.runs import Run

def make_run(**kw):
    base = dict(id="r1", workflow_type="dub", episode_id=456,
                callback_url="cb", audio_url="https://a.mp3", language="es",
                segments_key="dub-runs/r1/segments.json", source_lang="en",
                speaker_keys={"SPEAKER_00": "dub-stems/456/speaker_SPEAKER_00.wav"})
    base.update(kw); return Run(**base)

def test_dub_step_order_and_next():
    assert [s.name for s in wf.steps_for("dub")] == ["prep", "text", "synth", "mux"]
    assert wf.first_step("dub").name == "prep"
    assert wf.next_step("dub", "prep").name == "text"
    assert wf.next_step("dub", "mux") is None

def test_transcribe_is_prep_only():
    assert [s.name for s in wf.steps_for("transcribe")] == ["prep"]
    assert wf.next_step("transcribe", "prep") is None

def test_step_tiers_and_progress_labels():
    by = {s.name: s for s in wf.steps_for("dub")}
    assert by["prep"].tier == "gpu" and by["synth"].tier == "gpu"
    assert by["text"].tier == "cpu" and by["mux"].tier == "cpu"
    assert by["prep"].progress == "separating"

def test_build_input_prep_for_dub():
    r = make_run()
    assert wf.build_input(wf.first_step("dub"), r) == {
        "run_id": "r1", "episode_id": 456, "audio_url": "https://a.mp3", "extract": True}

def test_build_input_prep_extract_false_for_transcribe():
    r = make_run(workflow_type="transcribe")
    assert wf.build_input(wf.first_step("transcribe"), r)["extract"] is False

def test_build_input_text_synth_mux():
    r = make_run()
    assert wf.build_input(wf.next_step("dub", "prep"), r) == {
        "run_id": "r1", "episode_id": 456,
        "segments_key": "dub-runs/r1/segments.json", "source_lang": "en", "language": "es"}
    assert wf.build_input(wf.next_step("dub", "text"), r) == {
        "run_id": "r1", "episode_id": 456,
        "segments_key": "dub-runs/r1/segments.json", "language": "es",
        "speaker_keys": {"SPEAKER_00": "dub-stems/456/speaker_SPEAKER_00.wav"}}
    assert wf.build_input(wf.next_step("dub", "synth"), r) == {
        "run_id": "r1", "episode_id": 456,
        "segments_key": "dub-runs/r1/segments.json", "language": "es", "bg_volume": 0.15}
