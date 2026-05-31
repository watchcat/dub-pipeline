"""Declarative pipeline workflows composed from step descriptors."""
from dataclasses import dataclass
from src.orchestrator.runs import Run


@dataclass(frozen=True)
class Step:
    name: str         # prep | text | synth | mux
    tier: str         # gpu | cpu
    progress: str     # buzz-bot progress label posted when this step starts


PREP  = Step("prep",  "gpu", "separating")
TEXT  = Step("text",  "cpu", "translating")
SYNTH = Step("synth", "gpu", "synthesizing")
MUX   = Step("mux",   "cpu", "assembling")

WORKFLOWS: dict[str, list[Step]] = {
    "dub":        [PREP, TEXT, SYNTH, MUX],
    "transcribe": [PREP],
}


def steps_for(workflow_type: str) -> list[Step]:
    return WORKFLOWS[workflow_type]


def first_step(workflow_type: str) -> Step:
    return steps_for(workflow_type)[0]


def step_by_name(workflow_type: str, name: str) -> Step:
    for s in steps_for(workflow_type):
        if s.name == name:
            return s
    raise KeyError(name)


def next_step(workflow_type: str, completed: str) -> Step | None:
    steps = steps_for(workflow_type)
    names = [s.name for s in steps]
    i = names.index(completed)
    return steps[i + 1] if i + 1 < len(steps) else None


def build_input(step: Step, run: Run) -> dict:
    base = {"run_id": run.id, "episode_id": run.episode_id}
    if step.name == "prep":
        return {**base, "audio_url": run.audio_url, "extract": run.workflow_type == "dub"}
    if step.name == "text":
        return {**base, "segments_key": run.segments_key,
                "source_lang": run.source_lang, "language": run.language}
    if step.name == "synth":
        return {**base, "segments_key": run.segments_key,
                "language": run.language, "speaker_keys": run.speaker_keys}
    if step.name == "mux":
        return {**base, "segments_key": run.segments_key,
                "language": run.language, "bg_volume": run.bg_volume}
    raise ValueError(step.name)
