"""Run record + storage abstraction."""
from dataclasses import dataclass, replace
from typing import Optional, Protocol


@dataclass(frozen=True)
class Run:
    id: str
    workflow_type: str          # "dub" | "transcribe"
    episode_id: int
    callback_url: str
    dub_id: Optional[int] = None
    language: Optional[str] = None
    audio_url: Optional[str] = None
    bg_volume: float = 0.15
    status: str = "running"     # running | done | failed
    current_step: str = ""
    attempts: int = 0
    segments_key: Optional[str] = None
    source_lang: Optional[str] = None
    speaker_keys: Optional[dict] = None
    r2_url: Optional[str] = None
    duration_sec: Optional[float] = None
    segment_count: Optional[int] = None


class RunStore(Protocol):
    def create(self, run: Run) -> None: ...
    def get(self, run_id: str) -> Run: ...
    def update(self, run_id: str, **fields) -> Run: ...


class InMemoryRunStore:
    def __init__(self) -> None:
        self._runs: dict[str, Run] = {}

    def create(self, run: Run) -> None:
        self._runs[run.id] = run

    def get(self, run_id: str) -> Run:
        return self._runs[run_id]

    def update(self, run_id: str, **fields) -> Run:
        self._runs[run_id] = replace(self._runs[run_id], **fields)
        return self._runs[run_id]
