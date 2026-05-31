"""Advances a Run through its workflow, dispatching steps and reporting progress."""
from src import config
from src.orchestrator import auth, workflows
from src.orchestrator.runs import Run, RunStore

# Maps a step-completion callback's result fields onto Run columns.
_RESULT_FIELDS = ("segments_key", "source_lang", "speaker_keys",
                  "r2_url", "duration_sec", "segment_count")


class StateMachine:
    def __init__(self, store: RunStore, dispatchers: dict, reporter,
                 max_attempts: int = config.MAX_STEP_ATTEMPTS):
        self.store = store
        self.dispatchers = dispatchers
        self.reporter = reporter
        self.max_attempts = max_attempts

    def start(self, run: Run) -> None:
        self.store.create(run)
        self._dispatch(run, workflows.first_step(run.workflow_type))

    def handle_callback(self, run_id: str, step_name: str, ok: bool, result: dict) -> None:
        run = self.store.get(run_id)
        if not ok:
            self._on_failure(run, step_name, result.get("error"))
            return
        updates = {k: result[k] for k in _RESULT_FIELDS if k in result}
        run = self.store.update(run_id, attempts=0, **updates)
        nxt = workflows.next_step(run.workflow_type, step_name)
        if nxt is None:
            self._finalize(run)
        else:
            self._dispatch(run, nxt)

    # ── internals ──
    def _dispatch(self, run: Run, step) -> None:
        run = self.store.update(run.id, current_step=step.name)
        self.reporter.progress(run, step.progress)
        payload = workflows.build_input(step, run)
        token = auth.make_token(run.id, step.name)
        cb = (f"{config.ORCH_BASE_URL}/callback"
              f"?run_id={run.id}&step={step.name}&token={token}")
        self.dispatchers[step.tier].dispatch(step, run, payload, cb)

    def _on_failure(self, run: Run, step_name: str, error) -> None:
        if run.attempts + 1 < self.max_attempts:
            run = self.store.update(run.id, attempts=run.attempts + 1)
            self._dispatch(run, workflows.step_by_name(run.workflow_type, step_name))
        else:
            run = self.store.update(run.id, status="failed")
            self.reporter.failed(run, step_name, error)

    def _finalize(self, run: Run) -> None:
        run = self.store.update(run.id, status="done")
        if run.workflow_type == "transcribe":
            self.reporter.transcript_result(run)
        else:
            self.reporter.dub_result(run)
        self.reporter.progress(run, "complete", 100)
