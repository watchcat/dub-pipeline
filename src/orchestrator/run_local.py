"""Drive the full pipeline in-process using LocalDispatcher — no cloud, no GPU API.

Used for local end-to-end testing on a short clip. The orchestrator's /callback
HTTP hop is short-circuited: instead of an HTTP callback, results feed straight
back into the state machine via a loopback dispatcher.
"""
from src import config
from src.orchestrator.reporting import Reporter
from src.orchestrator.runs import InMemoryRunStore, Run
from src.orchestrator.state_machine import StateMachine
from src.workers import gpu_prep, gpu_synth, cpu_text, cpu_mux


class _LoopbackDispatcher:
    """LocalDispatcher variant that feeds results straight back into the state
    machine instead of doing an HTTP callback (keeps the e2e test cloud-free)."""
    def __init__(self, workers, sm_ref):
        self.workers = workers
        self.sm_ref = sm_ref

    def dispatch(self, step, run, payload, callback_url):
        try:
            result = self.workers[step.name](payload)
            self.sm_ref[0].handle_callback(run.id, step.name, True, result)
        except Exception as e:  # noqa: BLE001
            self.sm_ref[0].handle_callback(run.id, step.name, False, {"error": str(e)})


class _LocalReporter(Reporter):
    """Suppress intermediate progress posts during local in-process runs.

    The PROGRESS_URL endpoint is not reachable during local testing; only
    the final dub_result (and failed) callbacks are meaningful here.
    """
    def progress(self, run, label, pct=None):
        pass  # no-op: suppress HTTP progress posts for local runs


def run_dub_locally(run_id, dub_id, episode_id, audio_url, language, bg_volume=0.15):
    store = InMemoryRunStore()
    sm_ref: list = [None]
    workers = {"prep": gpu_prep.run, "synth": gpu_synth.run,
               "text": cpu_text.run, "mux": cpu_mux.run}
    disp = _LoopbackDispatcher(workers, sm_ref)
    sm = StateMachine(store, {"gpu": disp, "cpu": disp}, _LocalReporter())
    sm_ref[0] = sm
    sm.start(Run(id=run_id, workflow_type="dub", episode_id=episode_id,
                 callback_url=config.BUZZBOT_RESULT_URL, dub_id=dub_id,
                 language=language, audio_url=audio_url, bg_volume=bg_volume))
    return store
