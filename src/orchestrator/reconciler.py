# src/orchestrator/reconciler.py
"""Backstop for GPU jobs that die (OOM/preempt/crash) before posting /callback.

Scans runs whose GPU step deadline has passed with no callback, asks Nebius for
the job's real state, and either re-dispatches via the state machine's failure
path (which reuses R2 artifacts) or extends the deadline for a still-running job.
"""
import logging
import time
from datetime import datetime, timedelta, timezone
from src import config

log = logging.getLogger(__name__)


class Reconciler:
    def __init__(self, store, state_machine, nebius_client,
                 now=lambda: datetime.now(timezone.utc)):
        self.store = store
        self.sm = state_machine
        self.nebius = nebius_client
        self.now = now

    def tick(self) -> int:
        """One reconcile pass; returns how many runs it acted on."""
        acted = 0
        for run in self.store.due_for_reconcile(self.now()):
            status = self.nebius.get_status(run.nebius_job_id)
            if status == "running":
                self.store.update(run.id, step_deadline=self._next_deadline(run))
            else:
                # failed | gone | succeeded-without-callback: re-dispatch (R2 reuse
                # makes the retry cheap and idempotent).
                self.sm._on_failure(run, run.current_step,
                                    f"nebius job {status} without callback")
            acted += 1
        return acted

    def _next_deadline(self, run):
        secs = config.STEP_TIMEOUT.get(run.current_step, config.RECONCILER_INTERVAL_SEC)
        return self.now() + timedelta(seconds=secs)

    def run_forever(self, interval: int | None = None) -> None:
        interval = interval or config.RECONCILER_INTERVAL_SEC
        while True:
            try:
                self.tick()
            except Exception:  # noqa: BLE001 — the loop must never die
                log.exception("reconciler tick failed")
            time.sleep(interval)
