"""Dispatchers turn a (step, run, payload, callback) into an actual invocation.

- HttpDispatcher  : POST to a long-running CPU worker's /run endpoint.
- LocalDispatcher : run the worker function in-process (dev / local e2e).
- NebiusDispatcher: launches a Nebius GPU job — added in Plan 2.
"""
import logging
from typing import Callable, Protocol

import requests
from src.orchestrator.runs import Run
from src.orchestrator.workflows import Step

log = logging.getLogger(__name__)


class Dispatcher(Protocol):
    def dispatch(self, step: Step, run: Run, payload: dict, callback_url: str) -> None: ...


class HttpDispatcher:
    """Dispatch a step to a CPU worker reachable at a per-step URL."""
    def __init__(self, urls: dict[str, str]):
        self.urls = urls

    def dispatch(self, step: Step, run: Run, payload: dict, callback_url: str) -> None:
        requests.post(self.urls[step.name],
                      json={"input": payload, "callback_url": callback_url}, timeout=30)


class LocalDispatcher:
    """Run a worker's `run(input)->dict` in a background thread, then POST the callback."""
    def __init__(self, workers: dict[str, Callable[[dict], dict]]):
        self.workers = workers

    def dispatch(self, step: Step, run: Run, payload: dict, callback_url: str) -> None:
        worker = self.workers[step.name]

        def _go():
            try:
                result = worker(payload)
                requests.post(callback_url, json={"ok": True, **result}, timeout=30)
            except Exception as e:  # noqa: BLE001
                log.exception("local worker %s failed", step.name)
                requests.post(callback_url, json={"ok": False, "error": str(e)}, timeout=30)

        _go()  # synchronous in tests; see note for local e2e
