"""Post step progress to buzz-bot so SSE clients get live updates."""
import logging
import threading
import requests
from src import config

log = logging.getLogger(__name__)


def report(dub_id: int, step: str, pct: int | None = None):
    payload = {"dub_id": dub_id, "step": step}
    if pct is not None:
        payload["pct"] = pct
    threading.Thread(target=_send, args=(dub_id, step, payload), daemon=True).start()


def _send(dub_id: int, step: str, payload: dict):
    try:
        requests.post(config.PROGRESS_URL, json=payload, timeout=30)
    except Exception as e:
        log.debug(f"progress report failed (dub {dub_id}, step {step}): {e}")
