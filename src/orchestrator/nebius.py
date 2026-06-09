# src/orchestrator/nebius.py
"""Thin Nebius Jobs REST client. Endpoint shapes verified against Nebius docs at deploy."""
import logging
import requests
from src import config

log = logging.getLogger(__name__)
_TIMEOUT = 30

_TERMINAL = {"succeeded": "succeeded", "failed": "failed",
             "cancelled": "failed", "error": "failed"}


def _headers() -> dict:
    return {"Authorization": f"Bearer {config.NEBIUS_API_KEY}",
            "Content-Type": "application/json"}


def create_job(image: str, preset: str, env: dict, timeout_sec: int) -> str:
    """Create a Nebius GPU job; return its job id."""
    spec = {
        "project_id": config.NEBIUS_PROJECT_ID,
        "image": image,
        "preset": preset,
        "timeout_seconds": timeout_sec,
        "environment": env,
    }
    resp = requests.post(f"{config.NEBIUS_API_BASE}/jobs", json=spec,
                         headers=_headers(), timeout=_TIMEOUT)
    resp.raise_for_status()
    return resp.json()["id"]


def get_status(job_id: str) -> str:
    """Return one of: running | succeeded | failed | gone."""
    resp = requests.get(f"{config.NEBIUS_API_BASE}/jobs/{job_id}",
                        headers=_headers(), timeout=_TIMEOUT)
    if resp.status_code == 404:
        return "gone"
    resp.raise_for_status()
    state = (resp.json().get("status") or "").lower()
    return _TERMINAL.get(state, "running")
