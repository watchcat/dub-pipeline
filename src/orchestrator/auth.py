"""HMAC token scoping a callback to a single (run_id, step)."""
import hashlib
import hmac
from src import config


def make_token(run_id: str, step: str) -> str:
    msg = f"{run_id}:{step}".encode("utf-8")
    return hmac.new(config.ORCH_CALLBACK_SECRET.encode("utf-8"), msg, hashlib.sha256).hexdigest()


def verify_token(run_id: str, step: str, token: str) -> bool:
    return hmac.compare_digest(make_token(run_id, step), token or "")
