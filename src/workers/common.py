"""Shared worker helpers: callback POST + scratch-dir lifecycle."""
import shutil
import tempfile
import requests
from src import config


def post_callback(callback_url: str, body: dict) -> None:
    requests.post(callback_url, json=body, timeout=30)


def run_in_tempdir(body):
    """Call body(work_dir) in a fresh temp dir, always cleaning it up."""
    import os
    os.makedirs(config.TEMP_DIR, exist_ok=True)
    work_dir = tempfile.mkdtemp(dir=config.TEMP_DIR, prefix="dubstep_")
    try:
        return body(work_dir)
    finally:
        shutil.rmtree(work_dir, ignore_errors=True)
