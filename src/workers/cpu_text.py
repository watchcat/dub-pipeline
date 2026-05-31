"""CPU step group: split -> translate. Runs as an HTTP worker (FastAPI)."""
import logging

from fastapi import FastAPI, Request
from src import artifacts
from src.steps import split_segments, translate
from src.workers import common

log = logging.getLogger(__name__)


def run(inp: dict) -> dict:
    segments = artifacts.read_segments(inp["segments_key"])
    segments = split_segments.split_long_segments(segments)
    segments = translate.translate(segments, inp["source_lang"], inp["language"])
    return {"segments_key": artifacts.write_segments(inp["run_id"], segments)}


app = FastAPI()


@app.post("/run")
async def run_endpoint(request: Request):
    payload = await request.json()
    inp = payload["input"]
    callback_url = payload["callback_url"]
    try:
        common.post_callback(callback_url, {"ok": True, **run(inp)})
    except Exception as e:  # noqa: BLE001
        log.exception("cpu_text failed")
        common.post_callback(callback_url, {"ok": False, "error": str(e)})
    return {"accepted": True}


@app.get("/healthz")
async def healthz():
    return {"ok": True}
