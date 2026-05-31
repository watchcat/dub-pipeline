"""Orchestrator HTTP surface: POST /dispatch (start) and POST /callback (advance)."""
import logging

from fastapi import FastAPI, Request, Response
from src.orchestrator import auth
from src.orchestrator.runs import Run
from src.orchestrator.state_machine import StateMachine

log = logging.getLogger(__name__)
app = FastAPI()

_sm: StateMachine | None = None


def configure(store, dispatchers, reporter) -> None:
    """Wire the app to its collaborators (called at startup and from tests)."""
    global _sm
    _sm = StateMachine(store, dispatchers, reporter)


@app.post("/dispatch", status_code=202)
async def dispatch(request: Request):
    b = await request.json()
    run = Run(
        id=b["run_id"], workflow_type=b["workflow_type"], episode_id=b["episode_id"],
        callback_url=b["callback_url"], dub_id=b.get("dub_id"),
        language=b.get("language"), audio_url=b.get("audio_url"),
        bg_volume=b.get("bg_volume", 0.15))
    _sm.start(run)
    return {"run_id": run.id, "status": "started"}


@app.post("/callback")
async def callback(run_id: str, step: str, token: str, request: Request):
    if not auth.verify_token(run_id, step, token):
        return Response(status_code=401)
    body = await request.json()
    ok = body.pop("ok", False)
    _sm.handle_callback(run_id, step, ok, body)
    return {"ok": True}


@app.get("/healthz")
async def healthz():
    return {"ok": True}
