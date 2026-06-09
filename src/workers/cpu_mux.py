"""CPU step group: assemble -> mix -> upload final mp3. Runs as an HTTP worker."""
import logging
import os
import subprocess

from fastapi import FastAPI, Request
from src import artifacts, storage
from src.steps import assemble, mix
from src.workers import common

log = logging.getLogger(__name__)


def _ffprobe_duration(path: str) -> float:
    result = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration",
         "-of", "default=noprint_wrappers=1:nokey=1", path],
        capture_output=True, text=True, check=True)
    return float(result.stdout.strip())


def run(inp: dict) -> dict:
    run_id     = inp["run_id"]
    episode_id = inp["episode_id"]
    language   = inp["language"]
    bg_volume  = inp.get("bg_volume", 0.15)

    segments = artifacts.read_segments(inp["segments_key"])

    def body(work_dir: str) -> dict:
        background = os.path.join(work_dir, "background.wav")
        storage.download(artifacts.stem_key(episode_id, "background.wav"), background)

        # Pull each synthesized segment back to a local path for assembly.
        for seg in segments:
            key = seg.get("synth_r2_key")
            if key:
                local = os.path.join(work_dir, f"synth_{seg['idx']:04d}.wav")
                storage.download(key, local)
                seg["synth_wav"] = local
            else:
                seg["synth_wav"] = None

        dubbed_vocals, assembled = assemble.assemble(segments, work_dir)
        final_mp3 = mix.mix(dubbed_vocals, background, work_dir, bg_volume=bg_volume)

        r2_url = storage.upload(final_mp3, artifacts.dub_key(episode_id, language), "audio/mpeg")
        duration = _ffprobe_duration(final_mp3)
        count = len([s for s in assembled if s.get("synth_r2_key")])

        for seg in assembled:
            seg.pop("synth_wav", None)
        segments_key_out = artifacts.write_segments(run_id, assembled)
        return {"r2_url": r2_url, "duration_sec": duration,
                "segment_count": count, "segments_key": segments_key_out}

    return common.run_in_tempdir(body)


app = FastAPI()


@app.post("/run")
async def run_endpoint(request: Request):
    payload = await request.json()
    try:
        common.post_callback(payload["callback_url"], {"ok": True, **run(payload["input"])})
    except Exception as e:  # noqa: BLE001
        log.exception("cpu_mux failed")
        common.post_callback(payload["callback_url"], {"ok": False, "error": str(e)})
    return {"accepted": True}


@app.get("/healthz")
async def healthz():
    return {"ok": True}
