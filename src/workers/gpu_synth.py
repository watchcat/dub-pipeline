"""GPU step: synthesize translated segments with VoxCPM2. Nebius job entrypoint."""
import json
import logging
import os
import sys

from src import artifacts, storage
from src.steps import synthesize
from src.workers import common

log = logging.getLogger(__name__)


def run(inp: dict, on_progress=None) -> dict:
    run_id       = inp["run_id"]
    episode_id   = inp["episode_id"]
    language     = inp["language"]
    segments_key = inp["segments_key"]
    speaker_keys = inp.get("speaker_keys") or {}

    segments = artifacts.read_segments(segments_key)

    def body(work_dir: str) -> dict:
        speaker_samples: dict[str, str] = {}
        for speaker, key in speaker_keys.items():
            local = os.path.join(work_dir, f"speaker_{speaker}.wav")
            storage.download(key, local)
            speaker_samples[speaker] = local

        synthed = synthesize.synthesize(segments, speaker_samples, language, work_dir,
                                        on_progress=on_progress)

        out_segments = []
        for seg in synthed:
            seg = {**seg}
            wav = seg.pop("synth_wav", None)
            if wav:
                key = artifacts.stem_key(episode_id, f"synth_{language}_{seg['idx']:04d}.wav")
                storage.upload(wav, key, "audio/wav")
                seg["synth_r2_key"] = key
            out_segments.append(seg)

        return {"segments_key": artifacts.write_segments(run_id, out_segments)}

    return common.run_in_tempdir(body)


def main() -> None:
    inp = json.loads(os.environ["INPUT_JSON"])
    callback_url = os.environ["CALLBACK_URL"]
    progress_url = callback_url.replace("/callback", "/progress")
    last = {"pct": -1}

    def on_progress(done: int, total: int) -> None:
        pct = int(100 * done / total) if total else 0
        if pct != last["pct"]:
            last["pct"] = pct
            try:
                common.post_callback(progress_url, {"pct": pct})
            except Exception:  # noqa: BLE001 — progress is best-effort
                log.warning("synth progress post failed")

    try:
        common.post_callback(callback_url, {"ok": True, **run(inp, on_progress=on_progress)})
    except Exception as e:  # noqa: BLE001
        log.exception("gpu_synth failed")
        common.post_callback(callback_url, {"ok": False, "error": str(e)})
        sys.exit(1)


if __name__ == "__main__":
    main()
