"""GPU step: synthesize translated segments with VoxCPM2. Nebius job entrypoint."""
import json
import logging
import os
import sys

from src import artifacts, storage
from src.steps import synthesize
from src.workers import common

log = logging.getLogger(__name__)


def run(inp: dict) -> dict:
    run_id       = inp["run_id"]
    episode_id   = inp["episode_id"]
    language     = inp["language"]
    segments_key = inp["segments_key"]
    speaker_keys = inp.get("speaker_keys") or {}

    segments = artifacts.read_segments(segments_key)

    def body(work_dir: str) -> dict:
        # Download speaker samples referenced by key to local paths.
        speaker_samples: dict[str, str] = {}
        for speaker, key in speaker_keys.items():
            local = os.path.join(work_dir, f"speaker_{speaker}.wav")
            storage.download(key, local)
            speaker_samples[speaker] = local

        synthed = synthesize.synthesize(segments, speaker_samples, language, work_dir)

        out_segments = []
        for seg in synthed:
            wav = seg.get("synth_wav")
            if wav:
                key = artifacts.stem_key(episode_id, f"synth_{language}_{seg['idx']:04d}.wav")
                storage.upload(wav, key, "audio/wav")
                seg = {**seg, "synth_r2_key": key}
            seg.pop("synth_wav", None)  # local path not useful downstream
            out_segments.append(seg)

        segments_key_out = artifacts.write_segments(run_id, out_segments)
        return {"segments_key": segments_key_out}

    return common.run_in_tempdir(body)


def main() -> None:
    inp = json.loads(os.environ["INPUT_JSON"])
    callback_url = os.environ["CALLBACK_URL"]
    try:
        common.post_callback(callback_url, {"ok": True, **run(inp)})
    except Exception as e:  # noqa: BLE001
        log.exception("gpu_synth failed")
        common.post_callback(callback_url, {"ok": False, "error": str(e)})
        sys.exit(1)


if __name__ == "__main__":
    main()
