"""GPU step group: separate -> transcribe -> (extract). Nebius job entrypoint."""
import json
import logging
import os
import sys

from src import artifacts, storage
from src.steps import separate, transcribe, extract_samples
from src.workers import common

log = logging.getLogger(__name__)


def run(inp: dict) -> dict:
    run_id     = inp["run_id"]
    episode_id = inp["episode_id"]
    audio_url  = inp["audio_url"]
    do_extract = inp.get("extract", True)

    def body(work_dir: str) -> dict:
        vocals, background = separate.separate(audio_url, work_dir)
        storage.upload(vocals,     artifacts.stem_key(episode_id, "vocals.wav"),     "audio/wav")
        storage.upload(background, artifacts.stem_key(episode_id, "background.wav"), "audio/wav")

        segments, source_lang = transcribe.transcribe(vocals)

        speaker_keys: dict[str, str] = {}
        if do_extract:
            samples = extract_samples.extract_samples(segments, vocals, work_dir)
            for speaker, local_path in samples.items():
                key = artifacts.stem_key(episode_id, f"speaker_{speaker}.wav")
                storage.upload(local_path, key, "audio/wav")
                speaker_keys[speaker] = key

        segments_key = artifacts.write_segments(run_id, segments)
        return {"source_lang": source_lang, "speaker_keys": speaker_keys,
                "segments_key": segments_key}

    return common.run_in_tempdir(body)


def main() -> None:
    inp = json.loads(os.environ["INPUT_JSON"])
    callback_url = os.environ["CALLBACK_URL"]
    try:
        result = run(inp)
        common.post_callback(callback_url, {"ok": True, **result})
    except Exception as e:  # noqa: BLE001
        log.exception("gpu_prep failed")
        common.post_callback(callback_url, {"ok": False, "error": str(e)})
        sys.exit(1)


if __name__ == "__main__":
    main()
