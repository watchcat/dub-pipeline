"""Posts progress + final results back to buzz-bot (matches existing contract)."""
import logging
import requests
from src import artifacts, config
from src.orchestrator.runs import Run

log = logging.getLogger(__name__)
_TIMEOUT = 30


def _post(url: str, body: dict) -> None:
    try:
        requests.post(url, json=body, timeout=_TIMEOUT)
    except Exception as e:  # noqa: BLE001 — best-effort callback
        log.warning("callback failed (%s): %s", url, e)


class Reporter:
    def progress(self, run: Run, label: str, pct: int | None = None) -> None:
        body = {"dub_id": run.dub_id, "step": label}
        if pct is not None:
            body["pct"] = pct
        _post(config.PROGRESS_URL, body)

    def failed(self, run: Run, step_label: str, error) -> None:
        _post(run.callback_url, {
            "dub_id": run.dub_id,
            "success": False,
            "step": step_label,
            "error": str(error) if error is not None else None,
        })

    def dub_result(self, run: Run) -> None:
        segments = artifacts.read_segments(run.segments_key)
        _post(run.callback_url, {
            "dub_id": run.dub_id,
            "episode_id": run.episode_id,
            "language": run.language,
            "source_lang": run.source_lang,
            "success": True,
            "r2_url": run.r2_url,
            "duration_sec": run.duration_sec,
            "segment_count": run.segment_count,
            "speaker_count": len(run.speaker_keys or {}),
            "segments": [_segment_payload(s) for s in segments],
        })

    def transcript_result(self, run: Run) -> None:
        segments = artifacts.read_segments(run.segments_key)
        # Transcripts post to a dedicated config endpoint (per spec); the buzz-bot
        # transcribe consumer + its per-run callback wiring is Plan 2.
        _post(config.BUZZBOT_TRANSCRIPT_URL, {
            "episode_id": run.episode_id,
            "source_lang": run.source_lang,
            "segments": [_segment_payload(s) for s in segments],
        })


def _segment_payload(s: dict) -> dict:
    return {
        "idx": s["idx"],
        "start_sec": s.get("start_sec"),
        "end_sec": s.get("end_sec"),
        "speaker_id": s.get("speaker"),
        "text": s.get("text", ""),
        "words": s.get("words"),
        "translated_text": s.get("translated_text"),
        "synth_r2_key": s.get("synth_r2_key"),
        "synth_duration": s.get("synth_duration"),
        "synth_start_sec": s.get("synth_start_sec"),
    }
