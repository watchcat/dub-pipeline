"""R2 key conventions and segments.json artifact I/O for pipeline runs."""
import json
from src import storage


def stem_key(episode_id: int, filename: str) -> str:
    return f"dub-stems/{episode_id}/{filename}"


def dub_key(episode_id: int, language: str) -> str:
    return f"dubbed/{episode_id}/{language}.mp3"


def segments_key(run_id: str) -> str:
    return f"dub-runs/{run_id}/segments.json"


def write_segments(run_id: str, segments: list[dict]) -> str:
    key = segments_key(run_id)
    storage.upload_bytes(json.dumps(segments).encode("utf-8"), key, "application/json")
    return key


def read_segments(key: str) -> list[dict]:
    return json.loads(storage.download_bytes(key).decode("utf-8"))
