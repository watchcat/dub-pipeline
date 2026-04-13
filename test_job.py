"""Run a test dub job locally without RunPod or Redis.

Usage:
    python test_job.py [audio_url] [language]

Defaults:
    audio_url = https://pub-f72ec72a74374596b8e0b595f480860e.r2.dev/tmp/audio/41.mp3
    language  = ru
"""
import os
import secrets
import sys
from pathlib import Path

# Load .env before importing anything
env_file = Path(__file__).parent / ".env"
if env_file.exists():
    for line in env_file.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, _, v = line.partition("=")
            os.environ.setdefault(k.strip(), v.strip())

from src.worker import process_job
from src import config

AUDIO_URL = sys.argv[1] if len(sys.argv) > 1 else \
    "https://pub-f72ec72a74374596b8e0b595f480860e.r2.dev/tmp/audio/41.mp3"
LANGUAGE  = sys.argv[2] if len(sys.argv) > 2 else "ru"

job = {
    "job_id":       secrets.token_hex(16),
    "dub_id":       999999,
    "episode_id":   999999,
    "audio_url":    AUDIO_URL,
    "language":     LANGUAGE,
    "bg_volume":    0.15,
    "callback_url": "http://localhost:9999/internal/dub_result",  # intentionally unreachable
}

print(f"Running test job {job['job_id']}")
print(f"  audio:    {AUDIO_URL}")
print(f"  language: {LANGUAGE}")
print()

process_job(job)

print()
print("Done. Final MP3 uploaded to R2 at:")
print(f"  {config.R2_PUBLIC_URL}/dubbed/{job['episode_id']}/{LANGUAGE}.mp3")
