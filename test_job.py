"""Push a test dub job to the Redis queue and tail progress.

Usage:
    python test_job.py [audio_url] [language]

Defaults:
    audio_url = https://pub-f72ec72a74374596b8e0b595f480860e.r2.dev/tmp/audio/41.mp3
    language  = ru
"""
import json
import os
import secrets
import sys
import time

# Load .env before importing config
from pathlib import Path
env_file = Path(__file__).parent / ".env"
if env_file.exists():
    for line in env_file.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, _, v = line.partition("=")
            os.environ.setdefault(k.strip(), v.strip())

import redis
from src import config

AUDIO_URL = sys.argv[1] if len(sys.argv) > 1 else \
    "https://pub-f72ec72a74374596b8e0b595f480860e.r2.dev/tmp/audio/41.mp3"
LANGUAGE  = sys.argv[2] if len(sys.argv) > 2 else "ru"

job = {
    "job_id":       secrets.token_hex(16),
    "dub_id":       999999,           # fake ID — no DB row, callback will 404 gracefully
    "episode_id":   999999,
    "audio_url":    AUDIO_URL,
    "language":     LANGUAGE,
    "bg_volume":    0.15,
    "callback_url": "http://localhost:9999/internal/dub_result",  # intentionally unreachable
}

r = redis.from_url(config.REDIS_URL)

# Clear any stale test jobs first
r.delete(config.QUEUE_KEY)

payload = json.dumps(job)
r.rpush(config.QUEUE_KEY, payload)

print(f"Pushed test job {job['job_id']}")
print(f"  audio:    {AUDIO_URL}")
print(f"  language: {LANGUAGE}")
print(f"  queue:    {config.QUEUE_KEY}")
print()
print("Worker output will appear in the terminal where run-worker.sh is running.")
print("Final MP3 will be uploaded to R2 at:")
print(f"  {config.R2_PUBLIC_URL}/dubbed/{job['episode_id']}/{LANGUAGE}.mp3")
