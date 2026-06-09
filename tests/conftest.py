"""Pytest configuration.

`src/config.py` reads several required env vars at import time via
`os.environ[...]`, so importing any `src.*` module needs them present. Set
safe dummy defaults here (with setdefault, so a real environment still wins)
to keep the test suite hermetic and CI-safe — without depending on a real
`.env` file or leaking real secrets into the test process.
"""
import os

_DUMMY_ENV = {
    "PROGRESS_URL": "https://app.test/internal/dub_progress",
    "R2_ENDPOINT": "https://test.r2.cloudflarestorage.com",
    "R2_ACCESS_KEY_ID": "test-access-key",
    "R2_SECRET_ACCESS_KEY": "test-secret-key",
    "R2_BUCKET": "test-bucket",
    "R2_PUBLIC_URL": "https://pub-test.r2.dev",
    "GEMINI_API_KEY": "test-gemini-key",
    "HF_TOKEN": "test-hf-token",
}

for _k, _v in _DUMMY_ENV.items():
    os.environ.setdefault(_k, _v)
