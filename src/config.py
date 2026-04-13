import os

PROGRESS_URL     = os.environ["PROGRESS_URL"]   # https://app.buzz-bot.top/internal/dub_progress
TEMP_DIR         = os.environ.get("TEMP_DIR", "/tmp/dub-pipeline")
MODEL_DIR        = os.environ.get("MODEL_DIR", os.path.join(os.path.dirname(__file__), "..", "models"))

# R2 / S3-compatible storage
R2_ENDPOINT      = os.environ["R2_ENDPOINT"]    # https://<account>.r2.cloudflarestorage.com
R2_ACCESS_KEY    = os.environ["R2_ACCESS_KEY_ID"]
R2_SECRET_KEY    = os.environ["R2_SECRET_ACCESS_KEY"]
R2_BUCKET        = os.environ["R2_BUCKET"]
R2_PUBLIC_URL    = os.environ["R2_PUBLIC_URL"]  # https://pub-xxx.r2.dev

# Gemini translation model
GEMINI_API_KEY   = os.environ["GEMINI_API_KEY"]
GEMINI_MODEL     = os.environ.get("GEMINI_MODEL", "gemini-2.5-flash")

# HuggingFace token (required for pyannote speaker diarization models)
HF_TOKEN         = os.environ["HF_TOKEN"]

# Demucs model — htdemucs_ft for highest quality
DEMUCS_MODEL     = os.environ.get("DEMUCS_MODEL", "htdemucs_ft")

# WhisperX model — reuse large-v3 already downloaded for whisper-service
WHISPER_MODEL    = os.environ.get("WHISPER_MODEL", "large-v3")
WHISPER_DEVICE   = os.environ.get("WHISPER_DEVICE", "mps")   # mps | cuda | cpu
WHISPER_COMPUTE  = os.environ.get("WHISPER_COMPUTE", "float16")

# XTTS-v2 device
TTS_DEVICE       = os.environ.get("TTS_DEVICE", "mps")

# Assembly
MAX_DURATION_RATIO = float(os.environ.get("MAX_DURATION_RATIO", "1.5"))
BG_VOLUME_DEFAULT  = float(os.environ.get("BG_VOLUME_DEFAULT", "0.15"))

# Concurrency limits
MAX_SEGMENT_SEC  = 30.0   # split longer segments before synthesis
MIN_SEGMENT_SEC  = 0.5    # skip synthesis below this, insert silence
