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

# Checkpoint directory — persists across RunPod restarts via Network Volume
CHECKPOINT_DIR = os.environ.get("CHECKPOINT_DIR", "/tmp/dub-checkpoints")

# Assembly
BG_VOLUME_DEFAULT  = float(os.environ.get("BG_VOLUME_DEFAULT", "0.15"))

# Segment duration constraints
MAX_SEGMENT_SEC  = 30.0   # split longer segments before synthesis
MIN_SEGMENT_SEC  = 0.5    # skip synthesis below this, insert silence

# ── Orchestrator ────────────────────────────────────────────────────────────
DATABASE_URL          = os.environ.get("DATABASE_URL", "")
ORCH_CALLBACK_SECRET  = os.environ.get("ORCH_CALLBACK_SECRET", "dev-secret")
ORCH_BASE_URL         = os.environ.get("ORCH_BASE_URL", "http://localhost:8080")
BUZZBOT_RESULT_URL     = os.environ.get("BUZZBOT_RESULT_URL", "")
BUZZBOT_TRANSCRIPT_URL = os.environ.get("BUZZBOT_TRANSCRIPT_URL", "")
CPU_TEXT_URL          = os.environ.get("CPU_TEXT_URL", "")
CPU_MUX_URL           = os.environ.get("CPU_MUX_URL", "")
MAX_STEP_ATTEMPTS     = int(os.environ.get("MAX_STEP_ATTEMPTS", "3"))

# ── Nebius GPU jobs + reconciler ─────────────────────────────────────────────
NEBIUS_API_KEY    = os.environ.get("NEBIUS_API_KEY", "")
NEBIUS_PROJECT_ID = os.environ.get("NEBIUS_PROJECT_ID", "")
NEBIUS_API_BASE   = os.environ.get("NEBIUS_API_BASE", "https://api.nebius.cloud")
GPU_IMAGE         = os.environ.get("GPU_IMAGE", "")
NEBIUS_PRESET = {
    "prep":  os.environ.get("NEBIUS_PREP_PRESET", ""),
    "synth": os.environ.get("NEBIUS_SYNTH_PRESET", ""),
}
STEP_TIMEOUT = {
    "prep":  int(os.environ.get("STEP_TIMEOUT_PREP", "1800")),
    "synth": int(os.environ.get("STEP_TIMEOUT_SYNTH", "1200")),
}
RECONCILER_INTERVAL_SEC = int(os.environ.get("RECONCILER_INTERVAL_SEC", "60"))
