# RunPod Serverless worker — GPU image for dub-pipeline
# Models are NOT baked in; they live on a RunPod Network Volume at /runpod-volume/models
FROM runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04

WORKDIR /app

# Install system dependencies for audio processing
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir --ignore-installed -r requirements.txt

# Copy source
COPY src/ src/

# Models live on the RunPod Network Volume — set at runtime via env var
ENV MODEL_DIR=/runpod-volume/models
ENV TEMP_DIR=/tmp/dub-pipeline
ENV WHISPER_DEVICE=cuda
ENV WHISPER_COMPUTE=float16
ENV TTS_DEVICE=cuda
ENV COQUI_TOS_AGREED=1
ENV XDG_DATA_HOME=/runpod-volume/models
ENV HF_HOME=/runpod-volume/models/hf

CMD ["python", "-m", "src.worker"]
