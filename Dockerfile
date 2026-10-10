FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    HF_HOME=/tmp/huggingface \
    HF_HUB_CACHE=/tmp/huggingface/hub \
    TRANSFORMERS_CACHE=/tmp/huggingface/hub

WORKDIR /app

# Runtime requirements for CPU inference. No GPU is provisioned by this image.
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 ca-certificates \
    && rm -rf /var/lib/apt/lists/*

RUN python -m pip install --upgrade pip \
    && python -m pip install torch --index-url https://download.pytorch.org/whl/cpu

COPY requirements-api.txt ./requirements-api.txt
RUN python -m pip install -r requirements-api.txt

# Model weights are intentionally NOT downloaded during image build. The first
# request downloads them from Hugging Face to HF_HOME; subsequent requests reuse
# the in-memory model. The HF cache is temporary and may be lost on a cold start.
COPY api.py ./api.py

EXPOSE 8000
CMD ["sh", "-c", "uvicorn api:app --host 0.0.0.0 --port ${PORT:-8000}"]
