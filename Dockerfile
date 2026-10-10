FROM python:3.12-slim-bookworm

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    HF_HUB_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1 \
    TOKENIZERS_PARALLELISM=false \
    LLM_PROVIDER=groq \
    GROQ_MODEL=llama-3.3-70b-versatile \
    BUNDLE_DIR=/app/pipeline/data/spacenet-grounded \
    ENCODER_CHECKPOINT=/app/pipeline/data/models/SkyCLIP_ViT_B32_top50pct.safetensors \
    GROUNDING_MODEL_PATH=/app/pipeline/data/models/owlv2-cfd3195ba4ea9592eec887ded089f4c08eff231d \
    GROUNDING_DEVICE=cpu \
    JOBS_DB=/app/pipeline/data/state/conversations.sqlite3

WORKDIR /app

RUN apt-get update \
    && apt-get install --yes --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY pipeline/requirements.txt /tmp/requirements.txt
RUN python -m pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cpu "torch>=2.5,<3" "torchvision>=0.20,<1" \
    && python -m pip install --no-cache-dir -r /tmp/requirements.txt

COPY pipeline/terralens /app/pipeline/terralens
RUN mkdir -p /app/pipeline/data

EXPOSE 10000

CMD ["sh", "-c", "exec uvicorn terralens.api:app --app-dir /app/pipeline --host 0.0.0.0 --port \"${PORT:-10000}\" --workers 1"]
