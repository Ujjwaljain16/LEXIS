# CPU image for the LEXIS API (Hugging Face Spaces "Docker" SDK or any container host).
# Heavy work (embedding a corpus) is done offline on a GPU and stored in Qdrant Cloud; this image only
# serves queries, so it needs CPU torch and no CUDA.
FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1 HF_HOME=/tmp/hf
WORKDIR /app

COPY pyproject.toml ./
COPY src ./src
COPY config ./config
# torch and torchvision must come from the same (CPU) index, or `import transformers` fails at runtime.
RUN pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu \
    && pip install .

# Spaces runs as an unprivileged user on port 7860; override PORT elsewhere.
RUN useradd -m -u 1000 lexis && mkdir -p /app/data/bm25_index && chown -R lexis /app
USER lexis
ENV PORT=7860
EXPOSE 7860

# Required at runtime (never baked in): QDRANT_URL, QDRANT_API_KEY, GEMINI_API_KEY, LEXIS_API_KEYS.
CMD ["sh", "-c", "uvicorn lexis.serving.app:app --host 0.0.0.0 --port ${PORT}"]
