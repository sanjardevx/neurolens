# NeuroLens API: FastAPI + ONNX Runtime, no PyTorch, no GPU.
# Build from the repo root (it needs backend/ and the inference code in ml/):
#   docker build -f backend/Dockerfile -t neurolens-api .
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1 PIP_DISABLE_PIP_VERSION_CHECK=1
WORKDIR /app

# Torch-free inference code from ml/ (crop, infer_cls, infer_seg, measure). --no-deps: skip the training stack
COPY ml/pyproject.toml ml/pyproject.toml
COPY ml/neurolens_ml ml/neurolens_ml
RUN pip install --no-deps ./ml

COPY backend/pyproject.toml backend/pyproject.toml
COPY backend/app backend/app
RUN pip install ./backend

# Models: a mounted volume (docker compose), or downloaded at build from a Hugging Face model repo (the deploy
# workflow sets MODEL_REPO). The files carry their own version, so the API reports exactly what was downloaded.
ARG MODEL_REPO=""
RUN useradd --create-home app && mkdir -p /data/storage /models && chown -R app /data
RUN if [ -n "$MODEL_REPO" ]; then for f in classifier.onnx segmenter3d.onnx; do \
      python -c "import sys, urllib.request; urllib.request.urlretrieve(f'https://huggingface.co/{sys.argv[1]}/resolve/main/{sys.argv[2]}', f'/models/{sys.argv[2]}')" "$MODEL_REPO" "$f" \
      || exit 1; done; fi
USER app
WORKDIR /app/backend
ENV MODEL_DIR=/models STORAGE_DIR=/data/storage DATABASE_URL=sqlite:////data/neurolens.db

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')"
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
