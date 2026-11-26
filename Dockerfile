FROM python:3.11-slim

# Keep memory low on small (512 MB) instances:
# - one thread for ONNX Runtime / BLAS
# - fewer glibc malloc arenas (each thread arena reserves memory)
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    OMP_NUM_THREADS=1 \
    MALLOC_ARENA_MAX=2 \
    MODEL_THREADS=1 \
    MODEL_CACHE_DIR=/app/.model_cache

WORKDIR /app

COPY backend/requirements.txt backend/requirements.txt
RUN pip install --no-cache-dir -r backend/requirements.txt

COPY backend backend
COPY data data

# Build the SQLite timetable, then drop the CSV it was built from.
RUN python -m backend.train_db \
    && rm data/railway/Train_details_22122017.csv

# Download the ONNX embedding and reranker models into the image.
RUN python -m backend.rag.preload_models

EXPOSE 8000

# Render (and most hosts) set $PORT; default to 8000 for local runs.
CMD ["sh", "-c", "exec uvicorn backend.main:app --host 0.0.0.0 --port ${PORT:-8000} --workers 1"]
