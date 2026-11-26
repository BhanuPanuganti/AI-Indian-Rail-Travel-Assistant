import sys
import os
import atexit
from pathlib import Path

import numpy as np
from dotenv import load_dotenv
from fastembed import TextEmbedding
from qdrant_client import QdrantClient


load_dotenv()


for candidate in [
    Path(__file__).resolve().parent,
    *Path(__file__).resolve().parents
]:
    backend_dir = candidate / "backend"

    if backend_dir.exists():

        if str(candidate) not in sys.path:
            sys.path.insert(0, str(candidate))

        if str(backend_dir) not in sys.path:
            sys.path.insert(0, str(backend_dir))

        break


BASE_DIR = Path(__file__).parent

COLLECTION_NAME = "indian_railway_knowledge"

MODEL_NAME = "BAAI/bge-small-en-v1.5"

# ONNX model files are stored here. The Docker build downloads them into the
# image (see backend/rag/preload_models.py) so the server never fetches them
# at startup.
MODEL_CACHE_DIR = os.getenv(
    "MODEL_CACHE_DIR",
    str(Path(__file__).resolve().parents[2] / ".model_cache"),
)

# One inference thread keeps memory low on small instances.
MODEL_THREADS = int(os.getenv("MODEL_THREADS", "1"))

# ONNX Runtime's CPU memory arena keeps freed buffers for reuse, which grows
# resident memory by hundreds of MB. Off by default; set MODEL_CPU_MEM_ARENA=1
# to trade memory for slightly faster inference.
MODEL_CPU_MEM_ARENA = os.getenv("MODEL_CPU_MEM_ARENA", "0") == "1"


_model = None
_client = None


def get_embedding_model():

    global _model

    if _model is None:

        print(
            "Loading RAG embedding model..."
        )

        # fastembed runs the same bge-small-en-v1.5 model on ONNX Runtime,
        # which needs far less memory than PyTorch.
        _model = TextEmbedding(
            model_name=MODEL_NAME,
            cache_dir=MODEL_CACHE_DIR,
            threads=MODEL_THREADS,
            enable_cpu_mem_arena=MODEL_CPU_MEM_ARENA,
        )

    return _model


def _get_client():

    global _client

    if _client is None:

        qdrant_url = os.getenv("QDRANT_URL")
        qdrant_api_key = os.getenv("QDRANT_API_KEY")

        if not qdrant_url:
            raise RuntimeError(
                "QDRANT_URL environment variable is not set"
            )

        if not qdrant_api_key:
            raise RuntimeError(
                "QDRANT_API_KEY environment variable is not set"
            )

        _client = QdrantClient(
            url=qdrant_url,
            api_key=qdrant_api_key,
        )

    return _client


def close():

    global _client

    if _client is not None:

        try:
            _client.close()

        except Exception:
            pass

        _client = None


atexit.register(close)


def embed_texts(texts: list[str]):
    """Return L2-normalised embeddings (one row per text) as a numpy array."""

    vectors = np.asarray(list(get_embedding_model().embed(texts)), dtype=np.float32)
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    return vectors / np.where(norms == 0, 1.0, norms)


def retrieve(
    query: str,
    top_k: int = 5,
):

    query_vector = embed_texts([query])[0].tolist()

    client = _get_client()

    results = client.query_points(
        collection_name=COLLECTION_NAME,
        query=query_vector,
        limit=top_k,
    ).points

    return [
        {
            "score": result.score,
            "chunk_id": result.payload["chunk_id"],
            "text": result.payload["text"],
            "metadata": result.payload["metadata"],
        }
        for result in results
    ]