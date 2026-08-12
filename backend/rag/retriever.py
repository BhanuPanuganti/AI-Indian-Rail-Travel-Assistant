import sys
from pathlib import Path

from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer

for candidate in [Path(__file__).resolve().parent, *Path(__file__).resolve().parents]:
    backend_dir = candidate / "backend"
    if backend_dir.exists():
        if str(candidate) not in sys.path:
            sys.path.insert(0, str(candidate))
        if str(backend_dir) not in sys.path:
            sys.path.insert(0, str(backend_dir))
        break


BASE_DIR = Path(__file__).parent

QDRANT_PATH = BASE_DIR / "qdrant_data"

COLLECTION_NAME = "indian_railway_knowledge"

MODEL_NAME = "BAAI/bge-small-en-v1.5"


_model = None
_client = None


def get_embedding_model():

    global _model

    if _model is None:

        print(
            "Loading RAG embedding model..."
        )

        _model = SentenceTransformer(
            MODEL_NAME
        )

    return _model


def _get_client():
    global _client
    if _client is None:
        _client = QdrantClient(path=str(QDRANT_PATH))
    return _client


def close():
    global _client
    if _client is not None:
        try:
            _client.close()
        except Exception:
            pass
        _client = None


import atexit

atexit.register(close)


def retrieve(
    query: str,
    top_k: int = 5,
):

    model = get_embedding_model()

    query_vector = model.encode(
        query,
        normalize_embeddings=True,
    ).tolist()

    client = _get_client()

    results = client.query_points(
        collection_name=COLLECTION_NAME,
        query=query_vector,
        limit=top_k,
    ).points

    return [
        {
            "score": result.score,
            "chunk_id": result.payload[
                "chunk_id"
            ],
            "text": result.payload[
                "text"
            ],
            "metadata": result.payload[
                "metadata"
            ],
        }
        for result in results
    ]