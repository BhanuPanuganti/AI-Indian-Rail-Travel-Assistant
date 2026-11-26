from pathlib import Path
import json
import os
from dotenv import load_dotenv

from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    PointStruct,
    VectorParams,
)

try:
    from backend.rag.retriever import COLLECTION_NAME, embed_texts
except ModuleNotFoundError:
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from backend.rag.retriever import COLLECTION_NAME, embed_texts


BASE_DIR = Path(__file__).parent

CHUNKS_FILE = BASE_DIR / "chunks.jsonl"

load_dotenv()

QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")



def load_chunks():
    chunks = []

    with open(
        CHUNKS_FILE,
        encoding="utf-8",
    ) as file:

        for line in file:
            chunks.append(
                json.loads(line)
            )

    return chunks


def create_vector_store():

    chunks = load_chunks()

    print(
        f"Embedding {len(chunks)} chunks..."
    )

    texts = [
        chunk["text"]
        for chunk in chunks
    ]

    # Same embedding function the backend uses at query time.
    embeddings = embed_texts(texts)

    dimension = embeddings.shape[1]

    client = QdrantClient(
        url=QDRANT_URL,
        api_key=QDRANT_API_KEY,
    )

    if client.collection_exists(
        COLLECTION_NAME
    ):
        client.delete_collection(
            COLLECTION_NAME
        )

    client.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=VectorParams(
            size=dimension,
            distance=Distance.COSINE,
        ),
    )

    points = []

    for index, (
        chunk,
        embedding,
    ) in enumerate(
        zip(chunks, embeddings)
    ):

        points.append(
            PointStruct(
                id=index,
                vector=embedding.tolist(),
                payload={
                    "chunk_id": chunk[
                        "chunk_id"
                    ],
                    "text": chunk[
                        "text"
                    ],
                    "metadata": chunk[
                        "metadata"
                    ],
                },
            )
        )

    client.upsert(
        collection_name=COLLECTION_NAME,
        points=points,
    )

    print(
        f"Stored {len(points)} vectors."
    )

    print(
        f"Collection: {COLLECTION_NAME}"
    )


if __name__ == "__main__":
    create_vector_store()