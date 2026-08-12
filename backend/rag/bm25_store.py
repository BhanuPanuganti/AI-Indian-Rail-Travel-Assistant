import json
import re
from pathlib import Path

from rank_bm25 import BM25Okapi


BASE_DIR = Path(__file__).parent
CHUNKS_FILE = BASE_DIR / "chunks.jsonl"


def tokenize(text: str) -> list[str]:
    return re.findall(
        r"\b\w+\b",
        text.lower(),
    )


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


chunks = load_chunks()

tokenized_chunks = [
    tokenize(chunk["text"])
    for chunk in chunks
]

bm25 = BM25Okapi(
    tokenized_chunks
)


def bm25_search(
    query: str,
    top_k: int = 10,
):

    query_tokens = tokenize(query)

    scores = bm25.get_scores(
        query_tokens
    )

    ranked_indexes = sorted(
        range(len(scores)),
        key=lambda i: scores[i],
        reverse=True,
    )[:top_k]

    return [
        {
            "rank": rank + 1,
            "score": float(
                scores[index]
            ),
            "chunk_id": chunks[index][
                "chunk_id"
            ],
            "text": chunks[index][
                "text"
            ],
            "metadata": chunks[index][
                "metadata"
            ],
        }
        for rank, index in enumerate(
            ranked_indexes
        )
    ]