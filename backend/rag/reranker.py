import sys
from pathlib import Path

from sentence_transformers import CrossEncoder

for candidate in [Path(__file__).resolve().parent, *Path(__file__).resolve().parents]:
    backend_dir = candidate / "backend"
    if backend_dir.exists():
        if str(candidate) not in sys.path:
            sys.path.insert(0, str(candidate))
        if str(backend_dir) not in sys.path:
            sys.path.insert(0, str(backend_dir))
        break

try:
    from backend.rag.hybrid_retriever import hybrid_search
except ModuleNotFoundError:
    from hybrid_retriever import hybrid_search


MODEL_NAME = "cross-encoder/ms-marco-MiniLM-L6-v2"

_reranker = None


def get_reranker():

    global _reranker

    if _reranker is None:

        print(
            "Loading RAG reranker..."
        )

        _reranker = CrossEncoder(
            MODEL_NAME
        )

    return _reranker


def rerank(
    query: str,
    top_k: int = 5,
    candidate_k: int = 15,
):

    candidates = hybrid_search(
        query=query,
        top_k=candidate_k,
    )

    if not candidates:
        return []

    pairs = [
        (
            query,
            result["text"],
        )
        for result in candidates
    ]

    reranker = get_reranker()

    scores = reranker.predict(pairs)

    ranked = sorted(
        zip(candidates, scores),
        key=lambda item: float(item[1]),
        reverse=True,
    )

    results = []

    for rank, (
        result,
        score,
    ) in enumerate(
        ranked[:top_k],
        start=1,
    ):

        results.append(
            {
                **result,
                "rerank_score": float(score),
                "rank": rank,
            }
        )

    return results