import sys
from pathlib import Path

for candidate in [Path(__file__).resolve().parent, *Path(__file__).resolve().parents]:
    backend_dir = candidate / "backend"
    if backend_dir.exists():
        if str(candidate) not in sys.path:
            sys.path.insert(0, str(candidate))
        if str(backend_dir) not in sys.path:
            sys.path.insert(0, str(backend_dir))
        break

try:
    from backend.rag.bm25_store import bm25_search
except ModuleNotFoundError:
    from bm25_store import bm25_search

try:
    from backend.rag.retriever import retrieve
except ModuleNotFoundError:
    from retriever import retrieve


def reciprocal_rank_fusion(
    result_lists: list[list[dict]],
    k: int = 60,
):

    fused = {}

    for results in result_lists:

        for rank, result in enumerate(
            results,
            start=1,
        ):

            chunk_id = result[
                "chunk_id"
            ]

            if chunk_id not in fused:

                fused[chunk_id] = {
                    **result,
                    "rrf_score": 0.0,
                }

            fused[chunk_id][
                "rrf_score"
            ] += 1 / (
                k + rank
            )

    return sorted(
        fused.values(),
        key=lambda x: x["rrf_score"],
        reverse=True,
    )


def hybrid_search(
    query: str,
    top_k: int = 10,
):

    vector_results = retrieve(
        query,
        top_k=top_k,
    )

    keyword_results = bm25_search(
        query,
        top_k=top_k,
    )

    fused_results = reciprocal_rank_fusion(
        [
            vector_results,
            keyword_results,
        ]
    )

    return fused_results[:top_k]