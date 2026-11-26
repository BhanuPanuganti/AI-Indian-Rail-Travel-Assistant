import os
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

from fastembed.common.model_description import ModelSource
from fastembed.rerank.cross_encoder import TextCrossEncoder

try:
    from backend.rag.hybrid_retriever import hybrid_search
    from backend.rag.retriever import MODEL_CACHE_DIR, MODEL_CPU_MEM_ARENA, MODEL_THREADS
except ModuleNotFoundError:
    from hybrid_retriever import hybrid_search
    from retriever import MODEL_CACHE_DIR, MODEL_CPU_MEM_ARENA, MODEL_THREADS


# ONNX exports of cross-encoder/ms-marco-MiniLM-L-6-v2 (same model).
# "quantized" (default) is the int8 export from the same repository: about a
# quarter of the size and memory, used so the backend fits a 512 MB instance.
# "full" is the fp32 export.
_HF_REPO = "Xenova/ms-marco-MiniLM-L-6-v2"
_QUANTIZED_NAME = "local/ms-marco-MiniLM-L-6-v2-int8"

RERANKER_VARIANT = os.getenv("RERANKER_VARIANT", "quantized")
MODEL_NAME = _HF_REPO if RERANKER_VARIANT == "full" else _QUANTIZED_NAME

# Passages scored per ONNX run. Small batches cap the temporary memory spike
# of attention over 15 long passages; total work is the same.
RERANK_BATCH_SIZE = int(os.getenv("RERANK_BATCH_SIZE", "4"))

_reranker = None


def _register_quantized_model() -> None:
    known = {m["model"].lower() for m in TextCrossEncoder.list_supported_models()}
    if _QUANTIZED_NAME.lower() in known:
        return
    TextCrossEncoder.add_custom_model(
        model=_QUANTIZED_NAME,
        sources=ModelSource(hf=_HF_REPO),
        model_file="onnx/model_quantized.onnx",
        description="int8 quantized ms-marco-MiniLM-L-6-v2 cross-encoder",
        license="apache-2.0",
        size_in_gb=0.03,
    )


def get_reranker():

    global _reranker

    if _reranker is None:

        print(
            f"Loading RAG reranker ({RERANKER_VARIANT})..."
        )

        if MODEL_NAME == _QUANTIZED_NAME:
            _register_quantized_model()

        _reranker = TextCrossEncoder(
            model_name=MODEL_NAME,
            cache_dir=MODEL_CACHE_DIR,
            threads=MODEL_THREADS,
            enable_cpu_mem_arena=MODEL_CPU_MEM_ARENA,
        )

    return _reranker


def score(query: str, texts: list[str]) -> list[float]:
    """Cross-encoder relevance score for each text against the query."""

    if not texts:
        return []

    return [
        float(s)
        for s in get_reranker().rerank(query, texts, batch_size=RERANK_BATCH_SIZE)
    ]


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

    scores = score(
        query,
        [result["text"] for result in candidates],
    )

    ranked = sorted(
        zip(candidates, scores),
        key=lambda item: item[1],
        reverse=True,
    )

    return [
        {
            **result,
            "rerank_score": result_score,
            "rank": rank,
        }
        for rank, (result, result_score) in enumerate(
            ranked[:top_k],
            start=1,
        )
    ]
