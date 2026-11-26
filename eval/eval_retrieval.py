"""
Retrieval evaluation with an ablation over the RAG pipeline's stages.

Compares, on the golden question set:
  bm25            keyword search only
  vector          dense search only (bge-small-en-v1.5 via fastembed/ONNX, cosine)
  hybrid          vector + BM25 fused with Reciprocal Rank Fusion
  hybrid+rerank   hybrid candidates re-scored by the cross-encoder (production path)

Metrics: Hit@k, MRR, nDCG@k and page Recall@k for k in (1, 3, 5), plus mean
query latency. No LLM calls are made, so this run is free.

By default the vector index is built in memory from chunks.jsonl with the same
embedding model and cosine distance as the Qdrant collection, so results are
reproducible offline. Pass --vector qdrant to query the live Qdrant collection.

Usage:
    python -m eval.eval_retrieval
    python -m eval.eval_retrieval --vector qdrant
"""

import argparse
import time

import numpy as np

from eval.common import (
    DATASETS_DIR,
    load_chunks,
    load_jsonl,
    markdown_table,
    mean,
    page_key,
    save_results,
    validate_rag_labels,
)
from eval.metrics import hit_at_k, ndcg_at_k, page_recall_at_k, reciprocal_rank

from backend.rag.bm25_store import bm25_search
from backend.rag.hybrid_retriever import reciprocal_rank_fusion
from backend.rag.reranker import score
from backend.rag.retriever import embed_texts, retrieve

K_VALUES = (1, 3, 5)
CANDIDATE_K = 15   # matches rag_service.answer_question -> rerank(candidate_k=15)
FINAL_K = 5        # matches rerank(top_k=5)


class LocalVectorIndex:
    """In-memory cosine index over chunks.jsonl, equivalent to the Qdrant collection."""

    def __init__(self, chunks: list[dict]):
        self.chunks = chunks
        self.matrix = np.asarray(embed_texts([c["text"] for c in chunks]))

    def search(self, query: str, top_k: int) -> list[dict]:
        query_vector = np.asarray(embed_texts([query]))[0]
        scores = self.matrix @ query_vector
        order = np.argsort(-scores)[:top_k]
        return [
            {
                "score": float(scores[i]),
                "chunk_id": self.chunks[i]["chunk_id"],
                "text": self.chunks[i]["text"],
                "metadata": self.chunks[i]["metadata"],
            }
            for i in order
        ]


def build_methods(vector_search):
    def bm25(query):
        return bm25_search(query, top_k=CANDIDATE_K)

    def vector(query):
        return vector_search(query, CANDIDATE_K)

    def hybrid(query):
        fused = reciprocal_rank_fusion([vector(query), bm25(query)])
        return fused[:CANDIDATE_K]

    def hybrid_rerank(query):
        candidates = hybrid(query)
        scores = score(query, [c["text"] for c in candidates])
        order = sorted(range(len(candidates)), key=lambda i: float(scores[i]), reverse=True)
        return [candidates[i] for i in order[:FINAL_K]]

    return {
        "bm25": bm25,
        "vector": vector,
        "hybrid": hybrid,
        "hybrid+rerank": hybrid_rerank,
    }


def evaluate(methods: dict, dataset: list[dict], chunks: list[dict]) -> dict:
    corpus_pages = [page_key(c["metadata"]) for c in chunks]
    results = {}

    for name, search in methods.items():
        search("warm up")  # exclude one-time model/kernel warm-up from latency

        rows = []
        for item in dataset:
            relevant_pages = {(s, p) for s, p in item["relevant"]}
            total_relevant = sum(1 for page in corpus_pages if page in relevant_pages)

            start = time.perf_counter()
            hits = search(item["question"])
            latency_ms = (time.perf_counter() - start) * 1000

            retrieved_pages = [page_key(h["metadata"]) for h in hits]
            relevance = [page in relevant_pages for page in retrieved_pages]

            row = {
                "id": item["id"],
                "latency_ms": latency_ms,
                "mrr": reciprocal_rank(relevance[:FINAL_K]),
                "retrieved": [f"{s} p{p}" for s, p in retrieved_pages[:FINAL_K]],
            }
            for k in K_VALUES:
                row[f"hit@{k}"] = hit_at_k(relevance, k)
                row[f"ndcg@{k}"] = ndcg_at_k(relevance, k, total_relevant)
                row[f"recall@{k}"] = page_recall_at_k(retrieved_pages, relevant_pages, k)
            rows.append(row)

        summary = {
            metric: mean([r[metric] for r in rows])
            for metric in rows[0]
            if metric not in ("id", "retrieved")
        }
        results[name] = {"summary": summary, "per_question": rows}

    return results


def render_markdown(results: dict, n_questions: int, vector_backend: str) -> str:
    headers = ["Method", "Hit@1", "Hit@3", "Hit@5", "MRR@5", "nDCG@5", "Recall@5", "Latency (ms)"]
    rows = [
        [
            name,
            r["summary"]["hit@1"],
            r["summary"]["hit@3"],
            r["summary"]["hit@5"],
            r["summary"]["mrr"],
            r["summary"]["ndcg@5"],
            r["summary"]["recall@5"],
            f"{r['summary']['latency_ms']:.0f}",
        ]
        for name, r in results.items()
    ]

    production = results["hybrid+rerank"]["per_question"]
    misses = [r for r in production if r["hit@5"] == 0]
    miss_lines = "\n".join(
        f"- `{r['id']}` → retrieved {', '.join(r['retrieved'][:3])}" for r in misses
    ) or "- none"

    return (
        f"# Retrieval evaluation\n\n"
        f"{n_questions} answerable questions · vector backend: `{vector_backend}` · "
        f"relevance judged at (document, page) level.\n\n"
        f"{markdown_table(headers, rows)}\n\n"
        f"## Production path (hybrid+rerank) misses at k=5\n\n{miss_lines}\n"
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--vector", choices=["local", "qdrant"], default="local")
    args = parser.parse_args()

    chunks = load_chunks()
    dataset = load_jsonl(DATASETS_DIR / "rag_golden.jsonl")
    validate_rag_labels(dataset, chunks)
    dataset = [item for item in dataset if item.get("answerable", True)]

    if args.vector == "local":
        print(f"Embedding {len(chunks)} chunks for the local vector index...")
        index = LocalVectorIndex(chunks)
        vector_search = index.search
    else:
        def vector_search(query, top_k):
            return retrieve(query, top_k=top_k)

    results = evaluate(build_methods(vector_search), dataset, chunks)
    markdown = render_markdown(results, len(dataset), args.vector)

    json_path, md_path = save_results(
        "retrieval",
        {"vector_backend": args.vector, "n_questions": len(dataset), "methods": results},
        markdown,
    )

    print("\n" + markdown)
    print(f"Saved {json_path} and {md_path}")


if __name__ == "__main__":
    main()
