"""
Download the RAG embedding and reranker models into MODEL_CACHE_DIR.

Run at Docker build time so the server starts without network downloads:
    python -m backend.rag.preload_models
"""

from backend.rag.reranker import get_reranker, score
from backend.rag.retriever import MODEL_CACHE_DIR, embed_texts


def main():
    print(f"Caching models in {MODEL_CACHE_DIR}")
    embed_texts(["warm up"])
    get_reranker()
    score("warm up", ["warm up"])
    print("Models cached.")


if __name__ == "__main__":
    main()
