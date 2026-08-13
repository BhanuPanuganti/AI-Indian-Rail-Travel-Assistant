# RAG Pipeline — Technical Reference

This document gives a deeper technical explanation of the Retrieval-Augmented Generation (RAG) pipeline used to answer railway policy questions in AI Indian Rail Travel Assistant.

---

## Overview

When a user asks a question about railway rules, policies, Tatkal, cancellations, refunds, luggage, or concessions, the request is routed to the **RAG Agent** (`tool_graph.py`). The agent calls the `railway_knowledge_search` tool, which triggers the full RAG pipeline defined in the `backend/rag/` directory.

```
User question
  → railway_knowledge_search tool (tools.py)
    → answer_question() (rag_service.py)
      → rerank() (reranker.py)
        → hybrid_search() (hybrid_retriever.py)
          → retrieve() (retriever.py)       ← Qdrant dense vector search
          → bm25_search() (bm25_store.py)   ← BM25 keyword search
        ← Reciprocal Rank Fusion (top 15 candidates)
      ← cross-encoder reranking (top 5)
    ← context building
    → LLM synthesis (Gemini / Groq fallback)
  ← answer + source citations
```

---

## Knowledge Base

### Source Documents

Six official PDF documents are stored in `backend/rag/documents/`:

| File | Topic tag | Content |
|---|---|---|
| `irctc_eticket_faq.pdf` | `e_ticket` | e-Ticket booking, printing, and related FAQs |
| `irctc_terms_conditions.pdf` | `booking_rules` | IRCTC terms and conditions for online booking |
| `irctc_cancellation_refund_rules.pdf` | `cancellation_refund` | Cancellation fees, TDR, and refund rules |
| `indian_railways_luggage_rules.pdf` | `luggage` | Luggage allowances and excess luggage charges |
| `indian_railways_concession_rules.pdf` | `concessions` | Senior citizen, disability, and other concession categories |
| `irctc_tatkal_faq.pdf` | `tatkal` | Tatkal booking windows, charges, and rules |

### Document Ingestion (`ingest.py`)

The ingestion script is run **once** during setup (or whenever documents change):

```bash
python backend/rag/ingest.py
```

Steps performed:

1. Loads each PDF using `PyPDFLoader` (LangChain Community).
2. Cleans extracted text (collapses whitespace).
3. Attaches metadata to each page: `source` (filename), `topic`, `authority`, `document_type`.
4. Splits pages into overlapping chunks using `RecursiveCharacterTextSplitter`:
   - `chunk_size = 900` characters
   - `chunk_overlap = 150` characters
   - Separators in priority order: `\n\n`, `\n`, `. `, `? `, `! `, `; `, ` `
5. Writes each chunk as a JSON Lines record to `backend/rag/chunks.jsonl`.

`chunks.jsonl` is committed to the repository so the ingestion step does not need to re-run unless source PDFs change.

### Vector Store Build (`vector_store.py`)

Run **once** on every fresh clone:

```bash
python backend/rag/vector_store.py
```

1. Loads all chunks from `chunks.jsonl`.
2. Encodes each chunk with **`BAAI/bge-small-en-v1.5`** (384 dimensions, normalised).
3. Creates a Qdrant collection `indian_railway_knowledge` with cosine distance.
4. Upserts all vectors with payloads (`chunk_id`, `text`, `metadata`).
5. Writes to `backend/rag/qdrant_data/` (local on-disk Qdrant).

---

## Retrieval

### Dense Vector Search (`retriever.py`)

At query time:

1. The user's question is encoded with the same `BAAI/bge-small-en-v1.5` model (loaded once and cached globally).
2. Qdrant performs a cosine similarity search against all stored vectors.
3. Top `k` results are returned with scores, chunk IDs, text, and metadata.

**Why vector search?** It captures semantic meaning. A question like "What happens if I cancel my ticket?" will match chunks about "refund on cancellation" even without exact word overlap.

### BM25 Keyword Search (`bm25_store.py`)

At module import time, all chunks are loaded and a `BM25Okapi` index is built in memory.

At query time, the query is tokenised with a word-boundary regex and BM25 scores are computed for every chunk. The top `k` by score are returned.

**Why BM25?** It captures exact keyword matches. Domain-specific terms like "Tatkal", "TDR", or specific fee amounts ("Rs. 200") are matched precisely even when the vector model may not have strong signal on them.

### Hybrid Retrieval — Reciprocal Rank Fusion (`hybrid_retriever.py`)

Both result lists are merged using **Reciprocal Rank Fusion (RRF)**:

```
rrf_score(doc) += 1 / (k + rank)
```

where `k = 60` (standard smoothing constant) and `rank` is the document's 1-based position in a result list.

A document that ranks highly in **both** lists accumulates a higher RRF score than one ranked highly in only one. The merged list is sorted by RRF score descending, and the top **15 candidates** proceed to reranking.

---

## Reranking (`reranker.py`)

The 15 hybrid candidates are rescored by **`cross-encoder/ms-marco-MiniLM-L6-v2`**.

Unlike bi-encoder retrieval (which encodes query and document separately and compares their embeddings), a cross-encoder takes the query and a candidate document as a **joint input** and outputs a single relevance score. This is more precise because the model attends to the full relationship between the two texts.

Steps:

1. Form 15 `(query, chunk_text)` pairs.
2. `CrossEncoder.predict()` scores all pairs in one batch.
3. Sort descending by score; keep the top **5**.

**Why reranking?** Hybrid retrieval broadens recall (15 candidates). Reranking sharpens precision (top 5 for LLM context), discarding lower-relevance chunks and reducing noise in the prompt.

---

## Answer Synthesis (`rag_service.py`)

The 5 reranked chunks are assembled into a structured context block:

```
SOURCE: irctc_tatkal_faq.pdf
PAGE: 3
TOPIC: tatkal
CONTENT: <chunk text>
```

This context plus the user's question is sent to the LLM under a strict system prompt that:

- Prohibits inventing facts not in the documents.
- Requires clearly saying when the documents do not contain enough information.
- Requires preserving specific conditions, limits, deadlines, and amounts.
- Prohibits HTML tags in the response.

**Source citations** are appended to the answer:

```
📚 Sources
• irctc_tatkal_faq.pdf — Page 3
• irctc_tatkal_faq.pdf — Page 5
```

---

## Startup Preloading

On backend startup, the `lifespan` hook in `main.py` eagerly loads both models:

```python
get_embedding_model()   # BAAI/bge-small-en-v1.5
get_reranker()          # cross-encoder/ms-marco-MiniLM-L6-v2
```

This adds a few seconds to cold start but eliminates first-request latency.

---

## Production Deployment Note

The Qdrant vector store (`backend/rag/qdrant_data/`) is excluded from git and from the Docker build (`.gitignore` and `.dockerignore`). On Railway, the container starts without a Qdrant collection.

**Current workaround options:**

1. Run `python backend/rag/vector_store.py` as part of the container entrypoint (adds startup time but is self-contained).
2. Attach a Railway volume to persist `qdrant_data/` across deploys.
3. Migrate to a hosted Qdrant Cloud instance and update `retriever.py` to connect via URL rather than the local file path.
