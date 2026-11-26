# Retrieval evaluation

42 answerable questions · vector backend: `local` · relevance judged at (document, page) level.

| Method | Hit@1 | Hit@3 | Hit@5 | MRR@5 | nDCG@5 | Recall@5 | Latency (ms) |
|---|---|---|---|---|---|---|---|
| bm25 | 0.762 | 0.881 | 0.881 | 0.810 | 0.535 | 0.821 | 1 |
| vector | 0.786 | 0.905 | 0.929 | 0.851 | 0.583 | 0.871 | 20 |
| hybrid | 0.810 | 0.857 | 0.929 | 0.851 | 0.597 | 0.877 | 20 |
| hybrid+rerank | 0.905 | 0.976 | 0.976 | 0.937 | 0.652 | 0.931 | 996 |

## Production path (hybrid+rerank) misses at k=5

- `train_fully_cancelled` → retrieved irctc_terms_conditions.pdf p6, irctc_eticket_faq.pdf p2, irctc_cancellation_refund_rules.pdf p1
