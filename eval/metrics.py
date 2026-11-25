"""
Ranking and classification metrics.

Retrieval relevance is judged at page level: a retrieved chunk is relevant
when its (source, page) is in the question's golden set. That keeps the labels
valid even if the documents are re-chunked.
"""

import math
from collections import Counter


def hit_at_k(relevance: list[bool], k: int) -> float:
    """1.0 if any of the top-k results is relevant."""
    return 1.0 if any(relevance[:k]) else 0.0


def reciprocal_rank(relevance: list[bool]) -> float:
    """1 / rank of the first relevant result (0 if none)."""
    for rank, is_relevant in enumerate(relevance, start=1):
        if is_relevant:
            return 1.0 / rank
    return 0.0


def page_recall_at_k(retrieved_pages: list[tuple], relevant_pages: set, k: int) -> float:
    """Fraction of the golden pages that appear in the top-k results."""
    if not relevant_pages:
        return 0.0
    found = set(retrieved_pages[:k]) & relevant_pages
    return len(found) / len(relevant_pages)


def ndcg_at_k(relevance: list[bool], k: int, total_relevant: int) -> float:
    """
    Binary-relevance nDCG@k. `total_relevant` is the number of relevant
    chunks in the whole corpus, used to build the ideal ranking.
    """
    dcg = sum(
        1.0 / math.log2(rank + 1)
        for rank, is_relevant in enumerate(relevance[:k], start=1)
        if is_relevant
    )
    ideal_hits = min(k, total_relevant)
    idcg = sum(1.0 / math.log2(rank + 1) for rank in range(1, ideal_hits + 1))
    return dcg / idcg if idcg else 0.0


def classification_report(gold: list[str], predicted: list[str]) -> dict:
    """Accuracy, per-label precision/recall/F1, macro-F1 and a confusion matrix."""

    labels = sorted(set(gold) | set(predicted))
    pairs = Counter(zip(gold, predicted))

    per_label = {}
    for label in labels:
        tp = pairs[(label, label)]
        fp = sum(count for (g, p), count in pairs.items() if p == label and g != label)
        fn = sum(count for (g, p), count in pairs.items() if g == label and p != label)
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        per_label[label] = {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "support": sum(1 for g in gold if g == label),
        }

    gold_labels = sorted(set(gold))
    macro_f1 = sum(per_label[label]["f1"] for label in gold_labels) / len(gold_labels)

    return {
        "accuracy": sum(1 for g, p in zip(gold, predicted) if g == p) / len(gold),
        "macro_f1": macro_f1,
        "per_label": per_label,
        "labels": labels,
        "confusion": [[pairs[(g, p)] for p in labels] for g in labels],
    }
