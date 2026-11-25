import math

import pytest

from eval.common import load_chunks, load_jsonl, DATASETS_DIR, validate_rag_labels
from eval.metrics import (
    classification_report,
    hit_at_k,
    ndcg_at_k,
    page_recall_at_k,
    reciprocal_rank,
)


def test_hit_and_reciprocal_rank():
    relevance = [False, True, False]
    assert hit_at_k(relevance, 1) == 0.0
    assert hit_at_k(relevance, 2) == 1.0
    assert reciprocal_rank(relevance) == 0.5
    assert reciprocal_rank([False, False]) == 0.0


def test_ndcg_perfect_and_partial():
    assert ndcg_at_k([True, True, False], 3, total_relevant=2) == pytest.approx(1.0)
    expected = (1 / math.log2(3)) / (1 / math.log2(2))
    assert ndcg_at_k([False, True], 2, total_relevant=1) == pytest.approx(expected)
    assert ndcg_at_k([False, False], 2, total_relevant=0) == 0.0


def test_page_recall():
    retrieved = [("a.pdf", 1), ("b.pdf", 2), ("a.pdf", 1)]
    assert page_recall_at_k(retrieved, {("a.pdf", 1), ("c.pdf", 3)}, 3) == 0.5


def test_classification_report():
    report = classification_report(["X", "X", "Y"], ["X", "Y", "Y"])
    assert report["accuracy"] == pytest.approx(2 / 3)
    assert report["per_label"]["X"]["precision"] == 1.0
    assert report["per_label"]["X"]["recall"] == 0.5
    assert report["confusion"] == [[1, 1], [0, 1]]


def test_golden_rag_labels_point_at_real_pages():
    validate_rag_labels(load_jsonl(DATASETS_DIR / "rag_golden.jsonl"), load_chunks())


def test_golden_intent_labels_are_valid():
    from backend.models import ChatIntent

    valid = {intent.value for intent in ChatIntent}
    for item in load_jsonl(DATASETS_DIR / "intent_golden.jsonl"):
        assert item["intent"] in valid, item
