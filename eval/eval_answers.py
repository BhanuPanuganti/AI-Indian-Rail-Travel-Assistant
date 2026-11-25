"""
End-to-end RAG answer evaluation with an LLM judge.

For each golden question this runs the production pipeline
(rag_service.answer_question: hybrid retrieval -> rerank -> LLM answer) and then
asks a judge LLM to grade the answer against the retrieved passages.

Metrics
  faithfulness        share of the answer's claims supported by the retrieved passages
  answer relevance    how directly the answer addresses the question (0-1)
  key-fact recall     share of the golden key facts present in the answer
  citation precision  share of cited (document, page) pairs that are golden pages
  correct refusals    unanswerable questions the system declined to answer
  false refusals      answerable questions the system wrongly declined
  latency p50 / p95, LLM calls, tokens and Groq fallbacks

Each question costs two LLM calls (answer + judge). The judge is the same
Gemini model as the generator, so treat scores as indicative rather than
absolute; the retrieval eval (no LLM) is the objective baseline.

Usage:
    python -m eval.eval_answers                  # in-memory vector index
    python -m eval.eval_answers --vector qdrant  # live Qdrant collection
    python -m eval.eval_answers --limit 5 --delay 0
"""

import argparse
import time

from pydantic import BaseModel, Field

from eval.common import DATASETS_DIR, load_jsonl, markdown_table, mean, percentile, save_results, with_retry

from backend.llm_service import LLM_STATS, llm, reset_llm_stats
from backend.rag.rag_service import answer_question


class Judgement(BaseModel):
    faithfulness: float = Field(
        description="Fraction (0-1) of factual claims in the answer that are supported by the passages. 1.0 if the answer makes no claims."
    )
    answer_relevance: float = Field(
        description="0-1: how directly and completely the answer addresses the question."
    )
    key_facts_covered: list[bool] = Field(
        description="For each key fact, in the given order, whether the answer states it (paraphrase is fine)."
    )
    is_refusal: bool = Field(
        description="True if the answer says the documents do not contain the information instead of answering."
    )


JUDGE_PROMPT = """You are grading an Indian Railways assistant's answer.

QUESTION:
{question}

RETRIEVED PASSAGES (the only information the assistant was allowed to use):
{contexts}

ASSISTANT ANSWER:
{answer}

KEY FACTS (in order):
{key_facts}

Grade strictly. A claim is supported only if the passages state it. Return one
boolean per key fact, in the same order. If there are no key facts, return an
empty list."""


judge_llm = llm.with_structured_output(Judgement)


def judge(question: str, answer: str, contexts: list[str], key_facts: list[str]) -> Judgement:
    prompt = JUDGE_PROMPT.format(
        question=question,
        contexts="\n\n---\n\n".join(contexts) or "(none)",
        answer=answer,
        key_facts="\n".join(f"{i + 1}. {fact}" for i, fact in enumerate(key_facts)) or "(none)",
    )
    return judge_llm.invoke(prompt)


def citation_precision(sources: list[dict], relevant: set) -> float | None:
    cited = {(s["source"], int(s["page"])) for s in sources}
    if not cited:
        return None
    return len(cited & relevant) / len(cited)


def summarise(rows: list[dict], latencies: list[float], stats: dict) -> dict:
    answerable = [r for r in rows if r["answerable"] and not r["error"]]
    unanswerable = [r for r in rows if not r["answerable"] and not r["error"]]

    facts_total = sum(len(r["key_facts_covered"]) for r in answerable)
    facts_hit = sum(sum(r["key_facts_covered"]) for r in answerable)
    citation = [r["citation_precision"] for r in answerable if r["citation_precision"] is not None]

    return {
        "n_questions": len(rows),
        "n_errors": sum(1 for r in rows if r["error"]),
        "faithfulness": mean([r["faithfulness"] for r in answerable]),
        "answer_relevance": mean([r["answer_relevance"] for r in answerable]),
        "key_fact_recall": facts_hit / facts_total if facts_total else float("nan"),
        "citation_precision": mean(citation),
        "correct_refusal_rate": mean([1.0 if r["is_refusal"] else 0.0 for r in unanswerable]),
        "false_refusal_rate": mean([1.0 if r["is_refusal"] else 0.0 for r in answerable]),
        "latency_p50_s": percentile(latencies, 50),
        "latency_p95_s": percentile(latencies, 95),
        "llm_calls": stats["calls"],
        "llm_fallbacks": stats["fallbacks"],
        "llm_errors": stats["errors"],
        "input_tokens": stats["input_tokens"],
        "output_tokens": stats["output_tokens"],
    }


def render_markdown(summary: dict, rows: list[dict]) -> str:
    table = markdown_table(
        ["Metric", "Value"],
        [
            ["Faithfulness", summary["faithfulness"]],
            ["Answer relevance", summary["answer_relevance"]],
            ["Key-fact recall", summary["key_fact_recall"]],
            ["Citation precision", summary["citation_precision"]],
            ["Correct refusals (unanswerable)", summary["correct_refusal_rate"]],
            ["False refusals (answerable)", summary["false_refusal_rate"]],
            ["Answer latency p50 (s)", summary["latency_p50_s"]],
            ["Answer latency p95 (s)", summary["latency_p95_s"]],
            ["LLM calls (answer + judge)", summary["llm_calls"]],
            ["Groq fallbacks", summary["llm_fallbacks"]],
            ["Errors", summary["n_errors"]],
        ],
    )

    weak = [
        r for r in rows
        if not r["error"] and r["answerable"]
        and (r["faithfulness"] < 0.8 or (r["key_facts_covered"] and not all(r["key_facts_covered"])))
    ]
    weak_lines = "\n".join(
        f"- `{r['id']}` faithfulness {r['faithfulness']:.2f}, "
        f"key facts {sum(r['key_facts_covered'])}/{len(r['key_facts_covered'])}"
        for r in weak
    ) or "- none"

    return (
        f"# RAG answer evaluation (LLM judge)\n\n"
        f"{summary['n_questions']} questions · judge: same Gemini model as the generator "
        f"(scores are indicative).\n\n{table}\n\n"
        f"## Answers to review (faithfulness < 0.8 or missing key facts)\n\n{weak_lines}\n"
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--limit", type=int, default=None, help="evaluate only the first N questions")
    parser.add_argument("--delay", type=float, default=8.0, help="seconds to wait between questions (2 LLM calls each)")
    parser.add_argument(
        "--vector",
        choices=["local", "qdrant"],
        default="local",
        help="vector search backend: in-memory index from chunks.jsonl (default) or the live Qdrant collection",
    )
    args = parser.parse_args()

    if args.vector == "local":
        # Swap the Qdrant call inside the hybrid retriever for an equivalent
        # in-memory index; everything else is the production pipeline.
        import backend.rag.hybrid_retriever as hybrid_retriever
        from eval.common import load_chunks
        from eval.eval_retrieval import LocalVectorIndex

        vector_index = LocalVectorIndex(load_chunks())
        hybrid_retriever.retrieve = lambda query, top_k=5: vector_index.search(query, top_k)

    dataset = load_jsonl(DATASETS_DIR / "rag_golden.jsonl")[: args.limit]
    reset_llm_stats()

    rows, latencies = [], []
    for index, item in enumerate(dataset, start=1):
        answerable = item.get("answerable", True)
        relevant = {(s, p) for s, p in item["relevant"]}
        row = {"id": item["id"], "question": item["question"], "answerable": answerable, "error": None}

        try:
            start = time.perf_counter()
            result = with_retry(answer_question, item["question"])
            latencies.append(time.perf_counter() - start)

            verdict = with_retry(judge, item["question"], result["answer"], result.get("contexts", []), item["key_facts"])
            covered = list(verdict.key_facts_covered)[: len(item["key_facts"])]
            covered += [False] * (len(item["key_facts"]) - len(covered))

            row.update(
                {
                    "answer": result["answer"],
                    "sources": result["sources"],
                    "faithfulness": max(0.0, min(1.0, verdict.faithfulness)),
                    "answer_relevance": max(0.0, min(1.0, verdict.answer_relevance)),
                    "key_facts_covered": covered,
                    "is_refusal": verdict.is_refusal,
                    "citation_precision": citation_precision(result["sources"], relevant) if answerable else None,
                }
            )
            status = "refused" if verdict.is_refusal else f"faith {row['faithfulness']:.2f}"
        except Exception as exc:
            row["error"] = str(exc)[:300]
            status = "ERROR"

        rows.append(row)
        print(f"[{index}/{len(dataset)}] {status:<12} {item['question'][:60]}")

        if args.delay and index < len(dataset):
            time.sleep(args.delay)

    summary = summarise(rows, latencies, dict(LLM_STATS))
    markdown = render_markdown(summary, rows)
    json_path, md_path = save_results("answers", {"summary": summary, "rows": rows}, markdown)

    print("\n" + markdown)
    print(f"Saved {json_path} and {md_path}")


if __name__ == "__main__":
    main()
