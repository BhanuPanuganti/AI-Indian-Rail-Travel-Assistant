"""
Intent-router evaluation.

Runs the production intent classifier (intent_service.extract_user_intent) over
a labelled set of messages, including multi-turn follow-ups such as
"which one is faster?", and reports accuracy, macro-F1, per-intent
precision/recall/F1, a confusion matrix, ranking-preference accuracy and
latency.

This makes one LLM call per example. Use --delay to stay under your Gemini
rate limit (the default of 4 s keeps a run under 15 requests/minute).

Usage:
    python -m eval.eval_intent
    python -m eval.eval_intent --limit 10 --delay 0
"""

import argparse
import time

from langchain_core.messages import AIMessage, HumanMessage

from eval.common import DATASETS_DIR, load_jsonl, markdown_table, mean, percentile, save_results, with_retry
from eval.metrics import classification_report

from backend.intent_service import extract_user_intent
from backend.llm_service import LLM_STATS, reset_llm_stats
from backend.train_ranker import normalize_preference


def build_messages(item: dict) -> list:
    messages = []
    for role, text in item.get("history", []):
        messages.append(HumanMessage(content=text) if role == "human" else AIMessage(content=text))
    messages.append(HumanMessage(content=item["message"]))
    return messages


def render_markdown(report: dict, rows: list[dict], preference_acc, latencies, stats) -> str:
    labels = report["labels"]

    per_label = markdown_table(
        ["Intent", "Precision", "Recall", "F1", "Support"],
        [
            [label, m["precision"], m["recall"], m["f1"], m["support"]]
            for label, m in report["per_label"].items()
        ],
    )
    confusion = markdown_table(
        ["gold \\ predicted", *labels],
        [[label, *counts] for label, counts in zip(labels, report["confusion"])],
    )
    errors = "\n".join(
        f"- \"{r['message']}\" → expected `{r['gold']}`, got `{r['predicted']}`"
        for r in rows
        if r["gold"] != r["predicted"]
    ) or "- none"

    pref_line = (
        f"{preference_acc:.3f}" if preference_acc is not None else "n/a"
    )

    return (
        f"# Intent router evaluation\n\n"
        f"{len(rows)} examples · accuracy **{report['accuracy']:.3f}** · "
        f"macro-F1 **{report['macro_f1']:.3f}** · preference accuracy {pref_line}\n\n"
        f"Latency p50 {percentile(latencies, 50):.2f}s · p95 {percentile(latencies, 95):.2f}s · "
        f"LLM calls {stats['calls']} · Groq fallbacks {stats['fallbacks']} · errors {stats['errors']}\n\n"
        f"## Per intent\n\n{per_label}\n\n"
        f"## Confusion matrix\n\n{confusion}\n\n"
        f"## Misclassified\n\n{errors}\n"
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--limit", type=int, default=None, help="evaluate only the first N examples")
    parser.add_argument("--delay", type=float, default=4.0, help="seconds to wait between LLM calls")
    args = parser.parse_args()

    dataset = load_jsonl(DATASETS_DIR / "intent_golden.jsonl")[: args.limit]
    reset_llm_stats()

    rows, latencies = [], []
    for index, item in enumerate(dataset, start=1):
        start = time.perf_counter()
        try:
            result = with_retry(extract_user_intent, build_messages(item))
            predicted = result.intent.value
            preference = normalize_preference(result.preference) if result.preference else None
            error = None
        except Exception as exc:
            predicted, preference, error = "ERROR", None, str(exc)[:200]
        latency = time.perf_counter() - start
        latencies.append(latency)

        rows.append(
            {
                "message": item["message"],
                "gold": item["intent"],
                "predicted": predicted,
                "gold_preference": item.get("preference"),
                "predicted_preference": preference,
                "latency_s": latency,
                "error": error,
            }
        )
        mark = "✓" if predicted == item["intent"] else "✗"
        print(f"[{index}/{len(dataset)}] {mark} {predicted:<15} {item['message'][:60]}")

        if args.delay and index < len(dataset):
            time.sleep(args.delay)

    report = classification_report([r["gold"] for r in rows], [r["predicted"] for r in rows])

    with_pref = [r for r in rows if r["gold_preference"]]
    preference_acc = (
        mean([1.0 if r["predicted_preference"] == r["gold_preference"] else 0.0 for r in with_pref])
        if with_pref
        else None
    )

    stats = dict(LLM_STATS)
    markdown = render_markdown(report, rows, preference_acc, latencies, stats)
    json_path, md_path = save_results(
        "intent",
        {
            "n_examples": len(rows),
            "report": report,
            "preference_accuracy": preference_acc,
            "latency_p50_s": percentile(latencies, 50),
            "latency_p95_s": percentile(latencies, 95),
            "llm_stats": stats,
            "rows": rows,
        },
        markdown,
    )

    print("\n" + markdown)
    print(f"Saved {json_path} and {md_path}")


if __name__ == "__main__":
    main()
