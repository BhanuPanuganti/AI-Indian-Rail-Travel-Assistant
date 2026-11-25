"""Shared helpers for the offline evaluation scripts."""

import json
import statistics
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
EVAL_DIR = ROOT / "eval"
DATASETS_DIR = EVAL_DIR / "datasets"
RESULTS_DIR = EVAL_DIR / "results"
CHUNKS_FILE = ROOT / "backend" / "rag" / "chunks.jsonl"

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# The backend prints emoji status lines; make that safe on Windows consoles.
for stream in (sys.stdout, sys.stderr):
    try:
        stream.reconfigure(encoding="utf-8")
    except Exception:
        pass


def load_jsonl(path: Path) -> list[dict]:
    with open(path, encoding="utf-8") as file:
        return [json.loads(line) for line in file if line.strip()]


def load_chunks() -> list[dict]:
    return load_jsonl(CHUNKS_FILE)


def page_key(metadata: dict) -> tuple[str, int]:
    return (metadata["source"], int(metadata["page"]))


def validate_rag_labels(dataset: list[dict], chunks: list[dict]) -> None:
    """Fail fast if a golden label points at a page that has no chunks."""

    known_pages = {page_key(chunk["metadata"]) for chunk in chunks}
    problems = [
        f"{item['id']}: {source} p{page}"
        for item in dataset
        for source, page in item.get("relevant", [])
        if (source, page) not in known_pages
    ]
    if problems:
        raise ValueError("Golden labels reference unknown pages:\n" + "\n".join(problems))


def with_retry(fn, *args, attempts: int = 3, backoff_s: float = 10.0, **kwargs):
    """Call fn, retrying transient provider errors (timeouts, 5xx, rate limits)."""

    transient = ("DEADLINE_EXCEEDED", "504", "503", "500", "UNAVAILABLE", "429", "RESOURCE_EXHAUSTED", "timed out")
    for attempt in range(1, attempts + 1):
        try:
            return fn(*args, **kwargs)
        except Exception as exc:
            if attempt == attempts or not any(t.lower() in str(exc).lower() for t in transient):
                raise
            print(f"  transient error, retrying in {backoff_s * attempt:.0f}s: {str(exc)[:80]}")
            import time
            time.sleep(backoff_s * attempt)


def percentile(values: list[float], pct: float) -> float:
    if not values:
        return float("nan")
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, round(pct / 100 * (len(ordered) - 1))))
    return ordered[index]


def mean(values: list[float]) -> float:
    return statistics.fmean(values) if values else float("nan")


def save_results(name: str, payload: dict, markdown: str) -> tuple[Path, Path]:
    """Write <name>.json and <name>.md to eval/results/."""

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    payload = {"generated_at": datetime.now(timezone.utc).isoformat(), **payload}

    json_path = RESULTS_DIR / f"{name}.json"
    md_path = RESULTS_DIR / f"{name}.md"

    json_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    md_path.write_text(markdown, encoding="utf-8")

    return json_path, md_path


def markdown_table(headers: list[str], rows: list[list]) -> str:
    def fmt(value):
        if isinstance(value, float):
            return f"{value:.3f}"
        return str(value)

    lines = [
        "| " + " | ".join(headers) + " |",
        "|" + "|".join("---" for _ in headers) + "|",
    ]
    lines += ["| " + " | ".join(fmt(v) for v in row) + " |" for row in rows]
    return "\n".join(lines)
