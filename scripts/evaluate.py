"""Small smoke-evaluation harness for retrieval, citations, and latency.

Edit evals/questions.json so expected_source matches an uploaded PDF filename.
The keyword check is intentionally simple; manually inspect generated answers as well.
"""
import json
import statistics
import time
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parents[1]
DATASET_PATH = ROOT / "evals" / "questions.json"
REPORT_PATH = ROOT / "evals" / "latest_report.json"
API_URL = "http://localhost:8000"


def main() -> None:
    cases = json.loads(DATASET_PATH.read_text(encoding="utf-8"))
    rows = []
    for case in cases:
        started = time.perf_counter()
        response = requests.post(
            f"{API_URL}/ask",
            json={"question": case["question"]},
            timeout=180,
        )
        response.raise_for_status()
        result = response.json()
        elapsed_ms = round((time.perf_counter() - started) * 1000)
        cited_sources = {citation["source"] for citation in result.get("citations", [])}
        expected_source = case.get("expected_source", "")
        source_hit = bool(expected_source) and expected_source in cited_sources
        expected_keywords = [word.lower() for word in case.get("expected_answer_keywords", [])]
        answer_lower = result.get("answer", "").lower()
        keyword_coverage = (
            sum(word in answer_lower for word in expected_keywords) / len(expected_keywords)
            if expected_keywords else None
        )
        rows.append({
            "question": case["question"],
            "expected_source": expected_source,
            "source_hit": source_hit,
            "citation_count": len(result.get("citations", [])),
            "keyword_coverage": keyword_coverage,
            "latency_ms": elapsed_ms,
            "answer": result.get("answer", ""),
        })
        print(f"[{ 'PASS' if source_hit else 'CHECK' }] {case['question']} ({elapsed_ms} ms)")

    source_scores = [row["source_hit"] for row in rows if row["expected_source"]]
    keyword_scores = [row["keyword_coverage"] for row in rows if row["keyword_coverage"] is not None]
    report = {
        "num_questions": len(rows),
        "expected_source_hit_rate": (sum(source_scores) / len(source_scores)) if source_scores else None,
        "mean_keyword_coverage": (statistics.mean(keyword_scores)) if keyword_scores else None,
        "mean_latency_ms": round(statistics.mean(row["latency_ms"] for row in rows), 2) if rows else None,
        "results": rows,
        "note": "Keyword coverage is only a rough smoke check, not a faithfulness or correctness metric.",
    }
    REPORT_PATH.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\nSaved report to {REPORT_PATH}")
    print(json.dumps({key: value for key, value in report.items() if key != "results"}, indent=2))


if __name__ == "__main__":
    main()
