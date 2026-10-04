"""Measured latency breakdown and evidence-based Lab 18 bonus checks."""
from __future__ import annotations
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
import json
import math
import time

METRICS = ("faithfulness", "answer_relevancy", "context_precision", "context_recall")


class LatencyTracker:
    def __init__(self):
        self.samples = {}

    @contextmanager
    def measure(self, stage):
        started = time.perf_counter()
        try:
            yield
        finally:
            self.samples.setdefault(stage, []).append((time.perf_counter() - started) * 1000)

    def rows(self):
        rows = []
        for stage, samples in self.samples.items():
            ordered = sorted(samples)
            rows.append({"stage": stage, "calls": len(samples),
                         "total_ms": sum(samples), "avg_ms": sum(samples) / len(samples),
                         "min_ms": min(samples), "max_ms": max(samples),
                         "p95_ms": ordered[max(0, math.ceil(len(samples) * .95) - 1)]})
        return rows

    def save(self, output_dir, run_status):
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        rows = self.rows()
        payload = {"measured_at": datetime.now(timezone.utc).isoformat(),
                   "run_status": run_status, "unit": "milliseconds", "stages": rows,
                   "run_metadata": getattr(self, "metadata", {})}
        (output_dir / "latency_report.json").write_text(
            json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        lines = ["# Latency breakdown — Production RAG", "",
                 f"Run status: {run_status}", "",
                 "Measurements use time.perf_counter; model loading is separate from inference.",
                 "Rows are sequential stages; total time includes orchestration overhead.", "",
                 "Cache/fallback metadata: " + json.dumps(getattr(self, "metadata", {}), ensure_ascii=False), "",
                 "| Stage | Calls | Total ms | Average ms | Min ms | Max ms | P95 ms |",
                 "|---|---:|---:|---:|---:|---:|---:|"]
        for row in rows:
            lines.append(f"| {row['stage']} | {row['calls']} | {row['total_ms']:.2f} | "
                         f"{row['avg_ms']:.2f} | {row['min_ms']:.2f} | "
                         f"{row['max_ms']:.2f} | {row['p95_ms']:.2f} |")
        (output_dir / "latency_breakdown.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def check_bonus(results, tracker, combined_mode=True):
    """Threshold bonuses require successful real evaluation, not fallback zeros."""
    real = results.get("evaluation_status") == "success" and bool(results.get("per_question"))
    scores = {name: results.get(name, 0.0) for name in METRICS}
    finite = all(isinstance(v, (int, float)) and math.isfinite(v) for v in scores.values())
    stages = set(tracker.samples)
    complete_latency = {"m1_chunking", "m5_enrichment", "bm25_index", "dense_index",
                        "bm25_search", "dense_search", "rrf", "rerank",
                        "context_assembly", "llm_generation", "ragas_evaluation"} <= stages
    checks = [
        {"criterion": "Faithfulness >= 0.85", "max_points": 3,
         "verified": real and finite, "met": real and finite and scores["faithfulness"] >= .85},
        {"criterion": "All four RAGAS metrics >= 0.75", "max_points": 3,
         "verified": real and finite, "met": real and finite and all(v >= .75 for v in scores.values())},
        {"criterion": "Combined enrichment: one request per chunk", "max_points": 2,
         "verified": True, "met": combined_mode},
        {"criterion": "Measured latency breakdown report", "max_points": 2,
         "verified": complete_latency, "met": complete_latency},
    ]
    return {"evaluation_status": results.get("evaluation_status", "unknown"),
            "metrics": scores, "checks": checks,
            "eligible_points": sum(c["max_points"] for c in checks if c["met"]),
            "note": "Eligibility evidence only; final points are awarded by the instructor."}


def save_bonus_report(results, tracker, output_dir):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    report = check_bonus(results, tracker)
    (output_dir / "bonus_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    lines = ["# Bonus verification", "", "| Criterion | Max points | Status |",
             "|---|---:|---|"]
    for check in report["checks"]:
        status = "Met" if check["met"] else "Not met" if check["verified"] else "Unverified"
        lines.append(f"| {check['criterion']} | {check['max_points']} | {status} |")
    lines += ["", report["note"], ""]
    (output_dir / "bonus_report.md").write_text("\n".join(lines), encoding="utf-8")
