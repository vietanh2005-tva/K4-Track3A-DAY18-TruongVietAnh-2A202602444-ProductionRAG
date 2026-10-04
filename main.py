"""
Lab 18: Production RAG Pipeline — Main Entry Point
===================================================
Chạy toàn bộ pipeline: naive baseline → production → so sánh → report.

Usage:
    python main.py
"""

import json
import os
import sys
import time

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")


def main(compare_only=False):
    print("=" * 60)
    print("LAB 18: PRODUCTION RAG PIPELINE")
    print("=" * 60)
    start = time.time()

    os.makedirs("reports", exist_ok=True)

    if not compare_only:
        # Step 1: Basic Baseline
        print("\n📌 STEP 1: Running Basic RAG Baseline...")
        print("-" * 40)
        from naive_baseline import main as run_baseline
        run_baseline()

        # Step 2: Production Pipeline
        print("\n📌 STEP 2: Running Production Pipeline...")
        print("-" * 40)
        from src.pipeline import build_pipeline, evaluate_pipeline
        from src.m4_eval import load_test_set
        search, reranker = build_pipeline(evaluation_queries=[item["question"] for item in load_test_set()])
        prod_results = evaluate_pipeline(search, reranker)
        if prod_results.get("evaluation_status") != "success":
            raise RuntimeError("Production evaluation failed; reports contain placeholders, not measured scores.")
    else:
        print("Comparing existing verified reports; no new generation/evaluation API calls.")

    # Ensure reports are located in reports/
    for f in ["ragas_report.json", "naive_baseline_report.json"]:
        if os.path.exists(f):
            os.replace(f, f"reports/{f}")

    # Step 3: Comparison
    print("\n📌 STEP 3: Comparison")
    print("-" * 40)
    naive_path = "reports/naive_baseline_report.json"
    prod_path = "reports/ragas_report.json"

    if os.path.exists(naive_path) and os.path.exists(prod_path):
        with open(naive_path, encoding="utf-8") as f:
            naive = json.load(f)
        with open(prod_path, encoding="utf-8") as f:
            prod = json.load(f)

        if any(report.get("evaluation_status") != "success" or report.get("num_questions") != 20
               for report in (naive, prod)):
            raise RuntimeError("Comparison requires successful real evaluations of all 20 questions.")
        if any(naive.get(key) != prod.get(key) for key in ("evaluation_model", "evaluation_embeddings")):
            raise RuntimeError("Comparison requires the same evaluator and embedding model.")

        print(f"\n{'Metric':<25} {'Basic':>8} {'Production':>12} {'Δ':>8}")
        print("-" * 55)
        comparison = ["# Baseline vs Production", "", "Both reports contain 20 successfully evaluated questions.",
                      f"Evaluator: {prod.get('evaluation_model')}; embeddings: {prod.get('evaluation_embeddings')}", "",
                      "| Metric | Baseline | Production | Delta |", "|---|---:|---:|---:|"]
        for m in ["faithfulness", "answer_relevancy", "context_precision", "context_recall"]:
            n = naive.get("aggregate", {}).get(m, 0)
            p = prod.get("aggregate", {}).get(m, 0)
            d = p - n
            status = "✓" if p >= 0.75 else " "
            print(f"{status} {m:<23} {n:>8.4f} {p:>12.4f} {d:>+8.4f}")
            comparison.append(f"| {m} | {n:.4f} | {p:.4f} | {d:+.4f} |")
        with open("reports/comparison_report.md", "w", encoding="utf-8") as f:
            f.write("\n".join(comparison) + "\n")
    else:
        raise RuntimeError("Missing evaluation reports; run the full pipeline first.")

    elapsed = time.time() - start
    print(f"\n⏱️  Total time: {elapsed:.1f}s")
    print("\n📋 Next steps:")
    print("  1. Điền analysis/failure_analysis.md")
    print("  2. Viết analysis/reflections/reflection_[HọTên].md")
    print("  3. Chạy: python check_lab.py")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--compare-only", action="store_true",
                        help="Compare existing successful reports without repeating API calls")
    main(parser.parse_args().compare_only)
