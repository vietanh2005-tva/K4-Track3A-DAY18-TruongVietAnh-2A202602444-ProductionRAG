from __future__ import annotations

"""Module 4: RAGAS Evaluation — 4 metrics + failure analysis."""

import os, sys, json
import math
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")
from dataclasses import dataclass, asdict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import TEST_SET_PATH, OPENAI_API_KEY, OPENAI_BASE_URL, LLM_MODEL, EMBEDDING_MODEL


@dataclass
class EvalResult:
    question: str
    answer: str
    contexts: list[str]
    ground_truth: str
    faithfulness: float
    answer_relevancy: float
    context_precision: float
    context_recall: float


def load_test_set(path: str = TEST_SET_PATH) -> list[dict]:
    """Load test set from JSON. (Đã implement sẵn)"""
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def evaluate_ragas(questions: list[str], answers: list[str],
                   contexts: list[list[str]], ground_truths: list[str]) -> dict:
    """Run RAGAS evaluation."""
    names = ("faithfulness", "answer_relevancy", "context_precision", "context_recall")
    if not (len(questions) == len(answers) == len(contexts) == len(ground_truths)):
        raise ValueError("Evaluation inputs must have equal lengths")
    empty = {**dict.fromkeys(names, 0.0), "per_question": []}
    if not questions:
        return {**empty, "evaluation_status": "empty"}
    if not OPENAI_API_KEY:
        print("  RAGAS skipped: missing API key. Zero placeholders are not real scores.")
        return {**empty, "evaluation_status": "skipped", "evaluation_error": "Missing API key"}
    try:
        from ragas import evaluate
        from ragas.metrics import faithfulness, answer_relevancy, context_precision, context_recall
        from ragas.run_config import RunConfig
        from datasets import Dataset
        from langchain_openai import ChatOpenAI
        from src.local_embeddings import LocalMultilingualEmbeddings
        # Explicitly supply both models: Gemini's endpoint does not serve
        # OpenAI embedding model names such as text-embedding-ada-002.
        llm = ChatOpenAI(model=LLM_MODEL, api_key=OPENAI_API_KEY,
                         base_url=OPENAI_BASE_URL, temperature=0,
                         timeout=90, max_retries=0)
        if "generativelanguage.googleapis.com" in (OPENAI_BASE_URL or ""):
            from src.gemini_ragas import SingleCandidateLLMWrapper
            llm = SingleCandidateLLMWrapper(llm)
        embeddings = LocalMultilingualEmbeddings()
        dataset = Dataset.from_dict({"question": questions, "answer": answers,
                                     "contexts": contexts, "ground_truth": ground_truths})
        result = evaluate(dataset, metrics=[faithfulness, answer_relevancy,
                         context_precision, context_recall], llm=llm, embeddings=embeddings,
                         run_config=RunConfig(timeout=180, max_retries=3, max_workers=2),
                         raise_exceptions=True)
        rows = result.to_pandas().to_dict(orient="records")
        if len(rows) != len(questions):
            raise ValueError("RAGAS returned an unexpected number of rows")
        per_question = []
        for i, row in enumerate(rows):
            scores = {name: float(row[name]) for name in names}
            if not all(math.isfinite(value) for value in scores.values()):
                raise ValueError("RAGAS returned an invalid metric value")
            per_question.append(EvalResult(questions[i], answers[i], contexts[i],
                                           ground_truths[i], **scores))
        return {**{name: sum(getattr(r, name) for r in per_question) / len(per_question)
                   for name in names}, "per_question": per_question,
                "evaluation_status": "success", "evaluation_model": LLM_MODEL,
                "evaluation_embeddings": EMBEDDING_MODEL}
    except Exception as exc:
        error = type(exc).__name__
        detail = str(exc).replace(OPENAI_API_KEY, "<redacted>")[:800]
        print(f"  RAGAS evaluation failed ({error}); zero placeholders are not real scores.")
        print(f"  Service detail: {detail}")
        return {**empty, "evaluation_status": "failed", "evaluation_error": error}


def failure_analysis(eval_results: list[EvalResult], bottom_n: int = 10) -> list[dict]:
    """Analyze bottom-N worst questions using Diagnostic Tree."""
    if bottom_n <= 0:
        return []
    tree = {
        "faithfulness": ("LLM hallucinating", "Tighten prompt, lower temperature to 0"),
        "context_recall": ("Missing relevant chunks", "Improve chunking or add BM25"),
        "context_precision": ("Too many irrelevant chunks", "Add reranking or metadata filter"),
        "answer_relevancy": ("Answer does not match question", "Improve prompt template"),
    }
    failures = []
    for result in eval_results:
        scores = {name: float(getattr(result, name)) for name in tree}
        worst = min(scores, key=scores.get)
        diagnosis, fix = tree[worst]
        failures.append({"question": result.question, "answer": result.answer,
                         "contexts": result.contexts, "ground_truth": result.ground_truth,
                         "worst_metric": worst, "score": sum(scores.values()) / len(scores),
                         "diagnosis": diagnosis, "suggested_fix": fix})
    return sorted(failures, key=lambda item: item["score"])[:bottom_n]


def save_report(results: dict, failures: list[dict], path: str = "reports/ragas_report.json"):
    """Save evaluation report to JSON. (Đã implement sẵn)"""
    parent_dir = os.path.dirname(path)
    if parent_dir:
        os.makedirs(parent_dir, exist_ok=True)
    report = {
        "aggregate": {k: results.get(k, 0.0) for k in (
            "faithfulness", "answer_relevancy", "context_precision", "context_recall")},
        "num_questions": len(results.get("per_question", [])),
        "per_question": [asdict(item) if isinstance(item, EvalResult) else item
                         for item in results.get("per_question", [])],
        "evaluation_status": results.get("evaluation_status", "unknown"),
        "evaluation_error": results.get("evaluation_error"),
        "evaluation_model": results.get("evaluation_model"),
        "evaluation_embeddings": results.get("evaluation_embeddings"),
        "generation_cache_hits": results.get("generation_cache_hits"),
        "failures": failures,
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"Report saved to {path}")


if __name__ == "__main__":
    test_set = load_test_set()
    print(f"Loaded {len(test_set)} test questions")
    print("Run pipeline.py first to generate answers, then call evaluate_ragas().")
