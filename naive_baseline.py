"""
Basic RAG Baseline — Chạy TRƯỚC để có scores so sánh.
=====================================================
Basic = paragraph chunking + dense-only search (không hybrid, không rerank, không enrichment).
Đây là RAG đã học ở buổi trước — hôm nay sẽ cải thiện từng bước.
"""

import sys, os, time, json
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.m1_chunking import load_documents, chunk_basic
from src.m2_search import DenseSearch
from src.m4_eval import load_test_set, evaluate_ragas, save_report
from config import NAIVE_COLLECTION


def main():
    print("=" * 60)
    print("BASIC RAG BASELINE")
    print("(paragraph chunking + dense-only, no rerank, no enrichment)")
    print("=" * 60)

    docs = load_documents()
    chunks = []
    for doc in docs:
        for c in chunk_basic(doc["text"], metadata=doc["metadata"]):
            chunks.append({"text": c.text, "metadata": c.metadata})
    print(f"  {len(chunks)} basic paragraph chunks")

    search = DenseSearch()
    search.index(chunks, collection=NAIVE_COLLECTION)

    test_set = load_test_set()
    questions, answers, all_contexts, ground_truths = [], [], [], []

    from config import OPENAI_API_KEY, OPENAI_BASE_URL, LLM_MODEL
    llm_client = None
    if OPENAI_API_KEY:
        from openai import OpenAI
        llm_client = OpenAI(api_key=OPENAI_API_KEY, base_url=OPENAI_BASE_URL, timeout=90, max_retries=0)

    os.makedirs("reports", exist_ok=True)
    checkpoint_path = "reports/naive_evaluation_inputs.json"
    if os.path.exists(checkpoint_path):
        with open(checkpoint_path, encoding="utf-8") as f:
            cached = json.load(f)
        count = len(cached.get("questions", []))
        if (cached.get("model") == LLM_MODEL and
                cached.get("questions") == [x["question"] for x in test_set[:count]] and
                cached.get("ground_truths") == [x["ground_truth"] for x in test_set[:count]] and
                all(len(cached.get(key, [])) == count for key in ("answers", "contexts"))):
            questions, answers = cached["questions"], cached["answers"]
            all_contexts, ground_truths = cached["contexts"], cached["ground_truths"]
            print(f"Resuming {count} saved answers for the same model and test set.", flush=True)

    for i, item in enumerate(test_set):
        if i < len(questions):
            continue
        results = search.search(item["question"], top_k=3, collection=NAIVE_COLLECTION)
        contexts = [r.text for r in results]

        if llm_client and contexts:
            try:
                context_str = "\n\n".join(contexts)
                from src.llm_requests import create_completion
                resp = create_completion(llm_client, model=LLM_MODEL, temperature=0, messages=[
                    {"role": "system", "content": "Trả lời CHỈ dựa trên context. Nếu không có → nói 'Không tìm thấy.'"},
                    {"role": "user", "content": f"Context:\n{context_str}\n\nCâu hỏi: {item['question']}"},
                ])
                answer = resp.choices[0].message.content
            except Exception as exc:
                print(f"Generation failed ({type(exc).__name__}); stopping without scoring fallback text.", flush=True)
                raise
        else:
            answer = contexts[0] if contexts else "Không tìm thấy."

        answers.append(answer)
        questions.append(item["question"])
        all_contexts.append(contexts)
        ground_truths.append(item["ground_truth"])
        with open(checkpoint_path, "w", encoding="utf-8") as f:
            json.dump({"model": LLM_MODEL, "questions": questions, "answers": answers,
                       "contexts": all_contexts, "ground_truths": ground_truths}, f, ensure_ascii=False, indent=2)
        print(f"  [{i+1}/{len(test_set)}] {item['question'][:50]}...", flush=True)

    os.makedirs("reports", exist_ok=True)
    with open("reports/naive_evaluation_inputs.json", "w", encoding="utf-8") as f:
        json.dump({"model": LLM_MODEL, "questions": questions, "answers": answers,
                   "contexts": all_contexts, "ground_truths": ground_truths}, f, ensure_ascii=False, indent=2)
    results = evaluate_ragas(questions, answers, all_contexts, ground_truths)
    print("\nBASIC BASELINE SCORES")
    for m in ["faithfulness", "answer_relevancy", "context_precision", "context_recall"]:
        print(f"  {m}: {results.get(m, 0):.4f}")
    save_report(results, [], path="reports/naive_baseline_report.json")
    if results.get("evaluation_status") != "success":
        raise RuntimeError("Baseline evaluation failed; saved inputs can be used to retry scoring.")
    print("\nDone! Now implement advanced modules and run: python main.py")


if __name__ == "__main__":
    start = time.time()
    main()
    print(f"Total: {time.time() - start:.1f}s")
