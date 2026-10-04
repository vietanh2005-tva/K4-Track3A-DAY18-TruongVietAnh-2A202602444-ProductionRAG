from __future__ import annotations

"""Module 3: Reranking — Cross-encoder top-20 → top-3 + latency benchmark."""

import os, sys, time
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")
from dataclasses import dataclass
from functools import lru_cache

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import RERANK_TOP_K


@dataclass
class RerankResult:
    text: str
    original_score: float
    rerank_score: float
    metadata: dict
    rank: int


@lru_cache(maxsize=2)
def _cross_encoder(model_name):
    from sentence_transformers import CrossEncoder
    import torch
    # The official model is unchanged; half precision limits memory pressure
    # while both the embedding model and reranker are resident on Windows.
    dtype_name = os.getenv("RERANKER_DTYPE", "float16")
    if dtype_name not in ("float16", "float32", "bfloat16"):
        raise ValueError("Unsupported RERANKER_DTYPE")
    return CrossEncoder(model_name, model_kwargs={"dtype": getattr(torch, dtype_name)})


class CrossEncoderReranker:
    def __init__(self, model_name: str = "BAAI/bge-reranker-v2-m3"):
        self.model_name = model_name
        self._model = None

    def _load_model(self):
        if self._model is None:
            self._model = _cross_encoder(self.model_name)
        return self._model

    def rerank(self, query: str, documents: list[dict], top_k: int = RERANK_TOP_K) -> list[RerankResult]:
        """Rerank documents: top-20 → top-k."""
        if not documents or top_k <= 0:
            return []
        import numpy as np
        pairs = [(query, doc["text"]) for doc in documents]
        scores = np.asarray(self._load_model().predict(pairs)).reshape(-1)
        if len(scores) != len(documents):
            raise ValueError("Expected one reranking score per document")
        scored = sorted(zip(scores, documents), key=lambda pair: float(pair[0]), reverse=True)
        return [RerankResult(text=doc["text"], original_score=float(doc.get("score", 0.0)),
                             rerank_score=float(score), metadata=dict(doc.get("metadata", {})),
                             rank=i)
                for i, (score, doc) in enumerate(scored[:top_k])]


class FlashrankReranker:
    """Lightweight alternative (<5ms). Optional."""
    def __init__(self):
        self._model = None

    def rerank(self, query: str, documents: list[dict], top_k: int = RERANK_TOP_K) -> list[RerankResult]:
        if not documents or top_k <= 0:
            return []
        from flashrank import Ranker, RerankRequest
        if self._model is None:
            self._model = Ranker(cache_dir=os.path.join(
                os.path.dirname(os.path.dirname(os.path.abspath(__file__))), ".cache", "flashrank"))
        passages = [{"id": i, "text": doc["text"]} for i, doc in enumerate(documents)]
        results = self._model.rerank(RerankRequest(query=query, passages=passages))
        results = sorted(results, key=lambda r: float(r["score"]), reverse=True)[:top_k]
        return [RerankResult(text=documents[r["id"]]["text"],
                             original_score=float(documents[r["id"]].get("score", 0.0)),
                             rerank_score=float(r["score"]),
                             metadata=dict(documents[r["id"]].get("metadata", {})), rank=i)
                for i, r in enumerate(results)]


def benchmark_reranker(reranker, query: str, documents: list[dict], n_runs: int = 5) -> dict:
    """Benchmark latency over n_runs. (Đã implement sẵn)"""
    if n_runs <= 0:
        raise ValueError("n_runs must be positive")
    # Exclude initial model download/load from inference latency.
    reranker.rerank(query, documents)
    times = []
    for _ in range(n_runs):
        start = time.perf_counter()
        reranker.rerank(query, documents)
        elapsed = (time.perf_counter() - start) * 1000
        times.append(elapsed)
    return {"avg_ms": sum(times) / len(times), "min_ms": min(times), "max_ms": max(times)}


if __name__ == "__main__":
    query = "Nhân viên được nghỉ phép bao nhiêu ngày?"
    docs = [
        {"text": "Nhân viên được nghỉ 12 ngày/năm.", "score": 0.8, "metadata": {}},
        {"text": "Mật khẩu thay đổi mỗi 90 ngày.", "score": 0.7, "metadata": {}},
        {"text": "Thời gian thử việc là 60 ngày.", "score": 0.75, "metadata": {}},
    ]
    reranker = CrossEncoderReranker()
    for r in reranker.rerank(query, docs):
        print(f"[{r.rank}] {r.rerank_score:.4f} | {r.text}")
