from __future__ import annotations

"""Module 2: Hybrid Search — BM25 (Vietnamese) + Dense + RRF."""

import os, sys
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")
from dataclasses import dataclass
from functools import lru_cache

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import (QDRANT_HOST, QDRANT_PORT, COLLECTION_NAME, EMBEDDING_MODEL,
                    EMBEDDING_DIM, BM25_TOP_K, DENSE_TOP_K, HYBRID_TOP_K)


@lru_cache(maxsize=1)
def shared_sentence_encoder(model_name=EMBEDDING_MODEL):
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer(model_name)


@dataclass
class SearchResult:
    text: str
    score: float
    metadata: dict
    method: str  # "bm25", "dense", "hybrid"


def segment_vietnamese(text: str) -> str:
    """Segment Vietnamese words and normalize compound-word separators."""
    if not text.strip():
        return ""
    from underthesea import word_tokenize
    return word_tokenize(text, format="text").replace("_", " ")


class BM25Search:
    def __init__(self):
        self.corpus_tokens = []
        self.documents = []
        self.bm25 = None

    def index(self, chunks: list[dict]) -> None:
        """Build a new BM25 index; empty input resets the index."""
        from rank_bm25 import BM25Okapi
        self.documents = list(chunks)
        self.corpus_tokens = [segment_vietnamese(c["text"]).lower().split()
                              for c in self.documents]
        self.bm25 = (BM25Okapi(self.corpus_tokens)
                     if any(self.corpus_tokens) else None)

    def search(self, query: str, top_k: int = BM25_TOP_K) -> list[SearchResult]:
        """Return positive-scoring documents, highest score first."""
        if self.bm25 is None or top_k <= 0 or not query.strip():
            return []
        tokens = segment_vietnamese(query).lower().split()
        scores = self.bm25.get_scores(tokens)
        indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
        return [SearchResult(text=self.documents[i]["text"], score=float(scores[i]),
                             metadata=dict(self.documents[i].get("metadata", {})),
                             method="bm25")
                for i in indices if scores[i] > 0][:top_k]


class DenseSearch:
    def __init__(self):
        from qdrant_client import QdrantClient
        try:
            self.client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT, timeout=2)
            self.client.get_collections()
        except Exception:
            self.client = QdrantClient(":memory:")
        self._encoder = None
        self._query_vectors = {}

    def _get_encoder(self):
        if self._encoder is None:
            self._encoder = shared_sentence_encoder()
        return self._encoder

    def prepare_queries(self, queries: list[str]) -> None:
        """Batch evaluation query embeddings before loading the large reranker."""
        unique = list(dict.fromkeys(queries))
        if unique:
            vectors = self._get_encoder().encode(unique, batch_size=4, show_progress_bar=True)
            self._query_vectors = {query: vector.tolist() for query, vector in zip(unique, vectors)}

    def release_encoder(self) -> None:
        import gc
        self._encoder = None
        shared_sentence_encoder.cache_clear()
        gc.collect()

    def index(self, chunks: list[dict], collection: str = COLLECTION_NAME) -> None:
        """Rebuild the lab collection with cosine vectors and document payloads."""
        from qdrant_client.models import Distance, VectorParams, PointStruct
        # Encode first: a failed model load must not clear the previous index.
        vectors = None
        if chunks:
            vectors = self._get_encoder().encode(
                [c["text"] for c in chunks], batch_size=8, show_progress_bar=True)
            if any(len(v) != EMBEDDING_DIM for v in vectors):
                raise ValueError(f"Expected embedding dimension {EMBEDDING_DIM}")
        if self.client.collection_exists(collection_name=collection):
            self.client.delete_collection(collection_name=collection)
        self.client.create_collection(
            collection_name=collection,
            vectors_config=VectorParams(size=EMBEDDING_DIM, distance=Distance.COSINE))
        if not chunks:
            return
        for start in range(0, len(chunks), 64):
            points = [PointStruct(id=i, vector=vectors[i].tolist(),
                                  payload={**chunks[i].get("metadata", {}),
                                           "text": chunks[i]["text"]})
                      for i in range(start, min(start + 64, len(chunks)))]
            self.client.upsert(collection_name=collection, points=points, wait=True)

    def search(self, query: str, top_k: int = DENSE_TOP_K,
               collection: str = COLLECTION_NAME) -> list[SearchResult]:
        """Query Qdrant using the same encoder as indexing."""
        if top_k <= 0 or not query.strip():
            return []
        if not self.client.collection_exists(collection_name=collection):
            return []
        if self.client.count(collection_name=collection, exact=True).count == 0:
            return []
        query_vector = self._query_vectors.get(query)
        if query_vector is None:
            query_vector = self._get_encoder().encode(query).tolist()
        response = self.client.query_points(
            collection_name=collection, query=query_vector, limit=top_k,
            with_payload=True)
        results = []
        for point in response.points:
            payload = dict(point.payload or {})
            text = payload.pop("text", "")
            results.append(SearchResult(text=text, score=float(point.score),
                                        metadata=payload, method="dense"))
        return results


def reciprocal_rank_fusion(results_list: list[list[SearchResult]], k: int = 60,
                           top_k: int = HYBRID_TOP_K) -> list[SearchResult]:
    """Merge ranked lists: score(d) = sum(1 / (k + zero_based_rank + 1))."""
    if k < 0:
        raise ValueError("k must be non-negative")
    if top_k <= 0:
        return []
    scores, originals = {}, {}
    for results in results_list:
        seen = set()
        for rank, result in enumerate(results):
            if result.text in seen:
                continue
            seen.add(result.text)
            originals.setdefault(result.text, result)
            scores[result.text] = scores.get(result.text, 0.0) + 1.0 / (k + rank + 1)
    ranked = sorted(scores, key=scores.get, reverse=True)[:top_k]
    return [SearchResult(text=text, score=scores[text],
                         metadata=dict(originals[text].metadata), method="hybrid")
            for text in ranked]


class HybridSearch:
    """Combine BM25 and dense candidates using reciprocal rank fusion."""
    def __init__(self):
        self.bm25 = BM25Search()
        self.dense = DenseSearch()

    def index(self, chunks: list[dict]) -> None:
        self.bm25.index(chunks)
        self.dense.index(chunks)

    def search(self, query: str, top_k: int = HYBRID_TOP_K) -> list[SearchResult]:
        bm25_results = self.bm25.search(query, top_k=BM25_TOP_K)
        dense_results = self.dense.search(query, top_k=DENSE_TOP_K)
        return reciprocal_rank_fusion([bm25_results, dense_results], top_k=top_k)


if __name__ == "__main__":
    print("Original:  Nhân viên được nghỉ phép năm")
    print(f"Segmented: {segment_vietnamese('Nhân viên được nghỉ phép năm')}")
