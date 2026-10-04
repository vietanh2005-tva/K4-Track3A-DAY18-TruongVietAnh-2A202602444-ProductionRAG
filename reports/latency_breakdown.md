# Latency breakdown — Production RAG

Run status: success

Measurements use time.perf_counter; model loading is separate from inference.
Rows are sequential stages; total time includes orchestration overhead.

Cache/fallback metadata: {"enrichment_chunks": 100, "enrichment_cache_hits": 100, "enrichment_fallbacks": 0, "combined_single_request": true, "reranker_dtype": "float32", "prepared_query_count": 20, "query_embeddings_measured_separately": true, "reranker_released_before_ragas": true, "generation_cache_hits": 20}

| Stage | Calls | Total ms | Average ms | Min ms | Max ms | P95 ms |
|---|---:|---:|---:|---:|---:|---:|
| m1_chunking | 1 | 0.43 | 0.43 | 0.43 | 0.43 | 0.43 |
| m5_enrichment | 1 | 47.82 | 47.82 | 47.82 | 47.82 | 47.82 |
| bm25_index | 1 | 17263.23 | 17263.23 | 17263.23 | 17263.23 | 17263.23 |
| dense_index | 1 | 39577.62 | 39577.62 | 39577.62 | 39577.62 | 39577.62 |
| dense_query_embedding_batch | 1 | 1019.35 | 1019.35 | 1019.35 | 1019.35 | 1019.35 |
| reranker_model_load | 1 | 6448.44 | 6448.44 | 6448.44 | 6448.44 | 6448.44 |
| bm25_search | 20 | 188.91 | 9.45 | 1.61 | 61.64 | 11.36 |
| dense_search | 20 | 1277.94 | 63.90 | 22.62 | 168.53 | 131.08 |
| rrf | 20 | 2.04 | 0.10 | 0.07 | 0.12 | 0.12 |
| rerank | 20 | 63619.39 | 3180.97 | 2628.18 | 3982.81 | 3556.94 |
| context_assembly | 20 | 2.36 | 0.12 | 0.04 | 0.82 | 0.43 |
| llm_generation | 20 | 0.05 | 0.00 | 0.00 | 0.01 | 0.00 |
| ragas_evaluation | 1 | 316950.76 | 316950.76 | 316950.76 | 316950.76 | 316950.76 |
| pipeline_total | 1 | 450518.86 | 450518.86 | 450518.86 | 450518.86 | 450518.86 |
