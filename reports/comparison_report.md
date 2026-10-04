# Baseline vs Production

Both reports contain 20 successfully evaluated questions.
Evaluator: gemini-3.1-flash-lite; embeddings: BAAI/bge-m3

| Metric | Baseline | Production | Delta |
|---|---:|---:|---:|
| faithfulness | 0.8833 | 0.9429 | +0.0595 |
| answer_relevancy | 0.7416 | 0.9040 | +0.1623 |
| context_precision | 0.7000 | 0.9000 | +0.2000 |
| context_recall | 0.8500 | 0.9333 | +0.0833 |
