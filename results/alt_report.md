# RelIntel - Alternative RAG Evaluation Report

Generated: 2026-05-27 02:25
Samples: 25 per experiment
Experiments: hybrid, dense, bm25, hybrid_no_compress
TLM enabled: False

> **Warning:** 100/100 samples failed generation. Generation metrics may be empty.

## Methods

| Method | Metrics | Requires |
|---|---|---|
| **Local lexical** | context recall/precision, token F1, numeric match | nothing extra |
| **Hallucination guard** | citation coverage, number grounding | generation |
| **Cleanlab TLM** | trustworthiness, context sufficiency, response helpfulness | CLEANLAB_TLM_API_KEY |

---

## Aggregate Results

| Metric | hybrid | dense | bm25 | hybrid_no_compress |
|---|---|---|---|---|
| **Context Recall
(local)** | 0.807 +/-0.311 | 0.853 +/-0.290 | 0.833 +/-0.297 | 0.807 +/-0.311 |
| **Context Precision
(local)** | 0.576 +/-0.381 | 0.585 +/-0.380 | 0.564 +/-0.379 | 0.576 +/-0.381 |
| **Answer Token F1** | 0.000 +/-0.000 | 0.000 +/-0.000 | 0.000 +/-0.000 | 0.000 +/-0.000 |
| **Numeric Match** | 0.000 +/-0.000 | 0.000 +/-0.000 | 0.000 +/-0.000 | 0.000 +/-0.000 |
| **Guard Pass** | 1.000 +/-0.000 | 1.000 +/-0.000 | 1.000 +/-0.000 | 1.000 +/-0.000 |

### hybrid

- **Context Recall
(local)**: 0.807
- **Context Precision
(local)**: 0.576
- **Answer Token F1**: 0.000
- **Numeric Match**: 0.000
- **Guard Pass**: 1.000

### dense

- **Context Recall
(local)**: 0.853
- **Context Precision
(local)**: 0.585
- **Answer Token F1**: 0.000
- **Numeric Match**: 0.000
- **Guard Pass**: 1.000

### bm25

- **Context Recall
(local)**: 0.833
- **Context Precision
(local)**: 0.564
- **Answer Token F1**: 0.000
- **Numeric Match**: 0.000
- **Guard Pass**: 1.000

### hybrid_no_compress

- **Context Recall
(local)**: 0.807
- **Context Precision
(local)**: 0.576
- **Answer Token F1**: 0.000
- **Numeric Match**: 0.000
- **Guard Pass**: 1.000
