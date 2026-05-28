# RelIntel - RAGAS Evaluation Report

Generated: 2026-05-28 03:02  
Samples: 25 per experiment  
Experiments: hybrid, dense, bm25, hybrid_no_compress

---

## Metric Definitions

| Metric | What it measures | Ideal |
|---|---|---|
| **Context Precision** | Of retrieved chunks, what fraction are relevant to the question? | High -> retriever is precise |
| **Context Recall** | Of ideal reference contexts, what fraction did we retrieve? | High -> retriever is comprehensive |
| **Faithfulness** | Are all claims in the answer grounded in the retrieved context? | High -> low hallucination |
| **Answer Relevance** | Does the answer address what was asked? | High -> no off-topic drift |
| **Answer Correctness** | Semantic overlap between generated answer and ground truth | High -> factually accurate |

---

## Aggregate Results

| Metric | hybrid | dense | bm25 | hybrid_no_compress |
|---|---|---|---|---|
| **Context Precision** | 0.127 +/-0.317 | 0.153 +/-0.340 | 0.120 +/-0.314 | 0.127 +/-0.317 |
| **Context Recall** | - | - | - | - |
| **Faithfulness** | 0.584 +/-0.274 | 0.613 +/-0.320 | 0.605 +/-0.327 | 0.542 +/-0.281 |
| **Answer Relevance** | - | - | - | - |
| **Answer Correctness** | 0.496 +/-0.255 | 0.481 +/-0.225 | 0.444 +/-0.250 | 0.468 +/-0.241 |

---

## Per-Experiment Analysis

### hybrid

- **Context Precision**: 0.127 - Notable irrelevant chunks reaching the LLM - consider tighter filters or re-ranking.
- **Faithfulness**: 0.584 - Hallucination detected - review guard thresholds and prompt constraints.
- **Answer Correctness**: 0.496

**By category:**

| Category | CP | CR | Faith | AR | AC |
|---|---|---|---|---|---|
| aggregation | 0.00 | - | 0.47 | - | 0.75 |
| comparison | 0.00 | - | 0.78 | - | 0.53 |
| deal_stage_filtered | 0.00 | - | - | - | - |
| factual_lookup | 0.33 | - | 0.46 | - | 0.68 |
| lexical | 0.50 | - | 0.75 | - | 0.66 |
| multi_hop_synthesis | 0.11 | - | - | - | - |
| negation_absence | 0.42 | - | 0.51 | - | 0.49 |
| sentiment_filtered | 0.00 | - | - | - | - |
| source_filtered | 0.00 | - | 0.50 | - | 0.17 |
| team_filtered | 0.00 | - | 0.86 | - | 0.13 |
| temporal | 0.00 | - | - | - | - |

### dense

- **Context Precision**: 0.153 - Notable irrelevant chunks reaching the LLM - consider tighter filters or re-ranking.
- **Faithfulness**: 0.613 - Hallucination detected - review guard thresholds and prompt constraints.
- **Answer Correctness**: 0.481

**By category:**

| Category | CP | CR | Faith | AR | AC |
|---|---|---|---|---|---|
| aggregation | 0.00 | - | 0.55 | - | 0.57 |
| comparison | 0.00 | - | - | - | - |
| deal_stage_filtered | 0.00 | - | - | - | - |
| factual_lookup | 0.33 | - | 0.43 | - | 0.64 |
| lexical | 0.50 | - | 0.83 | - | 0.50 |
| multi_hop_synthesis | 0.17 | - | 0.55 | - | - |
| negation_absence | 0.67 | - | 0.58 | - | 0.37 |
| sentiment_filtered | 0.00 | - | 0.89 | - | - |
| source_filtered | 0.00 | - | - | - | - |
| team_filtered | 0.00 | - | 0.86 | - | 0.13 |
| temporal | 0.00 | - | - | - | - |

### bm25

- **Context Precision**: 0.120 - Notable irrelevant chunks reaching the LLM - consider tighter filters or re-ranking.
- **Faithfulness**: 0.605 - Hallucination detected - review guard thresholds and prompt constraints.
- **Answer Correctness**: 0.444

**By category:**

| Category | CP | CR | Faith | AR | AC |
|---|---|---|---|---|---|
| aggregation | 0.00 | - | 1.00 | - | 0.33 |
| comparison | 0.00 | - | 0.69 | - | 0.47 |
| deal_stage_filtered | 0.00 | - | - | - | - |
| factual_lookup | 0.33 | - | 0.50 | - | 0.73 |
| lexical | 0.50 | - | 0.71 | - | 0.49 |
| multi_hop_synthesis | 0.06 | - | 0.40 | - | 0.27 |
| negation_absence | 0.42 | - | 0.39 | - | 0.44 |
| sentiment_filtered | 0.00 | - | 0.64 | - | - |
| source_filtered | 0.00 | - | - | - | 0.18 |
| team_filtered | 0.00 | - | 0.86 | - | 0.13 |
| temporal | 0.00 | - | - | - | - |

### hybrid_no_compress

- **Context Precision**: 0.127 - Notable irrelevant chunks reaching the LLM - consider tighter filters or re-ranking.
- **Faithfulness**: 0.542 - Hallucination detected - review guard thresholds and prompt constraints.
- **Answer Correctness**: 0.468

**By category:**

| Category | CP | CR | Faith | AR | AC |
|---|---|---|---|---|---|
| aggregation | 0.00 | - | 0.51 | - | 0.68 |
| comparison | 0.00 | - | 0.74 | - | 0.50 |
| deal_stage_filtered | 0.00 | - | - | - | - |
| factual_lookup | 0.33 | - | 0.46 | - | 0.67 |
| lexical | 0.50 | - | 0.50 | - | 0.53 |
| multi_hop_synthesis | 0.11 | - | 0.20 | - | 0.24 |
| negation_absence | 0.42 | - | 0.38 | - | 0.55 |
| sentiment_filtered | 0.00 | - | 0.69 | - | - |
| source_filtered | 0.00 | - | - | - | 0.16 |
| team_filtered | 0.00 | - | 1.00 | - | 0.21 |
| temporal | 0.00 | - | - | - | - |

---

## Key Insights

*(Fill in after running evaluation)*

- **Hybrid vs Dense**: Did RRF improve context_recall for lexical queries?
- **Hybrid vs BM25**: Did dense retrieval improve context_precision on semantic queries?
- **Compression effect**: Did `hybrid_no_compress` change faithfulness scores?
- **Category gaps**: Which question categories had the lowest scores? Why?
- **Failure analysis**: Which individual samples scored lowest? What went wrong?

---

## Recommended Next Steps

Based on evaluation results, consider:

1. If context_precision < 0.7: tighten metadata pre-filters or add cross-encoder re-ranking
2. If context_recall < 0.7: increase candidate_k or review chunking strategy
3. If faithfulness < 0.8: strengthen system prompt citation rules or add post-hoc NLI check
4. If answer_correctness < 0.6: improve ground truth construction or fine-tune generator