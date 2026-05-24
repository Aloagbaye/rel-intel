"""
Phase 5 — RAGAS Evaluation Pipeline
RelIntel RAG System

Metrics computed per sample and in aggregate:

  Retrieval metrics (measure the retriever):
    context_precision   — of retrieved chunks, what fraction are actually relevant?
                          (signal-to-noise in the context window)
    context_recall      — of the ideal reference contexts, what fraction did
                          we actually retrieve?
                          (are we missing key evidence?)

  Generation metrics (measure the LLM given the context it received):
    faithfulness        — are all claims in the answer supported by the context?
                          (hallucination rate; lower = more hallucination)
    answer_relevance    — does the answer actually address the question?
                          (measures off-topic or incomplete answers)

  End-to-end metric:
    answer_correctness  — semantic similarity of generated answer vs ground truth
                          (combines factual overlap + semantic similarity)

Experiments:
  EXP-A  Hybrid retrieval (RRF)       — primary system
  EXP-B  Dense-only retrieval         — ablation
  EXP-C  BM25-only retrieval          — ablation
  EXP-D  Hybrid, no compression       — ablation (contextual compression effect)

Output:
  results/ragas_results.json          — full per-sample metric scores
  results/ragas_summary.csv           — aggregate mean ± std per experiment
  results/ragas_report.md             — human-readable report with commentary

Usage:
  # Full evaluation (all experiments, all 25 samples):
  python src/evaluate.py

  # Quick smoke run (hybrid only, first 5 samples):
  python src/evaluate.py --quick

  # Single experiment:
  python src/evaluate.py --experiment hybrid

  # Skip generation metrics (retrieval only, no API calls):
  python src/evaluate.py --retrieval-only

Environment:
  ANTHROPIC_API_KEY  — required for generation metrics
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Optional

# ── RAGAS imports ─────────────────────────────────────────────────────────────
# ragas >= 0.2.x API
from ragas import evaluate
from ragas.dataset_schema import EvaluationDataset, SingleTurnSample
from ragas.llm import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas.metrics import (
    LLMContextPrecisionWithReference,
    LLMContextRecall,
    Faithfulness,
    ResponseRelevancy,
    AnswerCorrectness,
)

# ── LangChain wrappers (RAGAS 0.2.x requires these) ──────────────────────────
from langchain_anthropic import ChatAnthropic
from langchain_openai import OpenAIEmbeddings   # or use any HF embeddings

# ── RelIntel imports ───────────────────────────────────────────────────────────
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "src"))

from eval_dataset import EvalSample, load_dataset
from generator import (
    build_context_block,
    build_user_message,
    call_llm,
    parse_citations,
    run_hallucination_guard,
    DEFAULT_MODEL,
)
from retriever import HybridRetriever, RetrievalQuery, RetrievalResult

RESULTS_DIR = ROOT / "results"
RESULTS_DIR.mkdir(exist_ok=True)

# ─── Config ───────────────────────────────────────────────────────────────────

EVAL_MODEL        = "claude-haiku-4-5-20251001"    # judge LLM for RAGAS metrics
GENERATOR_MODEL   = "claude-haiku-4-5-20251001"    # answer generation LLM
SLEEP_BETWEEN_S   = 1.0     # rate-limit buffer between API calls
CANDIDATE_K       = 20      # retrieval candidate pool per retriever

# ─── Data Structures ─────────────────────────────────────────────────────────

@dataclass
class SampleResult:
    sample_id:          str
    category:           str
    question:           str
    ground_truth:       str
    experiment:         str
    retrieved_contexts: list[str]    # actual retrieved chunk texts
    answer:             str
    # RAGAS scores (None if metric not computed)
    context_precision:  Optional[float] = None
    context_recall:     Optional[float] = None
    faithfulness:       Optional[float] = None
    answer_relevance:   Optional[float] = None
    answer_correctness: Optional[float] = None
    # Guard
    guard_passed:       bool  = True
    guard_warnings:     list[str] = field(default_factory=list)
    # Timing
    retrieval_ms:       float = 0.0
    generation_ms:      float = 0.0
    # Retriever diagnostics
    n_retrieved:        int   = 0
    hybrid_count:       int   = 0
    dense_only_count:   int   = 0
    bm25_only_count:    int   = 0

    def scores(self) -> dict[str, Optional[float]]:
        return {
            "context_precision":  self.context_precision,
            "context_recall":     self.context_recall,
            "faithfulness":       self.faithfulness,
            "answer_relevance":   self.answer_relevance,
            "answer_correctness": self.answer_correctness,
        }


@dataclass
class ExperimentSummary:
    experiment: str
    n_samples:  int
    metrics:    dict[str, dict]   # metric → {mean, std, min, max}
    by_category: dict[str, dict]  # category → metric → mean


# ─── Retrieval Adapters ───────────────────────────────────────────────────────

def retrieve_hybrid(
    retriever: HybridRetriever,
    sample: EvalSample,
) -> list[RetrievalResult]:
    """Full hybrid retrieval: BM25 + dense + RRF + compression."""
    rq = RetrievalQuery(
        text        = sample.question,
        top_k       = sample.top_k,
        candidate_k = CANDIDATE_K,
        **_filter_kwargs(sample.retrieval_filters),
    )
    return retriever.search(rq)


def retrieve_dense_only(
    retriever: HybridRetriever,
    sample: EvalSample,
) -> list[RetrievalResult]:
    """Dense-only: discard BM25 contribution by zeroing out BM25 scores."""
    import chromadb, numpy as np, pickle
    from retriever import build_chroma_filter, compress, RetrievalResult as RR

    q_emb = retriever._embed_query(sample.question)
    chroma_filter = build_chroma_filter(
        RetrievalQuery(text=sample.question, **_filter_kwargs(sample.retrieval_filters))
    )
    kwargs = dict(
        query_embeddings=[q_emb],
        n_results=sample.top_k,
        include=["metadatas", "distances"],
    )
    if chroma_filter:
        kwargs["where"] = chroma_filter
    try:
        res = retriever.collection.query(**kwargs)
    except Exception:
        return []

    results = []
    for cid, dist, meta in zip(
        res["ids"][0], res["distances"][0], res["metadatas"][0]
    ):
        chunk = retriever.chunks_by_id.get(cid)
        if not chunk:
            continue
        results.append(RR(
            chunk_id        = cid,
            text            = chunk["text"],
            compressed_text = compress(chunk["text"], sample.question),
            metadata        = meta,
            dense_rank      = len(results) + 1,
            bm25_rank       = None,
            rrf_score       = 1 - dist,
            dense_sim       = 1 - dist,
        ))
    return results


def retrieve_bm25_only(
    retriever: HybridRetriever,
    sample: EvalSample,
) -> list[RetrievalResult]:
    """BM25-only retrieval."""
    from retriever import apply_bm25_filter, compress, RetrievalResult as RR

    rq = RetrievalQuery(
        text=sample.question,
        **_filter_kwargs(sample.retrieval_filters),
    )
    allowed = apply_bm25_filter(retriever.chunks_by_id, rq)
    if allowed is not None and len(allowed) == 0:
        return []

    ranked = retriever.bm25.search(
        sample.question, top_k=sample.top_k, allowed_ids=allowed
    )
    results = []
    for rank, (cid, score) in enumerate(ranked, 1):
        chunk = retriever.chunks_by_id.get(cid)
        if not chunk:
            continue
        results.append(RR(
            chunk_id        = cid,
            text            = chunk["text"],
            compressed_text = compress(chunk["text"], sample.question),
            metadata        = chunk["metadata"],
            dense_rank      = None,
            bm25_rank       = rank,
            rrf_score       = score,
            dense_sim       = 0.0,
        ))
    return results


def retrieve_hybrid_no_compression(
    retriever: HybridRetriever,
    sample: EvalSample,
) -> list[RetrievalResult]:
    """Hybrid retrieval with compression disabled — full body text passed to LLM."""
    results = retrieve_hybrid(retriever, sample)
    for r in results:
        # Replace compressed_text with the full body (after context prefix)
        body_start = r.text.find("\n\n") + 2
        r.compressed_text = r.text[body_start:].strip() if body_start > 1 else r.text
    return results


def _filter_kwargs(filters: dict) -> dict:
    """Map eval_dataset filter keys to RetrievalQuery field names."""
    mapping = {
        "company_id":  "company_id",
        "sector":      "sector",
        "source_type": "source_type",
        "sentiment":   "sentiment",
        "deal_stage":  "deal_stage",
        "date_from":   "date_from",
        "date_to":     "date_to",
        "logged_by":   "logged_by",
    }
    return {mapping[k]: v for k, v in filters.items() if k in mapping}


EXPERIMENTS = {
    "hybrid":               retrieve_hybrid,
    "dense":                retrieve_dense_only,
    "bm25":                 retrieve_bm25_only,
    "hybrid_no_compress":   retrieve_hybrid_no_compression,
}

# ─── RAGAS Metric Setup ───────────────────────────────────────────────────────

def build_ragas_metrics(api_key: str) -> tuple:
    """
    Instantiate RAGAS metrics with Anthropic as both judge LLM and embedder.

    Context metrics use reference_contexts (ground-truth ideal retrieval)
    so they need a judge LLM. Faithfulness and ResponseRelevancy use the
    retrieved contexts, so they also need an LLM.

    For embeddings (ResponseRelevancy cosine sim), we use OpenAI ada-002 if
    OPENAI_API_KEY is set, otherwise fall back to a sentence-transformers wrapper.
    """
    llm = LangchainLLMWrapper(
        ChatAnthropic(
            model        = EVAL_MODEL,
            api_key      = api_key,
            temperature  = 0.0,
            max_tokens   = 1024,
        )
    )

    # Embedding model for ResponseRelevancy — try OpenAI, fall back to HF
    openai_key = os.environ.get("OPENAI_API_KEY")
    if openai_key:
        embeddings = LangchainEmbeddingsWrapper(
            OpenAIEmbeddings(api_key=openai_key, model="text-embedding-3-small")
        )
    else:
        # Sentence-transformers wrapper — works offline once model is cached
        from langchain_community.embeddings import HuggingFaceEmbeddings
        embeddings = LangchainEmbeddingsWrapper(
            HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        )

    metrics = [
        LLMContextPrecisionWithReference(llm=llm),
        LLMContextRecall(llm=llm),
        Faithfulness(llm=llm),
        ResponseRelevancy(llm=llm, embeddings=embeddings),
        AnswerCorrectness(llm=llm, embeddings=embeddings),
    ]
    return metrics, llm, embeddings


# ─── Single Sample Evaluation ─────────────────────────────────────────────────

def evaluate_sample(
    sample:        EvalSample,
    retriever:     HybridRetriever,
    retrieve_fn:   callable,
    api_key:       str,
    metrics:       list,
    experiment:    str,
    retrieval_only: bool = False,
) -> SampleResult:
    """
    Run the full pipeline for one sample:
      1. Retrieve
      2. Generate
      3. RAGAS evaluation
    """
    # ── 1. Retrieve ───────────────────────────────────────────────────────
    t0 = time.perf_counter()
    results = retrieve_fn(retriever, sample)
    retrieval_ms = (time.perf_counter() - t0) * 1000

    retrieved_texts = [r.compressed_text for r in results]

    # Source attribution diagnostics
    hybrid_count     = sum(1 for r in results if r.dense_rank and r.bm25_rank)
    dense_only_count = sum(1 for r in results if r.dense_rank and not r.bm25_rank)
    bm25_only_count  = sum(1 for r in results if r.bm25_rank and not r.dense_rank)

    # ── 2. Generate ───────────────────────────────────────────────────────
    generation_ms = 0.0
    answer        = ""
    guard_passed  = True
    guard_warnings = []

    if not retrieval_only:
        t0 = time.perf_counter()
        if not results:
            answer = "No relevant context was found for this query."
        else:
            context_block = build_context_block(results)
            user_msg      = build_user_message(sample.question, context_block)
            try:
                answer, _, _ = call_llm(user_msg, api_key, model=GENERATOR_MODEL)
            except Exception as e:
                answer = f"[Generation error: {e}]"

        generation_ms = (time.perf_counter() - t0) * 1000

        # Run guard
        if results and answer and not answer.startswith("[Generation error"):
            citations = parse_citations(answer, results)
            guard     = run_hallucination_guard(answer, results, citations)
            guard_passed   = guard.passed
            guard_warnings = guard.warnings

    # ── 3. RAGAS ──────────────────────────────────────────────────────────
    sr = SampleResult(
        sample_id          = sample.id,
        category           = sample.category,
        question           = sample.question,
        ground_truth       = sample.ground_truth,
        experiment         = experiment,
        retrieved_contexts = retrieved_texts,
        answer             = answer,
        guard_passed       = guard_passed,
        guard_warnings     = guard_warnings,
        retrieval_ms       = retrieval_ms,
        generation_ms      = generation_ms,
        n_retrieved        = len(results),
        hybrid_count       = hybrid_count,
        dense_only_count   = dense_only_count,
        bm25_only_count    = bm25_only_count,
    )

    if retrieval_only or not answer or answer.startswith("[Generation error"):
        return sr

    # Build RAGAS SingleTurnSample
    ragas_sample = SingleTurnSample(
        user_input          = sample.question,
        retrieved_contexts  = retrieved_texts,
        response            = answer,
        reference           = sample.ground_truth,
        reference_contexts  = sample.reference_contexts,
    )

    ragas_dataset = EvaluationDataset(samples=[ragas_sample])

    try:
        ragas_result = evaluate(dataset=ragas_dataset, metrics=metrics)
        df = ragas_result.to_pandas()
        row = df.iloc[0]

        metric_map = {
            "llm_context_precision_with_reference": "context_precision",
            "llm_context_recall":                   "context_recall",
            "faithfulness":                         "faithfulness",
            "response_relevancy":                   "answer_relevance",
            "answer_correctness":                   "answer_correctness",
        }
        for ragas_col, our_col in metric_map.items():
            if ragas_col in df.columns:
                val = row[ragas_col]
                setattr(sr, our_col, float(val) if val is not None else None)

    except Exception as e:
        print(f"    ⚠ RAGAS eval error on {sample.id}: {e}")

    return sr


# ─── Aggregate Statistics ─────────────────────────────────────────────────────

def aggregate(results: list[SampleResult], experiment: str) -> ExperimentSummary:
    import statistics

    metric_keys = [
        "context_precision", "context_recall", "faithfulness",
        "answer_relevance", "answer_correctness",
    ]

    def stats(vals: list[float]) -> dict:
        vals = [v for v in vals if v is not None]
        if not vals:
            return {"mean": None, "std": None, "min": None, "max": None, "n": 0}
        return {
            "mean": round(statistics.mean(vals), 4),
            "std":  round(statistics.stdev(vals), 4) if len(vals) > 1 else 0.0,
            "min":  round(min(vals), 4),
            "max":  round(max(vals), 4),
            "n":    len(vals),
        }

    overall = {
        k: stats([getattr(r, k) for r in results])
        for k in metric_keys
    }

    # Per-category breakdown
    by_cat: dict[str, list[SampleResult]] = defaultdict(list)
    for r in results:
        by_cat[r.category].append(r)

    by_category = {}
    for cat, cat_results in sorted(by_cat.items()):
        by_category[cat] = {
            k: stats([getattr(r, k) for r in cat_results])["mean"]
            for k in metric_keys
        }

    return ExperimentSummary(
        experiment   = experiment,
        n_samples    = len(results),
        metrics      = overall,
        by_category  = by_category,
    )


# ─── Report Writer ────────────────────────────────────────────────────────────

def write_report(
    all_results:  dict[str, list[SampleResult]],
    summaries:    dict[str, ExperimentSummary],
    output_dir:   Path,
    retrieval_only: bool,
) -> None:
    """Write JSON, CSV, and Markdown report."""

    # ── JSON — full per-sample results ────────────────────────────────────
    json_path = output_dir / "ragas_results.json"
    with open(json_path, "w") as f:
        json.dump(
            {
                exp: [
                    {
                        **{k: getattr(r, k) for k in [
                            "sample_id", "category", "question", "ground_truth",
                            "experiment", "answer", "n_retrieved",
                            "retrieval_ms", "generation_ms",
                            "guard_passed", "guard_warnings",
                            "hybrid_count", "dense_only_count", "bm25_only_count",
                        ]},
                        **r.scores(),
                        "retrieved_contexts": r.retrieved_contexts,
                    }
                    for r in results
                ]
                for exp, results in all_results.items()
            },
            f, indent=2,
        )
    print(f"  Saved → {json_path}")

    # ── CSV — summary table ────────────────────────────────────────────────
    csv_path = output_dir / "ragas_summary.csv"
    metric_keys = [
        "context_precision", "context_recall", "faithfulness",
        "answer_relevance", "answer_correctness",
    ]
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        header = ["experiment", "n_samples"] + [
            f"{m}_mean" for m in metric_keys
        ] + [f"{m}_std" for m in metric_keys]
        writer.writerow(header)
        for exp, summary in summaries.items():
            row = [exp, summary.n_samples]
            for m in metric_keys:
                row.append(summary.metrics[m].get("mean", ""))
            for m in metric_keys:
                row.append(summary.metrics[m].get("std", ""))
            writer.writerow(row)
    print(f"  Saved → {csv_path}")

    # ── Markdown report ────────────────────────────────────────────────────
    md_path = output_dir / "ragas_report.md"
    lines = [
        f"# RelIntel — RAGAS Evaluation Report",
        f"",
        f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}  ",
        f"Samples: {next(iter(all_results.values())).__len__() if all_results else 0} per experiment  ",
        f"Experiments: {', '.join(all_results.keys())}",
        f"",
        f"---",
        f"",
        f"## Metric Definitions",
        f"",
        f"| Metric | What it measures | Ideal |",
        f"|---|---|---|",
        f"| **Context Precision** | Of retrieved chunks, what fraction are relevant to the question? | High → retriever is precise |",
        f"| **Context Recall** | Of ideal reference contexts, what fraction did we retrieve? | High → retriever is comprehensive |",
        f"| **Faithfulness** | Are all claims in the answer grounded in the retrieved context? | High → low hallucination |",
        f"| **Answer Relevance** | Does the answer address what was asked? | High → no off-topic drift |",
        f"| **Answer Correctness** | Semantic overlap between generated answer and ground truth | High → factually accurate |",
        f"",
        f"---",
        f"",
        f"## Aggregate Results",
        f"",
    ]

    # Comparison table
    exp_list = list(summaries.keys())
    metric_keys = [
        "context_precision", "context_recall", "faithfulness",
        "answer_relevance", "answer_correctness",
    ]

    if not retrieval_only:
        header = "| Metric | " + " | ".join(exp_list) + " |"
        sep    = "|---|" + "---|" * len(exp_list)
        lines += [header, sep]
        for m in metric_keys:
            row = f"| **{m.replace('_',' ').title()}** | "
            row += " | ".join(
                f"{summaries[exp].metrics[m].get('mean', 'N/A'):.3f} ±{summaries[exp].metrics[m].get('std', 0):.3f}"
                if summaries[exp].metrics[m].get("mean") is not None else "—"
                for exp in exp_list
            )
            row += " |"
            lines.append(row)
        lines.append("")
    else:
        lines.append("*Generation metrics not computed (--retrieval-only mode)*")
        lines.append("")

    # Per-experiment narrative
    lines += ["---", "", "## Per-Experiment Analysis", ""]
    for exp, summary in summaries.items():
        lines.append(f"### {exp}")
        lines.append(f"")
        cp = summary.metrics["context_precision"].get("mean")
        cr = summary.metrics["context_recall"].get("mean")
        fa = summary.metrics["faithfulness"].get("mean")
        ar = summary.metrics["answer_relevance"].get("mean")
        ac = summary.metrics["answer_correctness"].get("mean")

        if cp is not None:
            lines.append(f"- **Context Precision**: {cp:.3f} — "
                         + ("Retriever is precise; low noise in context window."
                            if cp >= 0.7 else
                            "Notable irrelevant chunks reaching the LLM — consider tighter filters or re-ranking."))
        if cr is not None:
            lines.append(f"- **Context Recall**: {cr:.3f} — "
                         + ("Good coverage of reference material."
                            if cr >= 0.7 else
                            "Missing relevant evidence — increase top_k or improve hybrid recall."))
        if fa is not None and not retrieval_only:
            lines.append(f"- **Faithfulness**: {fa:.3f} — "
                         + ("Low hallucination rate."
                            if fa >= 0.8 else
                            "Hallucination detected — review guard thresholds and prompt constraints."))
        if ar is not None and not retrieval_only:
            lines.append(f"- **Answer Relevance**: {ar:.3f} — "
                         + ("Answers on-topic."
                            if ar >= 0.8 else
                            "Some answers drifting off-topic — refine system prompt."))
        if ac is not None and not retrieval_only:
            lines.append(f"- **Answer Correctness**: {ac:.3f}")
        lines.append("")

        # Category breakdown table
        lines.append("**By category:**")
        lines.append("")
        cat_header = "| Category | CP | CR | Faith | AR | AC |"
        cat_sep    = "|---|---|---|---|---|---|"
        lines += [cat_header, cat_sep]
        for cat, cat_metrics in sorted(summary.by_category.items()):
            def fmt(v): return f"{v:.2f}" if v is not None else "—"
            lines.append(
                f"| {cat} "
                f"| {fmt(cat_metrics.get('context_precision'))} "
                f"| {fmt(cat_metrics.get('context_recall'))} "
                f"| {fmt(cat_metrics.get('faithfulness'))} "
                f"| {fmt(cat_metrics.get('answer_relevance'))} "
                f"| {fmt(cat_metrics.get('answer_correctness'))} |"
            )
        lines.append("")

    # Key insights section
    lines += [
        "---", "",
        "## Key Insights", "",
        "*(Fill in after running evaluation)*", "",
        "- **Hybrid vs Dense**: Did RRF improve context_recall for lexical queries?",
        "- **Hybrid vs BM25**: Did dense retrieval improve context_precision on semantic queries?",
        "- **Compression effect**: Did `hybrid_no_compress` change faithfulness scores?",
        "- **Category gaps**: Which question categories had the lowest scores? Why?",
        "- **Failure analysis**: Which individual samples scored lowest? What went wrong?",
        "",
        "---", "",
        "## Recommended Next Steps", "",
        "Based on evaluation results, consider:", "",
        "1. If context_precision < 0.7: tighten metadata pre-filters or add cross-encoder re-ranking",
        "2. If context_recall < 0.7: increase candidate_k or review chunking strategy",
        "3. If faithfulness < 0.8: strengthen system prompt citation rules or add post-hoc NLI check",
        "4. If answer_correctness < 0.6: improve ground truth construction or fine-tune generator",
    ]

    with open(md_path, "w") as f:
        f.write("\n".join(lines))
    print(f"  Saved → {md_path}")


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="RelIntel RAGAS evaluation")
    parser.add_argument("--experiment",     choices=list(EXPERIMENTS.keys()) + ["all"],
                        default="all", help="Which experiment to run (default: all)")
    parser.add_argument("--quick",          action="store_true",
                        help="Run hybrid only on first 5 samples")
    parser.add_argument("--retrieval-only", action="store_true",
                        help="Skip generation and generation metrics (no API calls)")
    parser.add_argument("--sample-ids",     nargs="+",
                        help="Run specific sample IDs only (e.g. S01 S05 S21)")
    parser.add_argument("--no-report",      action="store_true",
                        help="Skip writing report files")
    args = parser.parse_args()

    api_key = os.environ.get("ANTHROPIC_API_KEY", "")
    if not api_key and not args.retrieval_only:
        print("ERROR: ANTHROPIC_API_KEY not set.")
        print("  export ANTHROPIC_API_KEY=sk-ant-...")
        print("  Or run with --retrieval-only to skip generation metrics.")
        sys.exit(1)

    # ── Load ──────────────────────────────────────────────────────────────
    print("Loading retriever and dataset...")
    retriever = HybridRetriever.load()
    samples   = load_dataset()

    if args.quick:
        samples = [s for s in samples if s.id in ("S01","S02","S06","S19","S21")]
        experiments_to_run = ["hybrid"]
    elif args.experiment == "all":
        experiments_to_run = list(EXPERIMENTS.keys())
    else:
        experiments_to_run = [args.experiment]

    if args.sample_ids:
        samples = [s for s in samples if s.id in args.sample_ids]

    print(f"Running: {experiments_to_run}")
    print(f"Samples: {len(samples)}")
    print(f"Mode: {'retrieval-only' if args.retrieval_only else 'full (retrieval + generation + RAGAS)'}")
    print()

    # ── RAGAS metrics ─────────────────────────────────────────────────────
    metrics = []
    if not args.retrieval_only:
        print("Initialising RAGAS metrics...")
        metrics, _, _ = build_ragas_metrics(api_key)

    # ── Run experiments ───────────────────────────────────────────────────
    all_results:  dict[str, list[SampleResult]] = {}
    all_summaries: dict[str, ExperimentSummary] = {}

    for exp in experiments_to_run:
        retrieve_fn = EXPERIMENTS[exp]
        print(f"\n{'═'*60}")
        print(f"  Experiment: {exp}  ({len(samples)} samples)")
        print(f"{'═'*60}")

        exp_results = []
        for i, sample in enumerate(samples, 1):
            print(f"\n  [{i}/{len(samples)}] {sample.id} · {sample.category}")
            print(f"  Q: {sample.question[:80]}")

            sr = evaluate_sample(
                sample         = sample,
                retriever      = retriever,
                retrieve_fn    = retrieve_fn,
                api_key        = api_key,
                metrics        = metrics,
                experiment     = exp,
                retrieval_only = args.retrieval_only,
            )
            exp_results.append(sr)

            # Print per-sample scores
            scores_str = "  Scores: " + " | ".join(
                f"{k.split('_')[0]}={v:.3f}" if v is not None else f"{k.split('_')[0]}=—"
                for k, v in sr.scores().items()
            )
            print(scores_str)
            print(f"  Retrieved: {sr.n_retrieved} chunks "
                  f"(hybrid={sr.hybrid_count} dense={sr.dense_only_count} bm25={sr.bm25_only_count}) "
                  f"| {sr.retrieval_ms:.0f}ms retrieval "
                  f"| {sr.generation_ms:.0f}ms generation")
            if sr.guard_warnings:
                for w in sr.guard_warnings[:2]:
                    print(f"  ⚠ {w}")

            if i < len(samples):
                time.sleep(SLEEP_BETWEEN_S)

        all_results[exp]   = exp_results
        all_summaries[exp] = aggregate(exp_results, exp)

        # Print experiment summary
        s = all_summaries[exp]
        print(f"\n  ── {exp} summary ──")
        for m, stats in s.metrics.items():
            if stats["mean"] is not None:
                print(f"     {m:<25} mean={stats['mean']:.3f}  std={stats['std']:.3f}")

    # ── Write reports ──────────────────────────────────────────────────────
    if not args.no_report:
        print(f"\n{'─'*60}")
        print("Writing reports...")
        write_report(all_results, all_summaries, RESULTS_DIR, args.retrieval_only)

    print(f"\n{'═'*60}")
    print("  Evaluation complete ✅")
    print(f"{'═'*60}")


if __name__ == "__main__":
    main()
