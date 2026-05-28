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
  results/ragas_comparison.png        — bar chart comparing experiments

Usage:
  # Full evaluation (all experiments, all 25 samples):
  python src/evaluate.py

  # Quick smoke run (hybrid only, first 5 samples):
  python src/evaluate.py --quick

  # Single experiment:
  python src/evaluate.py --experiment hybrid

  # Skip generation metrics (retrieval only, no API calls):
  python src/evaluate.py --retrieval-only

  # Regenerate report + plot from saved results:
  python src/evaluate.py --plot-only

Environment:
  ANTHROPIC_API_KEY  — required for generation metrics
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
import time
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

from pydantic import BaseModel, Field

sys.stdout.reconfigure(encoding="utf-8")
sys.stderr.reconfigure(encoding="utf-8")

# ── RAGAS imports ─────────────────────────────────────────────────────────────
# ragas >= 0.2.x API
from ragas import evaluate
from ragas.dataset_schema import EvaluationDataset, SingleTurnSample
from ragas.llms import LangchainLLMWrapper
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
from eval_experiments import CANDIDATE_K, EXPERIMENTS
from dotenv import load_dotenv

load_dotenv()

RESULTS_DIR = ROOT / "results"
RESULTS_DIR.mkdir(exist_ok=True)

# ─── Config ───────────────────────────────────────────────────────────────────

EVAL_MODEL        = "claude-haiku-4-5-20251001"    # judge LLM for RAGAS metrics
GENERATOR_MODEL   = "claude-haiku-4-5-20251001"    # answer generation LLM
SLEEP_BETWEEN_S   = 1.0     # rate-limit buffer between API calls

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


METRIC_KEYS = [
    "context_precision", "context_recall", "faithfulness",
    "answer_relevance", "answer_correctness",
]

METRIC_LABELS = {
    "context_precision":  "Context\nPrecision",
    "context_recall":     "Context\nRecall",
    "faithfulness":       "Faithfulness",
    "answer_relevance":   "Answer\nRelevance",
    "answer_correctness": "Answer\nCorrectness",
}

EXPERIMENT_LABELS = {
    "hybrid":               "Hybrid (RRF)",
    "dense":                "Dense-only",
    "bm25":                 "BM25-only",
    "hybrid_no_compress":   "Hybrid (no compress)",
}

RETRIEVAL_METRICS  = ["context_precision", "context_recall"]
GENERATION_METRICS = ["faithfulness", "answer_relevance", "answer_correctness"]

EXPERIMENT_ORDER = ["hybrid", "dense", "bm25", "hybrid_no_compress"]


def write_utf8(path: Path, content: str) -> None:
    """Write text with explicit UTF-8 encoding (avoids Windows cp1252 errors)."""
    path.write_text(content, encoding="utf-8")


class SampleResultRecord(BaseModel):
    sample_id:          str
    category:           str
    question:           str
    ground_truth:       str
    experiment:         str
    answer:             str
    n_retrieved:        int
    retrieval_ms:       float
    generation_ms:      float
    guard_passed:       bool
    guard_warnings:     list[str]
    hybrid_count:       int
    dense_only_count:   int
    bm25_only_count:    int
    context_precision:  Optional[float] = None
    context_recall:     Optional[float] = None
    faithfulness:       Optional[float] = None
    answer_relevance:   Optional[float] = None
    answer_correctness: Optional[float] = None
    retrieved_contexts: list[str] = Field(default_factory=list)


class ExperimentSummaryRecord(BaseModel):
    experiment:  str
    n_samples:   int
    metrics:     dict[str, dict[str, Any]]
    by_category: dict[str, dict[str, Any]]


class EvaluationReport(BaseModel):
    """Structured evaluation output — serialised to JSON/CSV/Markdown as UTF-8."""

    generated_at:           str
    samples_per_experiment: int
    experiments:            list[str]
    retrieval_only:         bool
    results:                dict[str, list[SampleResultRecord]]
    summaries:              dict[str, ExperimentSummaryRecord]

    @classmethod
    def from_run(
        cls,
        all_results:    dict[str, list[SampleResult]],
        summaries:      dict[str, ExperimentSummary],
        retrieval_only: bool,
    ) -> "EvaluationReport":
        results = {
            exp: [
                SampleResultRecord(
                    sample_id          = r.sample_id,
                    category           = r.category,
                    question           = r.question,
                    ground_truth       = r.ground_truth,
                    experiment         = r.experiment,
                    answer             = r.answer,
                    n_retrieved        = r.n_retrieved,
                    retrieval_ms       = r.retrieval_ms,
                    generation_ms      = r.generation_ms,
                    guard_passed       = r.guard_passed,
                    guard_warnings     = r.guard_warnings,
                    hybrid_count       = r.hybrid_count,
                    dense_only_count   = r.dense_only_count,
                    bm25_only_count    = r.bm25_only_count,
                    retrieved_contexts = r.retrieved_contexts,
                    **r.scores(),
                )
                for r in exp_results
            ]
            for exp, exp_results in all_results.items()
        }
        summary_records = {
            exp: ExperimentSummaryRecord(
                experiment  = s.experiment,
                n_samples   = s.n_samples,
                metrics     = s.metrics,
                by_category = s.by_category,
            )
            for exp, s in summaries.items()
        }
        return cls(
            generated_at           = datetime.now().strftime("%Y-%m-%d %H:%M"),
            samples_per_experiment = len(next(iter(all_results.values()))) if all_results else 0,
            experiments            = list(all_results.keys()),
            retrieval_only         = retrieval_only,
            results                = results,
            summaries              = summary_records,
        )

    def generation_error_count(self) -> int:
        return sum(
            1
            for samples in self.results.values()
            for s in samples
            if s.answer.startswith("[Generation error")
        )

    def scored_sample_count(self) -> int:
        return sum(
            1
            for samples in self.results.values()
            for s in samples
            if s.context_precision is not None
        )

    def write_json(self, path: Path) -> None:
        write_utf8(path, self.model_dump_json(indent=2))

    def write_csv(self, path: Path) -> None:
        with open(path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            header = ["experiment", "n_samples"] + [
                f"{m}_mean" for m in METRIC_KEYS
            ] + [f"{m}_std" for m in METRIC_KEYS]
            writer.writerow(header)
            for exp, summary in self.summaries.items():
                row = [exp, summary.n_samples]
                for m in METRIC_KEYS:
                    row.append(summary.metrics[m].get("mean", ""))
                for m in METRIC_KEYS:
                    row.append(summary.metrics[m].get("std", ""))
                writer.writerow(row)

    def to_markdown(self) -> str:
        metric_defs = [
            ("Context Precision", "Of retrieved chunks, what fraction are relevant to the question?", "High -> retriever is precise"),
            ("Context Recall", "Of ideal reference contexts, what fraction did we retrieve?", "High -> retriever is comprehensive"),
            ("Faithfulness", "Are all claims in the answer grounded in the retrieved context?", "High -> low hallucination"),
            ("Answer Relevance", "Does the answer address what was asked?", "High -> no off-topic drift"),
            ("Answer Correctness", "Semantic overlap between generated answer and ground truth", "High -> factually accurate"),
        ]

        lines = [
            "# RelIntel - RAGAS Evaluation Report",
            "",
            f"Generated: {self.generated_at}  ",
            f"Samples: {self.samples_per_experiment} per experiment  ",
            f"Experiments: {', '.join(self.experiments)}",
        ]

        gen_errors = self.generation_error_count()
        scored = self.scored_sample_count()
        total  = sum(len(v) for v in self.results.values())
        if gen_errors:
            lines += [
                "",
                f"> **Warning:** {gen_errors}/{total} samples failed generation "
                f"(see `answer` fields in ragas_results.json). RAGAS scores are empty.",
            ]
        elif not self.retrieval_only and scored == 0:
            lines += [
                "",
                f"> **Warning:** No RAGAS scores were computed for {total} samples. "
                "Check evaluation logs for RAGAS errors.",
            ]

        lines += [
            "",
            "---",
            "",
            "## Metric Definitions",
            "",
            "| Metric | What it measures | Ideal |",
            "|---|---|---|",
        ]
        for name, desc, ideal in metric_defs:
            lines.append(f"| **{name}** | {desc} | {ideal} |")
        lines += ["", "---", "", "## Aggregate Results", ""]

        exp_list = list(self.summaries.keys())
        if not self.retrieval_only:
            header = "| Metric | " + " | ".join(exp_list) + " |"
            sep    = "|---|" + "---|" * len(exp_list)
            lines += [header, sep]
            for m in METRIC_KEYS:
                row = f"| **{m.replace('_', ' ').title()}** | "
                row += " | ".join(
                    f"{self.summaries[exp].metrics[m].get('mean', 'N/A'):.3f} "
                    f"+/-{self.summaries[exp].metrics[m].get('std', 0):.3f}"
                    if self.summaries[exp].metrics[m].get("mean") is not None else "-"
                    for exp in exp_list
                )
                row += " |"
                lines.append(row)
            lines.append("")
        else:
            lines += ["*Generation metrics not computed (--retrieval-only mode)*", ""]

        lines += ["---", "", "## Per-Experiment Analysis", ""]
        for exp, summary in self.summaries.items():
            lines.append(f"### {exp}")
            lines.append("")
            cp = summary.metrics["context_precision"].get("mean")
            cr = summary.metrics["context_recall"].get("mean")
            fa = summary.metrics["faithfulness"].get("mean")
            ar = summary.metrics["answer_relevance"].get("mean")
            ac = summary.metrics["answer_correctness"].get("mean")

            if cp is not None:
                lines.append(f"- **Context Precision**: {cp:.3f} - "
                             + ("Strong signal-to-noise in retrieved context."
                                if cp >= 0.7 else
                                "Notable irrelevant chunks reaching the LLM - consider tighter filters or re-ranking."))
            if cr is not None:
                lines.append(f"- **Context Recall**: {cr:.3f} - "
                             + ("Good coverage of reference material."
                                if cr >= 0.7 else
                                "Missing relevant evidence - increase top_k or improve hybrid recall."))
            if fa is not None and not self.retrieval_only:
                lines.append(f"- **Faithfulness**: {fa:.3f} - "
                             + ("Low hallucination rate."
                                if fa >= 0.8 else
                                "Hallucination detected - review guard thresholds and prompt constraints."))
            if ar is not None and not self.retrieval_only:
                lines.append(f"- **Answer Relevance**: {ar:.3f} - "
                             + ("Answers on-topic."
                                if ar >= 0.8 else
                                "Some answers drifting off-topic - refine system prompt."))
            if ac is not None and not self.retrieval_only:
                lines.append(f"- **Answer Correctness**: {ac:.3f}")
            lines.append("")

            lines += ["**By category:**", ""]
            cat_header = "| Category | CP | CR | Faith | AR | AC |"
            cat_sep    = "|---|---|---|---|---|---|"
            lines += [cat_header, cat_sep]
            for cat, cat_metrics in sorted(summary.by_category.items()):
                def fmt(v: Any) -> str:
                    return f"{v:.2f}" if v is not None else "-"
                lines.append(
                    f"| {cat} "
                    f"| {fmt(cat_metrics.get('context_precision'))} "
                    f"| {fmt(cat_metrics.get('context_recall'))} "
                    f"| {fmt(cat_metrics.get('faithfulness'))} "
                    f"| {fmt(cat_metrics.get('answer_relevance'))} "
                    f"| {fmt(cat_metrics.get('answer_correctness'))} |"
                )
            lines.append("")

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
        return "\n".join(lines)

    def write_markdown(self, path: Path) -> None:
        write_utf8(path, self.to_markdown())

    def write_plot(self, path: Path) -> None:
        """Grouped bar chart comparing RAGAS metrics across experiments."""
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np

        exp_order = [e for e in EXPERIMENT_ORDER if e in self.summaries]
        if not exp_order:
            exp_order = list(self.summaries.keys())
        if not exp_order:
            return

        panels: list[tuple[str, list[str], float]] = [
            ("Retrieval Metrics", RETRIEVAL_METRICS, 0.7),
        ]
        if not self.retrieval_only:
            panels.append(("Generation Metrics", GENERATION_METRICS, 0.8))

        fig, axes = plt.subplots(
            1, len(panels),
            figsize=(6 * len(panels), 5.5),
            squeeze=False,
        )

        palette = ["#2563eb", "#ea580c", "#16a34a", "#9333ea"]
        x_labels = [EXPERIMENT_LABELS.get(e, e) for e in exp_order]

        for ax, (title, metrics, threshold) in zip(axes[0], panels):
            x = np.arange(len(metrics))
            width = min(0.7 / len(exp_order), 0.18)

            for i, exp in enumerate(exp_order):
                summary = self.summaries[exp]
                means = [
                    summary.metrics[m].get("mean") if summary.metrics[m].get("mean") is not None else 0.0
                    for m in metrics
                ]
                stds = [
                    summary.metrics[m].get("std") or 0.0
                    for m in metrics
                ]
                offset = (i - (len(exp_order) - 1) / 2) * width
                ax.bar(
                    x + offset,
                    means,
                    width,
                    yerr=stds,
                    label=x_labels[i],
                    color=palette[i % len(palette)],
                    capsize=4,
                    edgecolor="white",
                    linewidth=0.5,
                )

            ax.set_xticks(x)
            ax.set_xticklabels([METRIC_LABELS[m] for m in metrics])
            ax.set_ylim(0, 1.05)
            ax.set_ylabel("Score (mean +/- std)")
            ax.set_title(title, fontweight="bold")
            ax.axhline(
                threshold, color="#64748b", linestyle="--",
                linewidth=1, alpha=0.6, label=f"target ({threshold:.1f})",
            )
            ax.grid(axis="y", alpha=0.25)
            ax.legend(loc="upper right", fontsize=8)

        fig.suptitle(
            f"RelIntel RAG Evaluation  |  {self.samples_per_experiment} samples per experiment",
            fontsize=13, fontweight="bold", y=1.02,
        )
        fig.tight_layout()
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)

    @classmethod
    def load_json(cls, path: Path) -> "EvaluationReport":
        return cls.model_validate_json(path.read_text(encoding="utf-8"))

    def write_all(self, output_dir: Path) -> None:
        self.write_json(output_dir / "ragas_results.json")
        self.write_csv(output_dir / "ragas_summary.csv")
        self.write_markdown(output_dir / "ragas_report.md")
        self.write_plot(output_dir / "ragas_comparison.png")


# ─── RAGAS Metric Setup ───────────────────────────────────────────────────────

def build_ragas_metrics() -> tuple:
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
                answer, _, _ = call_llm(user_msg, model=GENERATOR_MODEL)
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
    import math

    metric_keys = [
        "context_precision", "context_recall", "faithfulness",
        "answer_relevance", "answer_correctness",
    ]

    def stats(vals: list[float]) -> dict:
        # RAGAS can emit NaN when an LLM judge fails mid-run (timeouts, max_tokens, etc).
        # `statistics.stdev([nan, ...])` can crash inside CPython when NaNs propagate.
        vals = [v for v in vals if v is not None and isinstance(v, (int, float)) and math.isfinite(v)]
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
    """Write JSON, CSV, and Markdown report via Pydantic (UTF-8 encoded)."""
    report = EvaluationReport.from_run(all_results, summaries, retrieval_only)
    report.write_all(output_dir)
    print(f"  Saved -> {output_dir / 'ragas_results.json'}")
    print(f"  Saved -> {output_dir / 'ragas_summary.csv'}")
    print(f"  Saved -> {output_dir / 'ragas_report.md'}")
    print(f"  Saved -> {output_dir / 'ragas_comparison.png'}")


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
    parser.add_argument("--plot-only",      action="store_true",
                        help="Regenerate report/plot from existing ragas_results.json")
    args = parser.parse_args()

    if args.plot_only:
        json_path = RESULTS_DIR / "ragas_results.json"
        if not json_path.exists():
            print(f"ERROR: {json_path} not found. Run evaluation first.")
            sys.exit(1)
        print(f"Loading {json_path}...")
        report = EvaluationReport.load_json(json_path)
        report.write_all(RESULTS_DIR)
        print(f"  Saved -> {RESULTS_DIR / 'ragas_comparison.png'}")
        return

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
        metrics, _, _ = build_ragas_metrics()

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
