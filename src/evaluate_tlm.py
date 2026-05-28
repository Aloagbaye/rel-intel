"""
Phase 5 — Alternative RAG Evaluation (TLM + Local Metrics)
RelIntel RAG System

Lightweight evaluation pipeline that does NOT depend on RAGAS/LangChain judges.

Metrics (always computed — no RAGAS):
  context_recall_local     — fraction of reference chunks matched in retrieval
  context_precision_local  — fraction of retrieved chunks matching a reference
  answer_token_f1          — token F1 between answer and ground truth
  numeric_match            — fraction of GT numbers found in the answer
  guard_passed             — existing citation/grounding guard (0/1)

Metrics (optional — requires CLEANLAB_TLM_API_KEY):
  tlm_trustworthiness      — Cleanlab TrustworthyRAG overall score
  tlm_context_sufficiency  — TLM default eval for context quality
  tlm_response_helpfulness — TLM default eval for answer quality

Usage:
  python src/evaluate_tlm.py --quick          # hybrid, 5 samples
  python src/evaluate_tlm.py                  # all experiments, 25 samples
  python src/evaluate_tlm.py --no-tlm         # local metrics only (no Cleanlab API)
  python src/evaluate_tlm.py --retrieval-only # skip generation + TLM
  python src/evaluate_tlm.py --plot-only      # regenerate reports from alt_results.json

Environment:
  ANTHROPIC_API_KEY     — answer generation (unless --retrieval-only)
  CLEANLAB_TLM_API_KEY  — TLM scoring (unless --no-tlm); free key at tlm.cleanlab.ai
"""

from __future__ import annotations

import argparse
import csv
import os
import re
import statistics
import sys
import time
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

from dotenv import load_dotenv
from pydantic import BaseModel, Field

sys.stdout.reconfigure(encoding="utf-8")
sys.stderr.reconfigure(encoding="utf-8")

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "src"))

load_dotenv()

from eval_dataset import EvalSample, load_dataset
from generator import (
    SYSTEM_PROMPT,
    build_context_block,
    build_user_message,
    call_llm,
    parse_citations,
    run_hallucination_guard,
    DEFAULT_MODEL,
)
from retriever import HybridRetriever

from eval_experiments import EXPERIMENTS

RESULTS_DIR = ROOT / "results"
RESULTS_DIR.mkdir(exist_ok=True)

SLEEP_BETWEEN_S   = 1.0
GENERATOR_MODEL   = DEFAULT_MODEL

LOCAL_METRIC_KEYS = [
    "context_recall_local",
    "context_precision_local",
    "answer_token_f1",
    "numeric_match",
    "guard_pass_rate",
]

TLM_METRIC_KEYS = [
    "tlm_trustworthiness",
    "tlm_context_sufficiency",
    "tlm_response_helpfulness",
]

ALL_METRIC_KEYS = LOCAL_METRIC_KEYS + TLM_METRIC_KEYS

METRIC_LABELS = {
    "context_recall_local":     "Context Recall\n(local)",
    "context_precision_local":  "Context Precision\n(local)",
    "answer_token_f1":          "Answer Token F1",
    "numeric_match":            "Numeric Match",
    "guard_pass_rate":          "Guard Pass",
    "tlm_trustworthiness":      "TLM\nTrustworthiness",
    "tlm_context_sufficiency":  "TLM Context\nSufficiency",
    "tlm_response_helpfulness": "TLM Response\nHelpfulness",
}

EXPERIMENT_LABELS = {
    "hybrid":             "Hybrid (RRF)",
    "dense":              "Dense-only",
    "bm25":               "BM25-only",
    "hybrid_no_compress": "Hybrid (no compress)",
}

EXPERIMENT_ORDER = ["hybrid", "dense", "bm25", "hybrid_no_compress"]
OVERLAP_THRESHOLD = 0.25


# ─── Data structures ──────────────────────────────────────────────────────────

@dataclass
class AltSampleResult:
    sample_id:          str
    category:           str
    question:           str
    ground_truth:       str
    experiment:         str
    answer:             str
    retrieved_contexts: list[str]
    n_retrieved:        int
    retrieval_ms:       float
    generation_ms:      float
    guard_passed:       bool = True
    guard_warnings:     list[str] = field(default_factory=list)
    hybrid_count:       int = 0
    dense_only_count:   int = 0
    bm25_only_count:    int = 0
    # local metrics
    context_recall_local:     Optional[float] = None
    context_precision_local:  Optional[float] = None
    answer_token_f1:          Optional[float] = None
    numeric_match:            Optional[float] = None
    guard_pass_rate:          Optional[float] = None
    # TLM metrics
    tlm_trustworthiness:      Optional[float] = None
    tlm_context_sufficiency:  Optional[float] = None
    tlm_response_helpfulness: Optional[float] = None
    tlm_error:                Optional[str] = None

    def scores(self) -> dict[str, Optional[float]]:
        return {k: getattr(self, k) for k in ALL_METRIC_KEYS}


@dataclass
class AltExperimentSummary:
    experiment:  str
    n_samples:   int
    metrics:     dict[str, dict[str, Any]]
    by_category: dict[str, dict[str, Any]]


# ─── Local metric functions ───────────────────────────────────────────────────

def _tokens(text: str) -> set[str]:
    return set(re.findall(r"[a-z0-9]+", text.lower()))


def token_f1(a: str, b: str) -> float:
    ta, tb = _tokens(a), _tokens(b)
    if not ta or not tb:
        return 0.0
    inter = len(ta & tb)
    prec = inter / len(ta)
    rec  = inter / len(tb)
    if prec + rec == 0:
        return 0.0
    return 2 * prec * rec / (prec + rec)


def chunk_overlap(a: str, b: str) -> float:
    return token_f1(a, b)


def context_recall_local(retrieved: list[str], reference: list[str]) -> float:
    if not reference:
        return 1.0
    hits = sum(
        1 for ref in reference
        if any(chunk_overlap(ref, r) >= OVERLAP_THRESHOLD for r in retrieved)
    )
    return hits / len(reference)


def context_precision_local(retrieved: list[str], reference: list[str]) -> float:
    if not retrieved:
        return 0.0
    hits = sum(
        1 for r in retrieved
        if any(chunk_overlap(r, ref) >= OVERLAP_THRESHOLD for ref in reference)
    )
    return hits / len(retrieved)


def extract_numbers(text: str) -> list[str]:
    raw = re.findall(r"\d[\d,\.%$kmb]*", text.lower())
    return [n.replace(",", "").replace("$", "") for n in raw]


def numeric_match_score(answer: str, ground_truth: str) -> float:
    gt_nums = extract_numbers(ground_truth)
    if not gt_nums:
        return token_f1(answer, ground_truth)
    ans = answer.lower().replace(",", "")
    hits = sum(1 for n in gt_nums if n in ans)
    return hits / len(gt_nums)


def compute_local_metrics(
    sample:    EvalSample,
    retrieved: list[str],
    answer:    str,
) -> dict[str, float]:
    return {
        "context_recall_local":    context_recall_local(retrieved, sample.reference_contexts),
        "context_precision_local": context_precision_local(retrieved, sample.reference_contexts),
        "answer_token_f1":         token_f1(answer, sample.ground_truth) if answer else 0.0,
        "numeric_match":           numeric_match_score(answer, sample.ground_truth) if answer else 0.0,
    }


# ─── TLM scoring ──────────────────────────────────────────────────────────────

_tlm_evaluator = None


def get_tlm_evaluator():
    global _tlm_evaluator
    if _tlm_evaluator is None:
        from cleanlab_tlm import TrustworthyRAG
        _tlm_evaluator = TrustworthyRAG(quality_preset="medium")
    return _tlm_evaluator


def _tlm_metric_score(result: dict, key: str) -> Optional[float]:
    entry = result.get(key)
    if entry is None:
        return None
    if isinstance(entry, dict):
        score = entry.get("score")
        return float(score) if score is not None else None
    return None


def score_with_tlm(
    query:    str,
    context:  str,
    response: str,
    prompt:   str,
) -> dict[str, Any]:
    try:
        evaluator = get_tlm_evaluator()
        result = evaluator.score(
            query=query,
            context=context,
            response=response,
            prompt=prompt,
        )
        # TrustworthyRAGScore behaves like a dict
        if hasattr(result, "items"):
            data = dict(result)
        else:
            data = result

        return {
            "tlm_trustworthiness":      _tlm_metric_score(data, "trustworthiness"),
            "tlm_context_sufficiency":  _tlm_metric_score(data, "context_sufficiency"),
            "tlm_response_helpfulness": _tlm_metric_score(data, "response_helpfulness"),
            "tlm_raw":                  {k: v for k, v in data.items() if k != "tlm_raw"},
        }
    except Exception as e:
        return {"tlm_error": str(e)}


# ─── Sample + aggregate ───────────────────────────────────────────────────────

def evaluate_sample_alt(
    sample:         EvalSample,
    retriever:      HybridRetriever,
    retrieve_fn:    callable,
    experiment:     str,
    use_tlm:        bool = True,
    retrieval_only: bool = False,
) -> AltSampleResult:
    t0 = time.perf_counter()
    results = retrieve_fn(retriever, sample)
    retrieval_ms = (time.perf_counter() - t0) * 1000

    retrieved_texts = [r.compressed_text for r in results]
    hybrid_count     = sum(1 for r in results if r.dense_rank and r.bm25_rank)
    dense_only_count = sum(1 for r in results if r.dense_rank and not r.bm25_rank)
    bm25_only_count  = sum(1 for r in results if r.bm25_rank and not r.dense_rank)

    answer = ""
    generation_ms = 0.0
    guard_passed = True
    guard_warnings: list[str] = []
    full_prompt = ""

    if not retrieval_only:
        t0 = time.perf_counter()
        if not results:
            answer = "No relevant context was found for this query."
        else:
            context_block = build_context_block(results)
            user_msg      = build_user_message(sample.question, context_block)
            full_prompt   = f"{SYSTEM_PROMPT}\n\n{user_msg}"
            try:
                answer, _, _ = call_llm(user_msg, model=GENERATOR_MODEL)
            except Exception as e:
                answer = f"[Generation error: {e}]"
        generation_ms = (time.perf_counter() - t0) * 1000

        if results and answer and not answer.startswith("[Generation error"):
            citations = parse_citations(answer, results)
            guard     = run_hallucination_guard(answer, results, citations)
            guard_passed   = guard.passed
            guard_warnings = guard.warnings

    local = compute_local_metrics(sample, retrieved_texts, answer)

    sr = AltSampleResult(
        sample_id          = sample.id,
        category           = sample.category,
        question           = sample.question,
        ground_truth       = sample.ground_truth,
        experiment         = experiment,
        answer             = answer,
        retrieved_contexts = retrieved_texts,
        n_retrieved        = len(results),
        retrieval_ms       = retrieval_ms,
        generation_ms      = generation_ms,
        guard_passed       = guard_passed,
        guard_warnings     = guard_warnings,
        hybrid_count       = hybrid_count,
        dense_only_count   = dense_only_count,
        bm25_only_count    = bm25_only_count,
        context_recall_local    = local["context_recall_local"],
        context_precision_local = local["context_precision_local"],
        answer_token_f1         = local["answer_token_f1"],
        numeric_match           = local["numeric_match"],
        guard_pass_rate         = 1.0 if guard_passed else 0.0,
    )

    if (
        use_tlm
        and not retrieval_only
        and answer
        and not answer.startswith("[Generation error")
        and os.environ.get("CLEANLAB_TLM_API_KEY")
    ):
        context_str = "\n\n".join(retrieved_texts)
        if not full_prompt:
            full_prompt = f"{SYSTEM_PROMPT}\n\nQuestion: {sample.question}\n\nContext:\n{context_str}"
        tlm = score_with_tlm(sample.question, context_str, answer, full_prompt)
        sr.tlm_trustworthiness      = tlm.get("tlm_trustworthiness")
        sr.tlm_context_sufficiency    = tlm.get("tlm_context_sufficiency")
        sr.tlm_response_helpfulness   = tlm.get("tlm_response_helpfulness")
        sr.tlm_error                = tlm.get("tlm_error")

    return sr


def aggregate_alt(results: list[AltSampleResult], experiment: str) -> AltExperimentSummary:
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

    overall = {k: stats([getattr(r, k) for r in results]) for k in ALL_METRIC_KEYS}

    by_cat: dict[str, list[AltSampleResult]] = defaultdict(list)
    for r in results:
        by_cat[r.category].append(r)

    by_category = {
        cat: {k: stats([getattr(r, k) for r in cat_results])["mean"] for k in ALL_METRIC_KEYS}
        for cat, cat_results in sorted(by_cat.items())
    }

    return AltExperimentSummary(
        experiment  = experiment,
        n_samples   = len(results),
        metrics     = overall,
        by_category = by_category,
    )


# ─── Report (Pydantic + UTF-8) ────────────────────────────────────────────────

def write_utf8(path: Path, content: str) -> None:
    path.write_text(content, encoding="utf-8")


class AltSampleRecord(BaseModel):
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
    context_recall_local:     Optional[float] = None
    context_precision_local:  Optional[float] = None
    answer_token_f1:          Optional[float] = None
    numeric_match:            Optional[float] = None
    guard_passed_score:       Optional[float] = Field(None, alias="guard_passed_metric")
    tlm_trustworthiness:      Optional[float] = None
    tlm_context_sufficiency:  Optional[float] = None
    tlm_response_helpfulness: Optional[float] = None
    tlm_error:                Optional[str] = None
    retrieved_contexts:       list[str] = Field(default_factory=list)


class AltSummaryRecord(BaseModel):
    experiment:  str
    n_samples:   int
    metrics:     dict[str, dict[str, Any]]
    by_category: dict[str, dict[str, Any]]


class AltEvaluationReport(BaseModel):
    generated_at:           str
    samples_per_experiment: int
    experiments:            list[str]
    retrieval_only:         bool
    tlm_enabled:            bool
    results:                dict[str, list[dict[str, Any]]]
    summaries:              dict[str, AltSummaryRecord]

    @classmethod
    def from_run(
        cls,
        all_results:    dict[str, list[AltSampleResult]],
        summaries:      dict[str, AltExperimentSummary],
        retrieval_only: bool,
        tlm_enabled:    bool,
    ) -> "AltEvaluationReport":
        results_out = {}
        for exp, rows in all_results.items():
            results_out[exp] = []
            for r in rows:
                d = {
                    "sample_id": r.sample_id, "category": r.category,
                    "question": r.question, "ground_truth": r.ground_truth,
                    "experiment": r.experiment, "answer": r.answer,
                    "n_retrieved": r.n_retrieved,
                    "retrieval_ms": r.retrieval_ms, "generation_ms": r.generation_ms,
                    "guard_passed": r.guard_passed, "guard_warnings": r.guard_warnings,
                    "hybrid_count": r.hybrid_count,
                    "dense_only_count": r.dense_only_count,
                    "bm25_only_count": r.bm25_only_count,
                    "retrieved_contexts": r.retrieved_contexts,
                    "tlm_error": r.tlm_error,
                    **r.scores(),
                }
                results_out[exp].append(d)

        summary_records = {
            exp: AltSummaryRecord(
                experiment=s.experiment, n_samples=s.n_samples,
                metrics=s.metrics, by_category=s.by_category,
            )
            for exp, s in summaries.items()
        }

        return cls(
            generated_at           = datetime.now().strftime("%Y-%m-%d %H:%M"),
            samples_per_experiment = len(next(iter(all_results.values()))) if all_results else 0,
            experiments            = list(all_results.keys()),
            retrieval_only         = retrieval_only,
            tlm_enabled            = tlm_enabled,
            results                = results_out,
            summaries              = summary_records,
        )

    def write_json(self, path: Path) -> None:
        write_utf8(path, self.model_dump_json(indent=2))

    def write_csv(self, path: Path) -> None:
        active_keys = [
            k for k in ALL_METRIC_KEYS
            if any(
                self.summaries[e].metrics.get(k, {}).get("mean") is not None
                for e in self.summaries
            )
        ] or LOCAL_METRIC_KEYS

        with open(path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            header = ["experiment", "n_samples"] + [f"{m}_mean" for m in active_keys] + [f"{m}_std" for m in active_keys]
            writer.writerow(header)
            for exp, summary in self.summaries.items():
                row = [exp, summary.n_samples]
                for m in active_keys:
                    row.append(summary.metrics[m].get("mean", ""))
                for m in active_keys:
                    row.append(summary.metrics[m].get("std", ""))
                writer.writerow(row)

    def to_markdown(self) -> str:
        gen_errors = sum(
            1 for samples in self.results.values()
            for s in samples if str(s.get("answer", "")).startswith("[Generation error")
        )
        total = sum(len(v) for v in self.results.values())

        lines = [
            "# RelIntel - Alternative RAG Evaluation Report",
            "",
            f"Generated: {self.generated_at}",
            f"Samples: {self.samples_per_experiment} per experiment",
            f"Experiments: {', '.join(self.experiments)}",
            f"TLM enabled: {self.tlm_enabled}",
            "",
        ]
        if gen_errors:
            lines.append(
                f"> **Warning:** {gen_errors}/{total} samples failed generation. "
                "Generation metrics may be empty."
            )
            lines.append("")

        lines += [
            "## Methods",
            "",
            "| Method | Metrics | Requires |",
            "|---|---|---|",
            "| **Local lexical** | context recall/precision, token F1, numeric match | nothing extra |",
            "| **Hallucination guard** | citation coverage, number grounding | generation |",
            "| **Cleanlab TLM** | trustworthiness, context sufficiency, response helpfulness | CLEANLAB_TLM_API_KEY |",
            "",
            "---",
            "",
            "## Aggregate Results",
            "",
        ]

        exp_list = list(self.summaries.keys())
        active_keys = [
            k for k in ALL_METRIC_KEYS
            if any(self.summaries[e].metrics.get(k, {}).get("mean") is not None for e in exp_list)
        ]

        if active_keys:
            header = "| Metric | " + " | ".join(exp_list) + " |"
            sep    = "|---|" + "---|" * len(exp_list)
            lines += [header, sep]
            for m in active_keys:
                cells = []
                for exp in exp_list:
                    mean = self.summaries[exp].metrics[m].get("mean")
                    std  = self.summaries[exp].metrics[m].get("std", 0)
                    cells.append(f"{mean:.3f} +/-{std:.3f}" if mean is not None else "-")
                lines.append(f"| **{METRIC_LABELS.get(m, m)}** | " + " | ".join(cells) + " |")
            lines.append("")

        for exp, summary in self.summaries.items():
            lines += [f"### {exp}", ""]
            for m in active_keys:
                mean = summary.metrics[m].get("mean")
                if mean is not None:
                    lines.append(f"- **{METRIC_LABELS.get(m, m)}**: {mean:.3f}")
            lines.append("")

        return "\n".join(lines)

    def write_markdown(self, path: Path) -> None:
        write_utf8(path, self.to_markdown())

    def write_plot(self, path: Path) -> None:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np

        exp_order = [e for e in EXPERIMENT_ORDER if e in self.summaries] or list(self.summaries.keys())
        if not exp_order:
            return

        local_keys = [k for k in LOCAL_METRIC_KEYS if k != "guard_pass_rate"]
        local_keys.append("guard_pass_rate")
        tlm_keys   = [k for k in TLM_METRIC_KEYS if any(
            self.summaries[e].metrics.get(k, {}).get("mean") is not None for e in exp_order
        )]

        panels = [("Local Metrics", local_keys)]
        if tlm_keys:
            panels.append(("TLM Metrics", tlm_keys))

        fig, axes = plt.subplots(1, len(panels), figsize=(6 * len(panels), 5.5), squeeze=False)
        palette = ["#2563eb", "#ea580c", "#16a34a", "#9333ea"]

        for ax, (title, metrics) in zip(axes[0], panels):
            x = np.arange(len(metrics))
            width = min(0.7 / len(exp_order), 0.18)
            for i, exp in enumerate(exp_order):
                means = [
                    self.summaries[exp].metrics[m].get("mean") or 0.0 for m in metrics
                ]
                stds = [
                    self.summaries[exp].metrics[m].get("std") or 0.0 for m in metrics
                ]
                offset = (i - (len(exp_order) - 1) / 2) * width
                ax.bar(
                    x + offset, means, width, yerr=stds,
                    label=EXPERIMENT_LABELS.get(exp, exp),
                    color=palette[i % len(palette)], capsize=4,
                    edgecolor="white", linewidth=0.5,
                )
            ax.set_xticks(x)
            ax.set_xticklabels([METRIC_LABELS.get(m, m) for m in metrics], fontsize=8)
            ax.set_ylim(0, 1.05)
            ax.set_ylabel("Score")
            ax.set_title(title, fontweight="bold")
            ax.grid(axis="y", alpha=0.25)
            ax.legend(loc="upper right", fontsize=8)

        fig.suptitle(
            f"RelIntel Alt Evaluation  |  {self.samples_per_experiment} samples/experiment",
            fontsize=13, fontweight="bold", y=1.02,
        )
        fig.tight_layout()
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)

    def write_all(self, output_dir: Path) -> None:
        self.write_json(output_dir / "alt_results.json")
        self.write_csv(output_dir / "alt_summary.csv")
        self.write_markdown(output_dir / "alt_report.md")
        self.write_plot(output_dir / "alt_comparison.png")

    @classmethod
    def load_json(cls, path: Path) -> "AltEvaluationReport":
        return cls.model_validate_json(path.read_text(encoding="utf-8"))


def write_report_alt(
    all_results:    dict[str, list[AltSampleResult]],
    summaries:      dict[str, AltExperimentSummary],
    retrieval_only: bool,
    tlm_enabled:    bool,
    output_dir:     Path,
) -> None:
    report = AltEvaluationReport.from_run(all_results, summaries, retrieval_only, tlm_enabled)
    report.write_all(output_dir)
    for name in ("alt_results.json", "alt_summary.csv", "alt_report.md", "alt_comparison.png"):
        print(f"  Saved -> {output_dir / name}")


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="RelIntel alternative RAG evaluation (TLM + local)")
    parser.add_argument("--experiment", choices=list(EXPERIMENTS.keys()) + ["all"], default="all")
    parser.add_argument("--quick", action="store_true", help="Hybrid only, 5 samples")
    parser.add_argument("--retrieval-only", action="store_true", help="Skip generation and TLM")
    parser.add_argument("--no-tlm", action="store_true", help="Skip Cleanlab TLM scoring")
    parser.add_argument("--sample-ids", nargs="+")
    parser.add_argument("--no-report", action="store_true")
    parser.add_argument("--plot-only", action="store_true", help="Regenerate from alt_results.json")
    args = parser.parse_args()

    if args.plot_only:
        path = RESULTS_DIR / "alt_results.json"
        if not path.exists():
            print(f"ERROR: {path} not found.")
            sys.exit(1)
        report = AltEvaluationReport.load_json(path)
        report.write_all(RESULTS_DIR)
        return

    use_tlm = not args.no_tlm and not args.retrieval_only
    if use_tlm and not os.environ.get("CLEANLAB_TLM_API_KEY"):
        print("NOTE: CLEANLAB_TLM_API_KEY not set — running local metrics only.")
        print("      Get a free key at https://tlm.cleanlab.ai/")
        use_tlm = False

    if not args.retrieval_only and not os.environ.get("ANTHROPIC_API_KEY"):
        print("ERROR: ANTHROPIC_API_KEY not set (or use --retrieval-only).")
        sys.exit(1)

    print("Loading retriever and dataset...")
    retriever = HybridRetriever.load()
    samples   = load_dataset()

    if args.quick:
        samples = [s for s in samples if s.id in ("S01", "S02", "S06", "S19", "S21")]
        experiments = ["hybrid"]
    elif args.experiment == "all":
        experiments = list(EXPERIMENTS.keys())
    else:
        experiments = [args.experiment]

    if args.sample_ids:
        samples = [s for s in samples if s.id in args.sample_ids]

    print(f"Experiments: {experiments}")
    print(f"Samples: {len(samples)}")
    print(f"TLM: {'on' if use_tlm else 'off'}")
    print(f"Mode: {'retrieval-only' if args.retrieval_only else 'full'}")
    print()

    all_results:  dict[str, list[AltSampleResult]] = {}
    all_summaries: dict[str, AltExperimentSummary] = {}

    for exp in experiments:
        retrieve_fn = EXPERIMENTS[exp]
        print(f"\n{'='*60}\n  Experiment: {exp}  ({len(samples)} samples)\n{'='*60}")

        exp_results = []
        for i, sample in enumerate(samples, 1):
            print(f"\n  [{i}/{len(samples)}] {sample.id} · {sample.category}")
            sr = evaluate_sample_alt(
                sample         = sample,
                retriever      = retriever,
                retrieve_fn    = retrieve_fn,
                experiment     = exp,
                use_tlm        = use_tlm,
                retrieval_only = args.retrieval_only,
            )
            exp_results.append(sr)

            score_str = " | ".join(
                f"{k.split('_')[0]}={getattr(sr, k):.2f}"
                for k in LOCAL_METRIC_KEYS[:4]
                if getattr(sr, k) is not None
            )
            tlm_str = f" tlm={sr.tlm_trustworthiness:.2f}" if sr.tlm_trustworthiness is not None else ""
            print(f"  {score_str}{tlm_str}")
            if sr.answer.startswith("[Generation error"):
                print(f"  ! {sr.answer[:80]}")

            if i < len(samples):
                time.sleep(SLEEP_BETWEEN_S)

        all_results[exp]   = exp_results
        all_summaries[exp] = aggregate_alt(exp_results, exp)

        print(f"\n  -- {exp} summary --")
        for m, stats in all_summaries[exp].metrics.items():
            if stats.get("mean") is not None:
                print(f"     {m:<28} mean={stats['mean']:.3f}  std={stats['std']:.3f}")

    if not args.no_report:
        print(f"\n{'-'*60}\nWriting reports...")
        write_report_alt(all_results, all_summaries, args.retrieval_only, use_tlm, RESULTS_DIR)

    print(f"\n{'='*60}\n  Alternative evaluation complete\n{'='*60}")


if __name__ == "__main__":
    main()
