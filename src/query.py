"""
Phase 4 — End-to-End Query Runner
RelIntel RAG System

Wires together:
  HybridRetriever (Phase 3)  →  generate() (Phase 4)  →  GenerationResult

Usage:
  python src/query.py                          # runs demo queries
  python src/query.py --query "your question"  # single query
  python src/query.py --interactive            # REPL mode

Environment:
  ANTHROPIC_API_KEY  — required for LLM calls
"""

import argparse
import sys
from pathlib import Path
from typing import Optional

sys.path.insert(0, str(Path(__file__).parent))

from generator import generate, format_result
from retriever import HybridRetriever, RetrievalQuery

SEP  = "─" * 60
SEP2 = "═" * 60

# ─── Demo Queries ─────────────────────────────────────────────────────────────
# Each entry exercises a different retrieval + generation capability

DEMO_QUERIES = [
    {
        "label":       "Q1 — Portfolio traction overview (no filters)",
        "query":       "Which companies are showing the strongest ARR growth and NRR?",
        "filters":     {},
        "top_k":       5,
        "description": "Broad semantic query — tests generation quality across full corpus",
    },
    {
        "label":       "Q2 — Fintech due diligence summary",
        "query":       "Summarise the key findings from our Fintech due diligence calls and meetings.",
        "filters":     {"sector": "Fintech", "source_type": "meeting"},
        "top_k":       5,
        "description": "Sector + source_type filter — tests scoped retrieval + synthesis",
    },
    {
        "label":       "Q3 — Term sheet pipeline",
        "query":       "What are the latest updates on companies at the Term Sheet stage?",
        "filters":     {"deal_stage": "Term Sheet"},
        "top_k":       5,
        "description": "Deal stage filter — tests deal-centric retrieval",
    },
    {
        "label":       "Q4 — Recent investor calls (last 6 months)",
        "query":       "What did founders say in recent investor update calls about burn rate and runway?",
        "filters":     {"source_type": "call", "date_from": "2025-11-01"},
        "top_k":       5,
        "description": "Source type + date range filter — tests temporal retrieval",
    },
    {
        "label":       "Q5 — Competitive moat signals",
        "query":       "Which portfolio companies have the clearest competitive moat and defensibility?",
        "filters":     {"sentiment": "positive"},
        "top_k":       5,
        "description": "Sentiment filter — tests signal-focused synthesis",
    },
]

# ─── Pipeline ─────────────────────────────────────────────────────────────────

def run_query(
    retriever:   HybridRetriever,
    query:       str,
    filters:     Optional[dict] = None,
    top_k:       int = 5,
) -> None:
    """Run one full RAG query: retrieve → generate → display."""
    filters = filters or {}

    rq = RetrievalQuery(
        text        = query,
        top_k       = top_k,
        sector      = filters.get("sector"),
        source_type = filters.get("source_type"),
        sentiment   = filters.get("sentiment"),
        deal_stage  = filters.get("deal_stage"),
        date_from   = filters.get("date_from"),
        date_to     = filters.get("date_to"),
        company_id  = filters.get("company_id"),
        logged_by   = filters.get("logged_by"),
    )

    results = retriever.search(rq)

    print(f"\n  Retrieved {len(results)} chunks:")
    for i, r in enumerate(results, 1):
        m = r.metadata
        print(f"    [{i}] rrf={r.rrf_score:.4f} | {m['company_name'][:28]} | "
              f"{m['source_type']} | {m['date']} | {m['deal_stage'] or '—'}")

    print()
    result = generate(query, results)
    print(format_result(result))

# ─── Demo Runner ──────────────────────────────────────────────────────────────

def run_demo(retriever: HybridRetriever):
    print(SEP2)
    print("  RelIntel — Phase 4: Generation Demo")
    print(SEP2)

    for demo in DEMO_QUERIES:
        print(f"\n{SEP2}")
        print(f"  {demo['label']}")
        print(f"  {demo['description']}")
        print(SEP2)
        if demo["filters"]:
            print(f"  Filters: {demo['filters']}")

        run_query(
            retriever = retriever,
            query     = demo["query"],
            filters   = demo["filters"],
            top_k     = demo["top_k"],
        )

    print(f"\n{SEP2}")
    print("  Demo complete ✅")
    print(SEP2)

# ─── Interactive REPL ─────────────────────────────────────────────────────────

def run_interactive(retriever: HybridRetriever):
    print(SEP2)
    print("  RelIntel — Interactive Query Mode")
    print("  Type 'quit' or Ctrl+C to exit")
    print("  Filter syntax: query [sector=Fintech] [source=meeting] [stage=Term Sheet]")
    print(SEP2)

    while True:
        try:
            raw = input("\nQuery> ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nExiting.")
            break

        if not raw or raw.lower() in ("quit", "exit", "q"):
            break

        # Parse inline filter hints: [key=value]
        filters = {}
        query   = raw
        for m in re.finditer(r'\[(\w+)=([^\]]+)\]', raw):
            k, v = m.group(1), m.group(2).strip()
            key_map = {
                "sector": "sector", "source": "source_type", "type": "source_type",
                "stage": "deal_stage", "sentiment": "sentiment",
                "from": "date_from", "to": "date_to",
            }
            if k in key_map:
                filters[key_map[k]] = v
            query = query.replace(m.group(0), "").strip()

        run_query(retriever, query, filters=filters)

# ─── Entry Point ──────────────────────────────────────────────────────────────

import re

def main():
    parser = argparse.ArgumentParser(description="RelIntel query runner")
    parser.add_argument("--query",       type=str,  help="Single query to run")
    parser.add_argument("--interactive", action="store_true", help="Start REPL mode")
    parser.add_argument("--top-k",       type=int,  default=5)
    parser.add_argument("--sector",      type=str)
    parser.add_argument("--source-type", type=str)
    parser.add_argument("--deal-stage",  type=str)
    parser.add_argument("--sentiment",   type=str)
    parser.add_argument("--date-from",   type=str)
    parser.add_argument("--date-to",     type=str)
    args = parser.parse_args()

    # Load retriever
    print("Loading RelIntel retriever...")
    retriever = HybridRetriever.load()

    if args.interactive:
        run_interactive(retriever)

    elif args.query:
        filters = {k: v for k, v in {
            "sector":      args.sector,
            "source_type": args.source_type,
            "deal_stage":  args.deal_stage,
            "sentiment":   args.sentiment,
            "date_from":   args.date_from,
            "date_to":     args.date_to,
        }.items() if v}
        run_query(retriever, args.query, filters=filters, top_k=args.top_k)

    else:
        # Default: run demo queries
        run_demo(retriever)


if __name__ == "__main__":
    main()
