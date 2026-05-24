"""
Phase 5 — RAGAS Evaluation Dataset
RelIntel RAG System

25 evaluation samples covering:
  - Question types: factual lookup, multi-hop synthesis, comparison,
    temporal, negation/absence, list retrieval
  - Retrieval paths: semantic, lexical (proper noun), metadata-filtered,
    date-ranged, multi-entity
  - Retrieval strategies under test: dense-only, BM25-only, hybrid

Schema per sample (RAGAS-compatible):
  question          : str   — the user query
  ground_truth      : str   — canonical correct answer (used for answer similarity)
  reference_contexts: list  — the ideal chunk texts that SHOULD be retrieved
  retrieval_filters : dict  — metadata pre-filters to apply (mirrors RetrievalQuery)
  category          : str   — question type label (for stratified analysis)
  top_k             : int   — how many chunks to retrieve for this question
  notes             : str   — rationale, retrieval difficulty, known edge cases

Dataset is generated from the actual RelIntel corpus so every ground_truth
answer and reference_context is grounded in real synthetic data.

Usage:
    from eval_dataset import load_dataset, EvalSample
    samples = load_dataset()

    # Or run standalone to print a summary:
    python src/eval_dataset.py
"""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

ROOT     = Path(__file__).parent.parent
DATA_DIR = ROOT / "data"


# ─── Schema ───────────────────────────────────────────────────────────────────

@dataclass
class EvalSample:
    id:                str
    question:          str
    ground_truth:      str
    reference_contexts: list[str]           # ideal retrieved chunk texts
    retrieval_filters: dict                 # maps to RetrievalQuery fields
    category:          str
    top_k:             int = 5
    notes:             str = ""

    def to_ragas_dict(self) -> dict:
        """Convert to the dict format ragas.evaluate() expects."""
        return {
            "question":          self.question,
            "ground_truth":      self.ground_truth,
            "reference_contexts": self.reference_contexts,
        }


# ─── Dataset Builder ──────────────────────────────────────────────────────────

def build_dataset(chunks: list[dict], companies: list[dict], contacts: list[dict],
                  deals: list[dict], interactions: list[dict]) -> list[EvalSample]:
    """
    Build evaluation samples that are anchored to real corpus content.
    We look up actual data values so ground_truth answers are verifiable.
    """

    # Index lookups
    co_by_id  = {c["company_id"]:  c for c in companies}
    ct_by_id  = {c["contact_id"]:  c for c in contacts}
    dl_by_id  = {d["deal_id"]:     d for d in deals}
    ix_by_id  = {i["interaction_id"]: i for i in interactions}

    # Group interactions by company
    ix_by_co: dict[str, list[dict]] = {}
    for ix in interactions:
        ix_by_co.setdefault(ix["company_id"], []).append(ix)

    # Group deals by stage
    deals_by_stage: dict[str, list[dict]] = {}
    for d in deals:
        deals_by_stage.setdefault(d["stage"], []).append(d)

    # Pick concrete anchors from the data for determinism
    # (sorted for reproducibility — no random)
    fintech_cos  = sorted([c for c in companies if c["sector"] == "Fintech"],
                          key=lambda x: x["company_id"])
    health_cos   = sorted([c for c in companies if c["sector"] == "HealthTech"],
                          key=lambda x: x["company_id"])
    ts_deals     = sorted(deals_by_stage.get("Term Sheet", []),
                          key=lambda x: x["deal_id"])
    dd_deals     = sorted(deals_by_stage.get("Due Diligence", []),
                          key=lambda x: x["deal_id"])
    won_deals    = sorted(deals_by_stage.get("Closed Won", []),
                          key=lambda x: x["deal_id"])

    # Pick a company with ≥4 interactions for multi-turn questions
    prolific_co = sorted(
        [(cid, ixs) for cid, ixs in ix_by_co.items() if len(ixs) >= 4],
        key=lambda x: x[0]
    )[0]
    prolific_co_data = co_by_id[prolific_co[0]]
    prolific_ixs     = sorted(prolific_co[1], key=lambda x: x["date"])

    # Grab specific interaction bodies for reference_contexts
    def body_for(ix: dict) -> str:
        """Reconstruct the full chunk text as stored in ChromaDB."""
        co = co_by_id.get(ix["company_id"], {})
        ct_names = []
        for cid in ix.get("contact_ids", []):
            c = ct_by_id.get(cid, {})
            if c:
                ct_names.append(f"{c['first_name']} {c['last_name']} ({c['title']})")
        dl = dl_by_id.get(ix.get("deal_id"), {})
        ctx_lines = [
            f"Company: {co.get('name','Unknown')} [{co.get('sector','')}]",
            f"Interaction: {ix['type'].replace('_',' ').title()} on {ix['date'][:10]}",
        ]
        if ct_names:
            ctx_lines.append(f"Contacts: {', '.join(ct_names)}")
        if dl:
            ctx_lines.append(
                f"Deal: {dl.get('name','')} | Stage: {dl.get('stage','')} "
                f"| ${dl.get('amount_usd',0):,}"
            )
        return "\n".join(ctx_lines) + "\n\n" + ix["body"]

    samples: list[EvalSample] = []

    # ── CATEGORY 1: Factual Lookup ────────────────────────────────────────────
    # Tests precise retrieval of a single data point; ideal for context_precision

    # S01 — Burn rate from a specific meeting
    meeting_ixs = sorted(
        [ix for ix in interactions if ix["type"] == "meeting"
         and "burn rate" in ix["body"].lower()],
        key=lambda x: x["interaction_id"]
    )
    if meeting_ixs:
        m = meeting_ixs[0]
        co = co_by_id[m["company_id"]]
        # Extract burn figure from body
        import re
        burn_match = re.search(r'\$[\d,]+K/month', m["body"])
        burn_str = burn_match.group() if burn_match else "an undisclosed amount"
        runway_match = re.search(r'(\d+) months runway', m["body"])
        runway_str = runway_match.group() if runway_match else "unknown runway"
        samples.append(EvalSample(
            id     = "S01",
            category = "factual_lookup",
            question = f"What is {co['name']}'s current burn rate and runway?",
            ground_truth = (
                f"{co['name']}'s burn rate is {burn_str} with {runway_str}."
            ),
            reference_contexts = [body_for(m)],
            retrieval_filters  = {"company_id": co["company_id"]},
            top_k  = 5,
            notes  = "Single-fact lookup. Tests context_precision — irrelevant "
                     "chunks from the same company should rank below this one.",
        ))

    # S02 — NPS score lookup
    nps_ixs = sorted(
        [ix for ix in interactions if "NPS is" in ix["body"]],
        key=lambda x: x["interaction_id"]
    )
    if nps_ixs:
        m = nps_ixs[0]
        co = co_by_id[m["company_id"]]
        nps_match = re.search(r'NPS is (\d+)', m["body"])
        churn_match = re.search(r'churn is below (\d+%)', m["body"])
        nps_val   = nps_match.group(1)   if nps_match   else "N/A"
        churn_val = churn_match.group(1) if churn_match else "N/A"
        samples.append(EvalSample(
            id     = "S02",
            category = "factual_lookup",
            question = f"What is {co['name']}'s NPS score and customer churn rate?",
            ground_truth = (
                f"{co['name']} has an NPS of {nps_val} and customer churn below {churn_val}."
            ),
            reference_contexts = [body_for(m)],
            retrieval_filters  = {"company_id": co["company_id"]},
            top_k = 5,
            notes = "Factual extraction from a single interaction body.",
        ))

    # S03 — Patent count lookup
    patent_ixs = sorted(
        [ix for ix in interactions if "patents pending" in ix["body"]],
        key=lambda x: x["interaction_id"]
    )
    if patent_ixs:
        m = patent_ixs[0]
        co = co_by_id[m["company_id"]]
        pat_match = re.search(r'(\d+) patents pending', m["body"])
        pat_count = pat_match.group(1) if pat_match else "several"
        samples.append(EvalSample(
            id     = "S03",
            category = "factual_lookup",
            question = f"How many patents does {co['name']} have pending?",
            ground_truth = f"{co['name']} has {pat_count} patents pending.",
            reference_contexts = [body_for(m)],
            retrieval_filters  = {"company_id": co["company_id"]},
            top_k = 5,
            notes = "Numeric extraction. Validates ungrounded_number detection in guard.",
        ))

    # ── CATEGORY 2: Multi-hop Synthesis ──────────────────────────────────────
    # Requires combining signals from 2+ chunks; tests context_recall

    # S04 — Company relationship history summary
    co_data = prolific_co_data
    co_ixs  = prolific_ixs[:4]
    types_seen = list({ix["type"] for ix in co_ixs})
    samples.append(EvalSample(
        id     = "S04",
        category = "multi_hop_synthesis",
        question = (
            f"Summarise our full relationship history with {co_data['name']}, "
            f"including interaction types, key topics, and next steps."
        ),
        ground_truth = (
            f"Our relationship with {co_data['name']} ({co_data['sector']}) "
            f"spans multiple interaction types including "
            f"{', '.join(types_seen[:3])}. "
            f"Key topics covered: "
            + "; ".join(ix["body"][:60] for ix in co_ixs[:2]) + "."
        ),
        reference_contexts = [body_for(ix) for ix in co_ixs],
        retrieval_filters  = {"company_id": co_data["company_id"]},
        top_k = 8,
        notes = (
            "Multi-chunk synthesis. Tests context_recall — all 4 interaction "
            "chunks should be retrieved. Stresses RRF fusion diversity."
        ),
    ))

    # S05 — Synthesis across sector
    if fintech_cos:
        fc = fintech_cos[0]
        fc_ixs = sorted(ix_by_co.get(fc["company_id"], [])[:3],
                        key=lambda x: x["date"], reverse=True)
        samples.append(EvalSample(
            id     = "S05",
            category = "multi_hop_synthesis",
            question = (
                f"What are the key investment signals we have gathered for "
                f"{fc['name']} across all interaction types?"
            ),
            ground_truth = (
                f"{fc['name']} is a {fc['sector']} company at {fc['stage']} stage. "
                f"Interactions include: "
                + " | ".join(ix["body"][:80] for ix in fc_ixs[:2]) + "."
            ),
            reference_contexts = [body_for(ix) for ix in fc_ixs],
            retrieval_filters  = {
                "company_id": fc["company_id"],
                "sector":     "Fintech",
            },
            top_k = 6,
            notes = "Sector + company filter. Tests whether hybrid retrieval "
                    "surfaces both meeting notes and email updates for one company.",
        ))

    # S06 — ARR growth across companies (cross-company synthesis)
    arr_ixs = sorted(
        [ix for ix in interactions if "YoY growth" in ix["body"]],
        key=lambda x: x["interaction_id"]
    )[:3]
    arr_snippets = [ix["body"] for ix in arr_ixs]
    samples.append(EvalSample(
        id     = "S06",
        category = "multi_hop_synthesis",
        question = "Which companies have shown the strongest YoY growth? Give specific figures.",
        ground_truth = (
            "Several companies have reported strong YoY growth: "
            + "; ".join(
                f"{co_by_id[ix['company_id']]['name']} — {ix['body'][:80]}"
                for ix in arr_ixs[:3]
            ) + "."
        ),
        reference_contexts = [body_for(ix) for ix in arr_ixs[:3]],
        retrieval_filters  = {},
        top_k = 8,
        notes = (
            "Cross-company synthesis. Tests whether dense retrieval surfaces "
            "semantically similar growth narratives. Good faithfulness test — "
            "LLM must cite specific percentages from context."
        ),
    ))

    # ── CATEGORY 3: Comparison ────────────────────────────────────────────────
    # Tests ranking and differentiation across multiple entities

    # S07 — Compare two companies on competitive moat
    moat_ixs = sorted(
        [ix for ix in interactions if "proprietary data" in ix["body"]
         or "patents pending" in ix["body"]],
        key=lambda x: x["interaction_id"]
    )[:2]
    if len(moat_ixs) >= 2:
        co_a = co_by_id[moat_ixs[0]["company_id"]]
        co_b = co_by_id[moat_ixs[1]["company_id"]]
        samples.append(EvalSample(
            id     = "S07",
            category = "comparison",
            question = (
                f"Compare the competitive moats of {co_a['name']} and {co_b['name']}. "
                f"Which appears stronger and why?"
            ),
            ground_truth = (
                f"{co_a['name']}: {moat_ixs[0]['body']}. "
                f"{co_b['name']}: {moat_ixs[1]['body']}."
            ),
            reference_contexts = [body_for(moat_ixs[0]), body_for(moat_ixs[1])],
            retrieval_filters  = {},
            top_k = 6,
            notes = (
                "Two-entity comparison. Tests whether retriever surfaces chunks "
                "for BOTH companies. BM25 advantage: exact company name matching."
            ),
        ))

    # S08 — Deal stage comparison: Term Sheet vs Due Diligence
    ts_sample = ts_deals[0] if ts_deals else None
    dd_sample = dd_deals[0] if dd_deals else None
    if ts_sample and dd_sample:
        ts_co = co_by_id.get(ts_sample["company_id"], {})
        dd_co = co_by_id.get(dd_sample["company_id"], {})
        ts_ixs = sorted(ix_by_co.get(ts_sample["company_id"], [])[:1],
                        key=lambda x: x["date"])
        dd_ixs = sorted(ix_by_co.get(dd_sample["company_id"], [])[:1],
                        key=lambda x: x["date"])
        samples.append(EvalSample(
            id     = "S08",
            category = "comparison",
            question = (
                f"Compare the status of deals for {ts_co.get('name','N/A')} "
                f"(Term Sheet) and {dd_co.get('name','N/A')} (Due Diligence). "
                f"What are the key differences in our engagement with each?"
            ),
            ground_truth = (
                f"{ts_co.get('name')} is at Term Sheet stage "
                f"(${ts_sample['amount_usd']:,}). "
                f"{dd_co.get('name')} is under Due Diligence "
                f"(${dd_sample['amount_usd']:,})."
            ),
            reference_contexts = (
                [body_for(ix) for ix in ts_ixs] +
                [body_for(ix) for ix in dd_ixs]
            ),
            retrieval_filters  = {},
            top_k = 6,
            notes = "Deal-stage comparison. Tests deal_stage metadata filter effectiveness.",
        ))

    # ── CATEGORY 4: Temporal ─────────────────────────────────────────────────
    # Tests date-range filtering and recency awareness

    # S09 — Most recent interactions (last 90 days)
    recent_ixs = sorted(
        [ix for ix in interactions if ix["date"][:10] >= "2026-02-23"],
        key=lambda x: x["date"], reverse=True
    )[:4]
    recent_cos = [co_by_id[ix["company_id"]]["name"] for ix in recent_ixs[:4]]
    samples.append(EvalSample(
        id     = "S09",
        category = "temporal",
        question = "What are the most recent interactions in our pipeline from the last 90 days?",
        ground_truth = (
            f"The most recent interactions include activities with: "
            f"{', '.join(dict.fromkeys(recent_cos))}."
        ),
        reference_contexts = [body_for(ix) for ix in recent_ixs],
        retrieval_filters  = {"date_from": "2026-02-23"},
        top_k = 5,
        notes = (
            "Temporal filter test. Validates that date_from pre-filter is applied "
            "before both dense and BM25 retrieval. All returned chunks must be "
            "within the date range."
        ),
    ))

    # S10 — Investor update calls in a specific window
    update_calls = sorted(
        [ix for ix in interactions
         if ix["type"] == "call"
         and "update" in ix["body"].lower()
         and ix["date"][:10] >= "2025-07-01"
         and ix["date"][:10] <= "2026-01-01"],
        key=lambda x: x["date"], reverse=True
    )[:3]
    samples.append(EvalSample(
        id     = "S10",
        category = "temporal",
        question = (
            "What did portfolio founders say in investor update calls "
            "between July 2025 and January 2026?"
        ),
        ground_truth = (
            "Investor update calls in H2 2025 included: "
            + "; ".join(ix["body"][:80] for ix in update_calls[:3])
            + "."
        ),
        reference_contexts = [body_for(ix) for ix in update_calls[:3]],
        retrieval_filters  = {
            "source_type": "call",
            "date_from":   "2025-07-01",
            "date_to":     "2026-01-01",
        },
        top_k = 5,
        notes = "Date range + source_type compound filter. H2 2025 window.",
    ))

    # S11 — Oldest vs newest interactions for a company
    if prolific_ixs:
        oldest = prolific_ixs[0]
        newest = prolific_ixs[-1]
        samples.append(EvalSample(
            id     = "S11",
            category = "temporal",
            question = (
                f"How has our relationship with {prolific_co_data['name']} "
                f"evolved from our earliest to most recent interaction?"
            ),
            ground_truth = (
                f"Earliest: {oldest['date'][:10]} — {oldest['body'][:100]}. "
                f"Most recent: {newest['date'][:10]} — {newest['body'][:100]}."
            ),
            reference_contexts = [body_for(oldest), body_for(newest)],
            retrieval_filters  = {"company_id": prolific_co_data["company_id"]},
            top_k = 8,
            notes = (
                "Temporal ordering within a single company. Tests whether the "
                "retriever surfaces the extremes (oldest + newest) not just the "
                "most semantically central chunks."
            ),
        ))

    # ── CATEGORY 5: Source-Type Filtered ─────────────────────────────────────
    # Tests source_type metadata filter precision

    # S12 — Due diligence meetings only
    dd_meetings = sorted(
        [ix for ix in interactions
         if ix["type"] == "meeting"
         and "due diligence" in ix["body"].lower()],
        key=lambda x: x["interaction_id"]
    )[:3]
    samples.append(EvalSample(
        id     = "S12",
        category = "source_filtered",
        question = "Summarise the key findings from our due diligence meetings.",
        ground_truth = (
            "Due diligence meetings covered: "
            + "; ".join(
                f"{co_by_id[ix['company_id']]['name']}: {ix['body'][:80]}"
                for ix in dd_meetings[:3]
            ) + "."
        ),
        reference_contexts = [body_for(ix) for ix in dd_meetings],
        retrieval_filters  = {"source_type": "meeting"},
        top_k = 6,
        notes = "source_type=meeting filter. All returned chunks must be meetings.",
    ))

    # S13 — LinkedIn messages only
    li_ixs = sorted(
        [ix for ix in interactions if ix["type"] == "linkedin_message"],
        key=lambda x: x["interaction_id"]
    )[:3]
    samples.append(EvalSample(
        id     = "S13",
        category = "source_filtered",
        question = "What topics have come up in our LinkedIn message threads with founders?",
        ground_truth = (
            "LinkedIn messages covered: "
            + "; ".join(ix["body"][:80] for ix in li_ixs[:3])
            + "."
        ),
        reference_contexts = [body_for(ix) for ix in li_ixs],
        retrieval_filters  = {"source_type": "linkedin_message"},
        top_k = 5,
        notes = "source_type=linkedin_message. Tests exact filter match on enum field.",
    ))

    # ── CATEGORY 6: Sentiment Filtered ───────────────────────────────────────

    # S14 — Negative sentiment signals
    neg_ixs = sorted(
        [ix for ix in interactions if ix["sentiment"] == "negative"],
        key=lambda x: x["interaction_id"]
    )[:3]
    samples.append(EvalSample(
        id     = "S14",
        category = "sentiment_filtered",
        question = (
            "Are there any companies in our pipeline where we logged concerns "
            "or negative signals? What were they?"
        ),
        ground_truth = (
            "Negative sentiment interactions include: "
            + "; ".join(
                f"{co_by_id[ix['company_id']]['name']}: {ix['body'][:80]}"
                for ix in neg_ixs[:3]
            ) + "."
        ),
        reference_contexts = [body_for(ix) for ix in neg_ixs],
        retrieval_filters  = {"sentiment": "negative"},
        top_k = 5,
        notes = "sentiment=negative filter. Important for risk flagging use case.",
    ))

    # S15 — Positive sentiment for a specific sector
    pos_health = sorted(
        [ix for ix in interactions
         if ix["sentiment"] == "positive"
         and co_by_id.get(ix["company_id"], {}).get("sector") == "HealthTech"],
        key=lambda x: x["interaction_id"]
    )[:3]
    samples.append(EvalSample(
        id     = "S15",
        category = "sentiment_filtered",
        question = "What are the most promising HealthTech opportunities we've engaged with?",
        ground_truth = (
            "Positive HealthTech interactions: "
            + "; ".join(
                f"{co_by_id[ix['company_id']]['name']}: {ix['body'][:80]}"
                for ix in pos_health[:3]
            ) + "."
        ),
        reference_contexts = [body_for(ix) for ix in pos_health],
        retrieval_filters  = {"sector": "HealthTech", "sentiment": "positive"},
        top_k = 5,
        notes = "sector + sentiment compound filter. Tests pre-filter intersection.",
    ))

    # ── CATEGORY 7: Deal-Stage Filtered ──────────────────────────────────────

    # S16 — Term Sheet updates
    ts_ixs_all = sorted(
        [ix for ix in interactions
         if dl_by_id.get(ix.get("deal_id"), {}).get("stage") == "Term Sheet"],
        key=lambda x: x["interaction_id"]
    )[:4]
    samples.append(EvalSample(
        id     = "S16",
        category = "deal_stage_filtered",
        question = "Which deals are currently at Term Sheet stage and what are the latest updates?",
        ground_truth = (
            "Active Term Sheet deals: "
            + "; ".join(
                f"{co_by_id[ix['company_id']]['name']}: {ix['body'][:80]}"
                for ix in ts_ixs_all[:3]
            ) + "."
        ),
        reference_contexts = [body_for(ix) for ix in ts_ixs_all[:3]],
        retrieval_filters  = {"deal_stage": "Term Sheet"},
        top_k = 6,
        notes = "deal_stage filter. Critical for pipeline management use case.",
    ))

    # S17 — Closed Won retrospective
    won_ixs = sorted(
        [ix for ix in interactions
         if dl_by_id.get(ix.get("deal_id"), {}).get("stage") == "Closed Won"],
        key=lambda x: x["interaction_id"]
    )[:3]
    samples.append(EvalSample(
        id     = "S17",
        category = "deal_stage_filtered",
        question = (
            "For our Closed Won deals, what were the key signals that indicated "
            "strong investment potential?"
        ),
        ground_truth = (
            "Closed Won signals: "
            + "; ".join(
                f"{co_by_id[ix['company_id']]['name']}: {ix['body'][:80]}"
                for ix in won_ixs[:3]
            ) + "."
        ),
        reference_contexts = [body_for(ix) for ix in won_ixs],
        retrieval_filters  = {"deal_stage": "Closed Won"},
        top_k = 5,
        notes = "Retrospective analysis of won deals. Good faithfulness stress test.",
    ))

    # ── CATEGORY 8: Logged-By / Team Member ──────────────────────────────────

    # S18 — Interactions logged by a specific team member
    team_members = {
        "usr_001": "Priya Mehta",
        "usr_002": "James Okafor",
        "usr_003": "Sarah Chen",
    }
    tm_id   = "usr_001"
    tm_name = team_members[tm_id]
    tm_ixs  = sorted(
        [ix for ix in interactions if ix.get("logged_by") == tm_id],
        key=lambda x: x["date"], reverse=True
    )[:3]
    samples.append(EvalSample(
        id     = "S18",
        category = "team_filtered",
        question = f"What are the most recent interactions logged by {tm_name}?",
        ground_truth = (
            f"Recent interactions logged by {tm_name}: "
            + "; ".join(
                f"{co_by_id[ix['company_id']]['name']} ({ix['date'][:10]}): {ix['body'][:60]}"
                for ix in tm_ixs[:3]
            ) + "."
        ),
        reference_contexts = [body_for(ix) for ix in tm_ixs],
        retrieval_filters  = {"logged_by": tm_id},
        top_k = 5,
        notes = "logged_by filter. Team productivity / coverage audit use case.",
    ))

    # ── CATEGORY 9: Lexical / Proper Noun ────────────────────────────────────
    # BM25 should dominate here; tests hybrid complementarity

    # S19 — Exact company name lookup
    target_co = sorted(companies, key=lambda x: x["company_id"])[0]
    target_ixs = sorted(ix_by_co.get(target_co["company_id"], []),
                        key=lambda x: x["date"])[:3]
    samples.append(EvalSample(
        id     = "S19",
        category = "lexical",
        question = f"What do we know about {target_co['name']}?",
        ground_truth = (
            f"{target_co['name']} is a {target_co['sector']} company "
            f"({target_co['stage']} stage, {target_co['hq_country']}). "
            f"Recent interactions: "
            + "; ".join(ix["body"][:80] for ix in target_ixs[:2])
            + "."
        ),
        reference_contexts = [body_for(ix) for ix in target_ixs],
        retrieval_filters  = {"company_id": target_co["company_id"]},
        top_k = 6,
        notes = (
            "Exact company name query. BM25 should rank this top-1. "
            "Dense retrieval may struggle if company name is rare in embedding space. "
            "Tests hybrid complementarity."
        ),
    ))

    # S20 — Contact name lookup
    target_ct = sorted(contacts, key=lambda x: x["contact_id"])[0]
    ct_ixs    = sorted(
        [ix for ix in interactions if target_ct["contact_id"] in ix.get("contact_ids", [])],
        key=lambda x: x["date"]
    )[:3]
    if ct_ixs:
        samples.append(EvalSample(
            id     = "S20",
            category = "lexical",
            question = (
                f"What interactions have we had with "
                f"{target_ct['first_name']} {target_ct['last_name']}?"
            ),
            ground_truth = (
                f"Interactions with {target_ct['first_name']} {target_ct['last_name']} "
                f"({target_ct['title']} at {co_by_id[target_ct['company_id']]['name']}): "
                + "; ".join(f"{ix['date'][:10]}: {ix['body'][:80]}" for ix in ct_ixs[:2])
                + "."
            ),
            reference_contexts = [body_for(ix) for ix in ct_ixs],
            retrieval_filters  = {"company_id": target_ct["company_id"]},
            top_k = 5,
            notes = (
                "Person name lookup — strong BM25 signal. "
                "The contact name appears verbatim in interaction bodies. "
                "Tests whether BM25 outperforms dense on named-entity queries."
            ),
        ))

    # ── CATEGORY 10: Absence / Negation ──────────────────────────────────────
    # Tests whether the model correctly identifies missing information

    # S21 — Question about a nonexistent company
    samples.append(EvalSample(
        id     = "S21",
        category = "negation_absence",
        question = "What do we know about QuantumLeap Ventures?",
        ground_truth = (
            "There is no information about QuantumLeap Ventures in the database."
        ),
        reference_contexts = [],   # no relevant context exists
        retrieval_filters  = {},
        top_k = 3,
        notes = (
            "Nonexistent entity. The RAG system should return 'no relevant context' "
            "or clearly state it cannot find information. Faithfulness should be 1.0 "
            "(no hallucination) but answer_relevance may be low."
        ),
    ))

    # S22 — Question about data not in corpus
    samples.append(EvalSample(
        id     = "S22",
        category = "negation_absence",
        question = (
            "Which companies have received FDA approval for their products?"
        ),
        ground_truth = (
            "The available interaction data does not contain specific information "
            "about FDA approvals."
        ),
        reference_contexts = [],
        retrieval_filters  = {},
        top_k = 5,
        notes = (
            "Domain-specific question with no corpus coverage. "
            "Tests hallucination risk: LLM must say 'not in context' rather "
            "than inferring from general knowledge."
        ),
    ))

    # ── CATEGORY 11: Aggregation / List ──────────────────────────────────────

    # S23 — All companies in a sector
    deep_cos = sorted(
        [c for c in companies if c["sector"] == "Deep Tech"],
        key=lambda x: x["company_id"]
    )
    deep_names = [c["name"] for c in deep_cos[:5]]
    samples.append(EvalSample(
        id     = "S23",
        category = "aggregation",
        question = "List all Deep Tech companies in our pipeline.",
        ground_truth = (
            f"Deep Tech companies in the pipeline include: {', '.join(deep_names)}."
        ),
        reference_contexts = [
            body_for(ix_by_co[c["company_id"]][0])
            for c in deep_cos[:3]
            if c["company_id"] in ix_by_co and ix_by_co[c["company_id"]]
        ],
        retrieval_filters  = {"sector": "Deep Tech"},
        top_k = 10,
        notes = (
            "Exhaustive enumeration request. Context_recall is the key metric — "
            "the retriever must surface at least one chunk per Deep Tech company. "
            "Tests whether top_k=10 is sufficient for full coverage."
        ),
    ))

    # S24 — Companies with strong competitive moat
    moat_ixs_all = sorted(
        [ix for ix in interactions
         if "proprietary data" in ix["body"] and ix["sentiment"] == "positive"],
        key=lambda x: x["interaction_id"]
    )[:5]
    moat_cos = list(dict.fromkeys(
        co_by_id[ix["company_id"]]["name"] for ix in moat_ixs_all
    ))
    samples.append(EvalSample(
        id     = "S24",
        category = "aggregation",
        question = (
            "Which companies in our pipeline have strong competitive moats "
            "based on proprietary data advantages?"
        ),
        ground_truth = (
            f"Companies with proprietary data moats: {', '.join(moat_cos[:5])}."
        ),
        reference_contexts = [body_for(ix) for ix in moat_ixs_all[:3]],
        retrieval_filters  = {"sentiment": "positive"},
        top_k = 8,
        notes = "Multi-entity aggregation. Requires recalling multiple companies.",
    ))

    # S25 — Runway comparison across companies
    runway_ixs = sorted(
        [ix for ix in interactions if "months runway" in ix["body"]],
        key=lambda x: x["interaction_id"]
    )[:4]
    samples.append(EvalSample(
        id     = "S25",
        category = "aggregation",
        question = (
            "Which companies have the longest runway based on our call notes? "
            "Give specific figures."
        ),
        ground_truth = (
            "Companies with runway data: "
            + "; ".join(
                f"{co_by_id[ix['company_id']]['name']}: {ix['body'][:80]}"
                for ix in runway_ixs[:4]
            ) + "."
        ),
        reference_contexts = [body_for(ix) for ix in runway_ixs],
        retrieval_filters  = {},
        top_k = 8,
        notes = (
            "Numeric comparison across entities. Faithfulness check: answer must "
            "cite actual runway figures from chunks, not invented numbers."
        ),
    ))

    return samples


# ─── Load / Save ──────────────────────────────────────────────────────────────

def load_dataset() -> list[EvalSample]:
    """Load or (re)build the evaluation dataset from the corpus."""
    with open(DATA_DIR / "interactions.json") as f:
        interactions = json.load(f)
    with open(DATA_DIR / "companies.json") as f:
        companies = json.load(f)
    with open(DATA_DIR / "contacts.json") as f:
        contacts = json.load(f)
    with open(DATA_DIR / "deals.json") as f:
        deals = json.load(f)
    with open(DATA_DIR / "chunks.json") as f:
        chunks = json.load(f)

    return build_dataset(chunks, companies, contacts, deals, interactions)


def save_dataset(samples: list[EvalSample], path: Optional[Path] = None) -> Path:
    path = path or (DATA_DIR / "eval_dataset.json")
    with open(path, "w") as f:
        json.dump(
            [{"id": s.id, "category": s.category, "question": s.question,
              "ground_truth": s.ground_truth,
              "reference_contexts": s.reference_contexts,
              "retrieval_filters": s.retrieval_filters,
              "top_k": s.top_k, "notes": s.notes}
             for s in samples],
            f, indent=2,
        )
    return path


# ─── CLI Summary ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import re
    from collections import Counter

    print("Building evaluation dataset from corpus...")
    samples = load_dataset()

    path = save_dataset(samples)
    print(f"Saved {len(samples)} samples → {path}")

    cats = Counter(s.category for s in samples)
    print(f"\nCategory distribution:")
    for cat, n in sorted(cats.items()):
        print(f"  {cat:<28} {n}")

    print(f"\nSample preview (first 3):")
    for s in samples[:3]:
        print(f"\n  [{s.id}] {s.category}")
        print(f"  Q: {s.question[:90]}")
        print(f"  A: {s.ground_truth[:90]}")
        print(f"  Filters: {s.retrieval_filters}")
        print(f"  Ref contexts: {len(s.reference_contexts)}")
