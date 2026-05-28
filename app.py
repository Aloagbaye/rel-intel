"""
RelIntel — Relationship Intelligence RAG System
Hugging Face Spaces deployment

Architecture on Spaces:
  - ChromaDB loads from bundled data/chroma/ at startup (persisted in repo)
  - BM25 index rebuilds in-memory from data/chunks.json (~0.3s)
  - embedder.pkl (TF-IDF + LSA) loaded from data/ — no external downloads
  - Anthropic API key injected via HF Space Secret (ANTHROPIC_API_KEY)
  - Single-process, stateless per query — safe for shared Spaces
"""

import os
import sys
import time
import pickle
from pathlib import Path
from dotenv import load_dotenv

import gradio as gr

load_dotenv()

# ── Path setup ────────────────────────────────────────────────────────────────
ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "observability"))

from retriever import HybridRetriever, RetrievalQuery
from generator import generate, build_context_block, DEFAULT_MODEL

# ── Observability ─────────────────────────────────────────────────────────────
try:
    from event_store import log_event
    _OBSERVABILITY = True
except ImportError:
    _OBSERVABILITY = False

# ── Constants ─────────────────────────────────────────────────────────────────
SECTORS = [
    "Any", "Climate Tech", "Consumer Tech", "Cybersecurity", "Deep Tech",
    "EdTech", "Enterprise SaaS", "Fintech", "HealthTech",
    "Logistics Tech", "PropTech",
]
SOURCE_TYPES = [
    "Any", "call", "email", "event", "linkedin_message", "meeting",
]
DEAL_STAGES = [
    "Any", "Lead", "Qualified", "Due Diligence",
    "Term Sheet", "Closed Won", "Closed Lost", "Passed",
]
SENTIMENTS = ["Any", "positive", "neutral", "negative"]

EXAMPLE_QUERIES = [
    ["Which companies are showing the strongest ARR growth and NRR?",
     "Any", "Any", "Any", "Any", "", "", 5],
    ["Summarise the key findings from our due diligence meetings.",
     "Any", "meeting", "Due Diligence", "Any", "", "", 5],
    ["What are the latest updates on companies at the Term Sheet stage?",
     "Any", "Any", "Term Sheet", "Any", "", "", 6],
    ["What did founders say in investor update calls about burn rate and runway?",
     "Any", "call", "Any", "Any", "2025-11-01", "", 5],
    ["Which HealthTech companies have the strongest competitive moat?",
     "HealthTech", "Any", "Any", "positive", "", "", 5],
    ["Are there any companies where we logged concerns or negative signals?",
     "Any", "Any", "Any", "negative", "", "", 5],
]

# ── Load retriever once at startup ────────────────────────────────────────────
print("Loading RelIntel retriever...")
t0 = time.perf_counter()
RETRIEVER = HybridRetriever.load()
print(f"Retriever ready in {time.perf_counter()-t0:.2f}s")

# ── Core query function ───────────────────────────────────────────────────────

def run_query(
    question:    str,
    sector:      str,
    source_type: str,
    deal_stage:  str,
    sentiment:   str,
    date_from:   str,
    date_to:     str,
    top_k:       int,
):
    # The API key is provided as a Space secret (server-side env var).
    # Never pass it into model arguments or return it to the client.
    has_api_key = bool(os.environ.get("ANTHROPIC_API_KEY"))

    # ── Validate ──────────────────────────────────────────────────────────
    if not question.strip():
        return (
            "⚠️ Please enter a question.",
            "", "", ""
        )
    if not has_api_key:
        return (
            "⚠️ ANTHROPIC_API_KEY is not configured on this Space.\n"
            "Add it under Settings → Repository secrets.",
            "", "", ""
        )

    # ── Build filters ─────────────────────────────────────────────────────
    filters = {}
    if sector     and sector     != "Any": filters["sector"]      = sector
    if source_type and source_type != "Any": filters["source_type"] = source_type
    if deal_stage  and deal_stage  != "Any": filters["deal_stage"]  = deal_stage
    if sentiment   and sentiment   != "Any": filters["sentiment"]   = sentiment
    if date_from.strip(): filters["date_from"] = date_from.strip()
    if date_to.strip():   filters["date_to"]   = date_to.strip()

    rq = RetrievalQuery(
        text        = question,
        top_k       = int(top_k),
        candidate_k = 20,
        sector      = filters.get("sector"),
        source_type = filters.get("source_type"),
        deal_stage  = filters.get("deal_stage"),
        sentiment   = filters.get("sentiment"),
        date_from   = filters.get("date_from"),
        date_to     = filters.get("date_to"),
    )

    # ── Retrieve ──────────────────────────────────────────────────────────
    t0 = time.perf_counter()
    results = RETRIEVER.search(rq)
    retrieval_ms = (time.perf_counter() - t0) * 1000

    if not results:
        if _OBSERVABILITY:
            try:
                log_event(
                    question=question, filters=filters, top_k=int(top_k),
                    n_retrieved=0, hybrid_count=0, dense_only_count=0, bm25_only_count=0,
                    retrieval_ms=retrieval_ms, generation_ms=0.0,
                    guard_passed=True, citation_coverage=1.0,
                    n_citations=0, n_invalid_refs=0, n_ungrounded_nums=0,
                    prompt_tokens=0, output_tokens=0,
                    answer_length=0, empty_result=True,
                )
            except Exception:
                pass
        return (
            "No relevant context found for this query with the current filters.\n"
            "Try broadening your filters or rephrasing the question.",
            "",
            _format_retrieved_md([], retrieval_ms),
            "",
        )

    # ── Generate ──────────────────────────────────────────────────────────
    t0 = time.perf_counter()
    try:
        result = generate(question, results, model=DEFAULT_MODEL)
    except Exception as e:
        # Avoid leaking secrets in exception strings (e.g., if user passed a key somewhere).
        msg = str(e)
        if "sk-ant-" in msg:
            msg = msg.replace("sk-ant-", "sk-ant-***REDACTED***")
        return (
            f"⚠️ Generation error: {msg}",
            "",
            _format_retrieved_md(results, retrieval_ms),
            "",
        )
    generation_ms = (time.perf_counter() - t0) * 1000

    # ── Log event ─────────────────────────────────────────────────────────────
    if _OBSERVABILITY:
        try:
            log_event(
                question          = question,
                filters           = filters,
                top_k             = int(top_k),
                n_retrieved       = len(results),
                hybrid_count      = sum(1 for r in results if r.dense_rank and r.bm25_rank),
                dense_only_count  = sum(1 for r in results if r.dense_rank and not r.bm25_rank),
                bm25_only_count   = sum(1 for r in results if r.bm25_rank and not r.dense_rank),
                retrieval_ms      = retrieval_ms,
                generation_ms     = generation_ms,
                guard_passed      = result.guard.passed,
                citation_coverage = result.guard.citation_coverage,
                n_citations       = len(result.citations),
                n_invalid_refs    = len(result.guard.invalid_citations),
                n_ungrounded_nums = len(result.guard.ungrounded_numbers),
                prompt_tokens     = result.prompt_tokens,
                output_tokens     = result.output_tokens,
                answer_length     = len(result.answer),
                empty_result      = False,
            )
        except Exception:
            pass  # observability must never break the main query path

    # ── Format outputs ────────────────────────────────────────────────────
    answer_md   = _format_answer_md(result)
    sources_md  = _format_sources_md(result)
    retrieved_md = _format_retrieved_md(results, retrieval_ms)
    guard_md    = _format_guard_md(result, generation_ms)

    return answer_md, sources_md, retrieved_md, guard_md


# ── Output formatters ─────────────────────────────────────────────────────────

def _format_answer_md(result) -> str:
    return result.answer


def _format_sources_md(result) -> str:
    if not result.citations:
        return "*No citations found in answer.*"
    lines = []
    for c in result.citations:
        stage = f" · {c.deal_stage}" if c.deal_stage else ""
        lines.append(
            f"**[{c.index}] {c.company_name}**  \n"
            f"`{c.source_type}` · {c.date}{stage} · {c.sector}  \n"
            f"*\"{c.snippet}...\"*"
        )
    return "\n\n---\n\n".join(lines)


def _format_retrieved_md(results, retrieval_ms: float) -> str:
    if not results:
        return f"*No chunks retrieved. ({retrieval_ms:.0f}ms)*"
    lines = [f"**{len(results)} chunks retrieved** in {retrieval_ms:.0f}ms\n"]
    for i, r in enumerate(results, 1):
        m = r.metadata
        source_label = r.source_label()
        source_badge = {
            "hybrid":     "🔀 hybrid",
            "dense-only": "🧠 dense",
            "bm25-only":  "🔤 BM25",
        }.get(source_label, source_label)

        lines.append(
            f"**[{i}]** {m['company_name']} · `{m['source_type']}` · {m['date']}  \n"
            f"RRF `{r.rrf_score:.4f}` · sim `{r.dense_sim:.3f}` · {source_badge}  \n"
            f"Sector: {m['sector']} · Deal: {m['deal_stage'] or '—'} · "
            f"Sentiment: {m['sentiment']}  \n"
            f"*{r.compressed_text[:120]}...*"
        )
    return "\n\n".join(lines)


def _format_guard_md(result, generation_ms: float) -> str:
    g = result.guard
    status = "✅ PASS" if g.passed else "⚠️ WARN"
    lines = [
        f"**Guard: {status}**  ",
        f"Citation coverage: `{g.citation_coverage:.0%}` · "
        f"Invalid refs: `{len(g.invalid_citations)}` · "
        f"Ungrounded numbers: `{len(g.ungrounded_numbers)}`  ",
        f"Model: `{result.model}` · "
        f"Tokens: `{result.prompt_tokens}` in / `{result.output_tokens}` out · "
        f"Generation: `{generation_ms:.0f}ms`",
    ]
    if g.warnings:
        lines.append("\n**Warnings:**")
        for w in g.warnings[:5]:
            lines.append(f"- {w}")
    return "\n".join(lines)


# ── Gradio UI ─────────────────────────────────────────────────────────────────

CSS = """
/* ── Font imports ── */
@import url('https://fonts.googleapis.com/css2?family=DM+Mono:ital,wght@0,300;0,400;0,500;1,300&family=Syne:wght@400;600;700;800&display=swap');

/* ── Root tokens ── */
:root {
    --c-bg:         #0d0f14;
    --c-surface:    #13161e;
    --c-border:     #1e2330;
    --c-border-hi:  #2e3550;
    --c-accent:     #4f7cff;
    --c-accent-dim: #2a3f80;
    --c-gold:       #e8c84a;
    --c-gold-dim:   #7a6520;
    --c-text:       #d4d8e8;
    --c-text-dim:   #6b7394;
    --c-green:      #3ecf8e;
    --c-red:        #f0624e;
    --radius:       8px;
    --font-ui:      'Syne', sans-serif;
    --font-mono:    'DM Mono', monospace;
}

/* ── Base ── */
body, .gradio-container {
    background: var(--c-bg) !important;
    font-family: var(--font-ui) !important;
    color: var(--c-text) !important;
}

/* ── Header ── */
.ri-header {
    padding: 32px 0 24px;
    border-bottom: 1px solid var(--c-border);
    margin-bottom: 24px;
}
.ri-wordmark {
    font-family: var(--font-ui);
    font-size: 13px;
    font-weight: 600;
    letter-spacing: 0.25em;
    text-transform: uppercase;
    color: var(--c-gold);
    margin-bottom: 6px;
}
.ri-title {
    font-family: var(--font-ui);
    font-size: 32px;
    font-weight: 800;
    color: #fff;
    line-height: 1.1;
    letter-spacing: -0.02em;
}
.ri-subtitle {
    font-family: var(--font-mono);
    font-size: 12px;
    color: var(--c-text-dim);
    margin-top: 8px;
    letter-spacing: 0.04em;
}
.ri-pipeline-badge {
    display: inline-flex;
    gap: 8px;
    margin-top: 14px;
    flex-wrap: wrap;
}
.ri-badge {
    font-family: var(--font-mono);
    font-size: 10px;
    letter-spacing: 0.08em;
    padding: 3px 10px;
    border-radius: 2px;
    border: 1px solid;
    text-transform: uppercase;
}
.ri-badge-blue  { color: var(--c-accent); border-color: var(--c-accent-dim); background: rgba(79,124,255,0.06); }
.ri-badge-gold  { color: var(--c-gold);   border-color: var(--c-gold-dim);   background: rgba(232,200,74,0.06); }
.ri-badge-green { color: var(--c-green);  border-color: rgba(62,207,142,0.3);background: rgba(62,207,142,0.06);}

/* ── Panels ── */
.ri-panel {
    background: var(--c-surface) !important;
    border: 1px solid var(--c-border) !important;
    border-radius: var(--radius) !important;
}
.ri-panel-label {
    font-family: var(--font-mono);
    font-size: 10px;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    color: var(--c-text-dim);
    padding: 10px 14px 0;
}

/* ── Gradio component overrides ── */
.gradio-container .block {
    background: transparent !important;
}
label.svelte-1b6s6g, .label-wrap {
    font-family: var(--font-mono) !important;
    font-size: 11px !important;
    letter-spacing: 0.08em !important;
    text-transform: uppercase !important;
    color: var(--c-text-dim) !important;
}
textarea, input[type=text], input[type=number], .input-wrap {
    background: var(--c-bg) !important;
    border: 1px solid var(--c-border) !important;
    border-radius: var(--radius) !important;
    color: var(--c-text) !important;
    font-family: var(--font-ui) !important;
    font-size: 14px !important;
    transition: border-color 0.15s !important;
}
textarea:focus, input:focus {
    border-color: var(--c-accent) !important;
    box-shadow: 0 0 0 3px rgba(79,124,255,0.12) !important;
    outline: none !important;
}
select, .wrap-inner {
    background: var(--c-bg) !important;
    border: 1px solid var(--c-border) !important;
    border-radius: var(--radius) !important;
    color: var(--c-text) !important;
    font-family: var(--font-mono) !important;
    font-size: 12px !important;
}

/* ── Run button ── */
#run-btn {
    background: var(--c-accent) !important;
    border: none !important;
    border-radius: var(--radius) !important;
    font-family: var(--font-ui) !important;
    font-size: 14px !important;
    font-weight: 700 !important;
    letter-spacing: 0.04em !important;
    color: #fff !important;
    padding: 12px 28px !important;
    cursor: pointer !important;
    transition: background 0.15s, transform 0.1s !important;
}
#run-btn:hover {
    background: #6b92ff !important;
    transform: translateY(-1px) !important;
}
#run-btn:active { transform: translateY(0) !important; }

/* ── Clear button ── */
#clear-btn {
    background: transparent !important;
    border: 1px solid var(--c-border) !important;
    border-radius: var(--radius) !important;
    font-family: var(--font-mono) !important;
    font-size: 11px !important;
    color: var(--c-text-dim) !important;
    transition: border-color 0.15s, color 0.15s !important;
}
#clear-btn:hover {
    border-color: var(--c-border-hi) !important;
    color: var(--c-text) !important;
}

/* ── Markdown output ── */
.prose, .prose p, .prose li {
    font-family: var(--font-ui) !important;
    font-size: 14px !important;
    line-height: 1.7 !important;
    color: var(--c-text) !important;
}
.prose strong { color: #fff !important; }
.prose code {
    font-family: var(--font-mono) !important;
    font-size: 11px !important;
    background: rgba(79,124,255,0.1) !important;
    color: var(--c-accent) !important;
    padding: 1px 6px !important;
    border-radius: 3px !important;
}
.prose em { color: var(--c-text-dim) !important; }
.prose hr { border-color: var(--c-border) !important; margin: 16px 0 !important; }

/* ── Tabs ── */
.tab-nav button {
    font-family: var(--font-mono) !important;
    font-size: 11px !important;
    letter-spacing: 0.08em !important;
    text-transform: uppercase !important;
    color: var(--c-text-dim) !important;
    background: transparent !important;
    border: none !important;
    border-bottom: 2px solid transparent !important;
    padding: 10px 16px !important;
    transition: color 0.15s, border-color 0.15s !important;
}
.tab-nav button.selected {
    color: var(--c-accent) !important;
    border-bottom-color: var(--c-accent) !important;
}

/* ── Accordion ── */
.accordion > .label-wrap {
    background: var(--c-surface) !important;
    border: 1px solid var(--c-border) !important;
    border-radius: var(--radius) !important;
    padding: 10px 14px !important;
}
.accordion > .label-wrap:hover {
    border-color: var(--c-border-hi) !important;
}

/* ── Slider ── */
.range-slider .range-handle {
    background: var(--c-accent) !important;
}
input[type=range] { accent-color: var(--c-accent) !important; }

/* ── Examples ── */
.examples table {
    font-family: var(--font-mono) !important;
    font-size: 11px !important;
}
.examples td { color: var(--c-text-dim) !important; }
.examples tr:hover td { color: var(--c-text) !important; }

/* ── Divider lines ── */
.ri-divider {
    border: none;
    border-top: 1px solid var(--c-border);
    margin: 20px 0;
}

/* ── Status dots ── */
.ri-dot {
    display: inline-block;
    width: 6px; height: 6px;
    border-radius: 50%;
    margin-right: 6px;
    background: var(--c-green);
    box-shadow: 0 0 6px var(--c-green);
    animation: pulse 2s infinite;
}
@keyframes pulse {
    0%, 100% { opacity: 1; }
    50%       { opacity: 0.4; }
}
"""

HEADER_HTML = """
<div class="ri-header">
  <div class="ri-wordmark">RelIntel</div>
  <div class="ri-title">Relationship Intelligence</div>
  <div class="ri-subtitle">
    Hybrid RAG · BM25 + Dense + RRF · Metadata Pre-filtering · Citation-grounded answers
  </div>
  <div class="ri-pipeline-badge">
    <span class="ri-badge ri-badge-blue">Hybrid Retrieval</span>
    <span class="ri-badge ri-badge-blue">Reciprocal Rank Fusion</span>
    <span class="ri-badge ri-badge-blue">Contextual Compression</span>
    <span class="ri-badge ri-badge-gold">Claude Haiku</span>
    <span class="ri-badge ri-badge-green"><span class="ri-dot"></span>Live</span>
  </div>
</div>
"""

def build_ui() -> gr.Blocks:
    with gr.Blocks(
        css=CSS,
        title="RelIntel — Relationship Intelligence RAG",
        theme=gr.themes.Base(
            primary_hue="blue",
            neutral_hue="slate",
        ),
    ) as demo:

        gr.HTML(HEADER_HTML)

        with gr.Row(equal_height=False):

            # ── Left column: inputs ───────────────────────────────────────
            with gr.Column(scale=4):

                question = gr.Textbox(
                    label="Question",
                    placeholder=(
                        "e.g. Which companies have the strongest ARR growth and NRR?\n"
                        "     What are the latest updates on Term Sheet deals?\n"
                        "     Summarise HealthTech due diligence findings."
                    ),
                    lines=3,
                )

                with gr.Accordion("Metadata Filters", open=False):
                    with gr.Row():
                        sector     = gr.Dropdown(SECTORS,      label="Sector",      value="Any")
                        source_type = gr.Dropdown(SOURCE_TYPES, label="Source Type", value="Any")
                    with gr.Row():
                        deal_stage = gr.Dropdown(DEAL_STAGES,  label="Deal Stage",  value="Any")
                        sentiment  = gr.Dropdown(SENTIMENTS,   label="Sentiment",   value="Any")
                    with gr.Row():
                        date_from  = gr.Textbox(label="Date From (YYYY-MM-DD)", placeholder="2025-01-01")
                        date_to    = gr.Textbox(label="Date To (YYYY-MM-DD)",   placeholder="2026-05-24")

                top_k = gr.Slider(
                    minimum=3, maximum=10, value=5, step=1,
                    label="Chunks to retrieve (top-k)",
                )

                with gr.Row():
                    run_btn   = gr.Button("Run Query", variant="primary", elem_id="run-btn")
                    clear_btn = gr.Button("Clear", elem_id="clear-btn")

                gr.Examples(
                    examples=EXAMPLE_QUERIES,
                    inputs=[question, sector, source_type, deal_stage,
                            sentiment, date_from, date_to, top_k],
                    label="Example Queries",
                    examples_per_page=6,
                )

            # ── Right column: outputs ─────────────────────────────────────
            with gr.Column(scale=6):
                with gr.Tabs():
                    with gr.Tab("Answer"):
                        answer_out = gr.Markdown(
                            label="Answer",
                            value="*Run a query to see the answer here.*",
                        )
                    with gr.Tab("Sources"):
                        sources_out = gr.Markdown(
                            label="Cited Sources",
                            value="*Citation metadata will appear here.*",
                        )
                    with gr.Tab("Retrieved Chunks"):
                        retrieved_out = gr.Markdown(
                            label="Retrieval Debug",
                            value="*Retrieval diagnostics will appear here.*",
                        )
                    with gr.Tab("Guard"):
                        guard_out = gr.Markdown(
                            label="Hallucination Guard",
                            value="*Guard results will appear here.*",
                        )

        # ── Architecture note ─────────────────────────────────────────────
        gr.HTML("""
        <div style="margin-top:28px; padding:16px 20px;
                    background:#13161e; border:1px solid #1e2330;
                    border-radius:8px; font-family:'DM Mono',monospace;
                    font-size:11px; color:#6b7394; line-height:1.8;">
          <span style="color:#4f7cff; letter-spacing:.1em;">PIPELINE</span>
          &nbsp;·&nbsp;
          Query → Metadata Pre-filter → Dense (LSA 256-dim) + BM25 (Okapi)
          → RRF(k=60) → Contextual Compression → Claude Haiku → Citation Guard
          &nbsp;&nbsp;|&nbsp;&nbsp;
          <span style="color:#e8c84a; letter-spacing:.1em;">CORPUS</span>
          &nbsp;·&nbsp;
          500 interactions · 50 companies · 200 contacts · 150 deals
          &nbsp;&nbsp;|&nbsp;&nbsp;
          <a href="https://github.com/Aloagbaye/relintel"
             style="color:#4f7cff; text-decoration:none;"
             target="_blank">github ↗</a>
        </div>
        """)

        # ── Event wiring ──────────────────────────────────────────────────
        outputs = [answer_out, sources_out, retrieved_out, guard_out]

        run_btn.click(
            fn=run_query,
            inputs=[question, sector, source_type, deal_stage,
                    sentiment, date_from, date_to, top_k],
            outputs=outputs,
        )
        question.submit(
            fn=run_query,
            inputs=[question, sector, source_type, deal_stage,
                    sentiment, date_from, date_to, top_k],
            outputs=outputs,
        )
        clear_btn.click(
            fn=lambda: ["", "Any", "Any", "Any", "Any", "", "", 5,
                        "*Run a query to see the answer here.*",
                        "*Citation metadata will appear here.*",
                        "*Retrieval diagnostics will appear here.*",
                        "*Guard results will appear here.*"],
            inputs=[],
            outputs=[question, sector, source_type, deal_stage,
                     sentiment, date_from, date_to, top_k] + outputs,
        )

    return demo


if __name__ == "__main__":
    demo = build_ui()
    port = int(os.getenv("PORT", "7860"))
    demo.launch(
        server_name = "0.0.0.0",
        server_port = port,
        show_error  = True,
        share       = False,
    )
