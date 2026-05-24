"""
Phase 4 — Generation Layer
RelIntel RAG System

Responsibilities:
  1. Prompt assembly  — formats retrieved chunks into a structured context block
                        with numbered citations the LLM can reference
  2. LLM call         — Anthropic claude-haiku-4-5-20251001 via the official
                        anthropic Python SDK (automatic retries, typed errors)
  3. Citation parser  — extracts [1], [2] ... references from the raw response
                        and maps them back to source metadata
  4. Hallucination guard — three lightweight checks run before the response is
                        returned to the caller:
                          a. Citation coverage   — every claim should cite a source
                          b. Source grounding    — cited sources must exist in context
                          c. Factual triplet check — key numbers in the answer must
                             appear verbatim in at least one cited chunk

Design decisions (interview-ready):
  - Numbered citations in the prompt (not inline XML tags) because they produce
    cleaner prose and the LLM naturally writes "...as seen in [2]" without
    coaching, reducing prompt complexity.
  - Context prefix retained in chunks so the LLM has company/date grounding
    even when a chunk is retrieved from a different time period than the query.
  - Hallucination guard is intentionally lightweight — a full LLM-as-judge
    pipeline (e.g. G-Eval) is the Phase 5 evaluation story; the guard here
    is a fast pre-flight check that catches obvious failures before they
    reach the user.
  - Generation is stateless — no conversation history. Each query is fully
    self-contained, which simplifies multi-tenant isolation.
"""

import re
from dataclasses import dataclass, field
from typing import Optional
# from dotenv import load_dotenv
import anthropic

from retriever import RetrievalResult

# load_dotenv()

# ─── Config ───────────────────────────────────────────────────────────────────

DEFAULT_MODEL     = "claude-haiku-4-5-20251001"
MAX_TOKENS        = 1024
TEMPERATURE       = 0.1   # low — we want factual, grounded answers

# ─── Data Structures ─────────────────────────────────────────────────────────

@dataclass
class Citation:
    index:        int           # 1-based, matches [N] in response text
    chunk_id:     str
    company_name: str
    source_type:  str
    date:         str
    deal_stage:   str
    sector:       str
    snippet:      str           # compressed_text used in context

@dataclass
class GuardResult:
    passed:              bool
    citation_coverage:   float   # fraction of sentences with ≥1 citation
    uncited_sentences:   list[str]
    invalid_citations:   list[int]  # [N] refs that don't map to a source
    ungrounded_numbers:  list[str]  # numbers in answer not found in any cited chunk
    warnings:            list[str]

@dataclass
class GenerationResult:
    query:          str
    answer:         str
    citations:      list[Citation]
    guard:          GuardResult
    model:          str
    prompt_tokens:  int = 0
    output_tokens:  int = 0
    raw_response:   str = ""

# ─── Prompt Assembly ─────────────────────────────────────────────────────────

SYSTEM_PROMPT = """\
You are RelIntel, an AI assistant for relationship intelligence and private capital research.
You answer questions about portfolio companies, deal activity, and relationship history
using only the provided context. You are precise, concise, and always cite your sources.

Rules:
- Answer using ONLY information from the numbered context chunks below.
- Cite every factual claim using [N] notation, where N matches the chunk number.
- If multiple chunks support a claim, cite all relevant ones: [1][3].
- If the context does not contain enough information to answer, say so explicitly.
- Do not invent, infer, or extrapolate beyond what the context states.
- Keep answers under 250 words unless the question explicitly asks for more detail.
- Format: prose paragraphs, no bullet points unless the question asks for a list.\
"""

def build_context_block(results: list[RetrievalResult]) -> str:
    """
    Format retrieved chunks into a numbered context block for the prompt.
    Each chunk includes its structured metadata header so the LLM can
    identify which company/date/interaction type the chunk comes from.
    """
    lines = ["=== CONTEXT ===\n"]
    for i, r in enumerate(results, 1):
        m = r.metadata
        lines.append(f"[{i}] {m['company_name']} | {m['source_type'].replace('_',' ').title()} "
                     f"| {m['date']} | Sector: {m['sector']} | Deal: {m['deal_stage'] or 'N/A'}")
        lines.append(r.compressed_text)
        lines.append("")
    lines.append("=== END CONTEXT ===")
    return "\n".join(lines)

def build_user_message(query: str, context_block: str) -> str:
    return f"{context_block}\n\nQuestion: {query}"

# ─── LLM Call ─────────────────────────────────────────────────────────────────

def call_llm(
    user_message: str,
    model:        str = DEFAULT_MODEL,
    max_tokens:   int = MAX_TOKENS,
    temperature:  float = TEMPERATURE,
) -> tuple[str, int, int]:
    """
    Call the Anthropic Messages API via the official Python SDK.
    Returns (response_text, prompt_tokens, output_tokens).
    The SDK handles retries, timeouts, and typed API errors automatically.
    Reads ANTHROPIC_API_KEY from the environment automatically.
    """
    client = anthropic.Anthropic()
    message = client.messages.create(
        model       = model,
        max_tokens  = max_tokens,
        temperature = temperature,
        system      = SYSTEM_PROMPT,
        messages    = [{"role": "user", "content": user_message}],
    )
    text          = message.content[0].text
    prompt_tokens = message.usage.input_tokens
    output_tokens = message.usage.output_tokens
    return text, prompt_tokens, output_tokens

# ─── Citation Parser ──────────────────────────────────────────────────────────

def parse_citations(
    answer:  str,
    results: list[RetrievalResult],
) -> list[Citation]:
    """
    Extract all [N] references from the answer and map each to its source chunk.
    Returns only citations that actually appear in the answer text.
    """
    referenced = set(int(n) for n in re.findall(r'\[(\d+)\]', answer))
    citations  = []

    for idx in sorted(referenced):
        if 1 <= idx <= len(results):
            r = results[idx - 1]
            m = r.metadata
            citations.append(Citation(
                index        = idx,
                chunk_id     = r.chunk_id,
                company_name = m["company_name"],
                source_type  = m["source_type"],
                date         = m["date"],
                deal_stage   = m.get("deal_stage", ""),
                sector       = m.get("sector", ""),
                snippet      = r.compressed_text[:120],
            ))

    return citations

# ─── Hallucination Guard ──────────────────────────────────────────────────────

_SENTENCE_SPLIT = re.compile(r'(?<=[.!?])\s+')
_NUMBER_RE      = re.compile(r'\b\d[\d,\.%$MBK]*\b')

def _extract_numbers(text: str) -> set[str]:
    """Extract numeric tokens — dollar amounts, percentages, counts."""
    return set(_NUMBER_RE.findall(text))

def _sentences(text: str) -> list[str]:
    return [s.strip() for s in _SENTENCE_SPLIT.split(text) if s.strip()]

def _has_citation(sentence: str) -> bool:
    return bool(re.search(r'\[\d+\]', sentence))

def run_hallucination_guard(
    answer:   str,
    results:  list[RetrievalResult],
    citations: list[Citation],
) -> GuardResult:
    """
    Three lightweight checks:

    1. Citation coverage
       Every substantive sentence (>6 words) should contain at least one [N].
       Reports uncited sentences as warnings, not hard failures.

    2. Source grounding
       Every [N] in the answer must correspond to a real chunk index.
       Out-of-range citations are hard failures.

    3. Factual triplet — number grounding
       Any number appearing in the answer should appear verbatim in at least
       one of the cited chunks. Numbers that appear in the answer but in none
       of the cited sources are flagged as potential hallucinations.

    In production this would be augmented by:
       - G-Eval (LLM-as-judge faithfulness scoring)
       - NLI-based entailment check (answer sentence ↔ source chunk)
       - RAGAS faithfulness metric (Phase 5)
    """
    warnings           = []
    uncited_sentences  = []
    invalid_citations  = []
    ungrounded_numbers = []

    sentences = _sentences(answer)

    # ── Check 1: Citation coverage ────────────────────────────────────────
    substantive = [s for s in sentences if len(s.split()) > 6]
    covered     = [s for s in substantive if _has_citation(s)]
    coverage    = len(covered) / len(substantive) if substantive else 1.0

    for s in substantive:
        if not _has_citation(s):
            uncited_sentences.append(s)
            warnings.append(f"Uncited sentence: \"{s[:80]}...\"")

    # ── Check 2: Source grounding ──────────────────────────────────────────
    all_refs = set(int(n) for n in re.findall(r'\[(\d+)\]', answer))
    for ref in all_refs:
        if ref < 1 or ref > len(results):
            invalid_citations.append(ref)
            warnings.append(f"Invalid citation [{ref}] — only {len(results)} sources provided")

    # ── Check 3: Number grounding ─────────────────────────────────────────
    answer_numbers = _extract_numbers(answer)
    # Exclude citation bracket numbers like [1], [2]
    citation_numbers = set(str(i) for i in range(1, len(results) + 1))
    answer_numbers -= citation_numbers

    # Build the pool of text from cited chunks only
    cited_indices  = {c.index for c in citations}
    cited_chunks   = [results[i - 1] for i in cited_indices if 1 <= i <= len(results)]
    cited_text     = " ".join(r.text for r in cited_chunks)
    cited_numbers  = _extract_numbers(cited_text)

    for num in answer_numbers:
        if num not in cited_numbers:
            ungrounded_numbers.append(num)
            warnings.append(f"Number not found in cited sources: {num}")

    passed = (
        len(invalid_citations) == 0
        and len(ungrounded_numbers) == 0
        and coverage >= 0.5
    )

    return GuardResult(
        passed             = passed,
        citation_coverage  = coverage,
        uncited_sentences  = uncited_sentences,
        invalid_citations  = invalid_citations,
        ungrounded_numbers = ungrounded_numbers,
        warnings           = warnings,
    )

# ─── Main Generation Function ─────────────────────────────────────────────────

def generate(
    query:   str,
    results: list[RetrievalResult],
    model:   str = DEFAULT_MODEL,
) -> GenerationResult:
    """
    Full generation pipeline for one query:
      1. Build prompt from retrieved results
      2. Call LLM
      3. Parse citations
      4. Run hallucination guard
      5. Return GenerationResult
    """
    if not results:
        return GenerationResult(
            query     = query,
            answer    = "No relevant context was found for this query.",
            citations = [],
            guard     = GuardResult(True, 1.0, [], [], [], []),
            model     = model,
        )

    context_block = build_context_block(results)
    user_message  = build_user_message(query, context_block)

    raw_answer, prompt_tok, output_tok = call_llm(
        user_message, model=model
    )

    citations = parse_citations(raw_answer, results)
    guard     = run_hallucination_guard(raw_answer, results, citations)

    return GenerationResult(
        query         = query,
        answer        = raw_answer,
        citations     = citations,
        guard         = guard,
        model         = model,
        prompt_tokens = prompt_tok,
        output_tokens = output_tok,
        raw_response  = raw_answer,
    )

# ─── Display Helpers ──────────────────────────────────────────────────────────

def format_result(result: GenerationResult, show_guard_detail: bool = True) -> str:
    lines = []
    lines.append(f"\nQuery: {result.query}")
    lines.append("─" * 60)
    lines.append(result.answer)
    lines.append("")
    lines.append(f"Sources ({len(result.citations)}):")
    for c in result.citations:
        lines.append(
            f"  [{c.index}] {c.company_name} | {c.source_type} | {c.date}"
            + (f" | {c.deal_stage}" if c.deal_stage else "")
        )
        lines.append(f"       \"{c.snippet}...\"")

    lines.append("")
    g = result.guard
    status = "✅ PASS" if g.passed else "⚠️  WARN"
    lines.append(f"Guard: {status} | citation_coverage={g.citation_coverage:.0%} "
                 f"| invalid_refs={len(g.invalid_citations)} "
                 f"| ungrounded_numbers={len(g.ungrounded_numbers)}")

    if show_guard_detail and g.warnings:
        for w in g.warnings:
            lines.append(f"  ⚠  {w}")

    lines.append(f"Tokens: {result.prompt_tokens} in / {result.output_tokens} out "
                 f"| model={result.model}")
    return "\n".join(lines)
