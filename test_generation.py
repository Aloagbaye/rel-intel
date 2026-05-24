"""
Phase 4 — Generation Layer Test Suite
RelIntel RAG System

Tests run without hitting the Anthropic API — the LLM call is mocked so
the suite validates prompt assembly, citation parsing, and the hallucination
guard logic independently of network access.

Tests:
  T1  Prompt assembly — context block format and numbered citations
  T2  Citation parser — extracts [N] refs and maps to correct source chunks
  T3  Citation parser — handles out-of-range and duplicate refs
  T4  Guard — all-cited clean answer → PASS
  T5  Guard — uncited sentences → coverage warning
  T6  Guard — invalid [N] ref → hard FAIL
  T7  Guard — number in answer not in cited source → hard FAIL
  T8  Guard — number appears in cited chunk → PASS
  T9  Empty retrieval — graceful no-context response
  T10 End-to-end with mocked LLM — full pipeline smoke test
"""

import os
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

sys.stdout.reconfigure(encoding="utf-8")
sys.stderr.reconfigure(encoding="utf-8")

sys.path.insert(0, str(Path(__file__).parent))

from generator import (
    GenerationResult,
    build_context_block,
    build_user_message,
    format_result,
    generate,
    parse_citations,
    run_hallucination_guard,
)
from retriever import HybridRetriever, RetrievalQuery, RetrievalResult

SEP  = "─" * 60
SEP2 = "═" * 60

# ─── Fixtures ─────────────────────────────────────────────────────────────────

def make_result(
    chunk_id:     str = "ix_aabbccdd_c0",
    company_name: str = "Acme Fintech AI",
    source_type:  str = "meeting",
    date:         str = "2026-03-15",
    sector:       str = "Fintech",
    deal_stage:   str = "Due Diligence",
    sentiment:    str = "positive",
    body:         str = "Product is live with 250 enterprise customers and 88% NRR.",
    rrf_score:    float = 0.032,
    dense_rank:   int = 1,
    bm25_rank:    int = 2,
) -> RetrievalResult:
    """Create a minimal RetrievalResult fixture."""
    full_text = (
        f"Company: {company_name} [{sector}]\n"
        f"Interaction: {source_type.title()} on {date}\n\n"
        f"{body}"
    )
    return RetrievalResult(
        chunk_id        = chunk_id,
        text            = full_text,
        compressed_text = body,
        metadata        = {
            "interaction_id":    chunk_id.rsplit("_c", 1)[0],
            "source_type":       source_type,
            "date":              date,
            "date_ts":           int(date.replace("-", "")),
            "sentiment":         sentiment,
            "logged_by":         "usr_001",
            "subject":           f"Meeting: {company_name}",
            "tags":              "diligence,portfolio",
            "company_id":        "co_aabbccdd",
            "company_name":      company_name,
            "sector":            sector,
            "company_stage":     "Series B",
            "relationship_strength": "strong",
            "deal_id":           "dl_11223344",
            "deal_stage":        deal_stage,
            "deal_type":         "Series B",
            "amount_usd":        10_000_000,
            "primary_contact_id": "ct_aabb1122",
            "contact_count":     2,
            "chunk_index":       0,
            "chunk_count":       1,
            "word_count":        45,
        },
        rrf_score   = rrf_score,
        dense_rank  = dense_rank,
        bm25_rank   = bm25_rank,
        dense_sim   = 0.41,
    )

# ─── Tests ────────────────────────────────────────────────────────────────────

class TestPromptAssembly(unittest.TestCase):

    def setUp(self):
        self.r1 = make_result(company_name="Acme Fintech AI",   chunk_id="ix_aaaa_c0",
                               body="Product is live with 250 customers and 88% NRR.")
        self.r2 = make_result(company_name="Blue Climate Tech", chunk_id="ix_bbbb_c0",
                               sector="Climate Tech", body="ARR grew 180% YoY to $12M.")

    def test_T1_context_block_format(self):
        print(f"\n{SEP}\n  T1 · Context block format\n{SEP}")
        block = build_context_block([self.r1, self.r2])

        self.assertIn("[1]", block, "First chunk should be numbered [1]")
        self.assertIn("[2]", block, "Second chunk should be numbered [2]")
        self.assertIn("Acme Fintech AI",   block)
        self.assertIn("Blue Climate Tech", block)
        self.assertIn("=== CONTEXT ===",   block)
        self.assertIn("=== END CONTEXT ===", block)
        self.assertIn("88% NRR",  block)
        self.assertIn("$12M",     block)

        print(block)
        print("✓ T1 passed — context block correctly formatted\n")

    def test_T1b_user_message_includes_query(self):
        block = build_context_block([self.r1])
        msg   = build_user_message("What is the NRR?", block)
        self.assertIn("What is the NRR?", msg)
        self.assertIn("=== CONTEXT ===",  msg)


class TestCitationParser(unittest.TestCase):

    def setUp(self):
        self.r1 = make_result(company_name="Acme Fintech AI",   chunk_id="ix_aaaa_c0")
        self.r2 = make_result(company_name="Blue Climate Tech", chunk_id="ix_bbbb_c0")
        self.r3 = make_result(company_name="Gamma SaaS",        chunk_id="ix_cccc_c0")

    def test_T2_basic_citation_extraction(self):
        print(f"\n{SEP}\n  T2 · Basic citation extraction\n{SEP}")
        answer = "Acme has strong NRR [1]. Climate Tech is growing fast [2]."
        citations = parse_citations(answer, [self.r1, self.r2, self.r3])

        self.assertEqual(len(citations), 2)
        self.assertEqual(citations[0].index, 1)
        self.assertEqual(citations[0].company_name, "Acme Fintech AI")
        self.assertEqual(citations[1].index, 2)
        self.assertEqual(citations[1].company_name, "Blue Climate Tech")
        print(f"  Parsed: {[(c.index, c.company_name) for c in citations]}")
        print("✓ T2 passed\n")

    def test_T3_out_of_range_and_duplicates(self):
        print(f"\n{SEP}\n  T3 · Out-of-range and duplicate [N] refs\n{SEP}")
        # [99] is out of range; [1] appears twice
        answer = "See [1] and also [1] for more. Source [99] claimed this."
        citations = parse_citations(answer, [self.r1, self.r2, self.r3])

        indices = [c.index for c in citations]
        # [1] should appear once (deduped); [99] should be absent (out of range)
        self.assertIn(1, indices)
        self.assertNotIn(99, indices)
        self.assertEqual(len(citations), 1, "Duplicate [1] should be deduped")
        print(f"  Indices parsed: {indices}")
        print("✓ T3 passed — out-of-range excluded, duplicates deduped\n")


class TestHallucinationGuard(unittest.TestCase):

    def setUp(self):
        self.r1 = make_result(
            company_name = "Acme Fintech AI",
            chunk_id     = "ix_aaaa_c0",
            body         = "Product is live with 250 enterprise customers and 88% NRR.",
        )
        self.r2 = make_result(
            company_name = "Blue Climate Tech",
            chunk_id     = "ix_bbbb_c0",
            body         = "ARR grew 180% YoY to $12M with burn at $400K/month.",
        )

    def test_T4_clean_cited_answer_passes(self):
        print(f"\n{SEP}\n  T4 · Clean fully-cited answer → PASS\n{SEP}")
        answer = (
            "Acme Fintech has 250 customers and 88% NRR [1]. "
            "Blue Climate Tech grew ARR 180% to $12M [2]."
        )
        citations = parse_citations(answer, [self.r1, self.r2])
        guard = run_hallucination_guard(answer, [self.r1, self.r2], citations)

        print(f"  passed={guard.passed} | coverage={guard.citation_coverage:.0%} "
              f"| invalid={guard.invalid_citations} | ungrounded={guard.ungrounded_numbers}")
        self.assertTrue(guard.passed)
        self.assertEqual(len(guard.invalid_citations),  0)
        self.assertEqual(len(guard.ungrounded_numbers), 0)
        print("✓ T4 passed\n")

    def test_T5_uncited_sentences_warn_but_pass(self):
        print(f"\n{SEP}\n  T5 · Uncited long sentences → coverage warning\n{SEP}")
        answer = (
            "Acme has strong metrics [1]. "
            "There are many interesting companies in this space with no citation at all here."
        )
        citations = parse_citations(answer, [self.r1, self.r2])
        guard = run_hallucination_guard(answer, [self.r1, self.r2], citations)

        print(f"  passed={guard.passed} | coverage={guard.citation_coverage:.0%}")
        print(f"  uncited: {guard.uncited_sentences}")
        self.assertGreater(len(guard.uncited_sentences), 0)
        self.assertLess(guard.citation_coverage, 1.0)
        print("✓ T5 passed — uncited sentences flagged\n")

    def test_T6_invalid_citation_fails(self):
        print(f"\n{SEP}\n  T6 · Invalid [N] ref → hard FAIL\n{SEP}")
        answer = "This company has great metrics [1]. See also [99] for more detail."
        citations = parse_citations(answer, [self.r1, self.r2])
        guard = run_hallucination_guard(answer, [self.r1, self.r2], citations)

        print(f"  passed={guard.passed} | invalid_citations={guard.invalid_citations}")
        self.assertFalse(guard.passed)
        self.assertIn(99, guard.invalid_citations)
        print("✓ T6 passed — invalid citation correctly fails guard\n")

    def test_T7_ungrounded_number_fails(self):
        print(f"\n{SEP}\n  T7 · Number not in cited source → hard FAIL\n{SEP}")
        # Answer cites [1] but claims 999 customers — not in any chunk
        answer = "Acme has 999 enterprise customers and leads the market [1]."
        citations = parse_citations(answer, [self.r1, self.r2])
        guard = run_hallucination_guard(answer, [self.r1, self.r2], citations)

        print(f"  passed={guard.passed} | ungrounded={guard.ungrounded_numbers}")
        self.assertFalse(guard.passed)
        self.assertIn("999", guard.ungrounded_numbers)
        print("✓ T7 passed — hallucinated number correctly detected\n")

    def test_T8_grounded_number_passes(self):
        print(f"\n{SEP}\n  T8 · Number from cited source → PASS\n{SEP}")
        # "88%" and "250" are both verbatim in r1's body
        answer = "Acme has 250 enterprise customers with 88% NRR [1]."
        citations = parse_citations(answer, [self.r1])
        guard = run_hallucination_guard(answer, [self.r1], citations)

        print(f"  passed={guard.passed} | ungrounded={guard.ungrounded_numbers}")
        self.assertTrue(guard.passed)
        self.assertEqual(len(guard.ungrounded_numbers), 0)
        print("✓ T8 passed — grounded numbers correctly verified\n")


class TestEdgeCases(unittest.TestCase):

    def test_T9_empty_retrieval(self):
        print(f"\n{SEP}\n  T9 · Empty retrieval → graceful no-context response\n{SEP}")
        result = generate(
            query   = "What is the ARR of company X?",
            results = [],
        )
        self.assertIn("No relevant context", result.answer)
        self.assertTrue(result.guard.passed)
        self.assertEqual(len(result.citations), 0)
        print(f"  Answer: {result.answer}")
        print("✓ T9 passed — empty retrieval handled gracefully\n")

    def test_T10_end_to_end_mocked_llm(self):
        print(f"\n{SEP}\n  T10 · End-to-end with mocked LLM\n{SEP}")

        r1 = make_result(
            company_name = "Acme Fintech AI",
            body         = "Product is live with 250 enterprise customers and 88% NRR.",
        )
        r2 = make_result(
            company_name = "Blue Climate Tech",
            chunk_id     = "ix_bbbb_c0",
            body         = "ARR grew 180% YoY to $12M with burn at $400K/month.",
            sector       = "Climate Tech",
        )

        mocked_answer = (
            "Acme Fintech AI is performing well with 250 enterprise customers and 88% NRR [1]. "
            "Blue Climate Tech has seen exceptional growth, with ARR rising 180% to $12M [2]."
        )

        with patch("generator.call_llm", return_value=(mocked_answer, 320, 68)):
            result = generate(
                query   = "Which companies are growing fastest?",
                results = [r1, r2],
            )

        print(format_result(result))

        self.assertEqual(result.query, "Which companies are growing fastest?")
        self.assertIn("Acme Fintech AI",   result.answer)
        self.assertIn("Blue Climate Tech", result.answer)
        self.assertEqual(len(result.citations), 2)
        self.assertEqual(result.citations[0].company_name, "Acme Fintech AI")
        self.assertEqual(result.citations[1].company_name, "Blue Climate Tech")
        self.assertTrue(result.guard.passed,
                        f"Guard should pass. Warnings: {result.guard.warnings}")
        self.assertEqual(result.prompt_tokens, 320)
        self.assertEqual(result.output_tokens,  68)
        print("✓ T10 passed — end-to-end pipeline with mocked LLM\n")


# ─── Runner ───────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print(SEP2)
    print("  RelIntel — Phase 4: Generation Layer Tests")
    print(SEP2)

    loader = unittest.TestLoader()
    loader.sortTestMethodsUsing = None  # preserve definition order
    suite  = unittest.TestSuite()

    for cls in [TestPromptAssembly, TestCitationParser, TestHallucinationGuard, TestEdgeCases]:
        suite.addTests(loader.loadTestsFromTestCase(cls))

    runner = unittest.TextTestRunner(verbosity=0, stream=open(os.devnull, "w"))
    result = runner.run(suite)

    print(SEP2)
    if result.wasSuccessful():
        print(f"  ALL {result.testsRun} TESTS PASSED ✅")
        print(SEP2)
        print()
        print("  Phase 4 generation capabilities verified:")
        print("    ✓ Prompt assembly — numbered context block with metadata headers")
        print("    ✓ Citation parsing — [N] extraction + source mapping")
        print("    ✓ Citation deduplication + out-of-range exclusion")
        print("    ✓ Hallucination guard — citation coverage check")
        print("    ✓ Hallucination guard — invalid [N] reference detection")
        print("    ✓ Hallucination guard — ungrounded number detection")
        print("    ✓ Hallucination guard — grounded numbers pass cleanly")
        print("    ✓ Empty retrieval — graceful no-context response")
        print("    ✓ End-to-end pipeline with mocked LLM")
        print()
        print("  To run live queries against the Anthropic API:")
        print("    export ANTHROPIC_API_KEY=sk-ant-...")
        print("    python src/query.py                    # demo queries")
        print("    python src/query.py --interactive      # REPL mode")
        print('    python src/query.py --query "..." --sector Fintech')
    else:
        print(f"  FAILURES: {len(result.failures)} | ERRORS: {len(result.errors)}")
        print(SEP2)
        for test, tb in result.failures + result.errors:
            print(f"\n  FAILED: {test}")
            print(tb)
        sys.exit(1)
