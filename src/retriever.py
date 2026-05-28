"""
Phase 3 — Hybrid Retrieval Engine
RelIntel RAG System

Architecture:
  ┌─────────────────────────────────────────────────────┐
  │                   Query Interface                    │
  └───────────────────┬─────────────────────────────────┘
                      │
          ┌───────────▼────────────┐
          │  Metadata Pre-Filter   │  ← company_id, sector, date range,
          │  (reduces candidate    │    source_type, sentiment, deal_stage
          │   set before ANN)      │
          └───────┬────────────────┘
                  │
        ┌─────────┴──────────┐
        │                    │
   ┌────▼─────┐        ┌─────▼─────┐
   │  Dense   │        │   BM25    │
   │  (LSA)   │        │  (Okapi)  │
   │ top-k=20 │        │  top-k=20 │
   └────┬─────┘        └─────┬─────┘
        │                    │
        └─────────┬──────────┘
                  │
        ┌─────────▼──────────┐
        │  Reciprocal Rank   │  ← RRF(k=60): score = Σ 1/(k + rank_i)
        │  Fusion (RRF)      │    merges both ranked lists
        └─────────┬──────────┘
                  │
        ┌─────────▼──────────┐
        │  Contextual        │  ← strips context prefix, trims to
        │  Compression       │    most relevant sentence(s) per chunk
        └─────────┬──────────┘
                  │
        ┌─────────▼──────────┐
        │  Top-N Results     │  ← default N=5, configurable
        └────────────────────┘

Key design decisions:
  1. WHY hybrid: BM25 excels at exact lexical matches (proper nouns: company
     names, people, deal terms). Dense retrieval handles paraphrase and
     semantic similarity. Neither alone is sufficient for relationship data.

  2. WHY metadata pre-filter BEFORE ANN: Reduces the candidate set so both
     BM25 and dense search the same constrained pool. Avoids returning
     semantically similar results from the wrong company/sector/timeframe.
     Mirrors RelIntel's multi-tenant isolation requirement.

  3. WHY RRF over score normalisation: RRF is robust to score scale
     differences between BM25 (unbounded) and cosine similarity (0-1).
     k=60 is a well-validated default from the original RRF paper.

  4. WHY contextual compression: Interaction bodies contain preamble context
     (company name, date). Compressing to the most relevant sentence(s)
     reduces the LLM context window usage and cuts hallucination surface.
"""

import json
import pickle
import re
import string
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import chromadb
import numpy as np
from rank_bm25 import BM25Okapi

# ─── Paths ────────────────────────────────────────────────────────────────────

ROOT     = Path(__file__).parent.parent
DATA_DIR = ROOT / "data"
DB_DIR   = DATA_DIR / "chroma"

# ─── Data Structures ─────────────────────────────────────────────────────────

@dataclass
class RetrievalResult:
    chunk_id:       str
    text:           str
    compressed_text: str          # post contextual-compression snippet
    metadata:       dict
    dense_rank:     Optional[int] = None   # rank in dense results (1-based, None if absent)
    bm25_rank:      Optional[int] = None   # rank in BM25 results  (1-based, None if absent)
    rrf_score:      float = 0.0
    dense_sim:      float = 0.0   # raw cosine similarity from dense retrieval

    def source_label(self) -> str:
        if self.dense_rank and self.bm25_rank:
            return "hybrid"
        elif self.dense_rank:
            return "dense-only"
        else:
            return "bm25-only"

@dataclass
class RetrievalQuery:
    text:           str
    top_k:          int = 5
    candidate_k:    int = 20      # candidates per retriever before fusion
    # ── Metadata filters ──
    company_id:     Optional[str] = None
    sector:         Optional[str] = None
    source_type:    Optional[str] = None   # email | meeting | call | linkedin_message | event
    sentiment:      Optional[str] = None   # positive | neutral | negative
    deal_stage:     Optional[str] = None
    date_from:      Optional[str] = None   # YYYY-MM-DD
    date_to:        Optional[str] = None   # YYYY-MM-DD
    logged_by:      Optional[str] = None   # team member id
    tags_include:   Optional[list[str]] = None   # chunks whose tags contain ANY of these

# ─── BM25 Index ───────────────────────────────────────────────────────────────

class BM25Index:
    """
    Wraps rank_bm25.BM25Okapi over the full chunk corpus.
    Tokenisation: lowercase, strip punctuation, split on whitespace.
    The index is built once at startup from chunks.json.
    """

    def __init__(self, chunks: list[dict]):
        self.chunks = chunks
        self.ids    = [c["id"] for c in chunks]
        tokenised   = [self._tokenise(c["text"]) for c in chunks]
        self.index  = BM25Okapi(tokenised)

    @staticmethod
    def _tokenise(text: str) -> list[str]:
        text = text.lower()
        text = text.translate(str.maketrans("", "", string.punctuation))
        return text.split()

    def search(self, query: str, top_k: int = 20, allowed_ids: Optional[set] = None) -> list[tuple[str, float]]:
        """
        Returns [(chunk_id, bm25_score), ...] sorted by score descending.
        If allowed_ids is provided, scores for excluded chunks are zeroed out
        (faster than rebuilding the index per query — acceptable at this scale).
        """
        tokens = self._tokenise(query)
        scores = self.index.get_scores(tokens)

        if allowed_ids is not None:
            for i, cid in enumerate(self.ids):
                if cid not in allowed_ids:
                    scores[i] = 0.0

        ranked = sorted(
            ((self.ids[i], float(scores[i])) for i in range(len(scores))),
            key=lambda x: -x[1],
        )
        return ranked[:top_k]

# ─── Contextual Compression ───────────────────────────────────────────────────

def compress(chunk_text: str, query: str, max_sentences: int = 3) -> str:
    """
    Lightweight local contextual compression — no LLM call required.

    Strategy:
      1. Strip the structured context prefix (everything before the first blank line).
      2. Split body into sentences.
      3. Score each sentence by query-term overlap (Jaccard-style).
      4. Return top `max_sentences` by score, preserving original order.

    In production this would be replaced by an LLMChainExtractor:
        from langchain.retrievers.document_compressors import LLMChainExtractor
    The interface here is identical — same input/output, swap the implementation.
    """
    # Strip context prefix (everything up to the first blank line)
    body_start = chunk_text.find("\n\n")
    body = chunk_text[body_start + 2:] if body_start != -1 else chunk_text

    sentences = re.split(r'(?<=[.!?])\s+', body.strip())
    if len(sentences) <= max_sentences:
        return body.strip()

    # Query term set
    stopwords = {"the", "a", "an", "is", "are", "was", "were", "and", "or",
                 "of", "to", "in", "for", "with", "about", "at", "by", "from"}
    query_terms = {
        t.lower().strip(string.punctuation)
        for t in query.split()
        if t.lower() not in stopwords and len(t) > 2
    }

    # Score sentences
    def score_sentence(s: str) -> float:
        words = {w.lower().strip(string.punctuation) for w in s.split()}
        return len(query_terms & words) / (len(query_terms) + 1e-9)

    scored = [(i, score_sentence(s), s) for i, s in enumerate(sentences)]
    top_idx = sorted(
        sorted(scored, key=lambda x: -x[1])[:max_sentences],
        key=lambda x: x[0],   # restore original order
    )

    return " ".join(s for _, _, s in top_idx)

# ─── Metadata Filter Builder ──────────────────────────────────────────────────

def build_chroma_filter(query: RetrievalQuery) -> Optional[dict]:
    """
    Translate RetrievalQuery filter fields into a ChromaDB `where` clause.
    ChromaDB supports: $eq, $ne, $gt, $gte, $lt, $lte, $and, $or
    All metadata values are scalar (lists were comma-joined at ingest time).
    """
    conditions = []

    if query.company_id:
        conditions.append({"company_id": {"$eq": query.company_id}})
    if query.sector:
        conditions.append({"sector": {"$eq": query.sector}})
    if query.source_type:
        conditions.append({"source_type": {"$eq": query.source_type}})
    if query.sentiment:
        conditions.append({"sentiment": {"$eq": query.sentiment}})
    if query.deal_stage:
        conditions.append({"deal_stage": {"$eq": query.deal_stage}})
    if query.logged_by:
        conditions.append({"logged_by": {"$eq": query.logged_by}})
    if query.date_from:
        ts_from = int(query.date_from.replace("-", ""))
        conditions.append({"date_ts": {"$gte": ts_from}})
    if query.date_to:
        ts_to = int(query.date_to.replace("-", ""))
        conditions.append({"date_ts": {"$lte": ts_to}})

    if not conditions:
        return None
    if len(conditions) == 1:
        return conditions[0]
    return {"$and": conditions}

def apply_bm25_filter(chunks_by_id: dict, query: RetrievalQuery) -> Optional[set]:
    """
    Return set of chunk IDs that pass the metadata filters, for use in BM25.
    Tags filter (contains-any) is only practical here — ChromaDB can't do substring.
    """
    allowed = set()
    for cid, chunk in chunks_by_id.items():
        m = chunk["metadata"]

        if query.company_id  and m.get("company_id")  != query.company_id:  continue
        if query.sector      and m.get("sector")       != query.sector:      continue
        if query.source_type and m.get("source_type")  != query.source_type: continue
        if query.sentiment   and m.get("sentiment")    != query.sentiment:   continue
        if query.deal_stage  and m.get("deal_stage")   != query.deal_stage:  continue
        if query.logged_by   and m.get("logged_by")    != query.logged_by:   continue
        if query.date_from:
            ts_from = int(query.date_from.replace("-", ""))
            if m.get("date_ts", 0) < ts_from:                                continue
        if query.date_to:
            ts_to = int(query.date_to.replace("-", ""))
            if m.get("date_ts", 0) > ts_to:                                  continue
        if query.tags_include:
            chunk_tags = set(m.get("tags", "").split(","))
            if not any(t in chunk_tags for t in query.tags_include):         continue

        allowed.add(cid)

    # Return the set regardless of whether it is empty.
    # Returning None means "no filter active"; returning an empty set means
    # "filters active but no chunks pass" — the search() early-exit depends on this.
    has_any_filter = any([
        query.company_id, query.sector, query.source_type, query.sentiment,
        query.deal_stage, query.logged_by, query.date_from, query.date_to,
        query.tags_include,
    ])
    return allowed if has_any_filter else None

# ─── RRF Fusion ───────────────────────────────────────────────────────────────

def reciprocal_rank_fusion(
    dense_ranked:  list[tuple[str, float]],   # [(chunk_id, cosine_sim), ...]
    bm25_ranked:   list[tuple[str, float]],   # [(chunk_id, bm25_score), ...]
    k: int = 60,
) -> list[tuple[str, float]]:
    """
    Reciprocal Rank Fusion (Cormack et al., 2009).

    score(d) = Σ_r  1 / (k + rank_r(d))

    k=60 is the canonical default. Higher k → more weight to lower-ranked docs.
    Returns [(chunk_id, rrf_score), ...] sorted descending.
    """
    scores: dict[str, float] = {}

    for rank, (cid, _) in enumerate(dense_ranked, start=1):
        scores[cid] = scores.get(cid, 0.0) + 1.0 / (k + rank)

    for rank, (cid, _) in enumerate(bm25_ranked, start=1):
        scores[cid] = scores.get(cid, 0.0) + 1.0 / (k + rank)

    return sorted(scores.items(), key=lambda x: -x[1])

# ─── Hybrid Retriever ─────────────────────────────────────────────────────────

class HybridRetriever:
    """
    Main retrieval interface for RelIntel.

    Usage:
        retriever = HybridRetriever.load()
        results = retriever.search(RetrievalQuery(
            text="companies with strong ARR growth",
            sector="Fintech",
            top_k=5,
        ))
    """

    def __init__(self, collection, embedder, chunks: list[dict]):
        self.collection  = collection
        self.embedder    = embedder
        self.chunks      = chunks
        self.chunks_by_id = {c["id"]: c for c in chunks}
        self.bm25        = BM25Index(chunks)
        print(f"[HybridRetriever] Ready — {len(chunks)} chunks indexed")

    @classmethod
    def load(cls) -> "HybridRetriever":
        # ChromaDB
        client     = chromadb.PersistentClient(path=str(DB_DIR))
        collection = client.get_collection("relintel_interactions")

        # Embedder
        with open(DATA_DIR / "embedder.pkl", "rb") as f:
            embedder = pickle.load(f)

        # Chunks
        with open(DATA_DIR / "chunks.json") as f:
            chunks = json.load(f)

        return cls(collection, embedder, chunks)

    def _embed_query(self, text: str) -> list[float]:
        vec = self.embedder.transform([text]).astype("float32")
        # L2 normalise (embedder pipeline already does this, but be explicit)
        norm = np.linalg.norm(vec)
        if norm > 0:
            vec = vec / norm
        return vec[0].tolist()

    def search(self, query: RetrievalQuery) -> list[RetrievalResult]:
        """
        Full hybrid retrieval pipeline:
          1. Build metadata filters
          2. Dense retrieval (ChromaDB ANN, filtered)
          3. BM25 retrieval (in-memory, filtered)
          4. RRF fusion
          5. Contextual compression on top-N
          6. Return RetrievalResult objects
        """
        # ── 1. Metadata filters ────────────────────────────────────────────
        chroma_filter = build_chroma_filter(query)
        bm25_allowed  = apply_bm25_filter(self.chunks_by_id, query)

        n_candidates = query.candidate_k

        # ── Early exit: filter matches nothing ────────────────────────────
        # bm25_allowed is an empty set (not None) only when filters are active
        # but no chunks pass — safe to bail before both retrievers run.
        if bm25_allowed is not None and len(bm25_allowed) == 0:
            return []

        # ── 2. Dense retrieval ─────────────────────────────────────────────
        q_emb = self._embed_query(query.text)
        dense_kwargs = dict(
            query_embeddings=[q_emb],
            n_results=n_candidates,
            include=["metadatas", "distances"],
        )
        if chroma_filter:
            dense_kwargs["where"] = chroma_filter

        try:
            dense_res  = self.collection.query(**dense_kwargs)
            dense_ids  = dense_res["ids"][0]
            dense_sims = [1 - d for d in dense_res["distances"][0]]   # cosine dist → sim
            dense_ranked = list(zip(dense_ids, dense_sims))
        except Exception as e:
            # Filter may exclude all docs (e.g. no matching sector)
            dense_ranked = []

        # ── 3. BM25 retrieval ──────────────────────────────────────────────
        bm25_ranked = self.bm25.search(
            query.text,
            top_k=n_candidates,
            allowed_ids=bm25_allowed,
        )

        # ── 4. RRF fusion ──────────────────────────────────────────────────
        fused = reciprocal_rank_fusion(dense_ranked, bm25_ranked, k=60)

        # Build lookup for dense ranks and sims
        dense_rank_map = {cid: (rank + 1, sim) for rank, (cid, sim) in enumerate(dense_ranked)}
        bm25_rank_map  = {cid: rank + 1        for rank, (cid, _)   in enumerate(bm25_ranked)}

        # ── 5. Compress & assemble top-N ───────────────────────────────────
        results = []
        for cid, rrf_score in fused[:query.top_k]:
            chunk = self.chunks_by_id.get(cid)
            if not chunk:
                continue

            d_rank, d_sim = dense_rank_map.get(cid, (None, 0.0))
            b_rank        = bm25_rank_map.get(cid)

            compressed = compress(chunk["text"], query.text)

            results.append(RetrievalResult(
                chunk_id        = cid,
                text            = chunk["text"],
                compressed_text = compressed,
                metadata        = chunk["metadata"],
                dense_rank      = d_rank,
                bm25_rank       = b_rank,
                rrf_score       = rrf_score,
                dense_sim       = d_sim,
            ))

        return results

    def format_results(self, results: list[RetrievalResult], verbose: bool = False) -> str:
        """Human-readable result formatter for debugging and evaluation."""
        lines = []
        for i, r in enumerate(results, 1):
            m = r.metadata
            lines.append(
                f"[{i}] rrf={r.rrf_score:.4f} | sim={r.dense_sim:.3f} | "
                f"source={r.source_label()} | "
                f"dense_r={r.dense_rank or '—'} bm25_r={r.bm25_rank or '—'}"
            )
            lines.append(
                f"     {m['company_name'][:30]} | {m['source_type']} | "
                f"{m['date']} | sector={m['sector']} | "
                f"deal={m['deal_stage'] or '—'} | sentiment={m['sentiment']}"
            )
            lines.append(f"     \"{r.compressed_text[:150]}\"")
            if verbose:
                lines.append(f"     [full] \"{r.text[:200]}\"")
            lines.append("")
        return "\n".join(lines)
