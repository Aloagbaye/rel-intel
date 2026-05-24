"""
Phase 2 — Ingestion Pipeline
RelIntel RAG System

Steps:
  1. Load all 4 entity files from data/
  2. Build enriched documents — structured context joined onto each interaction
  3. Chunk: word-count-aware chunking (sentence boundaries), passthrough for short docs
  4. Embed: TF-IDF + LSA (256-dim) — fully local, no external downloads required
             NOTE: swap SentenceTransformer('all-MiniLM-L6-v2') in production
  5. Store: ChromaDB (persistent) with rich metadata payload for pre-filtering
  6. Verify: smoke tests covering semantic queries + metadata filter queries

Output:
  data/chroma/      — persistent ChromaDB vector store
  data/chunks.json  — serialised chunks + metadata (for BM25 index in Phase 3)
  data/embedder.pkl — fitted TF-IDF/LSA pipeline (for query-time encoding)
"""

import json
import pickle
import re
import sys
import time
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")
sys.stderr.reconfigure(encoding="utf-8")

import chromadb
import numpy as np
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import Normalizer

# ─── Paths ────────────────────────────────────────────────────────────────────

ROOT     = Path(__file__).parent
DATA_DIR = ROOT / "data"
DB_DIR   = DATA_DIR / "chroma"
DB_DIR.mkdir(parents=True, exist_ok=True)

# ─── Config ───────────────────────────────────────────────────────────────────

EMBED_DIM        = 256
COLLECTION_NAME  = "relintel_interactions"
MAX_CHUNK_WORDS  = 120   # soft ceiling per chunk (word count)
CHUNK_OVERLAP_W  = 15    # word overlap between consecutive chunks

# ─── Helpers ─────────────────────────────────────────────────────────────────

def load_json(name: str) -> list[dict]:
    with open(DATA_DIR / f"{name}.json") as f:
        return json.load(f)

def word_count(text: str) -> int:
    return len(text.split())

def chunk_text(text: str, max_words: int = MAX_CHUNK_WORDS, overlap: int = CHUNK_OVERLAP_W) -> list[str]:
    """
    Sentence-boundary-aware chunking with word-count ceiling and overlap.
    Short texts (< max_words) are returned as a single chunk.
    """
    if word_count(text) <= max_words:
        return [text]

    sentences = re.split(r'(?<=[.!?])\s+', text.strip()) # split text into sentences
    chunks, current = [], []
    current_wc = 0

    for sent in sentences:
        swc = word_count(sent)
        if current_wc + swc > max_words and current:
            chunks.append(" ".join(current))
            # Keep last `overlap` words worth of sentences for continuity
            overlap_sents, owc = [], 0
            for s in reversed(current):
                wc = word_count(s)
                if owc + wc <= overlap:
                    overlap_sents.insert(0, s)
                    owc += wc
                else:
                    break
            current = overlap_sents + [sent]
            current_wc = sum(word_count(s) for s in current)
        else:
            current.append(sent)
            current_wc += swc

    if current:
        chunks.append(" ".join(current))
    return chunks or [text]

# ─── Document Builder ─────────────────────────────────────────────────────────

def build_documents(interactions, companies, contacts, deals) -> list[dict]:
    """
    Enrich each interaction with structured context, then chunk.
    Returns flat list of chunk dicts: {id, text, metadata}

    Design decisions:
    - Context prefix is prepended to every chunk so retrieved snippets are
      self-contained — the LLM doesn't need a separate lookup to know which
      company/deal a chunk belongs to.
    - Metadata is fully scalar (ChromaDB constraint) — lists are comma-joined.
    - date_ts is stored as YYYYMMDD int for range filtering without date parsing.
    """
    documents = []

    for ix in interactions:
        company = companies.get(ix["company_id"], {})
        deal_id = ix.get("deal_id")
        deal    = deals.get(deal_id, {}) if deal_id else {}

        # Resolve contact display names
        contact_names = []
        for cid in ix.get("contact_ids", []):
            c = contacts.get(cid, {})
            if c:
                contact_names.append(f"{c['first_name']} {c['last_name']} ({c['title']})")

        # Context prefix — grounding for retrieval-time isolation
        ctx_lines = [
            f"Company: {company.get('name', 'Unknown')} [{company.get('sector', '')}]",
            f"Interaction: {ix['type'].replace('_', ' ').title()} on {ix['date'][:10]}",
        ]
        if contact_names:
            ctx_lines.append(f"Contacts: {', '.join(contact_names)}")
        if deal:
            ctx_lines.append(
                f"Deal: {deal.get('name', '')} | Stage: {deal.get('stage', '')} "
                f"| ${deal.get('amount_usd', 0):,}"
            )
        context_prefix = "\n".join(ctx_lines) + "\n\n"

        body_chunks = chunk_text(ix["body"])

        for idx, chunk_body in enumerate(body_chunks):
            full_text = context_prefix + chunk_body
            metadata  = {
                # Interaction
                "interaction_id":    ix["interaction_id"],
                "source_type":       ix["type"],
                "date":              ix["date"][:10],
                "date_ts":           int(ix["date"][:10].replace("-", "")),
                "sentiment":         ix["sentiment"],
                "logged_by":         ix.get("logged_by", ""),
                "subject":           ix.get("subject", ""),
                "tags":              ",".join(ix.get("tags", [])),
                # Company
                "company_id":        ix["company_id"],
                "company_name":      company.get("name", ""),
                "sector":            company.get("sector", ""),
                "company_stage":     company.get("stage", ""),
                "relationship_strength": company.get("relationship_strength", ""),
                # Deal
                "deal_id":           deal_id or "",
                "deal_stage":        deal.get("stage", ""),
                "deal_type":         deal.get("deal_type", ""),
                "amount_usd":        int(deal.get("amount_usd", 0)),
                # Contact
                "primary_contact_id": ix["contact_ids"][0] if ix["contact_ids"] else "",
                "contact_count":     len(ix["contact_ids"]),
                # Chunk
                "chunk_index":       idx,
                "chunk_count":       len(body_chunks),
                "word_count":        word_count(full_text),
            }
            documents.append({"id": f"{ix['interaction_id']}_c{idx}", "text": full_text, "metadata": metadata})

    return documents

# ─── Embedder ─────────────────────────────────────────────────────────────────

def build_embedder(corpus: list[str]) -> Pipeline:
    """
    TF-IDF (bigrams, sublinear TF) → TruncatedSVD (LSA, 256-dim) → L2 normalise.

    Why LSA here vs sentence-transformers in production:
    - No external downloads — runs fully offline in any environment.
    - Captures domain vocabulary (ARR, NRR, term sheet, carry) without fine-tuning.
    - 256-dim cosine space is sufficient for this corpus size.
    - In production: swap for Voyage-finance-2, OpenAI text-embedding-3-small,
      or a locally-served ONNX model for semantic generalisation.
    """
    pipe = Pipeline([
        ("tfidf", TfidfVectorizer(
            max_features=12_000,
            ngram_range=(1, 2),
            sublinear_tf=True,
            min_df=2,
        )),
        ("svd",  TruncatedSVD(n_components=EMBED_DIM, random_state=42, n_iter=7)),
        ("norm", Normalizer(norm="l2")),
    ])
    pipe.fit(corpus)
    return pipe

def embed(pipe: Pipeline, texts: list[str]) -> np.ndarray:
    return pipe.transform(texts).astype("float32")

# ─── Smoke Tests ──────────────────────────────────────────────────────────────

def run_smoke_tests(collection, pipe):
    queries = [
        {
            "label":   "Semantic — strong ARR growth traction",
            "query":   "companies with strong ARR growth and enterprise customer traction",
            "filters": None,
        },
        {
            "label":   "Semantic — due diligence technical architecture",
            "query":   "due diligence call about technical architecture and product roadmap",
            "filters": {"source_type": "meeting"},
        },
        {
            "label":   "Metadata filter — Fintech + positive sentiment",
            "query":   "strong fintech investment opportunity",
            "filters": {"$and": [{"sector": {"$eq": "Fintech"}}, {"sentiment": {"$eq": "positive"}}]},
        },
        {
            "label":   "Metadata filter — Term Sheet stage",
            "query":   "term sheet negotiation and deal update",
            "filters": {"deal_stage": {"$eq": "Term Sheet"}},
        },
    ]

    for q in queries:
        print(f"\n  ▸ {q['label']}")
        print(f"    Query: \"{q['query']}\"")
        if q["filters"]:
            print(f"    Filter: {q['filters']}")

        q_emb = embed(pipe, [q["query"]]).tolist()
        kwargs = dict(
            query_embeddings=q_emb,
            n_results=3,
            include=["documents", "metadatas", "distances"],
        )
        if q["filters"]:
            kwargs["where"] = q["filters"]

        try:
            res = collection.query(**kwargs)
        except Exception as e:
            print(f"    ⚠ Query failed: {e}")
            continue

        for rank, (doc, meta, dist) in enumerate(zip(
            res["documents"][0], res["metadatas"][0], res["distances"][0]
        )):
            sim = 1 - dist
            body_start = doc.find("\n\n") + 2
            snippet = doc[body_start:body_start + 110]
            print(f"\n    [{rank+1}] sim={sim:.3f} | {meta['company_name'][:28]} | "
                  f"{meta['source_type']} | {meta['date']} | deal={meta['deal_stage'] or '—'}")
            print(f"         \"{snippet}...\"")

# ─── Main ─────────────────────────────────────────────────────────────────────

def ingest():
    print("=" * 57)
    print("  RelIntel — Phase 2: Ingestion Pipeline")
    print("=" * 57)

    # 1. Load
    print("\n[1/5] Loading data...")
    interactions = load_json("interactions")
    companies    = {c["company_id"]: c for c in load_json("companies")}
    contacts     = {c["contact_id"]: c for c in load_json("contacts")}
    deals        = {d["deal_id"]:    d for d in load_json("deals")}
    print(f"      {len(interactions)} interactions | {len(companies)} companies | "
          f"{len(contacts)} contacts | {len(deals)} deals")

    # 2. Build chunks
    print("\n[2/5] Building enriched chunks...")
    t0 = time.time()
    documents = build_documents(interactions, companies, contacts, deals)
    n_multi = len([d for d in documents if d["metadata"]["chunk_count"] > 1])
    wcs = [d["metadata"]["word_count"] for d in documents]
    wcs_s = sorted(wcs)
    print(f"      {len(documents)} chunks from {len(interactions)} interactions  "
          f"({time.time()-t0:.1f}s)")
    print(f"      Word counts — min:{wcs_s[0]}  p50:{wcs_s[len(wcs_s)//2]}  "
          f"p95:{wcs_s[int(len(wcs_s)*.95)]}  max:{wcs_s[-1]}")
    print(f"      Multi-chunk docs: {n_multi}")

    # Save chunks for BM25 (Phase 3)
    with open(DATA_DIR / "chunks.json", "w") as f:
        json.dump(documents, f, indent=2)
    print("      Saved → chunks.json")

    # 3. Build embedder on corpus
    print(f"\n[3/5] Fitting TF-IDF + LSA embedder (dim={EMBED_DIM})...")
    t0 = time.time()
    corpus = [d["text"] for d in documents]
    pipe = build_embedder(corpus)
    explained = pipe["svd"].explained_variance_ratio_.sum()
    print(f"      Fitted in {time.time()-t0:.1f}s | vocab={len(pipe['tfidf'].vocabulary_):,} "
          f"| explained variance={explained:.1%}")

    # Persist embedder
    with open(DATA_DIR / "embedder.pkl", "wb") as f:
        pickle.dump(pipe, f)
    print("      Saved → embedder.pkl")

    # 4. Embed all chunks
    print(f"\n[4/5] Embedding {len(documents)} chunks...")
    t0 = time.time()
    embeddings = embed(pipe, corpus)
    print(f"      Done in {time.time()-t0:.2f}s | shape={embeddings.shape}")

    # 5. Load into ChromaDB
    print(f"\n[5/5] Loading into ChromaDB ({COLLECTION_NAME})...")
    t0 = time.time()
    client = chromadb.PersistentClient(path=str(DB_DIR))
    try:
        client.delete_collection(COLLECTION_NAME)
    except Exception:
        pass
    collection = client.create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )
    BATCH = 500
    for i in range(0, len(documents), BATCH):
        batch = documents[i:i+BATCH]
        collection.add(
            ids        = [d["id"]       for d in batch],
            documents  = [d["text"]     for d in batch],
            metadatas  = [d["metadata"] for d in batch],
            embeddings = embeddings[i:i+BATCH].tolist(),
        )
    print(f"      {collection.count()} vectors loaded in {time.time()-t0:.1f}s")
    print(f"      Persisted → {DB_DIR}")

    # Smoke tests
    print("\n── Smoke Tests ─────────────────────────────────────────")
    run_smoke_tests(collection, pipe)

    print("\n✅ Phase 2 complete.")
    return collection, pipe, documents

if __name__ == "__main__":
    ingest()
