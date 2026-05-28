"""Shared retrieval experiment adapters for evaluation scripts."""

from __future__ import annotations

from eval_dataset import EvalSample
from retriever import HybridRetriever, RetrievalQuery, RetrievalResult

CANDIDATE_K = 20


def _filter_kwargs(filters: dict) -> dict:
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


def retrieve_hybrid(retriever: HybridRetriever, sample: EvalSample) -> list[RetrievalResult]:
    rq = RetrievalQuery(
        text        = sample.question,
        top_k       = sample.top_k,
        candidate_k = CANDIDATE_K,
        **_filter_kwargs(sample.retrieval_filters),
    )
    return retriever.search(rq)


def retrieve_dense_only(retriever: HybridRetriever, sample: EvalSample) -> list[RetrievalResult]:
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
    for cid, dist, meta in zip(res["ids"][0], res["distances"][0], res["metadatas"][0]):
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


def retrieve_bm25_only(retriever: HybridRetriever, sample: EvalSample) -> list[RetrievalResult]:
    from retriever import apply_bm25_filter, compress, RetrievalResult as RR

    rq = RetrievalQuery(text=sample.question, **_filter_kwargs(sample.retrieval_filters))
    allowed = apply_bm25_filter(retriever.chunks_by_id, rq)
    if allowed is not None and len(allowed) == 0:
        return []

    ranked = retriever.bm25.search(sample.question, top_k=sample.top_k, allowed_ids=allowed)
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


def retrieve_hybrid_no_compression(retriever: HybridRetriever, sample: EvalSample) -> list[RetrievalResult]:
    results = retrieve_hybrid(retriever, sample)
    for r in results:
        body_start = r.text.find("\n\n") + 2
        r.compressed_text = r.text[body_start:].strip() if body_start > 1 else r.text
    return results


EXPERIMENTS = {
    "hybrid":             retrieve_hybrid,
    "dense":              retrieve_dense_only,
    "bm25":               retrieve_bm25_only,
    "hybrid_no_compress": retrieve_hybrid_no_compression,
}
