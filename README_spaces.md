---
title: RelIntel — Relationship Intelligence RAG
emoji: 🔗
colorFrom: blue
colorTo: indigo
sdk: gradio
sdk_version: "4.44.1"
python_version: "3.12"
app_file: app.py
pinned: false
license: mit
short_description: Hybrid BM25+dense RAG for relationship intelligence
---

# RelIntel — Relationship Intelligence RAG System

A production-quality RAG pipeline demonstrating hybrid retrieval, metadata
pre-filtering, and citation-grounded generation over a synthetic private
capital relationship dataset.

## What it does

Ask questions about portfolio companies, deal activity, and relationship
history. The system retrieves relevant interaction records using a hybrid
BM25 + dense retrieval pipeline fused via Reciprocal Rank Fusion, then
generates a grounded answer with Claude Haiku.

## Pipeline

```
Query
  → Metadata Pre-filter (sector / deal stage / source type / sentiment / date)
  → Dense retrieval (TF-IDF + LSA, 256-dim, ChromaDB HNSW)
     + BM25 retrieval (Okapi, in-memory)
  → Reciprocal Rank Fusion (k=60)
  → Contextual Compression
  → Claude Haiku (citation-grounded generation)
  → Hallucination Guard (citation coverage · source grounding · number check)
```

## Corpus

- 50 companies across 10 sectors (Fintech, HealthTech, Climate Tech, …)
- 200 contacts · 150 deals · 500 interaction notes
- Interaction types: emails, meetings, calls, LinkedIn messages, events

## Filters available

| Filter | Options |
|---|---|
| Sector | Fintech, HealthTech, Climate Tech, Enterprise SaaS, … |
| Source Type | call, email, meeting, linkedin_message, event |
| Deal Stage | Lead, Qualified, Due Diligence, Term Sheet, Closed Won, … |
| Sentiment | positive, neutral, negative |
| Date Range | YYYY-MM-DD from/to |

## Example queries

- *Which companies are showing the strongest ARR growth and NRR?*
- *Summarise key findings from our Fintech due diligence meetings.*
- *What are the latest updates on Term Sheet stage deals?*
- *Which HealthTech companies have the strongest competitive moat?*
- *Are there companies where we logged concerns or negative signals?*

## Setup (self-hosting)

```bash
git clone https://huggingface.co/spaces/YOUR_USERNAME/relintel
cd relintel
pip install -r requirements_spaces.txt
export ANTHROPIC_API_KEY=sk-ant-...
python app.py
```

## Running on your own HF Space

1. Fork this Space
2. Go to **Settings → Repository secrets**
3. Add `ANTHROPIC_API_KEY` with your key
4. The Space will restart and be live

## Project structure

```
relintel/
├── app.py                  ← Gradio app (this file)
├── requirements_spaces.txt ← Space dependencies
├── src/
│   ├── retriever.py        ← Hybrid retrieval engine (Phase 3)
│   └── generator.py        ← Generation + citation + guard (Phase 4)
└── data/
    ├── chroma/             ← Pre-built ChromaDB vector store
    ├── chunks.json         ← BM25 corpus
    ├── embedder.pkl        ← Fitted TF-IDF + LSA pipeline
    ├── companies.json
    ├── contacts.json
    ├── deals.json
    └── interactions.json
```

## Full project

The complete system (data generation, ingestion, retrieval, generation,
RAGAS evaluation) lives at:
[github.com/Aloagbaye/relintel](https://github.com/Aloagbaye/relintel)

