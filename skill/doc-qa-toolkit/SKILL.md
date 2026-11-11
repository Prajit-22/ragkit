---
name: doc-qa-toolkit
description: Answers questions over a folder of documents with cited sources, using a dependency-free RAG pipeline - sentence-aware chunking, BM25 / vector / hybrid retrieval, and a prompt builder for any LLM. Use whenever the agent must ground an answer in local files (notes, docs, runbooks, transcripts), build a small searchable knowledge base, or add retrieval to a script without vendor SDKs.
license: MIT
compatibility: Python 3.9+. No third-party packages. Runs fully offline by default; no API keys required. Optional real embedder/LLM plug-ins need network access to their provider.
metadata:
  author: Prajit Chaturvedi
  version: 1.0.0
---

# Document Q&A Toolkit

Chunk documents, index them, and answer questions with cited sources - all in
plain stdlib Python. The default pipeline runs entirely offline: BM25 lexical
retrieval plus an extractive answer, so it works with zero API keys. When you
want a generative answer, the CLI prints a ready-made retrieval prompt for any
LLM, and the library accepts any embedder or LLM through a one-method protocol.

## How you should use it as an agent

1. To answer from the user's documents: index the folder once
   (`scripts/rag.py index <folder>`), then ask (`scripts/rag.py ask "..."`).
   The answer prints to stdout; the cited sources with scores print to stderr.
2. To write the answer yourself: `scripts/rag.py ask --prompt-only "..."`
   prints the filled retrieval prompt - retrieved passages plus the question -
   for you to answer directly. Always cite the sources it lists on stderr.
3. To embed retrieval in a script: import the vendored package
   (`scripts/ragkit/`) and use `RAGPipeline` directly. See
   `references/plugging-in.md` for real embedder/LLM wiring.
4. After changing any code, run `scripts/selftest.py` and report failures
   honestly.

## Setup

No install step - everything is stdlib Python. The CLI keeps its index at
`~/.doc-qa-toolkit/index.json` (override with `DOC_QA_DIR`).

```bash
python3 scripts/selftest.py                       # 9 offline checks
python3 scripts/rag.py index ~/notes ./docs       # index .md/.txt files
python3 scripts/rag.py ask "When do deploys run?" # answer + sources
python3 scripts/rag.py status                     # chunk count, index path
python3 scripts/rag.py reset                      # drop the index
```

`index` accepts files and folders (folders are walked recursively for
`.md`/`.markdown`/`.txt`) and adds to the existing index, so you can grow it
folder by folder. Re-indexing the same file duplicates its chunks - `reset`
first for a clean rebuild.

## Retrieval modes

- **BM25 (default, persisted)** - lexical ranking with document-length
  normalization; the index snapshots to JSON and reloads without any embedder.
- **Vector** - cosine similarity over any embedder; ships with a deterministic
  `HashingEmbedder` and a corpus-fit `TfIdfEmbedder` for offline use.
  `search_mmr` diversifies results so near-duplicates stop filling the top-k.
- **Hybrid** - reciprocal rank fusion (k=60) over BM25 + vector rankings.
  The CLI persists BM25; vector and hybrid modes are one import away in
  library code (see `references/design.md`).

Retrieval quality with the bundled embedders is below a neural embedder; the
interfaces are identical, so swapping in a real one is a two-line change.

## Honest limits

- All 9 selftest checks run offline against the local package - no live
  embedding or LLM APIs are called at any point.
- The default `EchoLLM` answers extractively by quoting the retrieved context;
  it cannot hallucinate, but it also cannot synthesize. For generative
  answers, use `--prompt-only` or plug in a real LLM.
- The tokenizer handles Latin and Indic scripts well; text without spaces
  (Chinese, Japanese) comes back as one token per unbroken run - supply your
  own segmenter for word-level matching there.
- This is a general document-Q&A toolkit, not a personal-notes product: bring
  your own documents, and manage the index folder yourself.

## Files

- `scripts/rag.py` - CLI: index / ask / status / reset
- `scripts/ragkit/` - the vendored toolkit: chunking, embeddings, vector
  store, retrievers (BM25, vector, hybrid), pipeline
- `scripts/selftest.py` - 9 offline checks (chunking, retrieval, citations,
  persistence, filters, delete, TF-IDF vectors, hybrid fusion, CLI round-trip)
- `references/design.md` - retrieval semantics and tuning
- `references/plugging-in.md` - connecting real embedders and LLMs
