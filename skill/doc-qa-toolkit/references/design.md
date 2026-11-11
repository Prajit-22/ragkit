# Design: how retrieval works here

Read this before tuning retrieval quality, changing chunking, or explaining
the toolkit's behavior.

## The pipeline shape

chunk -> index -> retrieve -> prompt -> answer. Every stage is plain stdlib
Python, so the whole thing runs anywhere Python runs, including CI. The shape
matches production RAG systems, which is the point: swapping any stage for a
hosted equivalent changes one class, not the architecture.

## Chunking

Two chunkers, both overlap-aware:

- `fixed_size_chunks(text, chunk_size=500, overlap=50)` - character windows.
- `sentence_chunks(text, max_sentences=4, overlap_sentences=1)` - whole
  sentences, the CLI default. Sentence boundaries keep facts intact and
  usually retrieve better on prose.

Neither emits a trailing chunk wholly contained in the previous one, so
overlap does not create duplicate retrieval hits.

## Retrievers

- `BM25Retriever` - Okapi BM25 with k1=1.5, b=0.75 and document-length
  normalization. Zero-score documents are filtered out. `save`/`load`
  persist the index as JSON; loading rebuilds term statistics through the
  normal add path, so they always match the documents.
- `VectorRetriever` - cosine similarity over an `InMemoryVectorStore`, over
  any embedder with `dimensions` and `embed(text)`. The store snapshots to
  JSON with the vectors, so a loaded store searches without an embedder.
- `HybridRetriever` - reciprocal rank fusion (default k=60) over the lexical
  and dense rankings. Metadata filters apply to both halves.

All retrievers require `k > 0` and accept `metadata_filter={key: value}`
exact-match filtering.

## MMR diversification

`VectorRetriever.search_mmr` applies maximal marginal relevance over a plain
top-`fetch_k` pool: each pick maximizes
`lambda_mult * relevance - (1 - lambda_mult) * redundancy`. `lambda_mult=1`
reproduces plain ordering; lower values push near-duplicates out. Returned
scores stay cosine similarity to the query - lambda changes selection, not
what a score means. `RAGPipeline.retrieve_mmr`/`query_mmr` raise `TypeError`
on retrievers without `search_mmr` instead of silently degrading.

## Tokenization

`tokenize` lowercases, NFC-normalizes, and keeps letters, digits, combining
marks, and apostrophes from any script, so accented Latin and Indic scripts
index as whole words. Scripts written without spaces (Chinese, Japanese)
come back as one token per unbroken run. BM25 and both bundled embedders
share this tokenizer.

## The prompt

`RAGPipeline.build_prompt` formats retrieved chunks as `- ` bullets under a
`Context:` header, then the question. The default `EchoLLM` answers by
quoting the first sentence of that context block - deterministic and unable
to hallucinate, at the cost of no synthesis. `ask --prompt-only` prints the
same prompt for a real LLM.
