# ragkit

A small, dependency-free retrieval-augmented generation (RAG) toolkit for
Python. Chunking, BM25 lexical retrieval, vector retrieval over pluggable
embedders, reciprocal-rank-fusion hybrid search, and a pipeline that answers
with cited sources - all in plain stdlib Python, all covered by tests.

No vendor SDKs, no model downloads, no API keys required to learn, test, or
demo. Plug in your own embedder or LLM by implementing a one-method protocol.

## Why

Most RAG examples either call a hosted API end-to-end or pull in hundreds of
megabytes of dependencies before the first retrieval runs. ragkit keeps the
pipeline shape identical to production systems - chunk, index, retrieve,
generate - while staying runnable anywhere Python runs, including CI.

## Features

- **Chunking** - fixed-size windows with overlap, or sentence-aware chunks that never split a sentence.
- **BM25 retriever** - Okapi BM25 with document-length normalization and metadata filters.
- **Vector retriever** - in-memory cosine-similarity store over any embedder; ships with deterministic `HashingEmbedder` and corpus-fit `TfIdfEmbedder`.
- **Hybrid retriever** - reciprocal rank fusion over lexical + dense rankings.
- **RAG pipeline** - ingest, retrieve, prompt-build, generate; every answer carries its source chunks and scores for citation.
- **Pluggable LLM** - one-method protocol; `EchoLLM` answers offline for tests and demos.

## Quick start

Requires Python 3.9+. No dependencies to install.

```bash
git clone https://github.com/Prajit-22/ragkit.git
cd ragkit
python examples/quickstart.py
```

```python
from ragkit import BM25Retriever, EchoLLM, RAGPipeline, sentence_chunks

pipe = RAGPipeline(BM25Retriever(), EchoLLM())
for doc in open("my_notes.txt"):
    pipe.ingest(sentence_chunks(doc))

result = pipe.query("What did I write about deployment?")
print(result["answer"])
for source in result["sources"]:
    print(f"  [{source['score']}] {source['text'][:80]}")
```

## Plugging in a real embedder or LLM

```python
from ragkit import CallableLLM, RAGPipeline, VectorRetriever

class MyEmbedder:
    dimensions = 1536
    def embed(self, text):
        return my_embedding_api(text)  # any provider

pipe = RAGPipeline(
    VectorRetriever(MyEmbedder()),
    CallableLLM(lambda prompt: my_llm_api(prompt)),
)
```

## Project layout

```
ragkit/
  chunking.py      fixed-size and sentence-aware chunkers
  embeddings.py    Embedder protocol, HashingEmbedder, TfIdfEmbedder
  vectorstore.py   in-memory cosine-similarity store
  retrieval.py     BM25Retriever, VectorRetriever, HybridRetriever (RRF)
  llm.py           LLM protocol, EchoLLM, CallableLLM
  pipeline.py      RAGPipeline: ingest -> retrieve -> prompt -> answer
tests/             43 unittest tests, stdlib only
examples/          offline end-to-end demo with a small markdown corpus
```

## Running the tests

```bash
python -m unittest discover -s tests
```

## Design notes

- `EchoLLM` answers extractively from the retrieved context, so tests and demos are deterministic and can never hallucinate.
- `HashingEmbedder` uses feature hashing over tokens - retrieval quality is below a neural embedder, but the interface and pipeline are identical, so swapping in a real embedder is a two-line change.
- BM25 filters out zero-score documents; vector search returns nearest neighbors by cosine similarity; the hybrid retriever fuses both rankings with reciprocal rank fusion (k=60).

### Retriever boundaries

`VectorRetriever(embedder, store=store)` uses the exact store you pass in,
including an empty store. Adding or deleting through the retriever changes
that same store. Omit `store` to get a fresh, independent in-memory store.

All retrievers require `k > 0`. `HybridRetriever` requires `rrf_k >= 0`;
the default is 60. A zero constant is valid, but weights the first rank more
heavily. Hybrid metadata filters apply to both component rankings.

## License

MIT
