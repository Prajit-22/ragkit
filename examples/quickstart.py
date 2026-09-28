"""End-to-end ragkit demo: chunk, index, retrieve, answer - fully offline.

Run from the repository root:

    python examples/quickstart.py
"""

import json
from pathlib import Path

from ragkit import (BM25Retriever, EchoLLM, HashingEmbedder, HybridRetriever,
                    RAGPipeline, VectorRetriever, sentence_chunks)

DATA = Path(__file__).parent / "data"


def load_corpus():
    docs = []
    for path in sorted(DATA.glob("*.md")):
        for chunk in sentence_chunks(path.read_text(), max_sentences=3, overlap_sentences=1):
            docs.append((chunk, {"source": path.name}))
    return docs


def main():
    corpus = load_corpus()
    retriever = HybridRetriever(BM25Retriever(), VectorRetriever(HashingEmbedder()))
    pipe = RAGPipeline(retriever, EchoLLM())
    texts, metadatas = zip(*corpus)
    pipe.ingest(list(texts), metadatas=list(metadatas))
    print(f"Indexed {len(texts)} chunks from {len(list(DATA.glob('*.md')))} files\n")

    for question in [
        "How does BM25 rank documents?",
        "What does vector search capture that keyword search misses?",
        "Why combine retrieval with generation?",
    ]:
        result = pipe.query(question, k=2)
        print(f"Q: {question}")
        print(f"A: {result['answer']}")
        for source in result["sources"]:
            print(f"   - [{source['score']:.4f}] ({source['metadata']['source']}) {source['text'][:80]}...")
        print()


if __name__ == "__main__":
    main()
