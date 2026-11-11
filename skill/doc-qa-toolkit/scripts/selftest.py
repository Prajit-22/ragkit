#!/usr/bin/env python3
"""Offline smoke test for the doc-qa-toolkit skill.

Every check runs locally against the vendored ragkit package - no network,
no API keys. Run from anywhere:

    python3 scripts/selftest.py
"""

import os
import subprocess
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

CHECKS = []


def check(name):
    def wrap(fn):
        CHECKS.append((name, fn))
        return fn
    return wrap


@check("sentence chunking never splits a sentence")
def _():
    from ragkit import sentence_chunks
    text = "Deploys run at noon. Never deploy on Friday. Rollbacks are automatic."
    chunks = sentence_chunks(text, max_sentences=2, overlap_sentences=0)
    assert chunks, "no chunks produced"
    for chunk in chunks:
        assert "Never deploy" not in chunk or chunk.startswith("Never deploy") or chunk.endswith("."), chunk
    joined = " ".join(chunks)
    assert "Never deploy on Friday." in joined


@check("BM25 index + search ranks the relevant passage first")
def _():
    from ragkit import BM25Retriever, Document
    r = BM25Retriever()
    r.add(Document(id="a", text="The gateway retries with exponential backoff."))
    r.add(Document(id="b", text="Bananas ripen faster in a paper bag."))
    hits = r.search("gateway backoff retry", k=1)
    assert hits and hits[0].document.id == "a", hits


@check("pipeline answers with cited sources and scores")
def _():
    from ragkit import BM25Retriever, EchoLLM, RAGPipeline
    pipe = RAGPipeline(BM25Retriever(), EchoLLM())
    pipe.ingest(["Rate limits reset at midnight UTC."], [{"source": "notes.md"}])
    result = pipe.query("When do rate limits reset?")
    assert result["answer"].strip(), "empty answer"
    assert result["sources"], "no sources"
    assert result["sources"][0]["metadata"]["source"] == "notes.md"
    assert isinstance(result["sources"][0]["score"], float)


@check("BM25 snapshot save/load preserves retrieval results")
def _():
    from ragkit import BM25Retriever, Document
    r = BM25Retriever()
    r.add(Document(id="a", text="Quotas reset daily at midnight."))
    r.add(Document(id="b", text="The CLI prints sources to stderr."))
    before = r.search("quotas reset", k=2)
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "index.json")
        r.save(path)
        after = BM25Retriever.load(path).search("quotas reset", k=2)
    assert [h.document.id for h in before] == [h.document.id for h in after]


@check("metadata filters restrict retrieval")
def _():
    from ragkit import BM25Retriever, Document
    r = BM25Retriever()
    r.add(Document(id="a", text="deployment checklist item", metadata={"kind": "ops"}))
    r.add(Document(id="b", text="deployment checklist item", metadata={"kind": "draft"}))
    hits = r.search("deployment checklist", k=4, metadata_filter={"kind": "ops"})
    assert [h.document.id for h in hits] == ["a"], hits


@check("delete removes documents from future searches")
def _():
    from ragkit import BM25Retriever, Document
    r = BM25Retriever()
    r.add(Document(id="a", text="temporary scratch note about retries"))
    assert r.search("retries", k=1), "doc should be searchable before delete"
    assert r.delete("a") is True
    assert not r.search("retries", k=1), "doc should be gone after delete"


@check("TF-IDF vector retrieval works without a neural embedder")
def _():
    from ragkit import Document, TfIdfEmbedder, VectorRetriever
    docs = ["The worker caches responses in KV.", "Sourdough needs a warm kitchen."]
    embedder = TfIdfEmbedder().fit(docs)
    r = VectorRetriever(embedder)
    for i, text in enumerate(docs):
        r.add(Document(id=str(i), text=text))
    hits = r.search("cache responses", k=1)
    assert hits and hits[0].document.id == "0", hits


@check("hybrid retriever fuses lexical and dense rankings")
def _():
    from ragkit import BM25Retriever, Document, HybridRetriever, TfIdfEmbedder, VectorRetriever
    docs = ["Retries honor the Retry-After header.", "Bread baking needs steam."]
    embedder = TfIdfEmbedder().fit(docs)
    hybrid = HybridRetriever(BM25Retriever(), VectorRetriever(embedder))
    for i, text in enumerate(docs):
        hybrid.add(Document(id=str(i), text=text))
    hits = hybrid.search("Retry-After header", k=1)
    assert hits and hits[0].document.id == "0", hits


@check("rag.py CLI: index, ask, status, reset round-trip")
def _():
    env = dict(os.environ)
    with tempfile.TemporaryDirectory() as tmp:
        env["DOC_QA_DIR"] = tmp
        docs = os.path.join(tmp, "docs")
        os.makedirs(docs)
        with open(os.path.join(docs, "notes.md"), "w") as handle:
            handle.write("Deploys run at noon. The rollback runbook lives in the wiki.")
        rag = os.path.join(HERE, "rag.py")
        out = subprocess.run([sys.executable, rag, "index", docs],
                             capture_output=True, text=True, env=env)
        assert out.returncode == 0, out.stderr
        out = subprocess.run([sys.executable, rag, "ask", "When do deploys run?"],
                             capture_output=True, text=True, env=env)
        assert out.returncode == 0, out.stderr
        assert out.stdout.strip(), "empty answer"
        assert "sources" in out.stderr, out.stderr
        out = subprocess.run([sys.executable, rag, "status"],
                             capture_output=True, text=True, env=env)
        assert out.returncode == 0 and "chunks:" in out.stdout, out.stdout
        out = subprocess.run([sys.executable, rag, "reset"],
                             capture_output=True, text=True, env=env)
        assert out.returncode == 0, out.stderr


def main():
    failed = 0
    for name, fn in CHECKS:
        try:
            fn()
            print("PASS %s" % name)
        except Exception as exc:
            failed += 1
            print("FAIL %s: %s" % (name, exc))
    print("%s - %d of %d checks" % ("FAIL" if failed else "PASS", len(CHECKS) - failed, len(CHECKS)))
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
