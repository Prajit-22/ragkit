import unittest

from ragkit import (BM25Retriever, CallableLLM, EchoLLM, RAGPipeline,
                    sentence_chunks)

CORPUS = [
    "RAG combines retrieval with generation. The retriever finds relevant chunks. The LLM answers using only those chunks.",
    "BM25 is a lexical ranking function. It uses term frequency and inverse document frequency. It needs no embeddings.",
    "Vector search embeds text into dense vectors. Cosine similarity compares them. It captures semantic similarity.",
]


class PipelineTests(unittest.TestCase):
    def setUp(self):
        self.pipe = RAGPipeline(BM25Retriever(), EchoLLM())
        chunks = []
        for doc in CORPUS:
            chunks.extend(sentence_chunks(doc, max_sentences=2, overlap_sentences=0))
        self.ids = self.pipe.ingest(chunks)

    def test_ingest_assigns_ids_and_skips_blanks(self):
        self.assertEqual(len(self.ids), 6)
        self.assertEqual(self.pipe.ingest(["", "   "]), [])

    def test_query_returns_answer_with_cited_sources(self):
        out = self.pipe.query("How does BM25 use term frequency to rank documents?", k=2)
        self.assertIn("BM25", out["answer"])
        self.assertGreaterEqual(len(out["sources"]), 1)
        self.assertIn("term frequency", out["sources"][0]["text"])
        self.assertIn("Context:", out["prompt"])
        self.assertIn("How does BM25 use term frequency to rank documents?", out["prompt"])

    def test_no_context_answer_is_honest(self):
        out = self.pipe.query("What is the airspeed of a swallow?", k=1)
        # EchoLLM still has a context block; force the empty case instead.
        empty_pipe = RAGPipeline(BM25Retriever(), EchoLLM())
        out = empty_pipe.query("anything at all")
        self.assertEqual(out["answer"], "I don't have enough context to answer that.")
        self.assertEqual(out["sources"], [])

    def test_custom_llm_receives_built_prompt(self):
        seen = {}

        def fake_llm(prompt):
            seen["prompt"] = prompt
            return "canned answer"

        pipe = RAGPipeline(BM25Retriever(), CallableLLM(fake_llm))
        pipe.ingest(["The capital of France is Paris."])
        out = pipe.query("What is the capital of France?")
        self.assertEqual(out["answer"], "canned answer")
        self.assertIn("The capital of France is Paris.", seen["prompt"])

    def test_metadata_flows_through_ingest(self):
        pipe = RAGPipeline(BM25Retriever(), EchoLLM())
        pipe.ingest(["alpha content"], metadatas=[{"source": "handbook.pdf"}])
        out = pipe.query("alpha", k=1)
        self.assertEqual(out["sources"][0]["metadata"], {"source": "handbook.pdf"})


class IngestMetadataTests(unittest.TestCase):
    def test_mismatched_metadata_length_is_rejected_before_anything_is_added(self):
        pipe = RAGPipeline(BM25Retriever(), EchoLLM())
        with self.assertRaisesRegex(ValueError, "same length"):
            pipe.ingest(["alpha", "beta"], [{"source": "a"}])
        self.assertEqual(pipe.retrieve("alpha"), [])
        self.assertEqual(pipe.ingest(["alpha"]), ["doc-1"])

    def test_metadata_is_copied_per_document(self):
        pipe = RAGPipeline(BM25Retriever(), EchoLLM())
        shared = {"source": "notes"}
        pipe.ingest(["alpha one", "alpha two"], [shared, shared])
        shared["source"] = "changed"
        sources = pipe.query("alpha", k=2)["sources"]
        self.assertEqual([s["metadata"] for s in sources], [{"source": "notes"}] * 2)

    def test_documents_without_metadata_get_independent_dicts(self):
        pipe = RAGPipeline(BM25Retriever(), EchoLLM())
        pipe.ingest(["alpha one", "alpha two"])
        first, second = [r.document for r in pipe.retrieve("alpha", k=2)]
        first.metadata["tag"] = "x"
        self.assertEqual(second.metadata, {})


if __name__ == "__main__":
    unittest.main()
