import json
import os
import tempfile
import unittest

from ragkit import (BM25Retriever, Document, HashingEmbedder, HybridRetriever,
                    VectorRetriever)

DOCS = [
    Document("bm25", "BM25 ranks documents using term frequency and inverse document frequency."),
    Document("vec", "Vector search embeds text into dense vectors compared with cosine similarity."),
    Document("rag", "Retrieval augmented generation fetches relevant chunks before the LLM answers."),
    Document("pie", "Apple pie needs apples, sugar, butter, and a hot oven."),
]


class BM25Tests(unittest.TestCase):
    def setUp(self):
        self.retriever = BM25Retriever()
        for doc in DOCS:
            self.retriever.add(doc)

    def test_relevant_document_wins(self):
        results = self.retriever.search("how does BM25 use term frequency", k=1)
        self.assertEqual(results[0].document.id, "bm25")

    def test_no_match_returns_empty(self):
        self.assertEqual(self.retriever.search("quantum entanglement", k=2), [])

    def test_metadata_filter(self):
        docs = [Document("x", "shared words here", {"lang": "en"}),
                Document("y", "shared words here", {"lang": "hi"})]
        r = BM25Retriever()
        for d in docs:
            r.add(d)
        results = r.search("shared words", k=2, metadata_filter={"lang": "hi"})
        self.assertEqual([res.document.id for res in results], ["y"])

    def test_add_replaces_same_id(self):
        self.retriever.add(Document("bm25", "completely new zebra content"))
        results = self.retriever.search("zebra", k=1)
        self.assertEqual(results[0].document.id, "bm25")
        self.assertEqual(len(self.retriever), len(DOCS))

    def test_delete(self):
        self.assertTrue(self.retriever.delete("pie"))
        self.assertEqual(self.retriever.search("apple pie oven", k=1), [])


class VectorRetrieverTests(unittest.TestCase):
    def test_semantic_order(self):
        retriever = VectorRetriever(HashingEmbedder())
        for doc in DOCS:
            retriever.add(doc)
        results = retriever.search("dense embedding cosine search", k=1)
        self.assertEqual(results[0].document.id, "vec")


class HybridTests(unittest.TestCase):
    def test_fusion_prefers_docs_in_both_rankings(self):
        hybrid = HybridRetriever(BM25Retriever(), VectorRetriever(HashingEmbedder()))
        for doc in DOCS:
            hybrid.add(doc)
        results = hybrid.search("retrieval augmented generation chunks", k=2)
        self.assertEqual(results[0].document.id, "rag")
        # RRF scores are positive and sorted descending.
        scores = [r.score for r in results]
        self.assertEqual(scores, sorted(scores, reverse=True))
        self.assertTrue(all(s > 0 for s in scores))

    def test_delete_removes_from_both(self):
        hybrid = HybridRetriever(BM25Retriever(), VectorRetriever(HashingEmbedder()))
        for doc in DOCS:
            hybrid.add(doc)
        self.assertTrue(hybrid.delete("rag"))
        ids = [r.document.id for r in hybrid.search("retrieval augmented generation", k=4)]
        self.assertNotIn("rag", ids)


if __name__ == "__main__":
    unittest.main()


class VectorRetrieverMMRTests(unittest.TestCase):
    class _TableEmbedder:
        dimensions = 2

        def __init__(self, table):
            self.table = table

        def embed(self, text):
            return self.table[text]

    def test_search_mmr_diversifies_through_the_embedder(self):
        table = {
            "query": [1.0, 0.2],
            "near one": [1.0, 0.1],
            "near two": [1.0, 0.0],
            "far": [0.0, 1.0],
        }
        retriever = VectorRetriever(self._TableEmbedder(table))
        retriever.add(Document("n1", "near one"))
        retriever.add(Document("n2", "near two"))
        retriever.add(Document("f", "far"))
        results = retriever.search_mmr("query", k=2, lambda_mult=0.5)
        self.assertEqual([r.document.id for r in results], ["n1", "f"])


class BM25PersistenceTests(unittest.TestCase):
    def setUp(self):
        self.retriever = BM25Retriever(k1=1.2, b=0.5)
        for doc in DOCS:
            self.retriever.add(doc)

    def test_round_trip_preserves_ranking_and_scores(self):
        query = "how does BM25 use term frequency"
        before = [(r.document.id, r.score) for r in self.retriever.search(query, k=3)]
        restored = BM25Retriever.from_dict(self.retriever.to_dict())
        after = [(r.document.id, r.score) for r in restored.search(query, k=3)]
        self.assertEqual([doc_id for doc_id, _ in before],
                         [doc_id for doc_id, _ in after])
        for (_, score_a), (_, score_b) in zip(before, after):
            self.assertAlmostEqual(score_a, score_b)
        self.assertEqual(len(restored), len(DOCS))

    def test_round_trip_preserves_tuning(self):
        restored = BM25Retriever.from_dict(self.retriever.to_dict())
        self.assertEqual((restored.k1, restored.b), (1.2, 0.5))
        default_tuned = BM25Retriever.from_dict(BM25Retriever().to_dict())
        self.assertEqual((default_tuned.k1, default_tuned.b), (1.5, 0.75))

    def test_metadata_and_text_survive(self):
        r = BM25Retriever()
        r.add(Document("x", "shared words here", {"lang": "en"}))
        r.add(Document("y", "shared words here", {"lang": "hi"}))
        restored = BM25Retriever.from_dict(r.to_dict())
        results = restored.search("shared words", k=2, metadata_filter={"lang": "hi"})
        self.assertEqual([res.document.id for res in results], ["y"])
        self.assertEqual(results[0].document.text, "shared words here")

    def test_loaded_index_stays_consistent_after_edits(self):
        restored = BM25Retriever.from_dict(self.retriever.to_dict())
        restored.add(Document("zebra", "zebra stripes pattern"))
        restored.delete("bm25")
        self.assertEqual(len(restored), len(DOCS))
        self.assertEqual(restored.search("zebra", k=1)[0].document.id, "zebra")
        self.assertEqual(restored.search("term frequency", k=3), [])

    def test_save_and_load_through_a_json_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "bm25.json")
            self.retriever.save(path)
            with open(path, encoding="utf-8") as handle:
                raw = json.load(handle)
            self.assertEqual(raw["version"], 1)
            restored = BM25Retriever.load(path)
        self.assertEqual(len(restored), len(DOCS))
        self.assertEqual(restored.search("apple pie", k=1)[0].document.id, "pie")

    def test_load_missing_file_raises_oserror(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(OSError):
                BM25Retriever.load(os.path.join(tmp, "absent.json"))

    def test_malformed_snapshots_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "version 1"):
            BM25Retriever.from_dict({"version": 2, "documents": []})
        with self.assertRaisesRegex(ValueError, "version 1"):
            BM25Retriever.from_dict(["not", "a", "dict"])
        with self.assertRaisesRegex(ValueError, "documents"):
            BM25Retriever.from_dict({"version": 1, "k1": 1.5, "b": 0.75})
        with self.assertRaisesRegex(ValueError, "tuning"):
            BM25Retriever.from_dict({"version": 1, "documents": []})
        with self.assertRaisesRegex(ValueError, "malformed"):
            BM25Retriever.from_dict(
                {"version": 1, "k1": 1.5, "b": 0.75, "documents": [{"id": "a"}]})
