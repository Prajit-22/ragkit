"""Regression coverage for caller-owned stores and hybrid search boundaries."""
import unittest
from ragkit import (BM25Retriever, Document, HashingEmbedder, HybridRetriever,
                    InMemoryVectorStore, VectorRetriever)


class StoreOwnershipTests(unittest.TestCase):
    def test_empty_store_is_preserved_and_receives_documents(self):
        store = InMemoryVectorStore()
        retriever = VectorRetriever(HashingEmbedder(), store=store)
        self.assertIs(retriever.store, store)
        document = Document("one", "BM25 lexical retrieval")
        retriever.add(document)
        self.assertEqual(len(store), 1)
        self.assertEqual(store.get("one"), document)
        self.assertTrue(retriever.delete("one"))
        self.assertEqual(len(store), 0)

    def test_populated_store_is_preserved(self):
        store = InMemoryVectorStore()
        embedder = HashingEmbedder()
        doc = Document("one", "BM25 lexical retrieval")
        store.add(doc, embedder.embed(doc.text))
        retriever = VectorRetriever(embedder, store=store)
        self.assertIs(retriever.store, store)
        self.assertEqual(retriever.search("BM25", k=1)[0].document.id, "one")

    def test_default_stores_are_independent(self):
        first = VectorRetriever(HashingEmbedder())
        second = VectorRetriever(HashingEmbedder())
        first.add(Document("one", "BM25"))
        self.assertEqual(len(second), 0)
        self.assertIsNot(first.store, second.store)


class HybridBoundaryTests(unittest.TestCase):
    def make_hybrid(self, rrf_k=60):
        return HybridRetriever(BM25Retriever(), VectorRetriever(HashingEmbedder()),
                               rrf_k=rrf_k)

    def test_nonpositive_k_rejected_for_empty_and_populated_corpora(self):
        hybrid = self.make_hybrid()
        for populated in (False, True):
            if populated:
                hybrid.add(Document("one", "BM25"))
            for k in (0, -1):
                with self.subTest(populated=populated, k=k):
                    with self.assertRaisesRegex(ValueError, "k must be positive"):
                        hybrid.search("BM25", k=k)

    def test_negative_rrf_constant_rejected(self):
        with self.assertRaisesRegex(ValueError, "rrf_k must be nonnegative"):
            self.make_hybrid(rrf_k=-1)

    def test_zero_rrf_constant_is_valid(self):
        hybrid = self.make_hybrid(rrf_k=0)
        hybrid.add(Document("one", "BM25"))
        self.assertEqual(hybrid.search("BM25", k=1)[0].score, 2.0)

    def test_metadata_filter_applies_to_both_rankings(self):
        hybrid = self.make_hybrid()
        hybrid.add(Document("en", "shared text", {"lang": "en"}))
        hybrid.add(Document("hi", "shared text", {"lang": "hi"}))
        results = hybrid.search("shared", metadata_filter={"lang": "hi"})
        self.assertEqual([result.document.id for result in results], ["hi"])

    def test_empty_corpus_returns_empty(self):
        self.assertEqual(self.make_hybrid().search("BM25"), [])


if __name__ == "__main__":
    unittest.main()
