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
