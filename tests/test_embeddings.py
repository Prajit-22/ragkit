import unittest

from ragkit import HashingEmbedder, TfIdfEmbedder


class HashingEmbedderTests(unittest.TestCase):
    def test_deterministic_and_normalized(self):
        emb = HashingEmbedder(dimensions=128)
        v1 = emb.embed("retrieval augmented generation")
        v2 = emb.embed("retrieval augmented generation")
        self.assertEqual(v1, v2)
        self.assertEqual(len(v1), 128)
        self.assertAlmostEqual(sum(x * x for x in v1) ** 0.5, 1.0, places=6)

    def test_similar_texts_score_above_unrelated(self):
        from ragkit import cosine_similarity
        emb = HashingEmbedder()
        a = emb.embed("vector databases store embeddings")
        b = emb.embed("vector stores keep embeddings")
        c = emb.embed("banana bread recipe oven temperature")
        self.assertGreater(cosine_similarity(a, b), cosine_similarity(a, c))

    def test_empty_text_is_zero_vector(self):
        emb = HashingEmbedder()
        self.assertEqual(emb.embed(""), [0.0] * emb.dimensions)


class TfIdfEmbedderTests(unittest.TestCase):
    DOCS = [
        "retrieval augmented generation combines search and LLMs",
        "BM25 ranks documents by term frequency",
        "cosine similarity compares embedding vectors",
    ]

    def test_fit_then_embed_shape(self):
        emb = TfIdfEmbedder().fit(self.DOCS)
        v = emb.embed("retrieval and generation")
        self.assertEqual(len(v), emb.dimensions)
        self.assertGreater(sum(v), 0)

    def test_embed_before_fit_raises(self):
        with self.assertRaises(RuntimeError):
            TfIdfEmbedder().embed("anything")

    def test_rare_terms_get_more_weight(self):
        docs = ["common rare", "common", "common"]
        emb = TfIdfEmbedder().fit(docs)
        vector = emb.embed("common rare")
        self.assertGreater(
            vector[emb.vocabulary["rare"]],
            vector[emb.vocabulary["common"]],
        )


if __name__ == "__main__":
    unittest.main()
