import unittest

from ragkit import Document, InMemoryVectorStore, cosine_similarity


class CosineSimilarityTests(unittest.TestCase):
    def test_identical_vectors_score_one(self):
        self.assertAlmostEqual(cosine_similarity([1, 2, 3], [1, 2, 3]), 1.0)

    def test_orthogonal_vectors_score_zero(self):
        self.assertAlmostEqual(cosine_similarity([1, 0], [0, 1]), 0.0)

    def test_zero_vector_is_safe(self):
        self.assertEqual(cosine_similarity([0, 0], [1, 1]), 0.0)

    def test_mismatched_dimensions_raise(self):
        with self.assertRaises(ValueError):
            cosine_similarity([1], [1, 2])


class VectorStoreTests(unittest.TestCase):
    def setUp(self):
        self.store = InMemoryVectorStore()
        self.store.add(Document("a", "alpha", {"kind": "greek"}), [1.0, 0.0])
        self.store.add(Document("b", "beta", {"kind": "greek"}), [0.9, 0.1])
        self.store.add(Document("c", "gamma", {"kind": "other"}), [0.0, 1.0])

    def test_search_returns_closest_first(self):
        results = self.store.search([1.0, 0.0], k=2)
        self.assertEqual([r.document.id for r in results], ["a", "b"])
        self.assertGreaterEqual(results[0].score, results[1].score)

    def test_metadata_filter(self):
        results = self.store.search([1.0, 0.0], k=3, metadata_filter={"kind": "other"})
        self.assertEqual([r.document.id for r in results], ["c"])

    def test_delete(self):
        self.assertTrue(self.store.delete("a"))
        self.assertFalse(self.store.delete("a"))
        self.assertEqual(len(self.store), 2)

    def test_rejects_empty_vector(self):
        with self.assertRaises(ValueError):
            self.store.add(Document("x", "x"), [])


if __name__ == "__main__":
    unittest.main()
