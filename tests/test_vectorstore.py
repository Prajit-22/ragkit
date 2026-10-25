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

    def test_mismatched_dimensions_rejected_at_add_time(self):
        with self.assertRaisesRegex(ValueError, "dimension"):
            self.store.add(Document("x", "x"), [1.0, 0.0, 0.0])
        # The bad add must not partially replace an existing document either.
        with self.assertRaisesRegex(ValueError, "dimension"):
            self.store.add(Document("a", "alpha"), [1.0, 0.0, 0.0])
        self.assertEqual(self.store.get("a").text, "alpha")

    def test_replacing_the_only_vector_can_change_dimensions(self):
        store = InMemoryVectorStore()
        store.add(Document("a", "alpha"), [1.0, 0.0])
        store.add(Document("a", "alpha again"), [1.0, 0.0, 0.0])
        store.add(Document("b", "beta"), [0.0, 1.0, 0.0])
        with self.assertRaisesRegex(ValueError, "dimension"):
            store.add(Document("c", "gamma"), [1.0, 0.0])

    def test_delete_then_readd_can_change_dimensions(self):
        store = InMemoryVectorStore()
        store.add(Document("a", "alpha"), [1.0, 0.0])
        store.delete("a")
        store.add(Document("b", "beta"), [1.0, 0.0, 0.0])
        self.assertEqual(len(store.search([1.0, 0.0, 0.0])), 1)



class VectorStoreMMRTests(unittest.TestCase):
    def setUp(self):
        self.store = InMemoryVectorStore()
        self.store.add(Document("a", "alpha", {"kind": "near"}), [1.0, 0.0])
        self.store.add(Document("b", "beta", {"kind": "near"}), [1.0, 0.1])
        self.store.add(Document("c", "gamma", {"kind": "far"}), [0.0, 1.0])
        self.query = [1.0, 0.2]

    def test_plain_search_ranks_near_duplicates_together(self):
        results = self.store.search(self.query, k=3)
        self.assertEqual([r.document.id for r in results], ["b", "a", "c"])

    def test_mmr_diversifies_the_second_pick(self):
        results = self.store.search_mmr(self.query, k=2, lambda_mult=0.5)
        self.assertEqual([r.document.id for r in results], ["b", "c"])

    def test_lambda_one_reproduces_plain_search_order(self):
        plain = [r.document.id for r in self.store.search(self.query, k=3)]
        mmr = [r.document.id for r in self.store.search_mmr(self.query, k=3, lambda_mult=1.0)]
        self.assertEqual(plain, mmr)

    def test_mmr_respects_metadata_filter(self):
        results = self.store.search_mmr(self.query, k=2, lambda_mult=0.5,
                                        metadata_filter={"kind": "near"})
        self.assertEqual([r.document.id for r in results], ["b", "a"])

    def test_mmr_scores_are_query_similarities(self):
        results = self.store.search_mmr(self.query, k=2, lambda_mult=0.5)
        self.assertAlmostEqual(results[0].score, cosine_similarity(self.query, [1.0, 0.1]))
        self.assertAlmostEqual(results[1].score, cosine_similarity(self.query, [0.0, 1.0]))

    def test_mmr_rejects_bad_arguments(self):
        with self.assertRaisesRegex(ValueError, "k must be positive"):
            self.store.search_mmr(self.query, k=0)
        with self.assertRaisesRegex(ValueError, "lambda_mult"):
            self.store.search_mmr(self.query, k=2, lambda_mult=-0.1)
        with self.assertRaisesRegex(ValueError, "lambda_mult"):
            self.store.search_mmr(self.query, k=2, lambda_mult=1.1)
        with self.assertRaisesRegex(ValueError, "fetch_k"):
            self.store.search_mmr(self.query, k=3, fetch_k=2)

    def test_mmr_on_empty_store_returns_empty(self):
        self.assertEqual(InMemoryVectorStore().search_mmr(self.query, k=2), [])

if __name__ == "__main__":
    unittest.main()
