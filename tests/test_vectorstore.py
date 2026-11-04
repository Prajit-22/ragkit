import os
import tempfile
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

class VectorStorePersistenceTests(unittest.TestCase):
    def setUp(self):
        self.store = InMemoryVectorStore()
        self.store.add(Document("a", "deployment rollback steps", {"source": "runbook.md"}),
                       [1.0, 0.0, 0.5])
        self.store.add(Document("b", "composting kitchen scraps", {"source": "garden.md"}),
                       [0.0, 1.0, 0.0])

    def test_round_trip_preserves_search_results_exactly(self):
        query = [0.9, 0.1, 0.4]
        before = [(r.document.id, r.score) for r in self.store.search(query, k=2)]
        restored = InMemoryVectorStore.from_dict(self.store.to_dict())
        after = [(r.document.id, r.score) for r in restored.search(query, k=2)]
        self.assertEqual(before, after)
        self.assertEqual(len(restored), 2)

    def test_round_trip_preserves_documents_and_metadata(self):
        restored = InMemoryVectorStore.from_dict(self.store.to_dict())
        doc = restored.get("a")
        self.assertEqual(doc.text, "deployment rollback steps")
        self.assertEqual(doc.metadata, {"source": "runbook.md"})
        filtered = restored.search([1.0, 0.0, 0.5], k=2,
                                   metadata_filter={"source": "runbook.md"})
        self.assertEqual([r.document.id for r in filtered], ["a"])

    def test_save_and_load_through_a_json_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "store.json")
            self.store.save(path)
            restored = InMemoryVectorStore.load(path)
        self.assertEqual(len(restored), 2)
        self.assertEqual(restored.get("b").text, "composting kitchen scraps")

    def test_empty_store_round_trips_and_stays_usable(self):
        empty = InMemoryVectorStore.from_dict(InMemoryVectorStore().to_dict())
        self.assertEqual(len(empty), 0)
        empty.add(Document("x", "fresh start"), [0.5, 0.5])
        self.assertEqual(len(empty), 1)

    def test_wrong_version_and_missing_documents_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "version 1"):
            InMemoryVectorStore.from_dict({"version": 2, "documents": []})
        with self.assertRaisesRegex(ValueError, "documents"):
            InMemoryVectorStore.from_dict({"version": 1})
        with self.assertRaisesRegex(ValueError, "version 1"):
            InMemoryVectorStore.from_dict(["not", "a", "dict"])

    def test_malformed_entries_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "malformed"):
            InMemoryVectorStore.from_dict(
                {"version": 1, "documents": [{"id": "a"}]})
        # A dimension mismatch inside the snapshot fails at add time.
        with self.assertRaisesRegex(ValueError, "dimension"):
            InMemoryVectorStore.from_dict({
                "version": 1,
                "documents": [
                    {"id": "a", "text": "one", "metadata": {}, "vector": [1.0, 0.0]},
                    {"id": "b", "text": "two", "metadata": {}, "vector": [1.0]},
                ],
            })
