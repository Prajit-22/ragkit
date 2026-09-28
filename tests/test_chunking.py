import unittest

from ragkit import fixed_size_chunks, sentence_chunks


class FixedSizeChunkTests(unittest.TestCase):
    def test_splits_and_overlaps(self):
        text = "".join(chr(65 + i % 26) for i in range(1000))
        chunks = fixed_size_chunks(text, chunk_size=300, overlap=60)
        # step = 240 -> starts at 0, 240, 480, 720, 960
        self.assertEqual(len(chunks), 5)
        self.assertEqual(chunks[0], text[:300])
        # Overlap: each full-size chunk's start repeats the previous chunk's tail.
        for prev, nxt in zip(chunks, chunks[1:]):
            if len(nxt) >= 60:
                self.assertEqual(prev[-60:], nxt[:60])

    def test_short_text_returns_single_chunk(self):
        self.assertEqual(fixed_size_chunks("hello world", 500), ["hello world"])

    def test_empty_text_returns_nothing(self):
        self.assertEqual(fixed_size_chunks("", 100), [])

    def test_rejects_bad_parameters(self):
        with self.assertRaises(ValueError):
            fixed_size_chunks("x", chunk_size=0)
        with self.assertRaises(ValueError):
            fixed_size_chunks("x", chunk_size=100, overlap=100)


class SentenceChunkTests(unittest.TestCase):
    TEXT = ("The sky is blue. Water is wet. Python is a language. "
            "Tests are useful. Retrieval matters. Generation follows.")

    def test_respects_max_sentences(self):
        chunks = sentence_chunks(self.TEXT, max_sentences=2, overlap_sentences=0)
        self.assertTrue(all(chunk.count(".") <= 2 for chunk in chunks))
        self.assertEqual(len(chunks), 3)

    def test_overlap_carries_sentences(self):
        chunks = sentence_chunks(self.TEXT, max_sentences=3, overlap_sentences=1)
        first_tail = chunks[0].split(". ")[-1].rstrip(".")
        self.assertIn(first_tail, chunks[1])

    def test_never_splits_a_sentence(self):
        chunks = sentence_chunks(self.TEXT, max_sentences=2, overlap_sentences=0)
        for chunk in chunks:
            self.assertTrue(chunk.endswith("."))

    def test_empty_text(self):
        self.assertEqual(sentence_chunks("", 3), [])


if __name__ == "__main__":
    unittest.main()
