import unittest

from ragkit import fixed_size_chunks, sentence_chunks


class FixedSizeChunkTests(unittest.TestCase):
    def test_splits_and_overlaps(self):
        text = "".join(chr(65 + i % 26) for i in range(1000))
        chunks = fixed_size_chunks(text, chunk_size=300, overlap=60)
        # step = 240 -> starts at 0, 240, 480, 720; the window at 720 reaches the
        # end, so a start-960 chunk would only repeat its tail and is not emitted.
        self.assertEqual(len(chunks), 4)
        self.assertEqual(chunks[0], text[:300])
        self.assertEqual(chunks[-1], text[720:])
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


class NoRepeatTailTests(unittest.TestCase):
    def test_fixed_size_has_no_chunk_contained_in_the_previous_one(self):
        chunks = fixed_size_chunks("a" * 100, chunk_size=60, overlap=20)
        self.assertEqual([len(c) for c in chunks], [60, 60])

    def test_fixed_size_still_covers_a_short_final_window(self):
        chunks = fixed_size_chunks("a" * 101, chunk_size=60, overlap=20)
        self.assertEqual([len(c) for c in chunks], [60, 60, 21])

    def test_fixed_size_exact_fit_is_one_chunk(self):
        self.assertEqual(len(fixed_size_chunks("a" * 60, chunk_size=60, overlap=20)), 1)

    def test_sentence_chunks_have_no_repeat_only_tail(self):
        text = "A one. B two. C three. D four."
        self.assertEqual(sentence_chunks(text, max_sentences=4, overlap_sentences=1), [text])

    def test_sentence_chunks_keep_overlap_between_real_windows(self):
        text = "A one. B two. C three. D four. E five. F six."
        chunks = sentence_chunks(text, max_sentences=4, overlap_sentences=1)
        self.assertEqual(chunks, ["A one. B two. C three. D four.", "D four. E five. F six."])


if __name__ == "__main__":
    unittest.main()
