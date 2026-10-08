import unittest

from ragkit import BM25Retriever, HashingEmbedder
from ragkit.embeddings import tokenize
from ragkit.vectorstore import Document


class TokenizeTests(unittest.TestCase):
    def test_ascii_behavior_is_unchanged(self):
        self.assertEqual(tokenize("Don't stop: RAG-2 works!"), ["don't", "stop", "rag", "2", "works"])
        self.assertEqual(tokenize("snake_case"), ["snake", "case"])
        self.assertEqual(tokenize(""), [])

    def test_accented_words_stay_whole(self):
        self.assertEqual(tokenize("Caf\u00e9 na\u00efve r\u00e9sum\u00e9"), ["caf\u00e9", "na\u00efve", "r\u00e9sum\u00e9"])

    def test_composed_and_decomposed_forms_match(self):
        self.assertEqual(tokenize("cafe\u0301"), tokenize("caf\u00e9"))

    def test_devanagari_keeps_vowel_signs(self):
        self.assertEqual(tokenize("\u0928\u092e\u0938\u094d\u0924\u0947 \u0926\u0941\u0928\u093f\u092f\u093e"),
                         ["\u0928\u092e\u0938\u094d\u0924\u0947", "\u0926\u0941\u0928\u093f\u092f\u093e"])

    def test_unspaced_scripts_become_one_run(self):
        self.assertEqual(tokenize("\u65e5\u672c\u8a9e text"), ["\u65e5\u672c\u8a9e", "text"])

    def test_bm25_retrieves_non_english_documents(self):
        bm25 = BM25Retriever()
        bm25.add(Document(id="a", text="Le caf\u00e9 ouvre \u00e0 huit heures", metadata={}))
        bm25.add(Document(id="b", text="\u0928\u092e\u0938\u094d\u0924\u0947 \u0926\u0941\u0928\u093f\u092f\u093e", metadata={}))
        bm25.add(Document(id="c", text="unrelated english sentence", metadata={}))
        self.assertEqual([r.document.id for r in bm25.search("caf\u00e9")], ["a"])
        self.assertEqual([r.document.id for r in bm25.search("\u0928\u092e\u0938\u094d\u0924\u0947")], ["b"])

    def test_hashing_embedder_is_nonzero_for_non_english_text(self):
        vector = HashingEmbedder().embed("\u0928\u092e\u0938\u094d\u0924\u0947")
        self.assertGreater(sum(v * v for v in vector), 0.0)


if __name__ == "__main__":
    unittest.main()
