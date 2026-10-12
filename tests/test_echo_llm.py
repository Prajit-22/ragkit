import unittest

from ragkit import BM25Retriever, EchoLLM, RAGPipeline


def answer(text, question):
    pipe = RAGPipeline(BM25Retriever(), EchoLLM())
    pipe.ingest([text])
    return pipe.query(question)["answer"]


class EchoLLMTests(unittest.TestCase):
    def test_no_doubled_period_when_context_ends_with_one(self):
        self.assertEqual(answer("The server restarts at 3 AM.", "server"),
                         "Based on the provided context: The server restarts at 3 AM.")

    def test_adds_period_when_context_has_none(self):
        self.assertEqual(answer("The server restarts at 3 AM", "server"),
                         "Based on the provided context: The server restarts at 3 AM.")

    def test_keeps_question_mark_and_exclamation(self):
        self.assertEqual(answer("Is it done?", "done"),
                         "Based on the provided context: Is it done?")

    def test_leading_minus_sign_is_not_stripped(self):
        self.assertEqual(answer("-40 is where Celsius equals Fahrenheit. Cool fact.", "celsius"),
                         "Based on the provided context: -40 is where Celsius equals Fahrenheit.")

    def test_retrieved_text_containing_question_label_is_kept(self):
        self.assertEqual(answer("Question: what is alpha? Answer: alpha is first.", "alpha"),
                         "Based on the provided context: Question: what is alpha? Answer: alpha is first.")

    def test_retrieved_text_with_question_line_is_kept(self):
        text = "Interview notes about alpha\nQuestion: how does alpha scale"
        self.assertEqual(answer(text, "alpha"),
                         "Based on the provided context: Interview notes about alpha Question: how does alpha scale.")

    def test_no_context_still_declines(self):
        pipe = RAGPipeline(BM25Retriever(), EchoLLM())
        self.assertEqual(pipe.query("anything")["answer"],
                         "I don't have enough context to answer that.")


if __name__ == "__main__":
    unittest.main()
