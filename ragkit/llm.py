"""LLM interface for the generation step of a RAG pipeline.

ragkit deliberately ships no vendor SDK. Implement the ``LLM`` protocol for
your provider (OpenAI, Anthropic, Ollama, a local model, ...); ``EchoLLM``
keeps tests and examples fully offline.
"""

from typing import List, Protocol


class LLM(Protocol):
    """Anything that completes a prompt with text."""

    def generate(self, prompt: str) -> str: ...


class EchoLLM(LLM):
    """Offline stand-in that answers by quoting its retrieved context.

    Useful for pipeline tests and demos: the "answer" is extractive, taken
    straight from the context block, so it can never hallucinate.
    """

    def generate(self, prompt: str) -> str:
        context = _extract_context(prompt)
        if not context or context == "(none)":
            return "I don't have enough context to answer that."
        first_sentence = context.split(". ")[0].strip()
        if not first_sentence.endswith((".", "!", "?")):
            first_sentence += "."
        return f"Based on the provided context: {first_sentence}"


def _extract_context(prompt: str) -> str:
    """Return the retrieved context block of a pipeline prompt as one string.

    The block starts after the first ``Context:`` and ends at the last
    ``Question:`` line, so retrieved text that itself contains the word
    "Question:" is not cut short. Only the literal ``- `` bullet prefix is
    removed from each line; leading signs and dashes in the content stay.
    """
    marker = "Context:"
    if marker not in prompt:
        return ""
    after = prompt.split(marker, 1)[1]
    if "\nQuestion:" in after:
        after = after.rsplit("\nQuestion:", 1)[0]
    lines: List[str] = []
    for line in after.strip().splitlines():
        line = line.strip()
        if line.startswith("- "):
            line = line[2:].strip()
        if line:
            lines.append(line)
    return " ".join(lines)


class CallableLLM(LLM):
    """Wrap any callable(prompt) -> str as an LLM."""

    def __init__(self, fn):
        self._fn = fn

    def generate(self, prompt: str) -> str:
        return self._fn(prompt)
