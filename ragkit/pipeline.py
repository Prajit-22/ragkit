"""The retrieval-augmented generation pipeline."""

from typing import Dict, List, Optional

from .llm import LLM
from .vectorstore import Document, SearchResult

PROMPT_TEMPLATE = """Answer the question using only the context below. If the context does not contain the answer, say so.

Context:
{context}

Question: {question}
Answer:"""


class RAGPipeline:
    """Wire a retriever and an LLM together.

    ``ingest`` adds documents (already chunked, or raw text to be chunked by
    the caller's chosen chunker). ``query`` retrieves the top-k chunks and
    asks the LLM to answer from them. The answer always comes back with its
    sources so callers can show citations.
    """

    def __init__(self, retriever, llm: LLM, prompt_template: str = PROMPT_TEMPLATE):
        self.retriever = retriever
        self.llm = llm
        self.prompt_template = prompt_template
        self._counter = 0

    def ingest(self, texts: List[str], metadatas: Optional[List[Dict]] = None) -> List[str]:
        if metadatas is not None and len(metadatas) != len(texts):
            raise ValueError("metadatas must have the same length as texts")
        ids = []
        for i, text in enumerate(texts):
            if not text or not text.strip():
                continue
            self._counter += 1
            doc_id = f"doc-{self._counter}"
            metadata = dict(metadatas[i]) if metadatas is not None else {}
            self.retriever.add(Document(id=doc_id, text=text, metadata=metadata))
            ids.append(doc_id)
        return ids

    def retrieve(self, question: str, k: int = 4,
                 metadata_filter: Optional[Dict] = None) -> List[SearchResult]:
        return self.retriever.search(question, k=k, metadata_filter=metadata_filter)

    def build_prompt(self, question: str, sources: List[SearchResult]) -> str:
        context = "\n".join(f"- {result.document.text}" for result in sources)
        return self.prompt_template.format(context=context or "(none)", question=question)

    def query(self, question: str, k: int = 4,
              metadata_filter: Optional[Dict] = None) -> Dict:
        sources = self.retrieve(question, k=k, metadata_filter=metadata_filter)
        prompt = self.build_prompt(question, sources)
        answer = self.llm.generate(prompt)
        return {
            "answer": answer,
            "sources": [
                {
                    "id": result.document.id,
                    "text": result.document.text,
                    "score": round(result.score, 4),
                    "metadata": result.document.metadata,
                }
                for result in sources
            ],
            "prompt": prompt,
        }
