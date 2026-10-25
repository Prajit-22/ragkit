"""In-memory vector store with cosine similarity search."""

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence


@dataclass
class Document:
    id: str
    text: str
    metadata: Dict = field(default_factory=dict)


@dataclass
class SearchResult:
    document: Document
    score: float


def cosine_similarity(a: Sequence[float], b: Sequence[float]) -> float:
    if len(a) != len(b):
        raise ValueError("vectors must have the same dimensions")
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(y * y for y in b))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


class InMemoryVectorStore:
    """Minimal vector store: add, delete, and top-k cosine search."""

    def __init__(self):
        self._vectors: Dict[str, List[float]] = {}
        self._documents: Dict[str, Document] = {}

    def __len__(self) -> int:
        return len(self._documents)

    def add(self, document: Document, vector: Sequence[float]) -> None:
        if not vector:
            raise ValueError("vector must not be empty")
        # All vectors in the store must share one dimension. Check at insert
        # time so a bad vector fails here, not mid-search inside cosine math.
        others = [v for doc_id, v in self._vectors.items() if doc_id != document.id]
        if others and len(vector) != len(others[0]):
            raise ValueError(
                f"vector dimension {len(vector)} does not match the store's existing dimension {len(others[0])}"
            )
        self._documents[document.id] = document
        self._vectors[document.id] = list(vector)

    def delete(self, document_id: str) -> bool:
        existed = document_id in self._documents
        self._documents.pop(document_id, None)
        self._vectors.pop(document_id, None)
        return existed

    def get(self, document_id: str) -> Optional[Document]:
        return self._documents.get(document_id)

    def search(self, query_vector: Sequence[float], k: int = 4,
               metadata_filter: Optional[Dict] = None) -> List[SearchResult]:
        if k <= 0:
            raise ValueError("k must be positive")
        scored = []
        for doc_id, vector in self._vectors.items():
            document = self._documents[doc_id]
            if metadata_filter and not all(
                document.metadata.get(key) == value for key, value in metadata_filter.items()
            ):
                continue
            scored.append(SearchResult(document, cosine_similarity(query_vector, vector)))
        scored.sort(key=lambda r: r.score, reverse=True)
        return scored[:k]

    def search_mmr(self, query_vector: Sequence[float], k: int = 4,
                   fetch_k: Optional[int] = None, lambda_mult: float = 0.5,
                   metadata_filter: Optional[Dict] = None) -> List[SearchResult]:
        """Maximal marginal relevance: trade some relevance for diversity.

        Candidates come from a plain top-``fetch_k`` search; each pick then
        maximizes ``lambda_mult * relevance - (1 - lambda_mult) * redundancy``,
        where redundancy is the candidate's highest similarity to anything
        already picked. ``lambda_mult=1`` reproduces plain ``search`` ordering;
        lower values push near-duplicate chunks out of the results. Returned
        scores stay cosine similarity to the query - lambda only changes the
        selection, not what a score means.
        """
        if k <= 0:
            raise ValueError("k must be positive")
        if not 0.0 <= lambda_mult <= 1.0:
            raise ValueError("lambda_mult must be between 0 and 1")
        if fetch_k is None:
            fetch_k = max(k * 4, 20)
        if fetch_k < k:
            raise ValueError("fetch_k must be >= k")
        pool = [
            (result.document, self._vectors[result.document.id], result.score)
            for result in self.search(query_vector, k=fetch_k, metadata_filter=metadata_filter)
        ]
        selected: List[SearchResult] = []
        selected_vectors: List[List[float]] = []
        while pool and len(selected) < k:
            best_index, best_value = 0, None
            for i, (document, vector, relevance) in enumerate(pool):
                redundancy = max(
                    (cosine_similarity(vector, chosen) for chosen in selected_vectors),
                    default=0.0,
                )
                value = lambda_mult * relevance - (1 - lambda_mult) * redundancy
                if best_value is None or value > best_value:
                    best_index, best_value = i, value
            document, vector, relevance = pool.pop(best_index)
            selected.append(SearchResult(document, relevance))
            selected_vectors.append(vector)
        return selected
