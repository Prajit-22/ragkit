"""ragkit: a small, dependency-free retrieval-augmented generation toolkit."""

from .chunking import fixed_size_chunks, sentence_chunks
from .embeddings import Embedder, HashingEmbedder, TfIdfEmbedder, tokenize
from .llm import CallableLLM, EchoLLM, LLM
from .pipeline import RAGPipeline
from .retrieval import BM25Retriever, HybridRetriever, VectorRetriever
from .vectorstore import Document, InMemoryVectorStore, SearchResult, cosine_similarity

__version__ = "0.1.0"

__all__ = [
    "BM25Retriever",
    "CallableLLM",
    "Document",
    "EchoLLM",
    "Embedder",
    "HashingEmbedder",
    "HybridRetriever",
    "InMemoryVectorStore",
    "LLM",
    "RAGPipeline",
    "SearchResult",
    "TfIdfEmbedder",
    "VectorRetriever",
    "cosine_similarity",
    "fixed_size_chunks",
    "sentence_chunks",
    "tokenize",
    "__version__",
]
