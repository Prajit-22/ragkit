# Retrieval-augmented generation

RAG combines retrieval with generation. Instead of answering from memory, the
system first retrieves relevant chunks from a corpus and then asks the LLM to
answer using only those chunks. This grounds answers in real documents, cuts
hallucination, and gives every answer citable sources. Hybrid pipelines fuse
lexical and vector retrieval to get the strengths of both.
