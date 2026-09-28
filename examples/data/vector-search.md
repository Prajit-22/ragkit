# Vector search

Vector search embeds text into dense vectors and compares them with cosine
similarity. Unlike keyword search, it captures semantic similarity: a query
about "fixing a flat tire" can match a document about "repairing a puncture"
even when no words overlap. The trade-off is that it needs an embedding model,
which can mean downloads, cost, or an external API.
