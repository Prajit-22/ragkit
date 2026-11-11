# Plugging in a real embedder or LLM

The bundled `HashingEmbedder`, `TfIdfEmbedder` and `EchoLLM` exist so the
pipeline runs offline and tests stay deterministic. Production quality comes
from swapping in a real embedder and/or LLM - both are one-method protocols.

## A real LLM

Any object with `generate(prompt: str) -> str` works. `CallableLLM` wraps a
plain function:

```python
import os, sys
sys.path.insert(0, "scripts")
from ragkit import BM25Retriever, CallableLLM, RAGPipeline

def groq_generate(prompt):
    import json, urllib.request
    req = urllib.request.Request(
        "https://api.groq.com/openai/v1/chat/completions",
        data=json.dumps({
            "model": "openai/gpt-oss-120b",
            "messages": [{"role": "user", "content": prompt}],
        }).encode(),
        headers={"Authorization": "Bearer " + os.environ["GROQ_API_KEY"],
                 "Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req) as resp:
        return json.load(resp)["choices"][0]["message"]["content"]

pipe = RAGPipeline(BM25Retriever.load("index.json"), CallableLLM(groq_generate))
print(pipe.query("What does the runbook say about rollbacks?")["answer"])
```

(The model name above is an example - check the provider's current docs for
available models and terms.) If you also packaged the free-llm-failover
skill, its CLI can serve as the LLM step: retrieve here, pass the prompt
there, and you get citations plus provider failover.

## A real embedder

Any object with a `dimensions` int and `embed(text) -> list[float]`:

```python
from ragkit import Document, VectorRetriever

class MyEmbedder:
    dimensions = 1536
    def embed(self, text):
        return my_embedding_api(text)  # any provider

retriever = VectorRetriever(MyEmbedder())
retriever.add(Document(id="1", text="..."))
```

Vectors persist with the store (`store.save(path)`), and a loaded store
searches without the embedder - you only need it again to embed new
documents or new queries. For query-time search after a reload, the embedder
must be the same one that produced the stored vectors.

## Notes

- Keep chunk metadata small and flat (`source`, `kind`, `date`): metadata
  filters are exact matches.
- The pipeline returns `answer`, `sources` (id, text, score, metadata) and
  the filled `prompt` - keep the sources in whatever you present.
