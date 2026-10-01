# Local models with Ollama

[Docs](../README.md) · [Installation](installation.md) · [Quickstart](quickstart.md)

The engine is the same as the [quickstart](quickstart.md). The embedder and generator talk to Ollama on localhost instead of OpenAI.

Pull an embedding model and a chat model first. `nomic-embed-text` matches the embedder's default dimension of 768. If you use another embedding model, set `embedder.WithOllamaDimension` to that model's size. Qdrant also needs this number when you create the collection. The in-memory vector store does not.

```go
emb := embedder.NewOllamaEmbedder("nomic-embed-text")
gen := generator.NewOllamaGenerator("llama3.2", "")

chunks, err := chunker.NewCharacterChunker(800, 100, nil)
if err != nil {
    log.Fatal(err)
}

engine, err := ragout.NewEngine(
    ragout.WithReader(reader.NewByExtension(10<<20)),
    ragout.WithChunker(chunks),
    ragout.WithEmbedder(emb),
    ragout.WithVectorStore(store.NewVectorStore()),
    ragout.WithIndexStore(store.NewBM25Index()),
    ragout.WithGenerator(gen),
)
```

An empty endpoint uses the local defaults:

| Client | Default URL |
| --- | --- |
| `NewOllamaEmbedder` | `http://localhost:11434/api/embed` |
| `NewOllamaGenerator` | `http://localhost:11434/api/generate` |

Override them with `embedder.WithOllamaEndpoint` and by passing a non-empty endpoint to `NewOllamaGenerator`.

`Ingest` and `Query` are unchanged. The generator still builds a prompt with numbered chunks and asks the model to cite `[1]`, `[2]`. The Ollama generator does not add an extra system preamble; the OpenAI generator prepends "You are an accurate RAG assistant."

Both clients stream. `Query` still waits for the full answer. Use `QueryStream` or `QueryIter` if you want tokens as they arrive. See [Streaming](../concepts/streaming.md).

## Next

- [Embedder options](../components/embedders.md)
- [Generator options and the token budget](../components/generators.md)
