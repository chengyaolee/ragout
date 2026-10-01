# Installation

[Docs](../README.md) · [Quickstart](quickstart.md) · [Local models](local.md)

Add Ragout to a Go module:

```bash
go get github.com/chengyaolee/ragout
```

The module path is `github.com/chengyaolee/ragout`. The `go` version in `go.mod` is the minimum the library is built with.

## What you wire up

`NewEngine` needs at least one store.

| Goal | What to pass |
| --- | --- |
| Hybrid search | `WithVectorStore`, `WithEmbedder`, and `WithIndexStore` |
| Vectors only | `WithVectorStore` and `WithEmbedder` |
| Keywords only | `WithIndexStore` |
| Answers | `WithGenerator` |
| Indexing files | `WithReader` and `WithChunker` |

`Ingest` returns an error if the reader or chunker is missing. `Query` returns an error if the generator is missing. A vector store without an embedder fails at `NewEngine` (`ErrNilEmbedder`). No store at all fails with `ErrNoStores`.

The in-memory vector store and BM25 index need no extra process. OpenAI and Cohere need API keys. Ollama needs a local server. Chroma and Qdrant need their own servers; Ragout calls them with `net/http` and does not import their SDKs.

## Try it before you wire a model

From a clone of this repo, with no API key:

```bash
go test ./...
go run ./examples/demo
```

The demo indexes three notes in memory and prints the chunk hybrid search cited. The [quickstart](quickstart.md) is the same engine pointed at OpenAI. [Ollama](local.md) is the local-model version.

## Next

Build the [quickstart](quickstart.md), or run the same pipeline on [Ollama](local.md).
