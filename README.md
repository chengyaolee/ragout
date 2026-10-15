# Ragout

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Go Reference](https://pkg.go.dev/badge/github.com/chengyaolee/ragout.svg)](https://pkg.go.dev/github.com/chengyaolee/ragout)
[![test](https://github.com/chengyaolee/ragout/actions/workflows/test.yml/badge.svg)](https://github.com/chengyaolee/ragout/actions/workflows/test.yml)

<p align="center">
  <img src="ragout.png" alt="ragout: Go build your RAG pipeline end-to-end" width="720" />
</p>

## Why Ragout

A Go service that "does RAG" usually embeds a file, asks one vector database for neighbors, and pastes those chunks into a prompt. Paraphrases survive that. Exact terms often do not: an error code, a SKU, a function name. The prompt then grows until the model loses the passage in the middle, and the answer comes back with nothing to point at.

Ragout is that pipeline, already assembled, for code that has to stay in Go.

- **Keyword and vector search run together.** BM25 and a vector index fan out on every question. Reciprocal rank fusion merges the two rankings, so a paraphrase and an exact token can both surface. Leave either store out and the query still runs.
- **The answer names its sources.** Chunks compete for a token budget. The ones that fit are numbered `[1]`, `[2]`, and the strongest sit at both ends of the prompt, where models actually read them. `Answer.Sources[i]` is citation `[i+1]`.
- **Nothing else has to be running.** The vector index and the BM25 index live in the process. Snapshot them to a file, or point the same interface at Chroma or Qdrant when the data outgrows memory.
- **Chunks keep their context.** Markdown, HTML, and DOCX are split at headings, and every chunk is indexed with its title and section, so "Revenue grew 24%" still matches "Acme 2024 results". LLM-written context is one option away.
- **Questions can be rewritten, safely.** Follow-ups are condensed with the conversation, and rephrasings, sub-questions, or a HyDE passage can widen recall. A failed or slow rewrite falls back to the original question.
- **Each stage is one interface.** Swap the reader, the chunker, the contextualizer, the query transformer, the embedder, the store, the reranker, the expander, or the generator without touching the engine. OpenAI and Ollama clients ship in the module. A type you write yourself is the same shape.
- **You can measure it.** `Retrieve` runs search without generation, and the `eval` package scores hit rate, recall, and MRR on your own questions.

| Project | What it is |
| --- | --- |
| **Ragout** | The pipeline: ingest, hybrid search, optional rerank, cited generation, streaming. |
| [LangChainGo](https://github.com/tmc/langchaingo), [LinGoose](https://github.com/henomis/lingoose), [GoLC](https://github.com/hupe1980/golc) | LLM toolkits. You assemble retrieval from providers and vector-store adapters. |
| [chromem-go](https://github.com/philippgille/chromem-go) | An embeddable vector index. |
| Weaviate, Qdrant, Chroma | Databases. Ragout talks to Qdrant and Chroma over HTTP, and also ships an in-memory vector index plus BM25 so a service can run with no extra process. |

## Try it

No key, no database, no model server. From a clone of this repo:

```bash
go test ./...
go run ./examples/demo
```

`go test` is the full suite. [`examples/demo`](examples/demo/main.go) indexes three short notes in memory, runs hybrid search, and prints the cited chunk. The embedder is a local stand-in. Swap it for `embedder.NewOpenAIEmbedder` or `embedder.NewOllamaEmbedder` when you want real vectors.

## Overview

```bash
go get github.com/chengyaolee/ragout
```

You wire a reader, a chunker, an embedder, at least one store, and a generator. `Ingest` indexes a document. `Query` searches, fuses, and returns an answer plus the chunks it cited.

```go
chunks, err := chunker.NewCharacterChunker(800, 100, nil)
engine, err := ragout.NewEngine(
    ragout.WithReader(reader.NewByExtension(10<<20)),
    ragout.WithChunker(chunks),
    ragout.WithEmbedder(embedder.NewOpenAIEmbedder(os.Getenv("OPENAI_API_KEY"))),
    ragout.WithVectorStore(store.NewVectorStore()),
    ragout.WithIndexStore(store.NewBM25Index()),
    ragout.WithGenerator(generator.NewOpenAIGenerator(os.Getenv("OPENAI_API_KEY"))),
)

f, err := os.Open("notes.md")
defer f.Close()
err = engine.Ingest(ctx, "notes.md", f, nil)

answer, err := engine.Query(ctx, "What is this document about?")
fmt.Println(answer.Text)
for i, src := range answer.Sources {
    source, _ := src.Chunk.Metadata[ragout.MetadataSource].(string)
    fmt.Printf("[%d] %s\n", i+1, source)
}
```

The runnable copy is [`examples/quickstart`](examples/quickstart/main.go). It reads `OPENAI_API_KEY` and a file path. Ollama setup is [local models](docs/getting-started/local.md).

Readers cover `.txt`, `.md`, `.html`, `.pdf`, and `.docx`, with a byte cap. Chunk IDs stay stable across retries. If one store write fails, both stores drop that document. `WithReplaceSource` makes re-ingesting an edited file replace its old chunks, and `DeleteSource` removes a file. `Query`, `QueryStream`, and `QueryIter` share one retrieval path, and sources are filled before the first token. OpenTelemetry spans and Prometheus histograms record each stage.

## Architecture

```mermaid
flowchart TD
  subgraph ingest [Ingest]
    source[Source stream] --> reader[Reader: sections]
    reader --> chunker[Chunker]
    chunker --> context[Contextualizer]
    context --> embed[Embedder]
    embed --> vectorStore[Vector store]
    context --> bm25[BM25 index]
  end
  subgraph query [Query]
    question[Question and history] --> rewrite[Query transformer]
    rewrite --> embedQuery[Embed searches]
    embedQuery --> dense[Dense searches]
    rewrite --> sparse[Sparse searches]
    dense --> fusion[Reciprocal rank fusion]
    sparse --> fusion
    fusion --> rerank[Reranker]
    rerank --> expand[Expander]
    expand --> gen[Generator]
    gen --> answer[Answer and citations]
  end
```

Ingest splits the document by section, chunks it, adds each chunk's title and section to the indexed text, embeds the chunks on a bounded worker pool, and writes the vector store and the BM25 index at the same time. Query optionally rewrites the question, fans every search out to the indexes, fuses the rankings, optionally reranks and expands, and generates. With one result list, fusion is skipped. With no reranker, the engine keeps the first `topN` chunks. Stages you do not configure do not run.

## Components

| Piece | Package | Role |
| --- | --- | --- |
| Engine | [`ragout`](docs/components/engine.md) | `Ingest`, `Query`, `QueryStream`, `QueryIter`, `Retrieve`, `DeleteSource` |
| Readers | [`reader`](docs/components/readers.md) | Text, Markdown, HTML, PDF, DOCX (split by section), or dispatch by extension |
| Chunk context | [`contextual`](docs/how-to/chunk-context.md) | LLM-written context; the default title and section breadcrumb is built in |
| Query rewrite | [`transform`](docs/how-to/query-rewrite.md) | Condense, multi-query, decompose, HyDE, chain |
| Chunkers | [`chunker`](docs/components/chunkers.md) | Recursive character windows, or tiktoken windows |
| Embedders | [`embedder`](docs/components/embedders.md) | OpenAI, Ollama, and a rate limiter |
| Stores | [`store`](docs/components/stores.md) | In-memory vectors, BM25, Chroma, Qdrant |
| Rerankers | [`reranker`](docs/components/rerankers.md) | Cohere, or maximal marginal relevance |
| Expanders | [`expand`](docs/how-to/expand-hits.md) | Neighbor chunks, or the whole section |
| Generators | [`generator`](docs/components/generators.md) | OpenAI and Ollama, with a context budget and optional history |
| Evaluation | [`eval`](docs/how-to/evaluate.md) | Hit rate, recall, and MRR over your own questions |

## Documentation

- [Get started](docs/getting-started/installation.md) — install, [quickstart](docs/getting-started/quickstart.md), [local Ollama](docs/getting-started/local.md)
- [Concepts](docs/concepts/pipeline.md) — how a query runs, documents, hybrid retrieval, generation, streaming
- [How-to guides](docs/how-to/ingest-files.md) — files, updating sources, chunk context, filters, stores, query rewrite, reranking, expansion, evaluation, traces
- [Components](docs/components/engine.md) — constructors and options for each package
- [Roadmap](docs/roadmap.md) — local cross-encoder, agentic loops, GraphRAG
- [Changelog](CHANGELOG.md) — what changed between releases

The full index is [docs/README.md](docs/README.md).

## Roadmap

Shipped today: hybrid dense and BM25 retrieval with pluggable fusion, section-aware readers, chunk context (title and section by default, LLM-written on request), query rewriting with conversation history, Cohere and MMR rerankers, neighbor and section expansion, cited streaming answers, source replacement and deletion, retrieval evaluation, in-memory persistence, Chroma, and Qdrant.

Next, still unshipped: a local HTTP cross-encoder, a self-check query loop, and GraphRAG. Details are on the [roadmap](docs/roadmap.md).

## License

[MIT](LICENSE). Copyright (c) 2026 chengyaolee.
