# Ragout documentation

These pages follow the usual shape of a library's docs. Start with a working program, then read how the pipeline behaves, then use a how-to for one task, then look up a component when you need its options.

## Get started

Install the module, run `go test` and the offline demo, then point the same engine at OpenAI or Ollama.

- [Installation](getting-started/installation.md)
- [Quickstart](getting-started/quickstart.md)
- [Local models with Ollama](getting-started/local.md)

## Concepts

How a document becomes chunks, how a question becomes an answer, and what each stage does at runtime.

- [Pipeline](concepts/pipeline.md)
- [Documents and chunks](concepts/documents.md)
- [Hybrid retrieval](concepts/hybrid-retrieval.md)
- [Generation and citations](concepts/generation.md)
- [Streaming](concepts/streaming.md)

## How-to guides

Short recipes. Each one assumes you already have an `Engine`.

- [Ingest files](how-to/ingest-files.md)
- [Filter by metadata](how-to/metadata-filters.md)
- [Choose a store](how-to/stores.md)
- [Rerank results](how-to/rerank.md)
- [Traces and metrics](how-to/observability.md)

## Components

Constructors, options, and the path each package takes at runtime.

- [Engine](components/engine.md)
- [Readers](components/readers.md)
- [Chunkers](components/chunkers.md)
- [Embedders](components/embedders.md)
- [Stores](components/stores.md)
- [Rerankers](components/rerankers.md)
- [Generators](components/generators.md)

## Roadmap

What this release includes, and the milestones that are specified but not shipped: contextual retrieval, query transforms, a local cross-encoder, agentic loops, and GraphRAG.

- [Roadmap](roadmap.md)
