# Roadmap

[Docs](../README.md) · [Pipeline](concepts/pipeline.md)

This page is the public list of milestones. Each unshipped item is specified as a small interface the engine can skip when you do not configure it. None of the types below exist in this release. Importing them will not compile.

## Shipped

- Hybrid retrieval: in-memory vectors, BM25, reciprocal rank fusion, and optional dense-only or sparse-only engines.
- Rerankers: Cohere and maximal marginal relevance.
- Cited generation with a token budget, lost-in-the-middle ordering, and `Query` / `QueryStream` / `QueryIter`.
- Readers for text, Markdown, HTML, PDF, and DOCX. Character and tiktoken chunkers. OpenAI and Ollama clients. A rate-limited embedder.
- In-memory snapshots. Chroma and Qdrant over HTTP.
- OpenTelemetry spans and Prometheus stage histograms.
- Stable chunk IDs and a rollback when one store write fails.

## Contextual retrieval

Status: not in this release.

Chunks are embedded on their own text today. A short chunk like a revenue figure has no company name and no year, so both BM25 and the vector index miss obvious questions.

A `Contextualizer` will take the parent document and a chunk and return a sentence or two of situational context. Ingest will prepend that context before embedding, and keep the original chunk text for display and citation. The call will be batched and bounded by a worker count. With no contextualizer configured, ingest stays on the path it has now.

## Query transformation

Status: not in this release.

A query is sent to search as you wrote it. A `QueryTransformer` will sit in front of retrieval and produce one or more search strings:

- Multi-query expansion, several phrasings of the same question, searched concurrently and fused with the same reciprocal rank fusion the engine already uses.
- HyDE, a hypothetical answer that is embedded in place of the raw question.
- Subquery decomposition, for questions that need more than one lookup.

With no transformer configured, the question string is the only search input.

## Pluggable fusion

Status: not in this release.

The engine always fuses with reciprocal rank fusion when both sides return hits. `reranker.ConvexScoreCombination` already blends sigmoid-normalized dense and sparse scores, and the min-max and logistic helpers sit next to it, but `Engine` does not call them.

A `FusionStrategy` option will keep RRF as the default and let you select score blending, including an alpha weight, without forking the query path.

## Local cross-encoder

Status: not in this release.

Cohere and MMR are the rerankers you can pass to `WithReranker`. A third implementation will POST to an HTTP `/rerank` endpoint that speaks the text-embeddings-inference shape, so a local BGE reranker or any compatible server can run beside the process. It will be another `Reranker`. The Cohere and MMR constructors stay.

## Agentic loop

Status: not in this release.

`Query` is one pass: retrieve, rerank, generate. A later `QueryAgentic` will add a bounded loop:

1. Retrieve and draft.
2. A relevance check decides whether the retrieved chunks are enough to answer.
3. On a miss, the question is reformulated and the loop runs again, up to a hop limit.

The single-pass `Query` methods stay the default. The loop runs only when you call it.

## GraphRAG

Status: not in this release.

Indexes today are flat: a vector per chunk and a BM25 posting list. Graph retrieval adds three pieces:

- Entity and relation types extracted from chunks at ingest.
- A graph store, in memory first, with `k`-hop lookup from the entities a question mentions.
- A third branch in the query fan-out. Dense, sparse, and graph hits fuse into one candidate list.

With no graph store configured, that branch does not run.

## Scale

Status: not in this release. Chroma and Qdrant are already supported and are not part of this list.

- A pgvector `VectorStore`.
- Graph database adapters (Neo4j over Cypher, and Bolt or openCypher for stores in that family) behind the same graph interface as the in-memory graph.
- Ingest workers that pull contextualization and triple extraction off the request path.
- An embedding cache in front of `Embedder`, and a cache in front of query transformation, so repeated questions do not call the model again.

## Next

The behavior of the current pipeline is in [Concepts](concepts/pipeline.md). The constructors you can call today are under [Components](components/engine.md).
