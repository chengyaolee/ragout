# Roadmap

[Docs](../README.md) · [Pipeline](concepts/pipeline.md)

This page is the public list of milestones. Each unshipped item is specified as a small interface the engine can skip when you do not configure it. None of the types in the unshipped sections below exist in this release. Importing them will not compile.

## Shipped

- Hybrid retrieval: in-memory vectors, BM25, reciprocal rank fusion, and optional dense-only or sparse-only engines.
- Rerankers: Cohere and maximal marginal relevance.
- Cited generation with a token budget, lost-in-the-middle ordering, and `Query` / `QueryStream` / `QueryIter`.
- Readers for text, Markdown, HTML, PDF, and DOCX, with Markdown, HTML, and DOCX split by heading section. Character and tiktoken chunkers. OpenAI and Ollama clients. A rate-limited embedder.
- In-memory snapshots. Chroma and Qdrant over HTTP.
- OpenTelemetry spans and Prometheus stage histograms.
- Stable chunk IDs, a rollback when one store write fails, `DeleteSource`, and replace-on-ingest.
- Chunk context: a title and section breadcrumb by default, or LLM-written context (`contextual.New`, Anthropic's contextual retrieval). See [Give chunks context](how-to/chunk-context.md).
- Query rewriting: condense with history, multi-query, decomposition, and HyDE, with per-index searches, a timeout, and fallback. See [Rewrite questions](how-to/query-rewrite.md).
- Pluggable fusion over any number of result lists, RRF by default.
- Neighbor and section expansion before generation. See [Widen hits](how-to/expand-hits.md).
- Optional conversation history in the answer prompt.
- `Retrieve` and the `eval` package for measuring retrieval, with per-category scoring, nDCG, and a benchmark runner. Results for this repository's docs and BEIR SciFact are in [eval/results](../eval/results/). See [Measure retrieval](how-to/evaluate.md).
- A local cross-encoder reranker over HTTP: bge-reranker-v2-m3 by default, Qwen3-Reranker-4B as an option. See [Rerank results](how-to/rerank.md).
- Retrieval workflows (`flow`): a graph of swappable steps, with a corrective-RAG preset that grades evidence, searches again for what is missing, and abstains when it stays insufficient. See [Build retrieval workflows](how-to/workflows.md).

## Score-blending fusion

Status: not in this release.

`WithFusion` accepts any `FusionStrategy`, and RRF is the only built-in one. `reranker.ConvexScoreCombination` already blends sigmoid-normalized dense and sparse scores for one dense and one sparse list. A built-in convex strategy waits on a question this release does not answer: with several searches per question, there is no single dense and sparse score to blend.

## Workflow extensions

Status: not in this release.

- A web-search fallback step for CRAG, as in the original paper, behind the same `Retriever` interface.
- A tool-using agent step: a model that picks which retrieval or tool to call next, inside a flow. It needs tool-calling support in the generators first.

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

## Query rewrite follow-ups

Status: not in this release. Each waits on evidence from the evaluation that it is needed.

- Weighted fusion, so rephrasings count less than the question itself.
- A per-sub-question quota before the final cut, so `Decompose` evidence for one part of a question is not crowded out.
- Filter extraction from the question (self-query), with an explicit allow-list of fields.

## Next

The behavior of the current pipeline is in [Concepts](concepts/pipeline.md). The constructors you can call today are under [Components](components/engine.md).
