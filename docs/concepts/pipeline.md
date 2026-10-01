# Pipeline

[Docs](../README.md) · [Documents](documents.md) · [Hybrid retrieval](hybrid-retrieval.md) · [Generation](generation.md) · [Streaming](streaming.md)

An `Engine` is one ingest path and one query path. Stages you do not configure are skipped. A missing reranker does not fail the query. A missing vector store does not embed the question.

## Ingest

`Ingest(ctx, source, r, metadata)`:

1. Copies `metadata`, sets `source` to the source name, and asks the reader to turn the stream into documents. The source name is also how `reader.ByExtension` picks `.md`, `.pdf`, and the other formats.
2. Assigns a document ID from the source name and the text when the reader left `ID` empty. Re-ingesting the same file produces the same IDs, so the stores overwrite instead of duplicating.
3. Chunks each document.
4. If a vector store is configured, embeds the chunks on a worker pool. The default batch size is 32 and the default worker count is 8. Change them with `WithEmbedBatchSize` and `WithEmbedConcurrency`.
5. Writes the vector store and the BM25 index at the same time. If either write fails, both stores delete those chunk IDs, including when the failure is a cancelled context. A retry then starts clean.

`Ingest` returns `ErrNilReader`, `ErrNilChunker`, `ErrEmptyDocument`, or `ErrEmptyChunk` when that stage has nothing to do. A reader that hits its byte cap returns `ErrDocumentTooLarge`.

## Query

`Query`, `QueryStream`, and `QueryIter` share one retrieval path.

1. Rejects an empty question (`ErrEmptyQuery`) and a missing generator (`ErrNilGenerator`).
2. Fans out. The embedder embeds the question and the vector store runs cosine search. At the same time the BM25 index scores the question text. Both searches take the same metadata filter and the same `topKRecall` (default 50).
3. Fuses. When both sides return hits, reciprocal rank fusion (default `k=60`) builds one ranking and keeps `topKRecall` chunks. When only one side returns hits, that ranking is used as-is. Fusion is how the engine avoids comparing cosine scores with BM25 scores. See [Hybrid retrieval](hybrid-retrieval.md).
4. Reranks when a reranker is set, down to `topN` (default 5). With no reranker, the engine keeps the first `topN` of the fused list.
5. Hands that list to the generator. The generator drops chunks that do not fit the token budget, numbers the rest, and streams tokens. An empty retrieval list is `ErrNoResults`.

Per-query options override the engine defaults for that call only: `WithQueryFilter`, `WithQueryTopKRecall`, `WithQueryTopNRerank`.

## Defaults

| Knob | Default | Option |
| --- | --- | --- |
| Recall depth | 50 | `WithTopKRecall` |
| Chunks after rerank, or the cut when there is no reranker | 5 | `WithTopNRerank` |
| RRF constant `k` | 60 | `WithRRFK` |
| Embed batch size | 32 | `WithEmbedBatchSize` |
| Embed workers | 8 | `WithEmbedConcurrency` |

Values of 0 or less are ignored, and the default stays.

## What a trace looks like

A configured OpenTelemetry tracer records `ragout.Engine.Query`, then `query.retrieval`, `query.embed`, `query.dense_search`, `query.sparse_search`, `query.rrf_fusion` when both sides hit, `query.reranker` when a reranker is set, and `query.generation`. Prometheus histograms use the same stage names. See [Traces and metrics](../how-to/observability.md).

## Next

[Documents and chunks](documents.md) explains the types that move through this path.
