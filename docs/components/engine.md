# Engine

[Docs](../README.md) · [Pipeline](../concepts/pipeline.md) · [Readers](readers.md)

`ragout.Engine` is the only type most callers need. Construct it with functional options, then call `Ingest` and one of the query methods.

```go
engine, err := ragout.NewEngine(
    ragout.WithReader(reader.NewByExtension(10<<20)),
    ragout.WithChunker(chunks),
    ragout.WithEmbedder(emb),
    ragout.WithVectorStore(vectors),
    ragout.WithIndexStore(index),
    ragout.WithReranker(rr),
    ragout.WithGenerator(gen),
    ragout.WithTopKRecall(50),
    ragout.WithTopNRerank(5),
    ragout.WithRRFK(60),
    ragout.WithEmbedBatchSize(32),
    ragout.WithEmbedConcurrency(8),
)
```

`NewEngine` checks three things. At least one of the vector store and the index store must be set (`ErrNoStores`). A vector store requires an embedder (`ErrNilEmbedder`). `WithReplaceSource` requires stores that can delete by source (`ErrDeleteUnsupported`). Reader, chunker, and generator are checked later, when the call that needs them runs. Every other stage is optional.

## Options

| Option | Effect |
| --- | --- |
| `WithReader` | Required by `Ingest`. |
| `WithChunker` | Required by `Ingest`. |
| `WithEmbedder` | Required when a vector store is set. Also used to embed the question. |
| `WithVectorStore` | Dense half of retrieval. `Upsert`, `SearchDense`, `Delete`. |
| `WithIndexStore` | Sparse half. `Index`, `SearchSparse`, `Delete`. |
| `WithReranker` | Optional second pass. |
| `WithGenerator` | Required by every query method. |
| `WithContextualizer` | Text indexed with each chunk but never shown to the generator. Default `MetadataContext()` (title and section breadcrumb); `nil` turns it off. See [Give chunks context](../how-to/chunk-context.md). |
| `WithReplaceSource` | `Ingest` replaces a source instead of adding to it. See [Update and delete sources](../how-to/update-sources.md). |
| `WithQueryTransformer` | Rewrites each question before retrieval. Default none. See [Rewrite questions](../how-to/query-rewrite.md). |
| `WithTransformTimeout` | Deadline for the transformer; on timeout the query falls back to the original question. Default none. |
| `WithFusion` | Replaces the default `RRF(k)` fusion of result lists. |
| `WithExpander` | Widens hits before generation. Default none. See [Widen hits](../how-to/expand-hits.md). |
| `WithTopKRecall` | Hits requested from each store. Default 50. |
| `WithTopNRerank` | Hits kept after rerank, or the cut when there is no reranker. Default 5. |
| `WithRRFK` | Fusion smoothing constant. Default 60. |
| `WithEmbedBatchSize` | Texts per embed request. Default 32. |
| `WithEmbedConcurrency` | Embed workers. Default 8. |

Non-positive integers are ignored and the default remains.

## Query options

These apply to one call.

| Option | Effect |
| --- | --- |
| `WithQueryFilter` | Metadata filter for both stores. |
| `WithQueryTopKRecall` | Overrides recall depth. |
| `WithQueryTopNRerank` | Overrides the final cut. |
| `WithHistory` | Earlier turns of the conversation, oldest first. Read by the query transformer, and by a generator that implements `HistoryGenerator`. |

## Methods

| Method | Behavior |
| --- | --- |
| `Ingest(ctx, source, r, metadata)` | Read, chunk, embed, write both stores. Rolls both back if one write fails. |
| `Query(ctx, query, opts...) (Answer, error)` | Full answer. |
| `QueryStream(ctx, query, cb, opts...) (Answer, error)` | One callback per token. |
| `QueryIter(ctx, query, opts...) (*Answer, iter.Seq2[string, error])` | Sources filled before the iterator yields. |
| `Retrieve(ctx, query, opts...) (Answer, error)` | Everything before generation. Needs no generator. `Text` and `Sources` stay empty. |
| `DeleteSource(ctx, source) error` | Removes every chunk ingested from `source`, from every store. |

| `Answer` field | Meaning |
| --- | --- |
| `Text` | The generated string. |
| `Sources` | The citation list. `Sources[i]` is `[i+1]`. |
| `Retrieved` | Every chunk passed to the generator, after any expansion. |
| `Question` | The standalone question that was reranked and answered. |
| `Searches` | Every search requested, the question first. |
| `TransformErr` | The query transformer's error or timeout, if any. The query still ran. |

## Errors

| Sentinel | When |
| --- | --- |
| `ErrNoStores` | `NewEngine` with neither store. |
| `ErrNilEmbedder` | Vector store set, embedder not set. |
| `ErrNilReader` | `Ingest` with no reader. |
| `ErrNilChunker` | `Ingest` with no chunker. |
| `ErrEmptyDocument` | Reader produced no documents, or a reader saw empty text. |
| `ErrEmptyChunk` | Chunker produced nothing. |
| `ErrDocumentTooLarge` | Input exceeded the reader's byte cap. |
| `ErrUnsupportedFormat` | `ByExtension` has no reader for that extension. |
| `ErrEmptyQuery` | Question is empty or whitespace. |
| `ErrNilGenerator` | Query with no generator. |
| `ErrNoResults` | Both searches returned nothing. |
| `ErrNilCallback` | `QueryStream` with a nil callback. |
| `ErrEmptyEmbeddings` | Embedder returned no vectors. |
| `ErrCountMismatch` | Embedder returned a different count than it was given. |
| `ErrDimensionMismatch` | Query vector length does not match a stored vector. |
| `ErrNoSource` | `DeleteSource` with an empty name. |
| `ErrDeleteUnsupported` | A configured store does not implement `SourceDeleter`. |

Storage and HTTP failures are wrapped with `fmt.Errorf` and the stage name (`ragout.Ingest: ...`, `embed query failed`, `dense search failed`, `reranking failed`, `expansion failed`). A query transformer's error is not returned: it lands in `Answer.TransformErr`.

## Interfaces

Implement these to replace a built-in. Context is the first parameter.

```go
type Reader interface {
    Read(ctx context.Context, r io.Reader, metadata map[string]any) ([]Document, error)
}
type Chunker interface {
    Chunk(ctx context.Context, doc Document) ([]Chunk, error)
}
type Embedder interface {
    EmbedBatch(ctx context.Context, texts []string) ([][]float32, error)
    Dimension() int
}
type VectorStore interface {
    Upsert(ctx context.Context, chunks []Chunk) error
    SearchDense(ctx context.Context, vector []float32, topK int, filter map[string]any) ([]ScoredChunk, error)
    Delete(ctx context.Context, ids []string) error
}
type IndexStore interface {
    Index(ctx context.Context, chunks []Chunk) error
    SearchSparse(ctx context.Context, query string, topK int, filter map[string]any) ([]ScoredChunk, error)
    Delete(ctx context.Context, ids []string) error
}
type Reranker interface {
    Rerank(ctx context.Context, query string, candidates []ScoredChunk, topN int) ([]ScoredChunk, error)
}
type Generator interface {
    GenerateIter(ctx context.Context, query string, candidates []ScoredChunk) (sources []ScoredChunk, tokens iter.Seq2[string, error])
}
```

`GenerateIter` must return sources in citation order before the iterator runs. `sources[i]` is what the prompt calls `[i+1]`.

Optional stages, each skipped when not configured:

```go
type Contextualizer interface {
    Contextualize(ctx context.Context, doc Document, chunks []Chunk) ([]string, error)
}
type QueryTransformer interface {
    Transform(ctx context.Context, question string, history []Turn) (Rewrite, error)
}
type FusionStrategy interface {
    Fuse(lists [][]ScoredChunk, topK int) []ScoredChunk
}
type Expander interface {
    Expand(ctx context.Context, candidates []ScoredChunk) ([]ScoredChunk, error)
}
type Completer interface { // one-shot model call, used by contextual and transform
    Complete(ctx context.Context, prefix, prompt string) (string, error)
}
```

Optional capabilities the engine checks for at runtime:

| Interface | Implemented by | Used for |
| --- | --- | --- |
| `HistoryGenerator` | OpenAI and Ollama generators | History in the answer prompt |
| `ChunkGetter` | All four stores | `expand.Neighbors` and `expand.Section` |
| `SourceDeleter` | All four stores | `DeleteSource` and `WithReplaceSource` |

## Next

[Readers](readers.md) are the first stage of ingest.
