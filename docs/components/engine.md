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

`NewEngine` checks two things. At least one of the vector store and the index store must be set (`ErrNoStores`). A vector store requires an embedder (`ErrNilEmbedder`). Reader, chunker, reranker, and generator are checked later, when the call that needs them runs. The reranker is optional for the life of the engine.

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

## Methods

| Method | Behavior |
| --- | --- |
| `Ingest(ctx, source, r, metadata)` | Read, chunk, embed, write both stores. Rolls both back if one write fails. |
| `Query(ctx, query, opts...) (Answer, error)` | Full answer. |
| `QueryStream(ctx, query, cb, opts...) (Answer, error)` | One callback per token. |
| `QueryIter(ctx, query, opts...) (*Answer, iter.Seq2[string, error])` | Sources filled before the iterator yields. |

`Answer.Text` is the generated string. `Answer.Sources` is the citation list (`Sources[i]` is `[i+1]`). `Answer.Retrieved` is every chunk passed into the generator.

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

Storage and HTTP failures are wrapped with `fmt.Errorf` and the stage name (`ragout.Ingest: ...`, `embed query failed`, `dense search failed`, `reranking failed`).

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

## Next

[Readers](readers.md) are the first stage of ingest.
