# Stores

[Docs](../README.md) · [Choose a store](../how-to/stores.md) · [Hybrid retrieval](../concepts/hybrid-retrieval.md)

Two interfaces, four implementations.

```go
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
```

`SearchDense` fills `DenseScore` and `Score`. `SearchSparse` fills `SparseScore` and `Score`. `Delete` ignores unknown IDs. The engine calls `Delete` on both stores when one of the ingest writes fails.

## In-memory vectors

```go
vectors := store.NewVectorStore()
```

`Upsert` copies each chunk, including a clone of its metadata, and replaces any chunk with the same ID. Chunks with an empty ID are skipped. `SearchDense` scans every chunk, drops those that fail the filter, scores the rest with `store.CosineSimilarity`, and returns the top `topK`. A zero vector scores 0. Different lengths return `ErrDimensionMismatch`.

`CosineSimilarity` divides by the product of the L2 norms. Vectors already within `1e-4` of unit length use the dot product directly.

## BM25

```go
index := store.NewBM25Index(
    store.WithK1(1.5),
    store.WithB(0.75),
    store.WithTokenizer(store.Tokenize),
)
```

`Index` tokenizes each chunk, updates document frequency, and replaces a chunk that is indexed again under the same ID. `SearchSparse` runs Okapi BM25 over the query tokens and then applies the metadata filter. An empty index or a query that tokenizes to nothing returns an empty slice.

`Tokenize` lowercases and splits on runes that are not letters or digits. `TermsMatched(query)` counts how many query tokens occur in the inverted index. The query span reads that method when it is present.

`k1` saturates term frequency. `b` scales scores by document length relative to the average. The defaults are the usual Okapi settings, 1.5 and 0.75.

## Snapshots

`VectorStore` and `BM25Index` implement `store.Persistable` (`Save` / `Load`).

```go
err := store.SaveFile("vectors.gob", vectors)
err = store.LoadFile("vectors.gob", vectors)
```

`SaveFile` encodes to a buffer, writes `path.tmp`, and renames it over `path`. `LoadFile` returns nil when the file does not exist. Gob needs basic metadata types, or a prior `gob.Register` for anything else. Loading a BM25 snapshot re-indexes the chunks so the posting lists match the corpus statistics.

## Chroma

```go
cs, err := store.NewChromaStore(ctx, "http://localhost:8000", "notes",
    store.WithChromaTenant("default_tenant"),
    store.WithChromaDatabase("default_database"),
    store.WithChromaHTTPClient(client),
)
```

HTTP only, Chroma API v2. The constructor gets or creates the collection and keeps its id. Upsert sends ids, embeddings, documents, and metadata. `ragout_document_id` and `ragout_index` are reserved metadata keys used to rebuild `Chunk` on search. Filters are translated to a Chroma `where` document. See [Filter by metadata](../how-to/metadata-filters.md).

There is no BM25 inside Chroma. Keep a `BM25Index` beside it for hybrid search.

## Qdrant

```go
qs, err := store.NewQdrantStore(ctx, "http://localhost:6333", "notes", emb.Dimension(),
    store.WithQdrantHTTPClient(client),
)
```

HTTP only. The constructor PUTs the collection with cosine distance and the given size. Status 409 means the collection is already there. Upsert writes points whose payload holds the chunk text under `ragout_content`, plus `ragout_document_id`, `ragout_index`, and your metadata. Search uses Qdrant's query API. Filters become `must`, `must_not`, and `range`.

Do not reuse the reserved payload keys for your own fields.

## Next

[Rerankers](rerankers.md) reorder what these searches return.
