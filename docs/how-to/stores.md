# Choose a store

[Docs](../README.md) · [Stores](../components/stores.md) · [Hybrid retrieval](../concepts/hybrid-retrieval.md)

The engine accepts any `VectorStore` and any `IndexStore`. This release ships an in-memory vector index, an in-memory BM25 index, a Chroma client, and a Qdrant client. BM25 is the only `IndexStore` implementation.

## In memory

`store.NewVectorStore` and `store.NewBM25Index` need no server. They are safe for concurrent ingest and query. They are empty after a restart unless you snapshot them.

```go
vectors := store.NewVectorStore()
index := store.NewBM25Index()

if err := store.LoadFile("vectors.gob", vectors); err != nil {
    return err
}
if err := store.LoadFile("bm25.gob", index); err != nil {
    return err
}

// ... build the engine with these stores, ingest, query ...

if err := store.SaveFile("vectors.gob", vectors); err != nil {
    return err
}
if err := store.SaveFile("bm25.gob", index); err != nil {
    return err
}
```

`LoadFile` on a missing path returns nil and leaves the store as it is. `SaveFile` writes a temp file and renames it into place.

Snapshots are gob-encoded. Metadata values should be strings, numbers, bools, and slices or maps of those. Other concrete types must be registered with `gob.Register` before save and load.

`Save` and `Load` write to any `io.Writer` / `io.Reader` if you do not want a file. `BM25Index.Load` rebuilds the inverted index by indexing the decoded chunks again.

## Chroma

Chroma is a vector store. Pair it with `NewBM25Index` when you still want hybrid search. BM25 stays in the Go process.

```go
chromaStore, err := store.NewChromaStore(ctx, "http://localhost:8000", "notes")
if err != nil {
    return err
}
```

The client uses Chroma's v2 HTTP API, tenant `default_tenant` and database `default_database`. Override those with `WithChromaTenant` and `WithChromaDatabase`. `WithChromaHTTPClient` is how you attach a token: set a `RoundTripper` that adds the header.

`NewChromaStore` gets or creates the collection. Chunk text is the Chroma document. `ragout_document_id` and `ragout_index` are reserved metadata keys.

## Qdrant

```go
qdrantStore, err := store.NewQdrantStore(ctx, "http://localhost:6333", "notes", emb.Dimension())
if err != nil {
    return err
}
```

`dim` must match the embedder. The collection is created with cosine distance if it does not exist. A 409 from Qdrant means it already exists and is treated as success. `WithQdrantHTTPClient` is the hook for an `api-key` header.

Payload keys `ragout_content`, `ragout_document_id`, and `ragout_index` are reserved. Your metadata is stored beside them.

## Dense only, or sparse only

Omit `WithIndexStore` for vectors only. The engine will not run BM25 or fusion.

Omit `WithVectorStore` and `WithEmbedder` for keywords only. `NewEngine` allows that. Queries skip the embedder and use the BM25 ranking.

```go
engine, err := ragout.NewEngine(
    ragout.WithReader(reader.NewTextReader(10<<20)),
    ragout.WithChunker(chunks),
    ragout.WithIndexStore(store.NewBM25Index()),
    ragout.WithGenerator(gen),
)
```

## Next

[Rerank results](rerank.md) if the top of this list still needs a second pass.
