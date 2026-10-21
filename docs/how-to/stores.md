# Choose a store

[Docs](../README.md) · [Stores](../components/stores.md) · [Hybrid retrieval](../concepts/hybrid-retrieval.md)

The engine accepts any `VectorStore` and any `IndexStore`. This release ships an in-memory vector index, an in-memory BM25 index, a Chroma client, and a Qdrant client. BM25 is the only `IndexStore` implementation.

## In memory

`store.NewVectorStore` and `store.NewBM25Index` need no server. They are safe for concurrent ingest and query. They are empty after a restart unless you snapshot them.

Both stores snapshot the same list of chunks, so one file restores both. Save the vector store and load that file into each store:

```go
vectors := store.NewVectorStore()
index := store.NewBM25Index()

// Name the file after the embedding model, so switching models starts a fresh index.
const snapshot = "chunks.nomic-embed-text.gob"
for _, s := range []store.Persistable{vectors, index} {
    if err := store.LoadFile(snapshot, s); err != nil {
        return err
    }
}

// ... build the engine with these stores, ingest, query ...

if err := store.SaveFile(snapshot, vectors); err != nil {
    return err
}
```

`LoadFile` on a missing path returns nil and leaves the store as it is. `SaveFile` writes a temp file and renames it into place, so a single snapshot is never half-written. Two files, one per store, can be left out of step by a crash between the two saves.

The snapshot does not record which model made the vectors. A model with a different dimension fails queries with `ErrDimensionMismatch`. A model with the same dimension silently compares vectors from two models. Keep the model in the file name as above, and re-ingest from your source data after a switch.

Each save rewrites the whole file, and the stores live in one process. For a corpus that outgrows that, use Chroma or Qdrant.

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
