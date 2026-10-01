# Filter by metadata

[Docs](../README.md) · [Documents](../concepts/documents.md) · [Stores](stores.md)

`WithQueryFilter` restricts both the vector search and the BM25 search. Keys you do not mention are ignored. Every key in the filter must match. A chunk that lacks a key fails the filter.

```go
answer, err := engine.Query(ctx, "on-call rotation",
    ragout.WithQueryFilter(map[string]any{
        "department": "engineering",
    }),
)
```

The metadata you pass to `Ingest` is copied onto every chunk, along with `source`.

## Equality and sets

A plain value must be equal (`reflect.DeepEqual`).

A slice means "any of these values":

```go
ragout.WithQueryFilter(map[string]any{
    "team": []any{"sre", "platform"},
})
```

## Operators

A nested map is a set of operators on that field. All operators on a field must match.

| Operator | Matches when |
| --- | --- |
| `$eq` | Equal to the value. |
| `$ne` | Not equal. |
| `$in` | Equal to one element of the slice. |
| `$nin` | Equal to none of the elements. |
| `$gt`, `$gte`, `$lt`, `$lte` | Numeric comparison. Both sides are read as `float64`. |

```go
ragout.WithQueryFilter(map[string]any{
    "year": map[string]any{"$gte": 2024},
    "status": map[string]any{"$in": []any{"published", "reviewed"}},
})
```

An unknown operator name is treated as an equality check between the chunk's value and that operator's argument. Stick to the operators above.

## Where the filter runs

The in-memory vector store and the BM25 index evaluate this language directly. Chroma receives the same shape as a `where` clause (`$and` when there is more than one clause). Qdrant receives `must` / `must_not` / `range` conditions. A slice or `$in` becomes Qdrant `match.any`. `$ne` and `$nin` become `must_not`.

Filter at query time. The stores do not keep a separate filtered index.

## Next

[Choose a store](stores.md) if the filter has to survive a process restart, or run against Chroma or Qdrant.
