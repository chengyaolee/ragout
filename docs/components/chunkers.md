# Chunkers

[Docs](../README.md) · [Documents](../concepts/documents.md) · [Embedders](embedders.md)

A chunker implements `ragout.Chunker`:

```go
Chunk(ctx context.Context, doc ragout.Document) ([]ragout.Chunk, error)
```

Both built-in chunkers copy `doc.Metadata` onto each chunk, set `DocumentID` from `doc.ID`, and set `ID` with `ragout.ChunkID`. Call them after ingest has assigned `doc.ID`, which `Engine.Ingest` does. An empty document fails. A cancelled context fails.

## Character windows

```go
c, err := chunker.NewCharacterChunker(800, 100, nil)
```

`chunkSize` must be greater than 0. `chunkOverlap` must be greater than or equal to 0. Overlap must be less than `chunkSize` by the time `Chunk` runs.

`nil` or an empty separator list uses `["\n\n", "\n", " ", ""]`.

The splitter tries the first separator that occurs in the text. Pieces small enough to fit are merged, including the separator, up to `chunkSize` bytes. When the current merge would pass the limit, the finished piece is emitted and a suffix of about `chunkOverlap` bytes is kept as the start of the next piece. A piece that is still too large is split with the next separator. When no separator remains, the leftover text is cut on a rune window of `chunkSize` with step `chunkSize - chunkOverlap`.

`Chunk.Index` is `0, 1, 2, ...` in emission order.

Use this chunker for prose when you want breaks on paragraphs and lines. The size is a byte length, except for that final rune window.

## Token windows

```go
c, err := chunker.NewTokenChunker(400, 50, "cl100k_base")
```

`chunkSize` is a token count. Overlap must be greater than or equal to 0 and less than `chunkSize`. `encodingName` is a tiktoken encoding. `cl100k_base` matches GPT-3.5 and GPT-4. `o200k_base` matches GPT-4o. The constructor loads that encoding immediately and returns the load error.

`Chunk` encodes the whole document, then walks a window of `chunkSize` tokens with step `chunkSize - chunkOverlap`, and decodes each window back to text. `Chunk.Index` is the starting token offset, not a dense chunk number. Chunk IDs follow that index, so they stay stable for the same document and the same window size.

Use this chunker when the embedding model and the generator share a tokenizer and you want the window to match what they will count.

## Which one

Character chunking needs no vocabulary file and keeps paragraph boundaries. Token chunking matches a model's tokenizer and can split mid-sentence when the window fills. The generator's budgeter counts tokens again later, so a character-chunked index is still safe. It just makes the budgeter's job less predictable.

## Next

[Embedders](embedders.md) turn chunk text into vectors during ingest and turn the question into a vector during query.
