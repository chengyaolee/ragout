# Quickstart

[Docs](../README.md) · [Installation](installation.md) · [Local models](local.md)

This program indexes one file and prints a cited answer. It is the same program as [`examples/quickstart`](../../examples/quickstart/main.go).

You need `OPENAI_API_KEY`. The default embedder is `text-embedding-3-small` (1536 dimensions). The default generator is `gpt-4o-mini`.

```go
package main

import (
	"context"
	"fmt"
	"log"
	"os"

	"github.com/chengyaolee/ragout"
	"github.com/chengyaolee/ragout/chunker"
	"github.com/chengyaolee/ragout/embedder"
	"github.com/chengyaolee/ragout/generator"
	"github.com/chengyaolee/ragout/reader"
	"github.com/chengyaolee/ragout/store"
)

func main() {
	apiKey := os.Getenv("OPENAI_API_KEY")
	if apiKey == "" {
		log.Fatal("OPENAI_API_KEY is required")
	}
	if len(os.Args) != 2 {
		log.Fatal("usage: quickstart <file>")
	}

	ctx := context.Background()

	chunks, err := chunker.NewCharacterChunker(800, 100, nil)
	if err != nil {
		log.Fatal(err)
	}

	engine, err := ragout.NewEngine(
		ragout.WithReader(reader.NewByExtension(10<<20)),
		ragout.WithChunker(chunks),
		ragout.WithEmbedder(embedder.NewOpenAIEmbedder(apiKey)),
		ragout.WithVectorStore(store.NewVectorStore()),
		ragout.WithIndexStore(store.NewBM25Index()),
		ragout.WithGenerator(generator.NewOpenAIGenerator(apiKey)),
	)
	if err != nil {
		log.Fatal(err)
	}

	f, err := os.Open(os.Args[1])
	if err != nil {
		log.Fatal(err)
	}
	defer f.Close()

	if err := engine.Ingest(ctx, os.Args[1], f, nil); err != nil {
		log.Fatal(err)
	}

	answer, err := engine.Query(ctx, "What is this document about?")
	if err != nil {
		log.Fatal(err)
	}

	fmt.Println(answer.Text)
	for i, src := range answer.Sources {
		source, _ := src.Chunk.Metadata[ragout.MetadataSource].(string)
		fmt.Printf("[%d] %s\n", i+1, source)
	}
}
```

```bash
go run ./examples/quickstart notes.md
```

The file name needs an extension the reader knows (`.txt`, `.md`, `.markdown`, `.html`, `.htm`, `.pdf`, `.docx`). `Ingest` records that name on every chunk as metadata `source`.

## What each step does

1. `NewCharacterChunker(800, 100, nil)` splits on paragraphs, then lines, then words, with a 100-byte overlap. `nil` separators means those defaults.
2. `NewByExtension(10 << 20)` picks a reader from the file extension and refuses inputs larger than 10 MiB.
3. `NewVectorStore` and `NewBM25Index` keep the index in memory. Restarting the process drops it. See [Choose a store](../how-to/stores.md) for snapshots, Chroma, and Qdrant.
4. `Ingest` reads, chunks, embeds in batches, and writes both stores. The same file ingested again overwrites the same chunk IDs.
5. `Query` embeds the question, searches both indexes, fuses them with reciprocal rank fusion, and generates.

## The answer

`Answer` has three fields:

| Field | When it is set | What it is |
| --- | --- | --- |
| `Text` | After generation finishes | The full answer. |
| `Sources` | Before the first token | Chunks that were placed in the prompt. `Sources[i]` is citation `[i+1]`. |
| `Retrieved` | Before generation | Every candidate handed to the generator, best first. The budgeter may drop some of these from `Sources` when they do not fit the token limit. |

`Query` blocks until the model finishes. To print tokens as they arrive, use [streaming](../concepts/streaming.md).

## Next

- [How the pipeline runs](../concepts/pipeline.md)
- [Ingest more than one format](../how-to/ingest-files.md)
- [Run it locally](local.md)
