# Give chunks context

[Docs](../README.md) · [Documents and chunks](../concepts/documents.md) · [Readers](../components/readers.md)

A chunk cut from a long document often loses its subject. "Revenue grew 24%" does not say which company or which year, so neither BM25 nor the vector index matches it to "Acme 2024 results".

A contextualizer fixes that at ingest. It writes a short text into `Chunk.Context`. The embedder and the BM25 index see `Context` followed by `Content`. The generator, the token budget, and citations only ever see `Content`, so answers still quote the original text.

## The default: titles and sections

With no option set, the engine uses `ragout.MetadataContext()`. It makes no model call. Each chunk gets its document's breadcrumb:

```text
Annual Report > Financials > Q3
```

The breadcrumb comes from metadata the readers fill in:

| Key | Set by |
| --- | --- |
| `title` | Markdown frontmatter `title:`, or an HTML `<title>`. |
| `section` | Markdown, HTML, and DOCX readers, from the heading path. See [Readers](../components/readers.md). |
| `source` | `Ingest`. Used, minus its extension, when there is no title. |

Plain text and PDF have no headings, so their breadcrumb is the title or the file name.

`ragout.Breadcrumb(meta)` builds the same string if you want it elsewhere.

## Turn it off

```go
ragout.WithContextualizer(nil)
```

Chunks are then indexed on their text alone. If you already have an index built with the default, re-ingest after changing this so every chunk was indexed the same way.

## LLM-written context

`contextual.New` asks a model for a sentence or two that situates each chunk in its document, the method Anthropic calls contextual retrieval. The result keeps the breadcrumb in front of that sentence.

```go
llm := generator.NewOpenAIGenerator(os.Getenv("OPENAI_API_KEY"))
ragout.WithContextualizer(contextual.New(llm,
    contextual.WithConcurrency(8), // parallel calls, default 4
    contextual.WithWindow(16000),  // bytes of document per call, default 8000
))
```

It costs one model call per chunk at ingest and nothing at query time.

**Measure before switching.** On the [docs benchmark](../../eval/results/docs.md), LLM context scored the same hit rate as the free title-and-section default (0.738 at k=5). It took several minutes to index 231 chunks with a 3B model, against seconds for the default. Its gain is likely larger on documents whose chunks lack headings, like the revenue example above, and with a stronger model.

- **Shared prefix:** a document that fits in the window is sent whole, as a prefix that is identical on every call. OpenAI and Ollama reuse a repeated prefix automatically. The first chunk is sent alone so the cache exists before the rest run.
- **Larger documents:** each call gets a window of the document around its chunk instead. Cost per chunk stays bounded, but the prefix differs per chunk, so it is not cached.
- **Sections:** Markdown, HTML, and DOCX arrive split by heading, so the "document" the model sees is the chunk's section.
- **Ollama:** set `generator.WithOllamaNumCtx` at least as large as the window plus the prompt. Ollama cuts longer prompts without an error.
- **Errors:** any call that fails fails the ingest.

Any `Completer` works: `Complete(ctx, prefix, prompt string) (string, error)`. Both built-in generators implement it.

## Write your own

```go
type Contextualizer interface {
    Contextualize(ctx context.Context, doc Document, chunks []Chunk) ([]string, error)
}
```

Return one string per chunk, in order.

## Check that it helps

Run the same questions against an engine with and without the option. See [Measure retrieval](evaluate.md).

## Next

[Rewrite questions](query-rewrite.md) works on the other side of the index.
