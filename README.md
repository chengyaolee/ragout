# Ragout
<p align="center">
  <img src="ragout-stock.jpg" width="300" />
</p>
An extensive RAG library for Go



### Phase 1: Foundation, Data Contracts & Metadata Propagation
#### Deliverables: 
- Package skeletons, `Document`, `Chunk`, `ScoredChunk`, core interfaces, and error sentinels (`ErrContextExceeded`, `ErrEmptyStream`, `ErrRateLimited`).

#### Engineering Rules:

- `context.Context` must always be parameter zero.
- Metadata must propagate immutably from `Document` down into `Chunk.Metadata` to prevent provenance loss during ingestion.


### Phase 2: Ingestion Primitives & Streaming Chunkers
#### Deliverables: 
- Plain text, Markdown, HTML, and PDF readers (`io.Reader` target); token-aware and recursive character splitters.
#### Key Implementations:
- **Pure Go Streaming:** Never load raw bytes using `os.ReadFile`. Read streams via `bufio.Reader` or `io.LimitReader` to enforce strict maximum ingestion size limits (protecting against denial-of-service memory allocations).
- Token Boundary Chunking: Integrate `pkoukk/tiktoken-go` for BPE token counting. Implement chunking with a deterministic step window *($size - overlap$)*. Maintain byte offsets in chunk metadata for exact citation highlighting.

### Phase 3: Bounded Concurrent Embedding Engine
#### **Deliverables:** Embedding client implementations (OpenAI, Cohere, Ollama) and worker pool orchestrators.
#### Go Concurrency Architecture:
- To ingest thousands of chunks without unbounded resource usage or API rate-limit errors, use `golang.org/x/sync/errgroup` wrapped around a channel-based worker pool:

```Go
func EmbedConcurrent(ctx context.Context, embedder Embedder, chunks []Chunk, maxWorkers int, batchSize int) error {
	g, ctx := errgroup.WithContext(ctx)
	chunkChan := make(chan []Chunk)

	// Producer
	g.Go(func() error {
		defer close(chunkChan)
		for i := 0; i < len(chunks); i += batchSize {
			end := min(i+batchSize, len(chunks))
			select {
			case chunkChan <- chunks[i:end]:
			case <-ctx.Done():
				return ctx.Err()
			}
		}
		return nil
	})

	// Bounded Worker Pool
	for w := 0; w < maxWorkers; w++ {
		g.Go(func() error {
			for batch := range chunkChan {
				texts := make([]string, len(batch))
				for i, c := range batch {
					texts[i] = c.Content
				}
				vecs, err := embedder.Embed(ctx, texts)
				if err != nil {
					return err
				}
				for i, v := range vecs {
					batch[i].Embedding = v
				}
			}
			return nil
		})
	}
	return g.Wait()
}
```