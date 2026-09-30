package ragout

import (
	"context"
	"fmt"
	"io"

	"golang.org/x/sync/errgroup"
)

// Ingest streams raw data, chunks it, generates embeddings in parallel, and indexes into stores.
func (e *Engine) Ingest(ctx context.Context, r io.Reader, metadata map[string]any) error {
	if e.reader == nil {
		return ErrNilReader
	}
	if e.chunker == nil {
		return ErrNilChunker
	}
	if err := ctx.Err(); err != nil {
		return err
	}

	// 1. Read documents from stream with DoS boundaries
	docs, err := e.reader.Read(ctx, r, metadata)
	if err != nil {
		return fmt.Errorf("ragout.Ingest: read error: %w", err)
	}
	if len(docs) == 0 {
		return ErrEmptyDocument
	}

	// 2. Chunk documents
	var allChunks []Chunk
	for _, doc := range docs {
		chunks, err := e.chunker.Chunk(ctx, doc)
		if err != nil {
			return fmt.Errorf("ragout.Ingest: chunk error: %w", err)
		}
		allChunks = append(allChunks, chunks...)
	}
	if len(allChunks) == 0 {
		return ErrEmptyChunk
	}

	// 3. Batch concurrent embedding (if VectorStore is enabled)
	if e.vStore != nil && e.embedder != nil {
		// EmbedConcurrent uses bounded worker pool from Phase 3
		if err := embedConcurrent(ctx, e.embedder, allChunks, e.embedConcurrency, e.embedBatchSize); err != nil {
			return fmt.Errorf("ragout.Ingest: embedding error: %w", err)
		}
	}

	// 4. Concurrent write to VectorStore and IndexStore
	g, writeCtx := errgroup.WithContext(ctx)

	if e.vStore != nil {
		g.Go(func() error {
			return e.vStore.Upsert(writeCtx, allChunks)
		})
	}

	if e.iStore != nil {
		g.Go(func() error {
			return e.iStore.Index(writeCtx, allChunks)
		})
	}

	if err := g.Wait(); err != nil {
		return fmt.Errorf("ragout.Ingest: storage indexing error: %w", err)
	}

	IngestedChunksTotal.Add(float64(len(allChunks)))
	return nil
}
