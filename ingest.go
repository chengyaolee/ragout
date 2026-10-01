package ragout

import (
	"context"
	"errors"
	"fmt"
	"io"

	"golang.org/x/sync/errgroup"
)

// Ingest streams raw data from source (e.g. a filename, used to pick a reader by
// extension and recorded on every chunk's metadata[MetadataSource]), chunks it,
// generates embeddings in parallel, and indexes into stores.
func (e *Engine) Ingest(ctx context.Context, source string, r io.Reader, metadata map[string]any) error {
	if e.reader == nil {
		return ErrNilReader
	}
	if e.chunker == nil {
		return ErrNilChunker
	}
	if err := ctx.Err(); err != nil {
		return err
	}

	meta := CloneMetadata(metadata)
	meta[MetadataSource] = source

	// 1. Read documents from stream with DoS boundaries
	docs, err := e.reader.Read(ctx, r, meta)
	if err != nil {
		return fmt.Errorf("ragout.Ingest: read error: %w", err)
	}
	if len(docs) == 0 {
		return ErrEmptyDocument
	}

	// 2. Chunk documents. IDs are derived from (source, content) so that retrying a
	// failed ingest, or re-ingesting the same file, overwrites the same chunks instead
	// of duplicating them.
	var allChunks []Chunk
	for _, doc := range docs {
		if doc.ID == "" {
			doc.ID = DocumentID(source, doc.Content)
		}
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
		// One store may have already committed the chunks before the other failed.
		// Roll both back using the chunks' deterministic IDs so a retry can't leave
		// a duplicate or half-indexed document behind.
		return e.rollbackIngest(allChunks, err)
	}

	IngestedChunksTotal.Add(float64(len(allChunks)))
	return nil
}

func (e *Engine) rollbackIngest(chunks []Chunk, writeErr error) error {
	ids := make([]string, len(chunks))
	for i, c := range chunks {
		ids[i] = c.ID
	}

	// Use a fresh, uncancelled context: the write failed (or its context was
	// cancelled), but the rollback must still run to avoid leaving a partial index.
	rbCtx := context.WithoutCancel(context.Background())
	errs := []error{fmt.Errorf("ragout.Ingest: storage indexing error: %w", writeErr)}

	if e.vStore != nil {
		if err := e.vStore.Delete(rbCtx, ids); err != nil {
			errs = append(errs, fmt.Errorf("ragout.Ingest: vector store rollback failed: %w", err))
		}
	}
	if e.iStore != nil {
		if err := e.iStore.Delete(rbCtx, ids); err != nil {
			errs = append(errs, fmt.Errorf("ragout.Ingest: index store rollback failed: %w", err))
		}
	}

	return errors.Join(errs...)
}
