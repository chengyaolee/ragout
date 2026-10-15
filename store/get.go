package store

import (
	"context"
	"sync"

	"github.com/chengyaolee/ragout"
)

// getChunks returns copies of the stored chunks for ids, skipping unknown IDs.
func getChunks(ctx context.Context, mu *sync.RWMutex, chunks map[string]ragout.Chunk, ids []string) ([]ragout.Chunk, error) {
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	mu.RLock()
	defer mu.RUnlock()
	out := make([]ragout.Chunk, 0, len(ids))
	for _, id := range ids {
		if c, ok := chunks[id]; ok {
			c.Metadata = ragout.CloneMetadata(c.Metadata)
			out = append(out, c)
		}
	}
	return out, nil
}

// GetChunks returns the stored chunks for ids. Unknown IDs are skipped.
func (vs *VectorStore) GetChunks(ctx context.Context, ids []string) ([]ragout.Chunk, error) {
	return getChunks(ctx, &vs.mu, vs.chunks, ids)
}

// GetChunks returns the indexed chunks for ids. Unknown IDs are skipped.
func (idx *BM25Index) GetChunks(ctx context.Context, ids []string) ([]ragout.Chunk, error) {
	return getChunks(ctx, &idx.mu, idx.chunks, ids)
}
