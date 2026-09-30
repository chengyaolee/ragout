package embedder

import (
	"context"

	"github.com/chengyaolee/ragout"
)

// EmbedConcurrent embeds chunks with the engine's bounded worker pool.
func EmbedConcurrent(ctx context.Context, em ragout.Embedder, chunks []ragout.Chunk, maxWorkers int, batchSize int) error {
	return ragout.EmbedConcurrent(ctx, em, chunks, maxWorkers, batchSize)
}
