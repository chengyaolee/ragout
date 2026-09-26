package embedder

import (
	"context"
)

type Embedder interface {
	// EmbedBatch takes a slice of strings and returns their float32 vectors.
	EmbedBatch(ctx context.Context, texts []string) ([][]float32, error)
	Dimension() int
}


