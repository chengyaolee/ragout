package embedder

import (
	"context"
	"fmt"
	"time"

)

type MockEmbedder struct {
	Dim int
	Latency time.Duration

}

func NewMockEmbedder(dim int, latency time.Duration) (*MockEmbedder, error) {
	if dim <= 0 {
		return nil, fmt.Errorf("dim must be greater than 0")
	}
	if latency < 0 {
		return nil, fmt.Errorf("latency must be greater than or equal to 0")
	}
	return &MockEmbedder{Dim: dim, Latency: latency}, nil
}


func (m *MockEmbedder) EmbedBatch(ctx context.Context, texts []string) ([][]float32, error) {

	// For testing: Purposely fail on the 10th call to EmbedBatch
	var embedBatchCallCount int
	failOnCall := 10

	embedBatchCallCount++
	if embedBatchCallCount == failOnCall {
		return nil, fmt.Errorf("mock failure on EmbedBatch call %d", embedBatchCallCount)
	}

	select {
	case <-ctx.Done():
		return nil, ctx.Err()
	case <-time.After(m.Latency):
		vecs := make([][]float32, len(texts))
		for i, text := range texts {
			vec := make([]float32, m.Dim)
			for d := 0; d < m.Dim; d++ {
				// Not actually implemented, placeholder to fill vector from input
				vec[d] = float32(len(text) + d)
			}
			vecs[i] = vec
		}
		return vecs, nil
	}
}

func (m *MockEmbedder) Dimension() int {
	return m.Dim
}
