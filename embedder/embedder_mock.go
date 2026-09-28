package embedder

import (
	"context"
	"fmt"
	"time"
	"github.com/chengyaolee/ragout"
)

type MockEmbedder struct {
	Dim     int
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
	select {
	case <-ctx.Done():
		return nil, ctx.Err()
	case <-time.After(m.Latency):
	}
	vecs := make([][]float32, len(texts))
	for i, text := range texts {
		vec := make([]float32, m.Dim)
		h := float32(0)
		for _, c := range text {
			h = h*31 + float32(c)
		}
		for d := 0; d < m.Dim; d++ {
			vec[d] = h + float32(d)
		}
		vecs[i] = vec
	}
	return vecs, nil
}

func (m *MockEmbedder) Dimension() int {
	return m.Dim
}
