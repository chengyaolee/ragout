package embedder_test

import (
	"context"
	"errors"
	"fmt"
	"sync/atomic"
	"testing"
	"time"

	"github.com/chengyaolee/ragout"
)

func TestMockEmbedder(t *testing.T) {
	m, err := embedder.NewMockEmbedder(4, 50*time.Millisecond)
	if err != nil {
		t.Fatal(err)
	}
	if m.Dimension() != 4 {
		t.Fatalf("dim = %d, want 4", m.Dimension())
	}
	start := time.Now()
	first, err := m.EmbedBatch(context.Background(), []string{"cat", "cat", "dog"})
	if err != nil {
		t.Fatal(err)
	}
	if time.Since(start) < 40*time.Millisecond {
		t.Fatal("EmbedBatch returned before the configured latency")
	}
	second, err := m.EmbedBatch(context.Background(), []string{"cat", "cat", "dog"})
	if err != nil {
		t.Fatal(err)
	}
	if len(first) != 3 || len(first[0]) != 4 {
		t.Fatalf("vectors = %d x %d, want 3 x 4", len(first), len(first[0]))
	}
	for d := range first[0] {
		if first[0][d] != first[1][d] || first[0][d] != second[0][d] {
			t.Fatalf("same text produced different vectors")
		}
		if first[0][d] == first[2][d] {
			t.Fatalf("different texts produced the same vector component %d", d)
		}
	}
}

func BenchmarkEmbedConcurrent(b *testing.B) {
	const (
		n         = 1000
		batchSize = 32
	)
	for _, workers := range []int{1, 4, 16} {
		b.Run(fmt.Sprintf("workers=%d", workers), func(b *testing.B) {
			m, err := embedder.NewMockEmbedder(8, 50*time.Millisecond)
			if err != nil {
				b.Fatal(err)
			}
			chunks := makeChunks(n)
			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				if err := embedder.EmbedConcurrent(context.Background(), m, chunks, workers, batchSize); err != nil {
					b.Fatal(err)
				}
			}
		})
	}
}

func TestEmbedConcurrent(t *testing.T) {
	t.Run("ten chunks batch three", func(t *testing.T) {
		chunks := makeChunks(10)
		contents := contentsOf(chunks)
		m := &script{dim: 4}
		err := embedder.EmbedConcurrent(context.Background(), m, chunks, 2, 3)
		if err != nil {
			t.Fatalf("EmbedConcurrent: %v", err)
		}
		if got := m.calls.Load(); got != 4 {
			t.Fatalf("batches = %d, want 4", got)
		}
		for i, c := range chunks {
			if c.Content != contents[i] {
				t.Fatalf("chunk %d content changed", i)
			}
			if len(c.Embedding) != m.Dimension() {
				t.Fatalf("chunk %d embedding len = %d, want %d", i, len(c.Embedding), m.Dimension())
			}
		}
	})

	t.Run("batch larger than input", func(t *testing.T) {
		chunks := makeChunks(5)
		m := &script{dim: 4}
		if err := embedder.EmbedConcurrent(context.Background(), m, chunks, 2, 100); err != nil {
			t.Fatalf("EmbedConcurrent: %v", err)
		}
		if got := m.calls.Load(); got != 1 {
			t.Fatalf("batches = %d, want 1", got)
		}
		for i, c := range chunks {
			if len(c.Embedding) != m.Dimension() {
				t.Fatalf("chunk %d embedding len = %d", i, len(c.Embedding))
			}
		}
	})

	t.Run("cancel mid-run", func(t *testing.T) {
		ctx, cancel := context.WithCancel(context.Background())
		defer cancel()
		m := &script{dim: 4, latency: 50 * time.Millisecond}
		time.AfterFunc(20*time.Millisecond, cancel)

		err := callWithTimeout(t, time.Second, func() error {
			return embedder.EmbedConcurrent(ctx, m, makeChunks(10), 2, 3)
		})
		if !errors.Is(err, context.Canceled) && !errors.Is(err, ctx.Err()) {
			t.Fatalf("err = %v, want context.Canceled", err)
		}
	})

	t.Run("error on second call", func(t *testing.T) {
		want := errors.New("mock failure on call 2")
		m := &script{dim: 4, failOn: 2, fail: want}
		err := callWithTimeout(t, time.Second, func() error {
			return embedder.EmbedConcurrent(context.Background(), m, makeChunks(10), 2, 3)
		})
		if !errors.Is(err, want) {
			t.Fatalf("err = %v, want %v", err, want)
		}
	})

	t.Run("empty", func(t *testing.T) {
		m := &script{dim: 4}
		if err := embedder.EmbedConcurrent(context.Background(), m, nil, 2, 3); err != nil {
			t.Fatalf("EmbedConcurrent: %v", err)
		}
		if got := m.calls.Load(); got != 0 {
			t.Fatalf("calls = %d, want 0", got)
		}
	})

	t.Run("bad vectors", func(t *testing.T) {
		t.Run("count", func(t *testing.T) {
			m := &script{dim: 4, vecs: func(texts []string) [][]float32 {
				return [][]float32{make([]float32, 4)} // one vector for many texts
			}}
			err := embedder.EmbedConcurrent(context.Background(), m, makeChunks(3), 1, 3)
			if !errors.Is(err, ragout.ErrCountMismatch) {
				t.Fatalf("err = %v, want ErrCountMismatch", err)
			}
		})
		t.Run("dimension", func(t *testing.T) {
			m := &script{dim: 4, vecs: func(texts []string) [][]float32 {
				out := make([][]float32, len(texts))
				for i := range texts {
					out[i] = make([]float32, 2)
				}
				return out
			}}
			err := embedder.EmbedConcurrent(context.Background(), m, makeChunks(3), 1, 3)
			if !errors.Is(err, ragout.ErrDimensionMismatch) {
				t.Fatalf("err = %v, want ErrDimensionMismatch", err)
			}
		})
	})
}

type script struct {
	dim     int
	latency time.Duration
	failOn  int
	fail    error
	vecs    func([]string) [][]float32
	calls   atomic.Int32
}

func (s *script) Dimension() int { return s.dim }

func (s *script) EmbedBatch(ctx context.Context, texts []string) ([][]float32, error) {
	n := int(s.calls.Add(1))
	if s.latency > 0 {
		select {
		case <-ctx.Done():
			return nil, ctx.Err()
		case <-time.After(s.latency):
		}
	}
	if s.failOn > 0 && n == s.failOn {
		return nil, s.fail
	}
	if s.vecs != nil {
		return s.vecs(texts), nil
	}
	out := make([][]float32, len(texts))
	for i := range texts {
		out[i] = make([]float32, s.dim)
	}
	return out, nil
}

func makeChunks(n int) []ragout.Chunk {
	chunks := make([]ragout.Chunk, n)
	for i := range chunks {
		chunks[i].Content = string(rune('a' + i))
	}
	return chunks
}

func contentsOf(chunks []ragout.Chunk) []string {
	out := make([]string, len(chunks))
	for i, c := range chunks {
		out[i] = c.Content
	}
	return out
}

func callWithTimeout(t *testing.T, d time.Duration, fn func() error) error {
	t.Helper()
	done := make(chan error, 1)
	go func() { done <- fn() }()
	select {
	case err := <-done:
		return err
	case <-time.After(d):
		t.Fatal("EmbedConcurrent did not return")
		return nil
	}
}
