// Package ragouttest provides deterministic, offline stand-ins for the ragout
// interfaces, for tests and demos that run without an API key or a model server.
package ragouttest

import (
	"context"
	"fmt"
	"iter"
	"sort"
	"strings"
	"time"

	"github.com/chengyaolee/ragout"
)

// Embedder hashes each text into a fixed vector, so equal texts embed equally.
type Embedder struct {
	Dim     int
	Latency time.Duration
}

func NewEmbedder(dim int, latency time.Duration) (*Embedder, error) {
	if dim <= 0 {
		return nil, fmt.Errorf("dim must be greater than 0")
	}
	if latency < 0 {
		return nil, fmt.Errorf("latency must be greater than or equal to 0")
	}
	return &Embedder{Dim: dim, Latency: latency}, nil
}

func (m *Embedder) EmbedBatch(ctx context.Context, texts []string) ([][]float32, error) {
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

func (m *Embedder) Dimension() int {
	return m.Dim
}

// Reranker scores each candidate by how many query words its content contains.
type Reranker struct{}

func NewReranker() *Reranker {
	return &Reranker{}
}

func (m *Reranker) Rerank(ctx context.Context, query string, candidates []ragout.ScoredChunk, topN int) ([]ragout.ScoredChunk, error) {
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	if len(candidates) == 0 || topN <= 0 {
		return []ragout.ScoredChunk{}, nil
	}

	queryTokens := strings.Fields(strings.ToLower(query))
	results := make([]ragout.ScoredChunk, len(candidates))
	copy(results, candidates)

	for i := range results {
		content := strings.ToLower(results[i].Chunk.Content)
		matches := 0
		for _, q := range queryTokens {
			if strings.Contains(content, q) {
				matches++
			}
		}
		results[i].Score = float64(matches)
	}

	sort.Slice(results, func(i, j int) bool {
		return results[i].Score > results[j].Score
	})

	if topN > len(results) {
		topN = len(results)
	}
	return results[:topN], nil
}

// Generator streams scripted tokens, then the retrieved chunk IDs.
// Its sources are the candidates it was given, in order.
type Generator struct {
	tokensToStream []string
}

func NewGenerator(tokens []string) *Generator {
	return &Generator{tokensToStream: tokens}
}

func (mg *Generator) GenerateIter(ctx context.Context, query string, candidates []ragout.ScoredChunk) ([]ragout.ScoredChunk, iter.Seq2[string, error]) {
	sources := append([]ragout.ScoredChunk(nil), candidates...)
	return sources, func(yield func(string, error) bool) {
		for _, token := range mg.tokensToStream {
			if err := ctx.Err(); err != nil {
				yield("", err)
				return
			}
			if !yield(token, nil) {
				return
			}
		}
		for _, c := range candidates {
			if err := ctx.Err(); err != nil {
				yield("", err)
				return
			}
			if !yield(c.Chunk.ID, nil) {
				return
			}
		}
	}
}
