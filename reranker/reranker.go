package reranker

import (
	"context"
	"sort"
	"strings"

	"github.com/chengyaolee/ragout"
)

type MockReranker struct{}

func NewMockReranker() *MockReranker {
	return &MockReranker{}
}

func (m *MockReranker) Rerank(ctx context.Context, query string, candidates []ragout.ScoredChunk, topN int) ([]ragout.ScoredChunk, error) {
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
