package reranker

import (
	"context"
	"fmt"
	"math"

	"github.com/chengyaolee/ragout"
	"github.com/chengyaolee/ragout/store"
)

// MMROptions configures the Maximal Marginal Relevance algorithm.
type MMROptions struct {
	Lambda float64
}

// MMRReranker implements ragout.Reranker using Maximal Marginal Relevance.
type MMRReranker struct {
	embedder ragout.Embedder
	lambda   float64
}

// NewMMRReranker creates a new MMR-based reranker.
func NewMMRReranker(embedder ragout.Embedder, lambda float64) *MMRReranker {
	if lambda <= 0 || lambda > 1 {
		lambda = 0.7
	}
	return &MMRReranker{
		embedder: embedder,
		lambda:   lambda,
	}
}

// Rerank reranks candidates using Maximal Marginal Relevance.
func (m *MMRReranker) Rerank(ctx context.Context, query string, candidates []ragout.ScoredChunk, topN int) ([]ragout.ScoredChunk, error) {
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	if len(candidates) == 0 || topN <= 0 {
		return []ragout.ScoredChunk{}, nil
	}

	var queryVector []float32
	if m.embedder != nil {
		vecs, err := m.embedder.EmbedBatch(ctx, []string{query})
		if err != nil {
			return nil, fmt.Errorf("mmr: failed to embed query: %w", err)
		}
		if len(vecs) > 0 {
			queryVector = vecs[0]
		}
	}

	return MaximalMarginalRelevance(queryVector, candidates, topN, &MMROptions{Lambda: m.lambda})
}

// MaximalMarginalRelevance selects chunks that are maximally relevant to the query
// while minimizing redundancy with already selected chunks.
func MaximalMarginalRelevance(queryVector []float32, candidates []ragout.ScoredChunk, topN int, options *MMROptions) ([]ragout.ScoredChunk, error) {
	if len(candidates) == 0 || topN <= 0 {
		return []ragout.ScoredChunk{}, nil
	}
	if topN > len(candidates) {
		topN = len(candidates)
	}
	lambda := 0.7
	if options != nil && options.Lambda != 0 {
		lambda = options.Lambda
	}

	if lambda < 0 || lambda > 1 {
		return nil, fmt.Errorf("lambda must be between 0 and 1")
	}

	selected := make([]ragout.ScoredChunk, 0, topN)
	remaining := make([]ragout.ScoredChunk, len(candidates))
	copy(remaining, candidates)

	for len(selected) < topN && len(remaining) > 0 {
		bestIdx := -1
		bestMMR := -math.MaxFloat64

		for i, candidate := range remaining {
			var querySimilarity float64
			if len(queryVector) > 0 && len(candidate.Chunk.Embedding) > 0 {
				sim, err := store.CosineSimilarity(queryVector, candidate.Chunk.Embedding)
				if err == nil {
					querySimilarity = sim
				} else {
					querySimilarity = candidate.Score
				}
			} else {
				querySimilarity = candidate.Score
			}

			maxSelectedSim := -math.MaxFloat64
			for _, chosen := range selected {
				if len(candidate.Chunk.Embedding) == 0 || len(chosen.Chunk.Embedding) == 0 {
					continue
				}
				sim, err := store.CosineSimilarity(candidate.Chunk.Embedding, chosen.Chunk.Embedding)
				if err != nil {
					continue
				}
				maxSelectedSim = max(maxSelectedSim, sim)
			}

			if len(selected) == 0 || maxSelectedSim == -math.MaxFloat64 {
				maxSelectedSim = 0.0
			}

			mmrScore := lambda*querySimilarity - (1.0-lambda)*maxSelectedSim
			if mmrScore > bestMMR {
				bestMMR = mmrScore
				bestIdx = i
			}
		}

		if bestIdx == -1 {
			break
		}

		selected = append(selected, remaining[bestIdx])
		remaining = append(remaining[:bestIdx], remaining[bestIdx+1:]...)
	}

	return selected, nil
}
