package reranker

import (
	"fmt"
	"math"

	"github.com/chengyaolee/ragout"
	"github.com/chengyaolee/ragout/store"
)

// Maximal Marginal Relevance (MMR)
// Selects chunks that are maximally relevant to the query, while also being minimally redundant with previously selected chunks
// MMR is a greedy algorithm that iteratively selects the chunk that maximizes the marginal relevance to the query
// and is minimally redundant with previously selected chunks
// MMR is a good choice for reranking because it is a good balance between relevance and diversity

// Arguments:
// queryVector - vector of query terms
// candidates - list of chunks to rerank
// topN - number of chunks to return
// lambda - trade-off between relevance and diversity (Default: lambda = 0.7)
//
// Returns:
// list of reranked chunks
// error if any
type MMROptions struct {
    Lambda float64
}

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
			querySimilarity, err := store.CosineSimilarity(queryVector, candidate.Chunk.Embedding)
			if err != nil {
				return nil, fmt.Errorf("mmr: failed to calculate cosine similarity: %w", err)
			}

			maxSelectedSim := 0.0
			for _, chosen := range selected {
				sim, err := store.CosineSimilarity(candidate.Chunk.Embedding, chosen.Chunk.Embedding)
				if err != nil {
					return nil, fmt.Errorf("mmr: failed to calculate cosine similarity: %w", err)
				}
				maxSelectedSim = max(maxSelectedSim, sim)
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
