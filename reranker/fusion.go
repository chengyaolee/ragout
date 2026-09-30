package reranker

import (
	"math"
	"sort"

	"github.com/chengyaolee/ragout"
)

// Rank-based Fusion

// ReciprocalRankFusion combines two ranked lists of ScoredChunk (dense and sparse) using reciprocal rank fusion.
// Arguments:
//
//	dense - ranked slice of scored chunks from a dense retriever.
//	sparse - ranked slice of scored chunks from a sparse retriever.
//	topK - number of results to return
//	k - smoothing constant (default: k = 60)
//
// Returns:
//
//	A slice of ScoredChunk representing the fused ranking.
func ReciprocalRankFusion(dense []ragout.ScoredChunk, sparse []ragout.ScoredChunk, topK int, k int) []ragout.ScoredChunk {
	return ragout.ReciprocalRankFusion(dense, sparse, topK, k)
}

// Score-based Fusion

// MinMaxNormalization normalizes scores using min-max scaling to [0, 1].
func MinMaxNormalization(chunks []ragout.ScoredChunk) []ragout.ScoredChunk {
	if len(chunks) == 0 {
		return nil
	}

	minimum := chunks[0].Score
	maximum := chunks[0].Score
	for _, chunk := range chunks[1:] {
		minimum = min(minimum, chunk.Score)
		maximum = max(maximum, chunk.Score)
	}

	normalized := make([]ragout.ScoredChunk, len(chunks))
	copy(normalized, chunks)
	if maximum == minimum {
		for i := range normalized {
			normalized[i].Score = 1.0
		}
		return normalized
	}

	span := maximum - minimum
	for i := range normalized {
		normalized[i].Score = (normalized[i].Score - minimum) / span
	}
	return normalized
}

// Logistic Sigmoid Normalization
// Passes raw score through scaled logistic sigmoid
// Outlier spikes smoothly saturates at 1.0 without compressing scores of other chunks
// (vs MinMax which can compress scores of other chunks)
func LogisticSigmoidNormalization(chunks []ragout.ScoredChunk, temperature float64) []ragout.ScoredChunk {
	if len(chunks) == 0 {
		return nil
	}
	if temperature <= 0 {
		temperature = 1.0
	}

	// Copy to avoid mutating input
	normalized := make([]ragout.ScoredChunk, len(chunks))
	copy(normalized, chunks)
	// Apply logistic sigmoid normalization
	for i := range normalized {
		normalized[i].Score = 1.0 / (1.0 + math.Exp(-normalized[i].Score/temperature))
	}
	return normalized
}

// ConvexScoreCombination (alpha-Blending) normalizes scores and blends them:
// Score = alpha * DenseScore + (1 - alpha) * SparseScore
func ConvexScoreCombination(dense []ragout.ScoredChunk, sparse []ragout.ScoredChunk, topK int, alpha float64) []ragout.ScoredChunk {
	if alpha < 0 || alpha > 1 || topK <= 0 {
		return nil
	}

	weight := alpha
	combined := make(map[string]ragout.ScoredChunk)
	for _, chunk := range LogisticSigmoidNormalization(dense, 1) {
		got := combined[chunk.Chunk.ID]
		got.Chunk = chunk.Chunk
		got.DenseScore = chunk.DenseScore
		got.Score = weight * chunk.Score
		combined[chunk.Chunk.ID] = got
	}
	for _, chunk := range LogisticSigmoidNormalization(sparse, 1) {
		got := combined[chunk.Chunk.ID]
		got.Chunk = chunk.Chunk
		got.SparseScore = chunk.SparseScore
		got.Score += (1.0 - weight) * chunk.Score
		combined[chunk.Chunk.ID] = got
	}

	results := make([]ragout.ScoredChunk, 0, len(combined))
	for _, chunk := range combined {
		results = append(results, chunk)
	}
	sort.Slice(results, func(i, j int) bool {
		return results[i].Score > results[j].Score
	})
	if topK > len(results) {
		topK = len(results)
	}
	return results[:topK]
}
