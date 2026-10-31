package reranker

import (
	"math"

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

// ConvexScoreCombination blends one dense list and one sparse list. It is
// ragout.ConvexBlend(alpha, 1) on those two lists. alpha outside [0, 1] or a
// non-positive topK returns nil.
func ConvexScoreCombination(dense []ragout.ScoredChunk, sparse []ragout.ScoredChunk, topK int, alpha float64) []ragout.ScoredChunk {
	if alpha < 0 || alpha > 1 || topK <= 0 {
		return nil
	}
	return ragout.ConvexBlend(alpha, 1).Fuse(
		[]ragout.RankedList{{Dense: true, Hits: dense}, {Hits: sparse}},
		topK,
	)
}
