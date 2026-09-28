package store

import (
	"context"
	"sort"
	"sync"

	"github.com/chengyaolee/ragout"
)

// VectorStore is an in-memory thread-safe dense vector store.
type VectorStore struct {
	mu     sync.RWMutex
	chunks map[string]ragout.Chunk
}

// NewVectorStore creates an initialized in-memory vector store.
func NewVectorStore() *VectorStore {
	return &VectorStore{
		chunks: make(map[string]ragout.Chunk),
	}
}

// Upsert adds or updates chunks in the vector store.
func (vs *VectorStore) Upsert(ctx context.Context, chunks []ragout.Chunk) error {
	if err := ctx.Err(); err != nil {
		return err
	}

	vs.mu.Lock()
	defer vs.mu.Unlock()

	for _, chunk := range chunks {
		if err := ctx.Err(); err != nil {
			return err
		}
		if chunk.ID == "" {
			continue
		}

		vs.chunks[chunk.ID] = ragout.Chunk{
			ID:         chunk.ID,
			DocumentID: chunk.DocumentID,
			Index:      chunk.Index,
			Content:    chunk.Content,
			Embedding:  chunk.Embedding,
			Metadata:   ragout.CloneMetadata(chunk.Metadata),
		}
	}

	return nil
}

// SearchDense searches for the topK most similar chunks to queryVector using cosine similarity.
func (vs *VectorStore) SearchDense(ctx context.Context, queryVector []float32, topK int, filter map[string]any) ([]ragout.ScoredChunk, error) {
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	if topK <= 0 {
		return []ragout.ScoredChunk{}, nil
	}

	vs.mu.RLock()
	defer vs.mu.RUnlock()

	scoredResults := make([]ragout.ScoredChunk, 0, len(vs.chunks))
	for _, chunk := range vs.chunks {
		if !matches(chunk.Metadata, filter) {
			continue
		}

		sim, err := CosineSimilarity(queryVector, chunk.Embedding)
		if err != nil {
			return nil, err
		}

		scoredResults = append(scoredResults, ragout.ScoredChunk{
			Chunk:      chunk,
			DenseScore: sim,
			Score:      sim,
		})
	}

	// Sort results descending by score
	sort.Slice(scoredResults, func(i, j int) bool {
		return scoredResults[i].Score > scoredResults[j].Score
	})

	if topK > len(scoredResults) {
		topK = len(scoredResults)
	}

	return scoredResults[:topK], nil
}

// CosineSimilarity computes the dot product between two normalized float32 vectors.
func CosineSimilarity(a, b []float32) (float32, error) {
	if len(a) != len(b) {
		return 0.0, ragout.ErrDimensionMismatch
	}

	var dotProduct float32
	for i := range a {
		dotProduct += a[i] * b[i]
	}
	return dotProduct, nil
}