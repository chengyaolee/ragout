package store

import (
	"context"
	"math"
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

	ptr := ragout.AcquireCandidateSlice()
	scored := (*ptr)[:0]
	release := true
	defer func() {
		if release {
			*ptr = scored
			ragout.ReleaseCandidateSlice(ptr)
		}
	}()

	for _, chunk := range vs.chunks {
		if !matches(chunk.Metadata, filter) {
			continue
		}

		sim, err := CosineSimilarity(queryVector, chunk.Embedding)
		if err != nil {
			return nil, err
		}

		scored = append(scored, ragout.ScoredChunk{
			Chunk:      chunk,
			DenseScore: sim,
			Score:      sim,
		})
	}

	top := ragout.SelectTopK(scored, topK)
	*ptr = top
	release = false
	return *ptr, nil
}

// CosineSimilarity computes the true cosine similarity between two float32 vectors.
// If the vectors are already normalized (L2 norm = 1.0), it computes dot product directly.
// Otherwise, it normalizes by the product of their L2 norms.
func CosineSimilarity(a, b []float32) (float64, error) {
	if len(a) != len(b) {
		return 0.0, ragout.ErrDimensionMismatch
	}

	var dotProduct, normA, normB float64
	for i := range a {
		ai := float64(a[i])
		bi := float64(b[i])
		dotProduct += ai * bi
		normA += ai * ai
		normB += bi * bi
	}

	if normA == 0 || normB == 0 {
		return 0.0, nil
	}

	if math.Abs(normA-1.0) < 1e-4 && math.Abs(normB-1.0) < 1e-4 {
		return dotProduct, nil
	}

	return dotProduct / (math.Sqrt(normA) * math.Sqrt(normB)), nil
}
