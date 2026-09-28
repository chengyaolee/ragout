package store_test

import (
	"context"
	"math"
	"testing"

	"github.com/chengyaolee/ragout"
	"github.com/chengyaolee/ragout/store"
)

const floatTolerance = 1e-4

func almostEqual(a, b float32) bool {
	return math.Abs(float64(a-b)) <= floatTolerance
}

func TestCosineSimilarity(t *testing.T) {
	t.Run("IdenticalVectors", func(t *testing.T) {
		// Both normalized
		v1 := []float32{1.0, 0.0, 0.0}
		v2 := []float32{1.0, 0.0, 0.0}

		sim, err := store.CosineSimilarity(v1, v2)
		if err != nil {
			t.Fatalf("expected no error, got %v", err)
		}
		if !almostEqual(sim, 1.0) {
			t.Errorf("expected 1.0, got %f", sim)
		}
	})

	t.Run("OrthogonalVectors", func(t *testing.T) {
		v1 := []float32{1.0, 0.0, 0.0}
		v2 := []float32{0.0, 1.0, 0.0}

		sim, err := store.CosineSimilarity(v1, v2)
		if err != nil {
			t.Fatalf("expected no error, got %v", err)
		}
		if !almostEqual(sim, 0.0) {
			t.Errorf("expected 0.0, got %f", sim)
		}
	})

	t.Run("OppositeVectors", func(t *testing.T) {
		v1 := []float32{1.0, 0.0}
		v2 := []float32{-1.0, 0.0}

		sim, err := store.CosineSimilarity(v1, v2)
		if err != nil {
			t.Fatalf("expected no error, got %v", err)
		}
		if !almostEqual(sim, -1.0) {
			t.Errorf("expected -1.0, got %f", sim)
		}
	})

	t.Run("DimensionMismatch", func(t *testing.T) {
		v1 := []float32{1.0, 0.0}
		v2 := []float32{1.0, 0.0, 0.5}

		_, err := store.CosineSimilarity(v1, v2)
		if err == nil {
			t.Fatal("expected error on dimension mismatch, got nil")
		}
	})
}

func TestVectorStore_UpsertAndSearchDense(t *testing.T) {
	var vs ragout.VectorStore = store.NewVectorStore()

	chunks := []ragout.Chunk{
		{
			ID:        "chunk-1",
			Content:   "Quantum computing and entanglement",
			Embedding: []float32{0.9, 0.1, 0.0},
			Metadata:  map[string]any{"topic": "quantum"},
		},
		{
			ID:        "chunk-2",
			Content:   "Deep learning and neural networks",
			Embedding: []float32{0.1, 0.9, 0.0},
			Metadata:  map[string]any{"topic": "ai"},
		},
		{
			ID:        "chunk-3",
			Content:   "Database indexing with B-Trees",
			Embedding: []float32{0.0, 0.1, 0.9},
			Metadata:  map[string]any{"topic": "database"},
		},
	}

	ctx := context.Background()
	if err := vs.Upsert(ctx, chunks); err != nil {
		t.Fatalf("Upsert failed: %v", err)
	}

	// Query closest to chunk-2 (AI)
	queryVector := []float32{0.2, 0.8, 0.0}
	results, err := vs.SearchDense(ctx, queryVector, 2, nil)
	if err != nil {
		t.Fatalf("SearchDense failed: %v", err)
	}

	if len(results) != 2 {
		t.Fatalf("expected 2 results, got %d", len(results))
	}

	if results[0].Chunk.ID != "chunk-2" {
		t.Errorf("expected top chunk to be 'chunk-2', got %q", results[0].Chunk.ID)
	}

	if results[0].DenseScore <= results[1].DenseScore {
		t.Errorf("expected results to be sorted descending: got %f <= %f", results[0].DenseScore, results[1].DenseScore)
	}

	if results[0].Score != results[0].DenseScore {
		t.Errorf("expected Score to match DenseScore: got %f != %f", results[0].Score, results[0].DenseScore)
	}
}

func TestVectorStore_MetadataFilter(t *testing.T) {
	vs := store.NewVectorStore()

	chunks := []ragout.Chunk{
		{
			ID:        "chunk-1",
			Content:   "AI article public",
			Embedding: []float32{1.0, 0.0},
			Metadata:  map[string]any{"access": "public", "lang": "en"},
		},
		{
			ID:        "chunk-2",
			Content:   "AI article private",
			Embedding: []float32{1.0, 0.0},
			Metadata:  map[string]any{"access": "private", "lang": "en"},
		},
	}

	ctx := context.Background()
	if err := vs.Upsert(ctx, chunks); err != nil {
		t.Fatalf("Upsert failed: %v", err)
	}

	filter := map[string]any{"access": "public"}
	results, err := vs.SearchDense(ctx, []float32{1.0, 0.0}, 10, filter)
	if err != nil {
		t.Fatalf("SearchDense failed: %v", err)
	}

	if len(results) != 1 {
		t.Fatalf("expected 1 result after filter, got %d", len(results))
	}
	if results[0].Chunk.ID != "chunk-1" {
		t.Errorf("expected 'chunk-1', got %q", results[0].Chunk.ID)
	}
}

func TestVectorStore_EdgeCases(t *testing.T) {
	vs := store.NewVectorStore()
	ctx := context.Background()

	t.Run("EmptyStore", func(t *testing.T) {
		res, err := vs.SearchDense(ctx, []float32{1.0, 0.0}, 5, nil)
		if err != nil {
			t.Fatalf("expected no error on empty store, got %v", err)
		}
		if len(res) != 0 {
			t.Errorf("expected 0 results, got %d", len(res))
		}
	})

	t.Run("ZeroOrNegativeTopK", func(t *testing.T) {
		res, err := vs.SearchDense(ctx, []float32{1.0, 0.0}, 0, nil)
		if err != nil {
			t.Fatalf("expected no error, got %v", err)
		}
		if len(res) != 0 {
			t.Errorf("expected empty slice for topK=0, got %d", len(res))
		}
	})

	t.Run("ContextCanceled", func(t *testing.T) {
		canceledCtx, cancel := context.WithCancel(context.Background())
		cancel()

		_, err := vs.SearchDense(canceledCtx, []float32{1.0, 0.0}, 5, nil)
		if err == nil {
			t.Error("expected context canceled error, got nil")
		}
	})
}
