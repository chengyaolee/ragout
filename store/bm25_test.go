package store_test

import (
	"context"
	"slices"
	"testing"

	"github.com/chengyaolee/ragout"
	"github.com/chengyaolee/ragout/store"
)

func TestTokenize(t *testing.T) {
	input := "Hello, World! This is a Go-based (BM25) Tokenizer... Test 123."
	expected := []string{"hello", "world", "this", "is", "a", "go", "based", "bm25", "tokenizer", "test", "123"}

	tokens := store.Tokenize(input)
	if !slices.Equal(tokens, expected) {
		t.Fatalf("Tokenize mismatch:\ngot:  %v\nwant: %v", tokens, expected)
	}
}

func TestBM25Index_IndexAndSearchSparse(t *testing.T) {
	var idx ragout.IndexStore = store.NewBM25Index()

	chunks := []ragout.Chunk{
		{
			ID:       "chunk-1",
			Content:  "The quick brown fox jumps over the lazy dog.",
			Metadata: map[string]any{"category": "animals"},
		},
		{
			ID:       "chunk-2",
			Content:  "Deep learning and transformer neural networks revolutionize language models.",
			Metadata: map[string]any{"category": "ai"},
		},
		{
			ID:       "chunk-3",
			Content:  "Concurrent programming in Go with goroutines and channels.",
			Metadata: map[string]any{"category": "golang"},
		},
	}

	ctx := context.Background()
	if err := idx.Index(ctx, chunks); err != nil {
		t.Fatalf("Index failed: %v", err)
	}

	results, err := idx.SearchSparse(ctx, "neural networks transformer", 2, nil)
	if err != nil {
		t.Fatalf("SearchSparse failed: %v", err)
	}

	if len(results) == 0 {
		t.Fatal("expected at least 1 result, got 0")
	}

	if results[0].Chunk.ID != "chunk-2" {
		t.Errorf("expected top result to be 'chunk-2', got %q", results[0].Chunk.ID)
	}

	if results[0].SparseScore <= 0 {
		t.Errorf("expected positive SparseScore, got %f", results[0].SparseScore)
	}

	if results[0].Score != results[0].SparseScore {
		t.Errorf("expected Score to match SparseScore: %f != %f", results[0].Score, results[0].SparseScore)
	}
}

func TestBM25Index_RareTermIDF(t *testing.T) {
	idx := store.NewBM25Index()

	// "database" is common to all chunks; "astrophysics" is rare (only in chunk-1)
	chunks := []ragout.Chunk{
		{
			ID:      "chunk-1",
			Content: "Database systems for astrophysics research telemetry.",
		},
		{
			ID:      "chunk-2",
			Content: "Relational database systems with transactional ACID guarantees.",
		},
		{
			ID:      "chunk-3",
			Content: "Distributed database architecture for cloud storage systems.",
		},
	}

	ctx := context.Background()
	if err := idx.Index(ctx, chunks); err != nil {
		t.Fatalf("Index failed: %v", err)
	}

	// Query has both a common term and a rare term
	results, err := idx.SearchSparse(ctx, "astrophysics database", 3, nil)
	if err != nil {
		t.Fatalf("SearchSparse failed: %v", err)
	}

	if len(results) == 0 {
		t.Fatal("expected results, got 0")
	}

	// Chunk-1 must rank #1 because "astrophysics" has a much higher IDF
	if results[0].Chunk.ID != "chunk-1" {
		t.Errorf("expected rare-term chunk-1 to rank first, got %q", results[0].Chunk.ID)
	}
}

func TestBM25Index_LengthNormalization(t *testing.T) {
	idx := store.NewBM25Index()

	// Both chunks contain "microservices architecture", but chunk-short is concise
	chunks := []ragout.Chunk{
		{
			ID:      "chunk-short",
			Content: "Microservices architecture design patterns.",
		},
		{
			ID:      "chunk-long",
			Content: "Microservices architecture is an organizational approach to software development where software is composed of small independent services that communicate over well-defined APIs owned by small self-contained teams.",
		},
	}

	ctx := context.Background()
	if err := idx.Index(ctx, chunks); err != nil {
		t.Fatalf("Index failed: %v", err)
	}

	results, err := idx.SearchSparse(ctx, "microservices architecture", 2, nil)
	if err != nil {
		t.Fatalf("SearchSparse failed: %v", err)
	}

	if len(results) != 2 {
		t.Fatalf("expected 2 results, got %d", len(results))
	}

	// BM25 penalizes bloated document length, so concise chunk should rank higher
	if results[0].Chunk.ID != "chunk-short" {
		t.Errorf("expected concise document 'chunk-short' to rank #1 due to length normalization, got %q", results[0].Chunk.ID)
	}
}

func TestBM25Index_MetadataFilter(t *testing.T) {
	idx := store.NewBM25Index()

	chunks := []ragout.Chunk{
		{
			ID:       "chunk-public",
			Content:  "Kubernetes deployment guidelines.",
			Metadata: map[string]any{"env": "prod", "access": "public"},
		},
		{
			ID:       "chunk-private",
			Content:  "Kubernetes deployment guidelines.",
			Metadata: map[string]any{"env": "dev", "access": "private"},
		},
	}

	ctx := context.Background()
	if err := idx.Index(ctx, chunks); err != nil {
		t.Fatalf("Index failed: %v", err)
	}

	results, err := idx.SearchSparse(ctx, "kubernetes", 10, map[string]any{"access": "public"})
	if err != nil {
		t.Fatalf("SearchSparse failed: %v", err)
	}

	if len(results) != 1 {
		t.Fatalf("expected 1 filtered result, got %d", len(results))
	}
	if results[0].Chunk.ID != "chunk-public" {
		t.Errorf("expected 'chunk-public', got %q", results[0].Chunk.ID)
	}
}

func TestBM25Index_EdgeCases(t *testing.T) {
	idx := store.NewBM25Index()
	ctx := context.Background()

	t.Run("EmptyIndex", func(t *testing.T) {
		results, err := idx.SearchSparse(ctx, "anything", 5, nil)
		if err != nil {
			t.Fatalf("expected no error on empty index, got %v", err)
		}
		if len(results) != 0 {
			t.Errorf("expected 0 results, got %d", len(results))
		}
	})

	t.Run("NonExistentTerms", func(t *testing.T) {
		err := idx.Index(ctx, []ragout.Chunk{
			{ID: "c1", Content: "Apples and oranges fruit salad"},
		})
		if err != nil {
			t.Fatalf("Index failed: %v", err)
		}

		results, err := idx.SearchSparse(ctx, "quantum superconductor", 5, nil)
		if err != nil {
			t.Fatalf("SearchSparse failed: %v", err)
		}
		if len(results) != 0 {
			t.Errorf("expected 0 results for non-existent terms, got %d", len(results))
		}
	})

	t.Run("ZeroOrNegativeTopK", func(t *testing.T) {
		results, err := idx.SearchSparse(ctx, "apples", 0, nil)
		if err != nil {
			t.Fatalf("expected no error, got %v", err)
		}
		if len(results) != 0 {
			t.Errorf("expected empty slice for topK=0, got %d", len(results))
		}
	})

	t.Run("ContextCanceled", func(t *testing.T) {
		canceledCtx, cancel := context.WithCancel(context.Background())
		cancel()

		_, err := idx.SearchSparse(canceledCtx, "apples", 5, nil)
		if err == nil {
			t.Error("expected error with canceled context, got nil")
		}
	})
}

func TestBM25Index_TermsMatched(t *testing.T) {
	idx := store.NewBM25Index()
	if err := idx.Index(context.Background(), []ragout.Chunk{
		{ID: "1", Content: "The quick brown fox"},
	}); err != nil {
		t.Fatal(err)
	}
	if got := idx.TermsMatched("quick fox missing"); got != 2 {
		t.Fatalf("TermsMatched = %d, want 2", got)
	}
	if got := idx.TermsMatched("quick quick"); got != 1 {
		t.Fatalf("duplicate term count = %d, want 1", got)
	}
	if got := idx.TermsMatched("zzzz"); got != 0 {
		t.Fatalf("missing term count = %d, want 0", got)
	}
}
