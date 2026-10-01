package store_test

import (
	"bytes"
	"context"
	"path/filepath"
	"testing"

	"github.com/chengyaolee/ragout"
	"github.com/chengyaolee/ragout/store"
)

func testChunks() []ragout.Chunk {
	return []ragout.Chunk{
		{
			ID:        "chunk-1",
			Content:   "The quick brown fox jumps over the lazy dog.",
			Embedding: []float32{1, 0, 0},
			Metadata:  map[string]any{"category": "animals"},
		},
		{
			ID:        "chunk-2",
			Content:   "Deep learning and transformer neural networks.",
			Embedding: []float32{0, 1, 0},
			Metadata:  map[string]any{"category": "ai"},
		},
	}
}

func TestVectorStore_SaveLoadRoundTrip(t *testing.T) {
	ctx := context.Background()
	vs := store.NewVectorStore()
	if err := vs.Upsert(ctx, testChunks()); err != nil {
		t.Fatal(err)
	}

	var buf bytes.Buffer
	if err := vs.Save(&buf); err != nil {
		t.Fatalf("Save: %v", err)
	}

	loaded := store.NewVectorStore()
	if err := loaded.Load(&buf); err != nil {
		t.Fatalf("Load: %v", err)
	}

	hits, err := loaded.SearchDense(ctx, []float32{1, 0, 0}, 2, nil)
	if err != nil {
		t.Fatal(err)
	}
	if len(hits) != 2 || hits[0].Chunk.ID != "chunk-1" {
		t.Fatalf("hits after load = %+v, want chunk-1 ranked first", hits)
	}
}

func TestVectorStore_SaveLoadFile(t *testing.T) {
	ctx := context.Background()
	vs := store.NewVectorStore()
	if err := vs.Upsert(ctx, testChunks()); err != nil {
		t.Fatal(err)
	}

	path := filepath.Join(t.TempDir(), "vector.gob")
	if err := store.SaveFile(path, vs); err != nil {
		t.Fatalf("SaveFile: %v", err)
	}

	loaded := store.NewVectorStore()
	// A missing file must not be an error (first run before any save).
	if err := store.LoadFile(filepath.Join(t.TempDir(), "missing.gob"), loaded); err != nil {
		t.Fatalf("LoadFile(missing) = %v, want nil", err)
	}
	if err := store.LoadFile(path, loaded); err != nil {
		t.Fatalf("LoadFile: %v", err)
	}

	hits, err := loaded.SearchDense(ctx, []float32{0, 1, 0}, 2, nil)
	if err != nil {
		t.Fatal(err)
	}
	if len(hits) != 2 || hits[0].Chunk.ID != "chunk-2" {
		t.Fatalf("hits after LoadFile = %+v, want chunk-2 ranked first", hits)
	}
}

func TestBM25Index_SaveLoadRoundTrip(t *testing.T) {
	ctx := context.Background()
	idx := store.NewBM25Index()
	if err := idx.Index(ctx, testChunks()); err != nil {
		t.Fatal(err)
	}

	before, err := idx.SearchSparse(ctx, "neural networks transformer", 2, nil)
	if err != nil {
		t.Fatal(err)
	}

	var buf bytes.Buffer
	if err := idx.Save(&buf); err != nil {
		t.Fatalf("Save: %v", err)
	}

	loaded := store.NewBM25Index()
	if err := loaded.Load(&buf); err != nil {
		t.Fatalf("Load: %v", err)
	}

	after, err := loaded.SearchSparse(ctx, "neural networks transformer", 2, nil)
	if err != nil {
		t.Fatal(err)
	}
	if len(after) != len(before) || len(after) == 0 || after[0].Chunk.ID != before[0].Chunk.ID {
		t.Fatalf("search after load = %+v, want same ranking as before save: %+v", after, before)
	}
	if after[0].SparseScore != before[0].SparseScore {
		t.Fatalf("BM25 score after load = %v, want %v (stats rebuilt identically)", after[0].SparseScore, before[0].SparseScore)
	}
}

func TestBM25Index_Delete(t *testing.T) {
	ctx := context.Background()
	idx := store.NewBM25Index()
	if err := idx.Index(ctx, testChunks()); err != nil {
		t.Fatal(err)
	}

	if err := idx.Delete(ctx, []string{"chunk-2"}); err != nil {
		t.Fatalf("Delete: %v", err)
	}

	hits, err := idx.SearchSparse(ctx, "neural networks transformer", 5, nil)
	if err != nil {
		t.Fatal(err)
	}
	for _, h := range hits {
		if h.Chunk.ID == "chunk-2" {
			t.Fatalf("deleted chunk-2 still present in search results: %+v", hits)
		}
	}

	// Deleting an unknown ID is a no-op, not an error.
	if err := idx.Delete(ctx, []string{"does-not-exist"}); err != nil {
		t.Fatalf("Delete(unknown) = %v, want nil", err)
	}

	// The remaining document's statistics (avgdl, totalDocuments) must still be
	// consistent: re-indexing it should not change its score.
	before, err := idx.SearchSparse(ctx, "fox dog", 1, nil)
	if err != nil {
		t.Fatal(err)
	}
	if err := idx.Index(ctx, []ragout.Chunk{testChunks()[0]}); err != nil {
		t.Fatal(err)
	}
	after, err := idx.SearchSparse(ctx, "fox dog", 1, nil)
	if err != nil {
		t.Fatal(err)
	}
	if len(before) != 1 || len(after) != 1 || before[0].SparseScore != after[0].SparseScore {
		t.Fatalf("score drifted after delete+reindex: before=%+v after=%+v", before, after)
	}
}

func TestVectorStore_Delete(t *testing.T) {
	ctx := context.Background()
	vs := store.NewVectorStore()
	if err := vs.Upsert(ctx, testChunks()); err != nil {
		t.Fatal(err)
	}
	if err := vs.Delete(ctx, []string{"chunk-1"}); err != nil {
		t.Fatalf("Delete: %v", err)
	}
	hits, err := vs.SearchDense(ctx, []float32{1, 0, 0}, 10, nil)
	if err != nil {
		t.Fatal(err)
	}
	if len(hits) != 1 || hits[0].Chunk.ID != "chunk-2" {
		t.Fatalf("hits after delete = %+v, want only chunk-2", hits)
	}
}
