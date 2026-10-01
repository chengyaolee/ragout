package store_test

import (
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"github.com/chengyaolee/ragout"
	"github.com/chengyaolee/ragout/store"
)

func TestChromaStore_UpsertAndSearchDense(t *testing.T) {
	var gotUpsertBody map[string]any
	var gotQueryBody map[string]any

	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		switch {
		case r.Method == http.MethodPost && strings.HasSuffix(r.URL.Path, "/collections"):
			json.NewEncoder(w).Encode(map[string]any{"id": "col-1"})
		case r.Method == http.MethodPost && strings.HasSuffix(r.URL.Path, "/upsert"):
			json.NewDecoder(r.Body).Decode(&gotUpsertBody)
			w.WriteHeader(http.StatusOK)
		case r.Method == http.MethodPost && strings.HasSuffix(r.URL.Path, "/query"):
			json.NewDecoder(r.Body).Decode(&gotQueryBody)
			json.NewEncoder(w).Encode(map[string]any{
				"ids":       [][]string{{"chunk-1"}},
				"distances": [][]float64{{0.2}},
				"documents": [][]string{{"hello world"}},
				"metadatas": [][]map[string]any{{{"topic": "greeting", "ragout_document_id": "doc-1", "ragout_index": float64(0)}}},
			})
		default:
			t.Fatalf("unexpected request: %s %s", r.Method, r.URL.Path)
		}
	}))
	defer srv.Close()

	ctx := context.Background()
	cs, err := store.NewChromaStore(ctx, srv.URL, "test-collection")
	if err != nil {
		t.Fatalf("NewChromaStore: %v", err)
	}

	chunks := []ragout.Chunk{{
		ID:         "chunk-1",
		DocumentID: "doc-1",
		Index:      0,
		Content:    "hello world",
		Embedding:  []float32{0.1, 0.2},
		Metadata:   map[string]any{"topic": "greeting"},
	}}
	if err := cs.Upsert(ctx, chunks); err != nil {
		t.Fatalf("Upsert: %v", err)
	}
	if ids, _ := gotUpsertBody["ids"].([]any); len(ids) != 1 || ids[0] != "chunk-1" {
		t.Fatalf("upsert body ids = %v", gotUpsertBody["ids"])
	}

	hits, err := cs.SearchDense(ctx, []float32{0.1, 0.2}, 5, map[string]any{"topic": "greeting"})
	if err != nil {
		t.Fatalf("SearchDense: %v", err)
	}
	if len(hits) != 1 {
		t.Fatalf("hits = %+v, want 1", hits)
	}
	if hits[0].Chunk.ID != "chunk-1" || hits[0].Chunk.DocumentID != "doc-1" || hits[0].Chunk.Content != "hello world" {
		t.Fatalf("hit chunk = %+v", hits[0].Chunk)
	}
	if hits[0].Score != 0.8 { // 1 - distance
		t.Fatalf("score = %v, want 0.8", hits[0].Score)
	}
	if _, reserved := hits[0].Chunk.Metadata["ragout_document_id"]; reserved {
		t.Fatalf("reserved metadata key leaked into Chunk.Metadata: %+v", hits[0].Chunk.Metadata)
	}
	if where, _ := gotQueryBody["where"].(map[string]any); where == nil {
		t.Fatalf("query body missing where clause: %v", gotQueryBody)
	}
}

func TestChromaStore_Delete(t *testing.T) {
	var gotDeleteBody map[string]any
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		switch {
		case strings.HasSuffix(r.URL.Path, "/collections"):
			json.NewEncoder(w).Encode(map[string]any{"id": "col-1"})
		case strings.HasSuffix(r.URL.Path, "/delete"):
			json.NewDecoder(r.Body).Decode(&gotDeleteBody)
		default:
			t.Fatalf("unexpected request: %s", r.URL.Path)
		}
	}))
	defer srv.Close()

	ctx := context.Background()
	cs, err := store.NewChromaStore(ctx, srv.URL, "test-collection")
	if err != nil {
		t.Fatal(err)
	}
	if err := cs.Delete(ctx, []string{"chunk-1", "chunk-2"}); err != nil {
		t.Fatalf("Delete: %v", err)
	}
	ids, _ := gotDeleteBody["ids"].([]any)
	if len(ids) != 2 {
		t.Fatalf("delete body ids = %v, want 2 entries", gotDeleteBody["ids"])
	}
}
