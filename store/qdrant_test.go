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

func TestQdrantStore_UpsertAndSearchDense(t *testing.T) {
	var gotUpsertBody map[string]any
	var gotQueryBody map[string]any

	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		switch {
		case r.Method == http.MethodPut && strings.HasSuffix(r.URL.Path, "/points"):
			json.NewDecoder(r.Body).Decode(&gotUpsertBody)
		case r.Method == http.MethodPut:
			w.WriteHeader(http.StatusOK) // create collection
		case r.Method == http.MethodPost && strings.HasSuffix(r.URL.Path, "/points/query"):
			json.NewDecoder(r.Body).Decode(&gotQueryBody)
			json.NewEncoder(w).Encode(map[string]any{
				"result": map[string]any{
					"points": []map[string]any{{
						"id":    "chunk-1",
						"score": 0.9,
						"payload": map[string]any{
							"topic":              "greeting",
							"ragout_content":     "hello world",
							"ragout_document_id": "doc-1",
							"ragout_index":       float64(0),
						},
						"vector": []float32{0.1, 0.2},
					}},
				},
			})
		default:
			t.Fatalf("unexpected request: %s %s", r.Method, r.URL.Path)
		}
	}))
	defer srv.Close()

	ctx := context.Background()
	qs, err := store.NewQdrantStore(ctx, srv.URL, "test-collection", 2)
	if err != nil {
		t.Fatalf("NewQdrantStore: %v", err)
	}

	chunks := []ragout.Chunk{{
		ID:         "chunk-1",
		DocumentID: "doc-1",
		Index:      0,
		Content:    "hello world",
		Embedding:  []float32{0.1, 0.2},
		Metadata:   map[string]any{"topic": "greeting"},
	}}
	if err := qs.Upsert(ctx, chunks); err != nil {
		t.Fatalf("Upsert: %v", err)
	}
	points, _ := gotUpsertBody["points"].([]any)
	if len(points) != 1 {
		t.Fatalf("upsert body points = %v", gotUpsertBody["points"])
	}

	hits, err := qs.SearchDense(ctx, []float32{0.1, 0.2}, 5, map[string]any{"topic": "greeting"})
	if err != nil {
		t.Fatalf("SearchDense: %v", err)
	}
	if len(hits) != 1 {
		t.Fatalf("hits = %+v, want 1", hits)
	}
	if hits[0].Chunk.ID != "chunk-1" || hits[0].Chunk.DocumentID != "doc-1" || hits[0].Chunk.Content != "hello world" {
		t.Fatalf("hit chunk = %+v", hits[0].Chunk)
	}
	if hits[0].Score != 0.9 {
		t.Fatalf("score = %v, want 0.9", hits[0].Score)
	}
	if _, reserved := hits[0].Chunk.Metadata["ragout_content"]; reserved {
		t.Fatalf("reserved payload key leaked into Chunk.Metadata: %+v", hits[0].Chunk.Metadata)
	}
	if filter, _ := gotQueryBody["filter"].(map[string]any); filter == nil {
		t.Fatalf("query body missing filter: %v", gotQueryBody)
	}
}

func TestQdrantStore_Delete(t *testing.T) {
	var gotDeleteBody map[string]any
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		switch {
		case r.Method == http.MethodPut && strings.Contains(r.URL.Path, "/collections/"):
			w.WriteHeader(http.StatusOK)
		case strings.HasSuffix(r.URL.Path, "/points/delete"):
			json.NewDecoder(r.Body).Decode(&gotDeleteBody)
		default:
			t.Fatalf("unexpected request: %s %s", r.Method, r.URL.Path)
		}
	}))
	defer srv.Close()

	ctx := context.Background()
	qs, err := store.NewQdrantStore(ctx, srv.URL, "test-collection", 2)
	if err != nil {
		t.Fatal(err)
	}
	if err := qs.Delete(ctx, []string{"chunk-1", "chunk-2"}); err != nil {
		t.Fatalf("Delete: %v", err)
	}
	points, _ := gotDeleteBody["points"].([]any)
	if len(points) != 2 {
		t.Fatalf("delete body points = %v, want 2 entries", gotDeleteBody["points"])
	}
}

func TestQdrantStore_CollectionAlreadyExists(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusConflict)
	}))
	defer srv.Close()

	if _, err := store.NewQdrantStore(context.Background(), srv.URL, "exists", 2); err != nil {
		t.Fatalf("NewQdrantStore with an existing collection (409) = %v, want nil", err)
	}
}
