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

func TestInMemoryGetChunks(t *testing.T) {
	ctx := context.Background()
	chunks := []ragout.Chunk{{ID: "a", DocumentID: "d", Index: 0, Content: "alpha", Embedding: []float32{1}}}

	vs := store.NewVectorStore()
	idx := store.NewBM25Index()
	if err := vs.Upsert(ctx, chunks); err != nil {
		t.Fatal(err)
	}
	if err := idx.Index(ctx, chunks); err != nil {
		t.Fatal(err)
	}
	for name, g := range map[string]ragout.ChunkGetter{"vector": vs, "bm25": idx} {
		got, err := g.GetChunks(ctx, []string{"missing", "a"})
		if err != nil || len(got) != 1 || got[0].Content != "alpha" || got[0].DocumentID != "d" {
			t.Errorf("%s GetChunks = %+v, %v", name, got, err)
		}
	}
}

func TestChromaStore_GetChunks(t *testing.T) {
	var body map[string]any
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		switch {
		case strings.HasSuffix(r.URL.Path, "/collections"):
			json.NewEncoder(w).Encode(map[string]any{"id": "col-1"})
		case strings.HasSuffix(r.URL.Path, "/col-1/get"):
			json.NewDecoder(r.Body).Decode(&body)
			json.NewEncoder(w).Encode(map[string]any{
				"ids":       []string{"c1"},
				"documents": []string{"hello"},
				"metadatas": []map[string]any{{"topic": "x", "ragout_document_id": "d1", "ragout_index": float64(3)}},
			})
		default:
			t.Fatalf("unexpected request: %s %s", r.Method, r.URL.Path)
		}
	}))
	defer srv.Close()

	ctx := context.Background()
	cs, err := store.NewChromaStore(ctx, srv.URL, "c")
	if err != nil {
		t.Fatal(err)
	}
	got, err := cs.GetChunks(ctx, []string{"c1", "c2"})
	if err != nil {
		t.Fatal(err)
	}
	if len(got) != 1 || got[0].Content != "hello" || got[0].DocumentID != "d1" || got[0].Index != 3 || got[0].Metadata["topic"] != "x" {
		t.Fatalf("GetChunks = %+v", got)
	}
	if _, leaked := got[0].Metadata["ragout_index"]; leaked {
		t.Errorf("reserved key leaked: %+v", got[0].Metadata)
	}
	if ids, _ := body["ids"].([]any); len(ids) != 2 {
		t.Errorf("request ids = %v", body["ids"])
	}
}

func TestQdrantStore_GetChunks(t *testing.T) {
	var body map[string]any
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		switch {
		case r.Method == http.MethodPut:
			w.WriteHeader(http.StatusOK) // create collection
		case r.Method == http.MethodPost && strings.HasSuffix(r.URL.Path, "/collections/c/points"):
			json.NewDecoder(r.Body).Decode(&body)
			json.NewEncoder(w).Encode(map[string]any{"result": []map[string]any{{
				"id":      "p1",
				"payload": map[string]any{"ragout_content": "hello", "ragout_document_id": "d1", "ragout_index": float64(2)},
			}}})
		default:
			t.Fatalf("unexpected request: %s %s", r.Method, r.URL.Path)
		}
	}))
	defer srv.Close()

	ctx := context.Background()
	qs, err := store.NewQdrantStore(ctx, srv.URL, "c", 2)
	if err != nil {
		t.Fatal(err)
	}
	got, err := qs.GetChunks(ctx, []string{"p1"})
	if err != nil {
		t.Fatal(err)
	}
	if len(got) != 1 || got[0].Content != "hello" || got[0].DocumentID != "d1" || got[0].Index != 2 {
		t.Fatalf("GetChunks = %+v", got)
	}
	if body["with_payload"] != true {
		t.Errorf("request body = %v", body)
	}
}

func TestRemoteDeleteSource(t *testing.T) {
	ctx := context.Background()
	var chromaDeleted, qdrantBody map[string]any
	chroma := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		switch {
		case strings.HasSuffix(r.URL.Path, "/collections"):
			json.NewEncoder(w).Encode(map[string]any{"id": "col-1"})
		case strings.HasSuffix(r.URL.Path, "/get"):
			json.NewEncoder(w).Encode(map[string]any{"ids": []string{"old", "keep"}})
		case strings.HasSuffix(r.URL.Path, "/delete"):
			json.NewDecoder(r.Body).Decode(&chromaDeleted)
		}
	}))
	defer chroma.Close()
	cs, err := store.NewChromaStore(ctx, chroma.URL, "c")
	if err != nil {
		t.Fatal(err)
	}
	if err := cs.DeleteSource(ctx, "a.md", []string{"keep"}); err != nil {
		t.Fatal(err)
	}
	if ids, _ := chromaDeleted["ids"].([]any); len(ids) != 1 || ids[0] != "old" {
		t.Errorf("chroma deleted %v, want [old]", chromaDeleted["ids"])
	}

	qdrant := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if strings.HasSuffix(r.URL.Path, "/points/delete") {
			json.NewDecoder(r.Body).Decode(&qdrantBody)
		}
	}))
	defer qdrant.Close()
	qs, err := store.NewQdrantStore(ctx, qdrant.URL, "c", 2)
	if err != nil {
		t.Fatal(err)
	}
	if err := qs.DeleteSource(ctx, "a.md", []string{"keep"}); err != nil {
		t.Fatal(err)
	}
	b, _ := json.Marshal(qdrantBody)
	if !strings.Contains(string(b), `"has_id":["keep"]`) || !strings.Contains(string(b), `"value":"a.md"`) {
		t.Errorf("qdrant filter = %s", b)
	}
}
