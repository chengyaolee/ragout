package reranker

import (
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"github.com/chengyaolee/ragout"
)

var _ ragout.Reranker = (*HTTPReranker)(nil)

func candidates() []ragout.ScoredChunk {
	return []ragout.ScoredChunk{
		{Chunk: ragout.Chunk{ID: "a", Content: "cats sleep", Context: "Pets"}, Score: 0.9},
		{Chunk: ragout.Chunk{ID: "b", Content: "raft elects leaders"}, Score: 0.5},
		{Chunk: ragout.Chunk{ID: "c", Content: "paxos"}, Score: 0.1},
	}
}

func server(t *testing.T, reply string, got *map[string]any) *httptest.Server {
	t.Helper()
	return httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		json.NewDecoder(r.Body).Decode(got)
		if r.Header.Get("Authorization") != "" {
			(*got)["auth"] = r.Header.Get("Authorization")
		}
		w.Write([]byte(reply))
	}))
}

func TestHTTPReranker_CohereShape(t *testing.T) {
	var got map[string]any
	srv := server(t, `{"results":[{"index":2,"relevance_score":0.2},{"index":1,"relevance_score":0.95},{"index":7,"relevance_score":1}]}`, &got)
	defer srv.Close()

	out, err := NewHTTP(srv.URL, WithAPIKey("k")).Rerank(context.Background(), "who leads?", candidates(), 2)
	if err != nil {
		t.Fatal(err)
	}
	if len(out) != 2 || out[0].Chunk.ID != "b" || out[0].Score != 0.95 || out[1].Chunk.ID != "c" {
		t.Fatalf("want b then c by model score, out-of-range index ignored: %+v", out)
	}
	if got["model"] != BGERerankerV2M3.Name || got["auth"] != "Bearer k" {
		t.Errorf("request = %v", got)
	}
	docs, _ := got["documents"].([]any)
	if len(docs) != 3 || docs[0] != "Pets\n\ncats sleep" {
		t.Errorf("documents should be each chunk's indexed text: %v", docs)
	}
}

func TestHTTPReranker_TEIShape(t *testing.T) {
	var got map[string]any
	srv := server(t, `[{"index":0,"score":0.1},{"index":2,"score":0.8}]`, &got)
	defer srv.Close()

	out, err := NewHTTP(srv.URL, WithAPI(TEIAPI)).Rerank(context.Background(), "q", candidates(), 5)
	if err != nil {
		t.Fatal(err)
	}
	if len(out) != 2 || out[0].Chunk.ID != "c" {
		t.Fatalf("out = %+v", out)
	}
	if _, ok := got["texts"]; !ok || got["documents"] != nil {
		t.Errorf("TEI request must send texts: %v", got)
	}
}

func TestHTTPReranker_Qwen3Template(t *testing.T) {
	var got map[string]any
	srv := server(t, `{"results":[{"index":0,"relevance_score":0.5}]}`, &got)
	defer srv.Close()

	m := Qwen3Reranker("Qwen/Qwen3-Reranker-0.6B", "Find API docs")
	if _, err := NewHTTP(srv.URL, WithModel(m)).Rerank(context.Background(), "who leads?", candidates()[:1], 1); err != nil {
		t.Fatal(err)
	}
	q, _ := got["query"].(string)
	docs, _ := got["documents"].([]any)
	if !strings.HasPrefix(q, "<|im_start|>system") || !strings.Contains(q, "<Instruct>: Find API docs\n<Query>: who leads?\n") {
		t.Errorf("query not templated: %q", q)
	}
	if d, _ := docs[0].(string); !strings.HasPrefix(d, "<Document>: Pets") || !strings.HasSuffix(d, "</think>\n\n") {
		t.Errorf("document not templated: %q", d)
	}
	if !strings.Contains(Qwen3Reranker("q", "").FormatQuery("x"), qwen3Instruction) {
		t.Errorf("default instruction missing")
	}
	if Qwen3Reranker4B.FormatQuery != nil || Qwen3Reranker4B.FormatDocument != nil {
		t.Errorf("the llama.cpp preset must send raw text; the server applies the template")
	}
}

func TestHTTPReranker_Errors(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		http.Error(w, `{"error":"model not loaded"}`, http.StatusServiceUnavailable)
	}))
	defer srv.Close()
	if _, err := NewHTTP(srv.URL).Rerank(context.Background(), "q", candidates(), 2); err == nil || !strings.Contains(err.Error(), "model not loaded") {
		t.Errorf("err = %v", err)
	}
	var got map[string]any
	bad := server(t, `{"results":[{"index":0}]}`, &got)
	defer bad.Close()
	if _, err := NewHTTP(bad.URL).Rerank(context.Background(), "q", candidates(), 2); err == nil {
		t.Errorf("a result without a score must be an error")
	}
	if out, err := NewHTTP("http://unused").Rerank(context.Background(), "q", nil, 3); err != nil || len(out) != 0 {
		t.Errorf("no candidates: %v, %v", out, err)
	}
}
