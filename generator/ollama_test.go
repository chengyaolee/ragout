package generator

import (
	"context"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
	"time"

	"github.com/chengyaolee/ragout"
)

func TestOllamaGenerateIter(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Write([]byte("{\"model\":\"llama3\",\"response\":\"Hello\",\"done\":false}\n"))
		w.Write([]byte("{\"model\":\"llama3\",\"response\":\" world\",\"done\":false}\n"))
		w.Write([]byte("{\"model\":\"llama3\",\"response\":\"\",\"done\":true}\n"))
	}))
	defer srv.Close()

	g := NewOllamaGenerator("llama3", srv.URL)
	var b strings.Builder
	for token, err := range g.GenerateIter(context.Background(), "hi", []ragout.ScoredChunk{
		{Chunk: ragout.Chunk{ID: "a", Content: "fact"}},
	}) {
		if err != nil {
			t.Fatal(err)
		}
		b.WriteString(token)
	}
	if b.String() != "Hello world" {
		t.Fatalf("got %q, want %q", b.String(), "Hello world")
	}
}

func TestOllamaGenerateIter_BreakClosesBody(t *testing.T) {
	closed := make(chan struct{})
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		_, _ = io.Copy(io.Discard, r.Body)
		flusher := w.(http.Flusher)
		w.Write([]byte("{\"response\":\"Hello\",\"done\":false}\n"))
		flusher.Flush()
		select {
		case <-r.Context().Done():
			close(closed)
		case <-time.After(5 * time.Second):
		}
	}))
	defer srv.Close()

	g := NewOllamaGenerator("llama3", srv.URL)
	for token, err := range g.GenerateIter(context.Background(), "hi", nil) {
		if err != nil {
			t.Fatal(err)
		}
		if token == "Hello" {
			break
		}
	}

	select {
	case <-closed:
	case <-time.After(2 * time.Second):
		t.Fatal("break did not close the response body")
	}
}
