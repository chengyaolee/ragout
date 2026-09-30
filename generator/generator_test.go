package generator

import (
	"context"
	"fmt"
	"net/http"
	"net/http/httptest"
	"testing"

	"github.com/chengyaolee/ragout"
)

var (
	_ ragout.Generator = (*OpenAIGenerator)(nil)
	_ ragout.Generator = (*OllamaGenerator)(nil)
)

func TestOpenAIGenerateIter(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.Header.Get("Authorization") != "Bearer test-key" {
			http.Error(w, "unauthorized", http.StatusUnauthorized)
			return
		}
		fmt.Fprint(w, "data: {\"choices\":[{\"delta\":{\"content\":\"The\"}}]}\n\n")
		fmt.Fprint(w, "data: {\"choices\":[{\"delta\":{\"content\":\" answer\"}}]}\n\n")
		fmt.Fprint(w, "data: [DONE]\n\n")
	}))
	defer srv.Close()

	g := NewOpenAIGenerator("test-key", WithOpenAIEndpoint(srv.URL))
	got, err := g.Generate(context.Background(), "q", nil)
	if err != nil {
		t.Fatal(err)
	}
	if got != "The answer" {
		t.Fatalf("got %q, want %q", got, "The answer")
	}
}
