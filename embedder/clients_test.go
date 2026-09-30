package embedder_test

import (
	"bytes"
	"context"
	"io"
	"net/http"
	"testing"

	"github.com/chengyaolee/ragout/embedder"
)

type roundTripFunc func(req *http.Request) *http.Response

func (f roundTripFunc) RoundTrip(req *http.Request) (*http.Response, error) {
	return f(req), nil
}

func TestOpenAIEmbedder_Success(t *testing.T) {
	mockClient := &http.Client{
		Transport: roundTripFunc(func(req *http.Request) *http.Response {
			if req.Header.Get("Authorization") != "Bearer test-key" {
				return &http.Response{
					StatusCode: http.StatusUnauthorized,
					Body:       io.NopCloser(bytes.NewBufferString(`{"error": "unauthorized"}`)),
				}
			}
			jsonResp := `{
				"data": [
					{"index": 0, "embedding": [0.1, 0.2, 0.3]},
					{"index": 1, "embedding": [0.4, 0.5, 0.6]}
				]
			}`
			return &http.Response{
				StatusCode: http.StatusOK,
				Body:       io.NopCloser(bytes.NewBufferString(jsonResp)),
			}
		}),
	}

	emb := embedder.NewOpenAIEmbedder(
		"test-key",
		embedder.WithOpenAIHTTPClient(mockClient),
		embedder.WithOpenAIDimension(3),
	)

	vecs, err := emb.EmbedBatch(context.Background(), []string{"hello", "world"})
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if len(vecs) != 2 || len(vecs[0]) != 3 {
		t.Fatalf("expected 2 vectors of dim 3, got %#v", vecs)
	}
	if emb.Dimension() != 3 {
		t.Fatalf("expected dim 3, got %d", emb.Dimension())
	}
}

func TestOllamaEmbedder_Success(t *testing.T) {
	mockClient := &http.Client{
		Transport: roundTripFunc(func(req *http.Request) *http.Response {
			jsonResp := `{
				"model": "nomic-embed-text",
				"embeddings": [
					[0.1, 0.2, 0.3],
					[0.4, 0.5, 0.6]
				]
			}`
			return &http.Response{
				StatusCode: http.StatusOK,
				Body:       io.NopCloser(bytes.NewBufferString(jsonResp)),
			}
		}),
	}

	emb := embedder.NewOllamaEmbedder(
		"nomic-embed-text",
		embedder.WithOllamaHTTPClient(mockClient),
		embedder.WithOllamaDimension(3),
	)

	vecs, err := emb.EmbedBatch(context.Background(), []string{"foo", "bar"})
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if len(vecs) != 2 || len(vecs[0]) != 3 {
		t.Fatalf("expected 2 vectors of dim 3, got %#v", vecs)
	}
}
