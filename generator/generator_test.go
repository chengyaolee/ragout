package generator

import (
	"bytes"
	"context"
	"io"
	"net/http"
	"testing"

	"github.com/chengyaolee/ragout"
)

var (
	_ ragout.Generator = (*OpenAIGenerator)(nil)
	_ ragout.Generator = (*OllamaGenerator)(nil)
)

type roundTripFunc func(req *http.Request) *http.Response

func (f roundTripFunc) RoundTrip(req *http.Request) (*http.Response, error) {
	return f(req), nil
}

func TestOpenAIGenerateIter(t *testing.T) {
	mockClient := &http.Client{
		Transport: roundTripFunc(func(req *http.Request) *http.Response {
			if req.Header.Get("Authorization") != "Bearer test-key" {
				return &http.Response{
					StatusCode: http.StatusUnauthorized,
					Body:       io.NopCloser(bytes.NewBufferString("unauthorized")),
				}
			}
			sse := "data: {\"choices\":[{\"delta\":{\"content\":\"The\"}}]}\n\n" +
				"data: {\"choices\":[{\"delta\":{\"content\":\" answer\"}}]}\n\n" +
				"data: [DONE]\n\n"
			return &http.Response{
				StatusCode: http.StatusOK,
				Body:       io.NopCloser(bytes.NewBufferString(sse)),
			}
		}),
	}

	g := NewOpenAIGenerator("test-key", WithOpenAIHTTPClient(mockClient))
	got, err := g.Generate(context.Background(), "q", nil)
	if err != nil {
		t.Fatal(err)
	}
	if got != "The answer" {
		t.Fatalf("got %q, want %q", got, "The answer")
	}
}
