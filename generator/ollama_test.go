package generator

import (
	"bytes"
	"context"
	"io"
	"net/http"
	"strings"
	"testing"

	"github.com/chengyaolee/ragout"
)

type trackingCloser struct {
	io.Reader
	onClose func()
}

func (t *trackingCloser) Close() error {
	if t.onClose != nil {
		t.onClose()
	}
	return nil
}

func TestOllamaGenerateIter(t *testing.T) {
	mockClient := &http.Client{
		Transport: roundTripFunc(func(req *http.Request) *http.Response {
			stream := "{\"model\":\"llama3\",\"response\":\"Hello\",\"done\":false}\n" +
				"{\"model\":\"llama3\",\"response\":\" world\",\"done\":false}\n" +
				"{\"model\":\"llama3\",\"response\":\"\",\"done\":true}\n"
			return &http.Response{
				StatusCode: http.StatusOK,
				Body:       io.NopCloser(bytes.NewBufferString(stream)),
			}
		}),
	}

	g := NewOllamaGenerator("llama3", "", WithOllamaHTTPClient(mockClient))
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
	mockClient := &http.Client{
		Transport: roundTripFunc(func(req *http.Request) *http.Response {
			stream := "{\"response\":\"Hello\",\"done\":false}\n" +
				"{\"response\":\" world\",\"done\":false}\n"
			return &http.Response{
				StatusCode: http.StatusOK,
				Body: &trackingCloser{
					Reader: strings.NewReader(stream),
					onClose: func() {
						close(closed)
					},
				},
			}
		}),
	}

	g := NewOllamaGenerator("llama3", "", WithOllamaHTTPClient(mockClient))
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
	default:
		t.Fatal("expected response body to be closed on break")
	}
}
