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

func jsonClient(reply string, onReq func(body string, h http.Header)) *http.Client {
	return &http.Client{Transport: roundTripFunc(func(req *http.Request) *http.Response {
		b, _ := io.ReadAll(req.Body)
		onReq(string(b), req.Header)
		return &http.Response{StatusCode: http.StatusOK, Body: io.NopCloser(bytes.NewBufferString(reply))}
	})}
}

func TestCompleters(t *testing.T) {
	ctx := context.Background()

	var body string
	capture := func(b string, _ http.Header) { body = b }

	var _ ragout.Completer = (*OpenAIGenerator)(nil)
	var _ ragout.Completer = (*OllamaGenerator)(nil)

	o := NewOpenAIGenerator("k", WithOpenAIHTTPClient(jsonClient(`{"choices":[{"message":{"content":"hi"}}]}`, capture)))
	if got, err := o.Complete(ctx, "PREFIX", "ask"); err != nil || got != "hi" {
		t.Fatalf("openai = %q, %v", got, err)
	}
	if !strings.Contains(body, `"stream":false`) || !strings.Contains(body, "PREFIX\\n\\nask") {
		t.Errorf("openai body = %s", body)
	}

	l := NewOllamaGenerator("m", "", WithOllamaHTTPClient(jsonClient(`{"response":"yo","done":true}`, capture)))
	if got, err := l.Complete(ctx, "", "ask"); err != nil || got != "yo" {
		t.Fatalf("ollama = %q, %v", got, err)
	}

	// History turns are opt-in: the default generator ignores them.
	var hist = []ragout.Turn{{Role: "user", Content: "EARLIER TURN"}}
	sse := "data: [DONE]\n"
	g := NewOpenAIGenerator("k", WithOpenAIHTTPClient(jsonClient(sse, capture)))
	_, toks := g.GenerateIterWithHistory(ctx, "q", hist, nil)
	for range toks {
	}
	if strings.Contains(body, "EARLIER TURN") {
		t.Errorf("default generator must not include history")
	}
	g = NewOpenAIGenerator("k", WithOpenAIHistoryTurns(2), WithOpenAIHTTPClient(jsonClient(sse, capture)))
	_, toks = g.GenerateIterWithHistory(ctx, "q", hist, nil)
	for range toks {
	}
	if !strings.Contains(body, "EARLIER TURN") {
		t.Errorf("WithOpenAIHistoryTurns should include history: %s", body)
	}
}
