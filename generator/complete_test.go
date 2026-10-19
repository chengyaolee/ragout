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
	return statusClient(http.StatusOK, reply, onReq)
}

func statusClient(status int, reply string, onReq func(body string, h http.Header)) *http.Client {
	return &http.Client{Transport: roundTripFunc(func(req *http.Request) *http.Response {
		b, _ := io.ReadAll(req.Body)
		onReq(string(b), req.Header)
		return &http.Response{StatusCode: status, Body: io.NopCloser(bytes.NewBufferString(reply))}
	})}
}

func TestCompleters(t *testing.T) {
	ctx := context.Background()

	var body string
	capture := func(b string, _ http.Header) { body = b }

	var _ ragout.Completer = (*OpenAIGenerator)(nil)
	var _ ragout.Completer = (*OllamaGenerator)(nil)

	// Complete streams under the hood and joins the tokens.
	sseReply := "data: {\"choices\":[{\"delta\":{\"content\":\"h\"}}]}\n\ndata: {\"choices\":[{\"delta\":{\"content\":\"i\"}}]}\n\ndata: [DONE]\n"
	o := NewOpenAIGenerator("k", WithOpenAIHTTPClient(jsonClient(sseReply, capture)))
	if got, err := o.Complete(ctx, "PREFIX", "ask"); err != nil || got != "hi" {
		t.Fatalf("openai = %q, %v", got, err)
	}
	if !strings.Contains(body, `"stream":true`) || !strings.Contains(body, "PREFIX\\n\\nask") {
		t.Errorf("openai body = %s", body)
	}

	l := NewOllamaGenerator("m", "", WithOllamaHTTPClient(jsonClient("{\"response\":\"y\"}\n{\"response\":\"o\",\"done\":true}\n", capture)))
	if got, err := l.Complete(ctx, "", "ask"); err != nil || got != "yo" {
		t.Fatalf("ollama = %q, %v", got, err)
	}

	// Ollama failures must surface as errors, not as empty answers.
	l = NewOllamaGenerator("m", "", WithOllamaHTTPClient(statusClient(http.StatusNotFound, `{"error":"model 'm' not found"}`, capture)))
	if _, err := l.Complete(ctx, "", "ask"); err == nil || !strings.Contains(err.Error(), "not found") {
		t.Errorf("missing model: err = %v", err)
	}
	l = NewOllamaGenerator("m", "", WithOllamaHTTPClient(jsonClient("{\"response\":\"part\"}\n{\"error\":\"out of memory\"}\n", capture)))
	if _, err := l.Complete(ctx, "", "ask"); err == nil || !strings.Contains(err.Error(), "out of memory") {
		t.Errorf("mid-stream error: err = %v", err)
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

	// num_ctx is only sent when configured.
	l = NewOllamaGenerator("m", "", WithOllamaHTTPClient(jsonClient(`{"response":"yo"}`, capture)))
	l.Complete(ctx, "", "ask")
	if strings.Contains(body, "num_ctx") {
		t.Errorf("default request must not set num_ctx: %s", body)
	}
	l = NewOllamaGenerator("m", "", WithOllamaNumCtx(8192), WithOllamaHTTPClient(jsonClient(`{"response":"yo"}`, capture)))
	l.Complete(ctx, "", "ask")
	if !strings.Contains(body, `"num_ctx":8192`) {
		t.Errorf("num_ctx missing: %s", body)
	}
}
