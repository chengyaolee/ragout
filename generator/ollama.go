package generator

import (
	"bufio"
	"bytes"
	"context"
	"encoding/json"
	"iter"
	"net/http"
	"time"

	"github.com/chengyaolee/ragout"
)

type OllamaGenerator struct {
	endpoint   string
	model      string
	httpClient *http.Client
	budgeter   *ContextBudgeter
	history    int // earlier turns to include when a query passes history (default 0: none)
	numCtx     int // context window sent as options.num_ctx; 0 leaves Ollama's default
}

type OllamaOption func(*OllamaGenerator)

func WithOllamaHTTPClient(client *http.Client) OllamaOption {
	return func(g *OllamaGenerator) {
		if client != nil {
			g.httpClient = client
		}
	}
}

func WithOllamaBudgeter(b *ContextBudgeter) OllamaOption {
	return func(g *OllamaGenerator) {
		if b != nil {
			g.budgeter = b
		}
	}
}

// WithOllamaHistoryTurns lets the generator include up to n earlier turns in the prompt
// when a query passes ragout.WithHistory. The default, 0, keeps answers single-turn.
func WithOllamaHistoryTurns(n int) OllamaOption {
	return func(g *OllamaGenerator) { g.history = max(0, n) }
}

// WithOllamaNumCtx sets the context window (Ollama's num_ctx) for every request.
// Ollama truncates a prompt longer than its window without an error, so set this at
// least as large as the budgeter's token limit plus room for the answer.
func WithOllamaNumCtx(n int) OllamaOption {
	return func(g *OllamaGenerator) { g.numCtx = max(0, n) }
}

func NewOllamaGenerator(model string, endpoint string, options ...OllamaOption) *OllamaGenerator {
	if endpoint == "" {
		endpoint = "http://localhost:11434/api/generate"
	}
	customTransport := http.DefaultTransport.(*http.Transport).Clone()
	customTransport.ResponseHeaderTimeout = 30 * time.Second

	g := &OllamaGenerator{
		endpoint:   endpoint,
		model:      model,
		httpClient: &http.Client{Transport: customTransport},
		budgeter:   NewContextBudgeter(nil, 4096),
	}
	for _, opt := range options {
		opt(g)
	}
	return g
}

type ollamaStreamResponse struct {
	Response string `json:"response"`
	Done     bool   `json:"done"`
}

func (og *OllamaGenerator) GenerateIter(ctx context.Context, query string, candidates []ragout.ScoredChunk) ([]ragout.ScoredChunk, iter.Seq2[string, error]) {
	return og.GenerateIterWithHistory(ctx, query, nil, candidates)
}

// GenerateIterWithHistory is GenerateIter with up to WithOllamaHistoryTurns earlier turns
// in the prompt.
func (og *OllamaGenerator) GenerateIterWithHistory(ctx context.Context, query string, history []ragout.Turn, candidates []ragout.ScoredChunk) ([]ragout.ScoredChunk, iter.Seq2[string, error]) {
	prompt, sources := og.budgeter.BuildPromptWithHistory("", query, lastTurns(history, og.history), candidates)

	return sources, func(yield func(string, error) bool) {
		if err := ctx.Err(); err != nil {
			yield("", err)
			return
		}

		payload := og.payload(map[string]any{
			"model":  og.model,
			"prompt": prompt,
			"stream": true,
		})
		b, _ := json.Marshal(payload)

		req, err := http.NewRequestWithContext(ctx, http.MethodPost, og.endpoint, bytes.NewReader(b))
		if err != nil {
			yield("", err)
			return
		}

		resp, err := og.httpClient.Do(req)
		if err != nil {
			yield("", err)
			return
		}
		defer resp.Body.Close()

		scanner := bufio.NewScanner(resp.Body)
		for scanner.Scan() {
			var chunk ollamaStreamResponse
			if err := json.Unmarshal(scanner.Bytes(), &chunk); err != nil {
				continue
			}
			if chunk.Done {
				return
			}
			if chunk.Response != "" {
				if !yield(chunk.Response, nil) {
					return // Consumer stopped
				}
			}
		}
	}
}

// Complete runs a non-streaming generation. Ollama reuses its KV cache for an
// unchanged prompt prefix, so prefix is sent first and unchanged.
func (og *OllamaGenerator) Complete(ctx context.Context, prefix, prompt string) (string, error) {
	b, err := json.Marshal(og.payload(map[string]any{
		"model":  og.model,
		"prompt": joinPrompt(prefix, prompt),
		"stream": false,
	}))
	if err != nil {
		return "", err
	}
	req, err := http.NewRequestWithContext(ctx, http.MethodPost, og.endpoint, bytes.NewReader(b))
	if err != nil {
		return "", err
	}
	var out ollamaStreamResponse
	if err := doJSON(og.httpClient, req, "ollama", &out); err != nil {
		return "", err
	}
	return out.Response, nil
}

// payload adds request-wide options to an Ollama request body.
func (og *OllamaGenerator) payload(p map[string]any) map[string]any {
	if og.numCtx > 0 {
		p["options"] = map[string]any{"num_ctx": og.numCtx}
	}
	return p
}
