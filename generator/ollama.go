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

func (og *OllamaGenerator) GenerateIter(ctx context.Context, query string, candidates []ragout.ScoredChunk) iter.Seq2[string, error] {
	return func(yield func(string, error) bool) {
		if err := ctx.Err(); err != nil {
			yield("", err)
			return
		}

		prompt, _ := og.budgeter.BuildPrompt("", query, ReorderLostInTheMiddle(candidates))

		payload := map[string]any{
			"model":  og.model,
			"prompt": prompt,
			"stream": true,
		}
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

func (og *OllamaGenerator) GenerateStream(ctx context.Context, query string, context []ragout.ScoredChunk, cb ragout.StreamCallback) error {
	return StreamFromIter(og.GenerateIter(ctx, query, context), cb)
}

func (og *OllamaGenerator) Generate(ctx context.Context, query string, context []ragout.ScoredChunk) (string, error) {
	return CollectFromIter(og.GenerateIter(ctx, query, context))
}
