package generator

import (
	"bufio"
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"iter"
	"net/http"
	"strings"
	"time"

	"github.com/chengyaolee/ragout"
)

type OpenAIGenerator struct {
	apiKey     string
	model      string
	endpoint   string
	httpClient *http.Client
	budgeter   *ContextBudgeter
}

type OpenAIOption func(*OpenAIGenerator)

func WithOpenAIModel(model string) OpenAIOption {
	return func(g *OpenAIGenerator) { g.model = model }
}

func WithOpenAIEndpoint(endpoint string) OpenAIOption {
	return func(g *OpenAIGenerator) { g.endpoint = endpoint }
}

func WithOpenAIBudgeter(b *ContextBudgeter) OpenAIOption {
	return func(g *OpenAIGenerator) { g.budgeter = b }
}

func WithOpenAIHTTPClient(client *http.Client) OpenAIOption {
	return func(g *OpenAIGenerator) {
		if client != nil {
			g.httpClient = client
		}
	}
}

func NewOpenAIGenerator(apiKey string, options ...OpenAIOption) *OpenAIGenerator {
	customTransport := http.DefaultTransport.(*http.Transport).Clone()
	customTransport.ResponseHeaderTimeout = 30 * time.Second

	g := &OpenAIGenerator{
		apiKey:     apiKey,
		model:      "gpt-4o-mini",
		endpoint:   "https://api.openai.com/v1/chat/completions",
		httpClient: &http.Client{Transport: customTransport},
		budgeter:   NewContextBudgeter(nil, 4096),
	}
	for _, opt := range options {
		opt(g)
	}
	return g
}

type openAIChatRequest struct {
	Model    string              `json:"model"`
	Messages []openAIChatMessage `json:"messages"`
	Stream   bool                `json:"stream"`
}

type openAIChatMessage struct {
	Role    string `json:"role"`
	Content string `json:"content"`
}

type openAIStreamChunk struct {
	Choices []struct {
		Delta struct {
			Content string `json:"content"`
		} `json:"delta"`
	} `json:"choices"`
}

// GenerateIter assembles the prompt (returning the chunks placed in it, in citation
// order) and streams response tokens using modern Go iter.Seq2.
func (g *OpenAIGenerator) GenerateIter(ctx context.Context, query string, candidates []ragout.ScoredChunk) ([]ragout.ScoredChunk, iter.Seq2[string, error]) {
	prompt, sources := g.budgeter.BuildPrompt("You are an accurate RAG assistant.", query, candidates)

	return sources, func(yield func(string, error) bool) {
		if err := ctx.Err(); err != nil {
			yield("", err)
			return
		}

		// Prepare payload with "stream": true
		payload := openAIChatRequest{
			Model: g.model,
			Messages: []openAIChatMessage{
				{Role: "user", Content: prompt},
			},
			Stream: true,
		}

		body, err := json.Marshal(payload)
		if err != nil {
			yield("", fmt.Errorf("openai: marshal payload error: %w", err))
			return
		}

		req, err := http.NewRequestWithContext(ctx, http.MethodPost, g.endpoint, bytes.NewReader(body))
		if err != nil {
			yield("", err)
			return
		}
		req.Header.Set("Authorization", "Bearer "+g.apiKey)
		req.Header.Set("Content-Type", "application/json")

		resp, err := g.httpClient.Do(req)
		if err != nil {
			yield("", fmt.Errorf("openai: http request failed: %w", err))
			return
		}
		defer resp.Body.Close()

		if resp.StatusCode != http.StatusOK {
			yield("", fmt.Errorf("openai: bad status code %d", resp.StatusCode))
			return
		}

		// 4. Stream SSE lines
		scanner := bufio.NewScanner(resp.Body)
		for scanner.Scan() {
			line := scanner.Text()
			if !strings.HasPrefix(line, "data: ") {
				continue
			}
			data := strings.TrimPrefix(line, "data: ")
			if data == "[DONE]" {
				return // Stream completed cleanly
			}

			var chunk openAIStreamChunk
			if err := json.Unmarshal([]byte(data), &chunk); err != nil {
				continue // Skip empty heartbeats or malformed frames
			}

			if len(chunk.Choices) > 0 && chunk.Choices[0].Delta.Content != "" {
				token := chunk.Choices[0].Delta.Content
				// If caller executed 'break', yield returns false:
				if !yield(token, nil) {
					return // Abort stream immediately without goroutine leaks
				}
			}
		}

		if err := scanner.Err(); err != nil {
			yield("", fmt.Errorf("openai: sse stream read error: %w", err))
		}
	}
}
