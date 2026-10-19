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
	history    int // earlier turns to include when a query passes history (default 0: none)
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

// WithOpenAIHistoryTurns lets the generator include up to n earlier turns in the prompt
// when a query passes ragout.WithHistory. The default, 0, keeps answers single-turn.
func WithOpenAIHistoryTurns(n int) OpenAIOption {
	return func(g *OpenAIGenerator) { g.history = max(0, n) }
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
	return g.GenerateIterWithHistory(ctx, query, nil, candidates)
}

// GenerateIterWithHistory is GenerateIter with up to WithOpenAIHistoryTurns earlier turns
// in the prompt.
func (g *OpenAIGenerator) GenerateIterWithHistory(ctx context.Context, query string, history []ragout.Turn, candidates []ragout.ScoredChunk) ([]ragout.ScoredChunk, iter.Seq2[string, error]) {
	prompt, sources := g.budgeter.BuildPromptWithHistory("You are an accurate RAG assistant.", query, lastTurns(history, g.history), candidates)
	return sources, g.stream(ctx, prompt)
}

// stream sends prompt as one user message and yields the streamed tokens.
func (g *OpenAIGenerator) stream(ctx context.Context, prompt string) iter.Seq2[string, error] {
	return func(yield func(string, error) bool) {
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

// Complete runs one chat completion and returns the whole reply. It streams under the
// hood, so the client's response-header timeout bounds the wait for the first token,
// not the whole generation. OpenAI caches long shared prompt prefixes automatically,
// so prefix is sent first and unchanged to benefit from it.
func (g *OpenAIGenerator) Complete(ctx context.Context, prefix, prompt string) (string, error) {
	return collect(g.stream(ctx, joinPrompt(prefix, prompt)))
}
