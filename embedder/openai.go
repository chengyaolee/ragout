package embedder

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"time"

	"github.com/chengyaolee/ragout"
)

// OpenAIEmbedder implements ragout.Embedder for OpenAI embedding models.
type OpenAIEmbedder struct {
	apiKey     string
	model      string
	endpoint   string
	dimension  int
	httpClient *http.Client
}

type OpenAIOption func(*OpenAIEmbedder)

func WithOpenAIModel(model string) OpenAIOption {
	return func(o *OpenAIEmbedder) {
		o.model = model
		switch model {
		case "text-embedding-3-large":
			o.dimension = 3072
		case "text-embedding-3-small", "text-embedding-ada-002":
			o.dimension = 1536
		}
	}
}

func WithOpenAIEndpoint(endpoint string) OpenAIOption {
	return func(o *OpenAIEmbedder) {
		o.endpoint = endpoint
	}
}

func WithOpenAIDimension(dim int) OpenAIOption {
	return func(o *OpenAIEmbedder) {
		if dim > 0 {
			o.dimension = dim
		}
	}
}

func WithOpenAIHTTPClient(client *http.Client) OpenAIOption {
	return func(o *OpenAIEmbedder) {
		if client != nil {
			o.httpClient = client
		}
	}
}

// NewOpenAIEmbedder creates a new OpenAI embedding client.
func NewOpenAIEmbedder(apiKey string, options ...OpenAIOption) *OpenAIEmbedder {
	emb := &OpenAIEmbedder{
		apiKey:     apiKey,
		model:      "text-embedding-3-small",
		endpoint:   "https://api.openai.com/v1/embeddings",
		dimension:  1536,
		httpClient: &http.Client{Timeout: 30 * time.Second},
	}
	for _, opt := range options {
		opt(emb)
	}
	return emb
}

func (o *OpenAIEmbedder) Dimension() int {
	return o.dimension
}

type openAIEmbedRequest struct {
	Model string   `json:"model"`
	Input []string `json:"input"`
}

type openAIEmbedResponse struct {
	Data []struct {
		Index     int       `json:"index"`
		Embedding []float32 `json:"embedding"`
	} `json:"data"`
}

// EmbedBatch embeds a slice of strings using the OpenAI Embeddings API.
func (o *OpenAIEmbedder) EmbedBatch(ctx context.Context, texts []string) ([][]float32, error) {
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	if len(texts) == 0 {
		return [][]float32{}, nil
	}

	payload := openAIEmbedRequest{
		Model: o.model,
		Input: texts,
	}

	body, err := json.Marshal(payload)
	if err != nil {
		return nil, fmt.Errorf("openai embedder: marshal error: %w", err)
	}

	req, err := http.NewRequestWithContext(ctx, http.MethodPost, o.endpoint, bytes.NewReader(body))
	if err != nil {
		return nil, fmt.Errorf("openai embedder: new request error: %w", err)
	}

	req.Header.Set("Authorization", "Bearer "+o.apiKey)
	req.Header.Set("Content-Type", "application/json")

	resp, err := o.httpClient.Do(req)
	if err != nil {
		return nil, fmt.Errorf("openai embedder: http request failed: %w", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		respBytes, _ := io.ReadAll(resp.Body)
		return nil, fmt.Errorf("openai embedder: bad status code %d: %s", resp.StatusCode, string(respBytes))
	}

	var res openAIEmbedResponse
	if err := json.NewDecoder(resp.Body).Decode(&res); err != nil {
		return nil, fmt.Errorf("openai embedder: decode error: %w", err)
	}

	if len(res.Data) != len(texts) {
		return nil, ragout.ErrCountMismatch
	}

	vectors := make([][]float32, len(texts))
	for _, item := range res.Data {
		if item.Index < 0 || item.Index >= len(texts) {
			continue
		}
		vectors[item.Index] = item.Embedding
	}

	return vectors, nil
}
