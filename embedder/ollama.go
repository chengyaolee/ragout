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

// OllamaEmbedder implements ragout.Embedder for local Ollama embedding models.
type OllamaEmbedder struct {
	model      string
	endpoint   string
	dimension  int
	httpClient *http.Client
}

type OllamaOption func(*OllamaEmbedder)

func WithOllamaModel(model string) OllamaOption {
	return func(o *OllamaEmbedder) {
		o.model = model
	}
}

func WithOllamaEndpoint(endpoint string) OllamaOption {
	return func(o *OllamaEmbedder) {
		o.endpoint = endpoint
	}
}

func WithOllamaDimension(dim int) OllamaOption {
	return func(o *OllamaEmbedder) {
		if dim > 0 {
			o.dimension = dim
		}
	}
}

func WithOllamaHTTPClient(client *http.Client) OllamaOption {
	return func(o *OllamaEmbedder) {
		if client != nil {
			o.httpClient = client
		}
	}
}

// NewOllamaEmbedder creates a new Ollama embedder.
func NewOllamaEmbedder(model string, options ...OllamaOption) *OllamaEmbedder {
	emb := &OllamaEmbedder{
		model:      model,
		endpoint:   "http://localhost:11434/api/embed",
		dimension:  768, // Default for nomic-embed-text
		httpClient: &http.Client{Timeout: 60 * time.Second},
	}
	for _, opt := range options {
		opt(emb)
	}
	return emb
}

func (o *OllamaEmbedder) Dimension() int {
	return o.dimension
}

type ollamaEmbedRequest struct {
	Model string   `json:"model"`
	Input []string `json:"input"`
}

type ollamaEmbedResponse struct {
	Model      string      `json:"model"`
	Embeddings [][]float32 `json:"embeddings"`
}

// EmbedBatch sends texts to the Ollama embedding API.
func (o *OllamaEmbedder) EmbedBatch(ctx context.Context, texts []string) ([][]float32, error) {
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	if len(texts) == 0 {
		return [][]float32{}, nil
	}

	payload := ollamaEmbedRequest{
		Model: o.model,
		Input: texts,
	}

	body, err := json.Marshal(payload)
	if err != nil {
		return nil, fmt.Errorf("ollama embedder: marshal error: %w", err)
	}

	req, err := http.NewRequestWithContext(ctx, http.MethodPost, o.endpoint, bytes.NewReader(body))
	if err != nil {
		return nil, fmt.Errorf("ollama embedder: new request error: %w", err)
	}
	req.Header.Set("Content-Type", "application/json")

	resp, err := o.httpClient.Do(req)
	if err != nil {
		return nil, fmt.Errorf("ollama embedder: http request failed: %w", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		respBytes, _ := io.ReadAll(resp.Body)
		return nil, fmt.Errorf("ollama embedder: bad status code %d: %s", resp.StatusCode, string(respBytes))
	}

	var res ollamaEmbedResponse
	if err := json.NewDecoder(resp.Body).Decode(&res); err != nil {
		return nil, fmt.Errorf("ollama embedder: decode error: %w", err)
	}

	if len(res.Embeddings) != len(texts) {
		return nil, ragout.ErrCountMismatch
	}

	if len(res.Embeddings) > 0 && len(res.Embeddings[0]) > 0 {
		o.dimension = len(res.Embeddings[0])
	}

	return res.Embeddings, nil
}
