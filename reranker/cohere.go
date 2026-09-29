package reranker

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"sort"
	"time"

	"github.com/chengyaolee/ragout"
)

type CohereReranker struct {
	apiKey     string
	model      string
	httpClient *http.Client
}

type CohereOption func(*CohereReranker)

func WithCohereModel(model string) CohereOption {
	return func(r *CohereReranker) {
		r.model = model
	}
}

func WithCohereHTTPClient(client *http.Client) CohereOption {
	return func(r *CohereReranker) {
		r.httpClient = client
	}
}

func NewCohereReranker(apiKey string, options ...CohereOption) *CohereReranker {
	r := &CohereReranker{
		apiKey:     apiKey,
		model:      "rerank-v3.5",
		httpClient: &http.Client{Timeout: 10 * time.Second},
	}
	for _, option := range options {
		option(r)
	}
	return r
}

type cohereRerankRequest struct {
	Model     string   `json:"model"`
	Query     string   `json:"query"`
	Documents []string `json:"documents"`
	TopN      int      `json:"topN, omitempty"`
}

type cohereRerankResponse struct {
	Results []struct {
		Index          int     `json:"index"`
		RelevanceScore float64 `json:"relevance_score"`
	} `json:"results"`
}

func (r *CohereReranker) Rerank(ctx context.Context, query string, candidates []ragout.ScoredChunk, topN int) ([]ragout.ScoredChunk, error) {
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	if len(candidates) == 0 || topN <= 0 {
		return []ragout.ScoredChunk{}, nil
	}

	// Extract Text to send
	documents := make([]string, len(candidates))
	for i, candidate := range candidates {
		documents[i] = candidate.Chunk.Content
	}

	payload := cohereRerankRequest{
		Model:     r.model,
		Query:     query,
		Documents: documents,
		TopN:      topN,
	}

	body, err := json.Marshal(payload)
	if err != nil {
		return nil, fmt.Errorf("cohere: failed to marshal request: %w", err)
	}

	request, err := http.NewRequestWithContext(
		ctx,
		http.MethodPost,
		"https://api.cohere.ai/v1/rerank",
		bytes.NewBuffer(body),
	)
	if err != nil {
		return nil, fmt.Errorf("cohere: failed to create request: %w", err)
	}

	request.Header.Set("Content-Type", "application/json")
	request.Header.Set("Authorization", fmt.Sprintf("Bearer %s", r.apiKey))

	response, err := r.httpClient.Do(request)
	if err != nil {
		return nil, fmt.Errorf("cohere: failed to send request: %w", err)
	}
	defer response.Body.Close()

	if response.StatusCode != http.StatusOK {
		responseBytes, _ := io.ReadAll(response.Body)
		return nil, fmt.Errorf("cohere: failed to get response: %s: %s", response.Status, string(responseBytes))
	}

	var cohereResponse cohereRerankResponse
	if err := json.NewDecoder(response.Body).Decode(&cohereResponse); err != nil {
		return nil, fmt.Errorf("cohere: failed to decode response: %w", err)
	}

	// Process each item in the API response
	reranked := make([]ragout.ScoredChunk, 0, len(cohereResponse.Results))
	for _, result := range cohereResponse.Results {
		// Check array bounds
		if result.Index < 0 || result.Index >= len(candidates) {
			continue
		}
		// Get original chunk
		original := candidates[result.Index]
		// Construct new chunk with updated score
		reranked = append(reranked, ragout.ScoredChunk{
			Chunk:       original.Chunk,
			DenseScore:  original.DenseScore,
			SparseScore: original.SparseScore,
			Score:       result.RelevanceScore,
		})
	}

	// Return reordered slice
	sort.Slice(reranked, func(i, j int) bool {
		return reranked[i].Score > reranked[j].Score
	})

	// Return top N
	if len(reranked) > topN {
		reranked = reranked[:topN]
	}

	return reranked, nil
}
