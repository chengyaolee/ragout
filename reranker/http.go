package reranker

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"sort"
	"strings"
	"time"

	"github.com/chengyaolee/ragout"
)

// Model is a cross-encoder a rerank server can host: the name sent in the request, and
// how query and documents are wrapped before scoring. Models trained on a prompt
// template need the wrapping when the server does not apply it (vLLM does not).
type Model struct {
	Name           string
	FormatQuery    func(query string) string // nil sends the query as is
	FormatDocument func(doc string) string   // nil sends each document as is
}

// BGERerankerV2M3 is the default model: BAAI/bge-reranker-v2-m3, a 0.6B multilingual
// cross-encoder under Apache 2.0, small enough for a laptop or a modest server.
var BGERerankerV2M3 = Model{Name: "BAAI/bge-reranker-v2-m3"}

// Qwen3Reranker4B is Qwen/Qwen3-Reranker-4B (Apache 2.0): stronger and heavier than the
// default. It scores yes/no from a chat template, which is applied here with the
// model's default instruction; use Qwen3Reranker for a task-specific one. Serve it with
// vLLM (rerank API, with the hf_overrides from the model card) or a reranking GGUF in
// llama.cpp. If your server already applies the template, set FormatQuery and
// FormatDocument to nil.
var Qwen3Reranker4B = Qwen3Reranker("Qwen/Qwen3-Reranker-4B", "")

const (
	qwen3Instruction = "Given a web search query, retrieve relevant passages that answer the query"
	qwen3Prefix      = "<|im_start|>system\nJudge whether the Document meets the requirements based on the Query and the Instruct provided. Note that the answer can only be \"yes\" or \"no\".<|im_end|>\n<|im_start|>user\n"
	qwen3Suffix      = "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
)

// Qwen3Reranker returns a Qwen3-Reranker model (any size) wrapped in its prompt
// template. An empty instruction uses the model card's default. The model card
// recommends an English instruction written for your task; it typically adds 1-5%.
func Qwen3Reranker(name, instruction string) Model {
	if instruction == "" {
		instruction = qwen3Instruction
	}
	return Model{
		Name: name,
		FormatQuery: func(q string) string {
			return qwen3Prefix + "<Instruct>: " + instruction + "\n<Query>: " + q + "\n"
		},
		FormatDocument: func(d string) string { return "<Document>: " + d + qwen3Suffix },
	}
}

// API is the request and response shape a rerank server speaks.
type API int

const (
	// CohereAPI: {model, query, documents, top_n} -> {results: [{index, relevance_score}]}.
	// llama.cpp (llama-server --reranking), vLLM, Infinity, Jina and Cohere all accept it.
	CohereAPI API = iota
	// TEIAPI: Hugging Face text-embeddings-inference, {query, texts} -> [{index, score}].
	TEIAPI
)

// DefaultEndpoint is llama-server's rerank route on its default port.
const DefaultEndpoint = "http://localhost:8080/v1/rerank"

// HTTPReranker scores candidates with a cross-encoder behind an HTTP rerank endpoint,
// self-hosted or not. The default is BGERerankerV2M3 on DefaultEndpoint:
//
//	llama-server -hf gpustack/bge-reranker-v2-m3-GGUF --reranking --port 8080
//	ragout.WithReranker(reranker.NewHTTP(""))
//
// Switching models is one option:
//
//	reranker.NewHTTP(url, reranker.WithModel(reranker.Qwen3Reranker4B))
type HTTPReranker struct {
	endpoint   string
	model      Model
	api        API
	apiKey     string
	httpClient *http.Client
}

type HTTPOption func(*HTTPReranker)

// WithModel selects the model. Default BGERerankerV2M3.
func WithModel(m Model) HTTPOption { return func(r *HTTPReranker) { r.model = m } }

// WithAPI selects the request shape. Default CohereAPI.
func WithAPI(a API) HTTPOption { return func(r *HTTPReranker) { r.api = a } }

// WithAPIKey sends "Authorization: Bearer <key>", for hosted APIs or a secured server.
func WithAPIKey(key string) HTTPOption { return func(r *HTTPReranker) { r.apiKey = key } }

func WithHTTPClient(c *http.Client) HTTPOption {
	return func(r *HTTPReranker) {
		if c != nil {
			r.httpClient = c
		}
	}
}

// NewHTTP returns a reranker for endpoint, the full URL of the rerank route. An empty
// endpoint means DefaultEndpoint.
func NewHTTP(endpoint string, opts ...HTTPOption) *HTTPReranker {
	if endpoint == "" {
		endpoint = DefaultEndpoint
	}
	r := &HTTPReranker{
		endpoint: endpoint,
		model:    BGERerankerV2M3,
		// Generous: a local server may be loading the model on the first call.
		httpClient: &http.Client{Timeout: 2 * time.Minute},
	}
	for _, opt := range opts {
		opt(r)
	}
	return r
}

// Rerank scores every candidate against query and returns the topN best, best first.
// Each document is the chunk's indexed text (context breadcrumb plus content), so the
// model sees the title and section too; Score becomes the model's relevance score.
func (r *HTTPReranker) Rerank(ctx context.Context, query string, candidates []ragout.ScoredChunk, topN int) ([]ragout.ScoredChunk, error) {
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	if len(candidates) == 0 || topN <= 0 {
		return []ragout.ScoredChunk{}, nil
	}

	if r.model.FormatQuery != nil {
		query = r.model.FormatQuery(query)
	}
	docs := make([]string, len(candidates))
	for i, c := range candidates {
		docs[i] = c.Chunk.IndexText()
		if r.model.FormatDocument != nil {
			docs[i] = r.model.FormatDocument(docs[i])
		}
	}

	var payload any = map[string]any{"model": r.model.Name, "query": query, "documents": docs, "top_n": len(docs)}
	if r.api == TEIAPI {
		payload = map[string]any{"query": query, "texts": docs}
	}
	body, err := json.Marshal(payload)
	if err != nil {
		return nil, err
	}
	req, err := http.NewRequestWithContext(ctx, http.MethodPost, r.endpoint, bytes.NewReader(body))
	if err != nil {
		return nil, err
	}
	req.Header.Set("Content-Type", "application/json")
	if r.apiKey != "" {
		req.Header.Set("Authorization", "Bearer "+r.apiKey)
	}

	resp, err := r.httpClient.Do(req)
	if err != nil {
		return nil, fmt.Errorf("rerank: request failed: %w", err)
	}
	defer resp.Body.Close()
	raw, err := io.ReadAll(io.LimitReader(resp.Body, 8<<20))
	if err != nil {
		return nil, fmt.Errorf("rerank: read response: %w", err)
	}
	if resp.StatusCode != http.StatusOK {
		return nil, fmt.Errorf("rerank: bad status %d: %s", resp.StatusCode, truncate(raw, 512))
	}

	scores, err := decodeScores(raw)
	if err != nil {
		return nil, fmt.Errorf("rerank: %w", err)
	}
	out := make([]ragout.ScoredChunk, 0, len(scores))
	for _, s := range scores {
		if s.index < 0 || s.index >= len(candidates) {
			continue
		}
		c := candidates[s.index]
		c.Score = s.score
		out = append(out, c)
	}
	sort.SliceStable(out, func(i, j int) bool { return out[i].Score > out[j].Score })
	return out[:min(topN, len(out))], nil
}

type indexScore struct {
	index int
	score float64
}

// decodeScores reads either response shape: a Cohere-style {"results": [...]} object
// or TEI's bare array, so a server that answers in the other shape still works.
func decodeScores(raw []byte) ([]indexScore, error) {
	type item struct {
		Index          int      `json:"index"`
		RelevanceScore *float64 `json:"relevance_score"`
		Score          *float64 `json:"score"`
	}
	var items []item
	if trimmed := bytes.TrimSpace(raw); len(trimmed) > 0 && trimmed[0] == '[' {
		if err := json.Unmarshal(trimmed, &items); err != nil {
			return nil, fmt.Errorf("decode response: %w", err)
		}
	} else {
		var obj struct {
			Results []item `json:"results"`
		}
		if err := json.Unmarshal(raw, &obj); err != nil {
			return nil, fmt.Errorf("decode response: %w", err)
		}
		items = obj.Results
	}
	out := make([]indexScore, 0, len(items))
	for _, it := range items {
		switch {
		case it.RelevanceScore != nil:
			out = append(out, indexScore{it.Index, *it.RelevanceScore})
		case it.Score != nil:
			out = append(out, indexScore{it.Index, *it.Score})
		default:
			return nil, fmt.Errorf("result %d has no score", it.Index)
		}
	}
	return out, nil
}

func truncate(b []byte, n int) string {
	return strings.ToValidUTF8(string(b[:min(n, len(b))]), "")
}
