package store

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"reflect"
	"time"

	"github.com/chengyaolee/ragout"
)

// Reserved Qdrant payload keys used to recover chunk fields that have no dedicated
// Qdrant column.
const (
	qdrantPayloadContent    = "ragout_content"
	qdrantPayloadDocumentID = "ragout_document_id"
	qdrantPayloadIndex      = "ragout_index"
)

// QdrantStore implements ragout.VectorStore against a Qdrant server over its REST API
// (net/http only; no client SDK dependency).
type QdrantStore struct {
	baseURL    string
	collection string
	httpClient *http.Client
}

type QdrantOption func(*QdrantStore)

// WithQdrantHTTPClient overrides the HTTP client. Use it to set an api-key header via
// a custom http.RoundTripper.
func WithQdrantHTTPClient(client *http.Client) QdrantOption {
	return func(q *QdrantStore) {
		if client != nil {
			q.httpClient = client
		}
	}
}

// NewQdrantStore connects to baseURL (e.g. "http://localhost:6333") and creates the
// named collection (cosine distance, size dim) if it doesn't already exist.
func NewQdrantStore(ctx context.Context, baseURL, collection string, dim int, opts ...QdrantOption) (*QdrantStore, error) {
	qs := &QdrantStore{
		baseURL:    baseURL,
		collection: collection,
		httpClient: &http.Client{Timeout: 30 * time.Second},
	}
	for _, opt := range opts {
		opt(qs)
	}

	payload := map[string]any{
		"vectors": map[string]any{
			"size":     dim,
			"distance": "Cosine",
		},
	}
	body, err := json.Marshal(payload)
	if err != nil {
		return nil, fmt.Errorf("qdrant: marshal create collection: %w", err)
	}
	url := fmt.Sprintf("%s/collections/%s", qs.baseURL, qs.collection)
	req, err := http.NewRequestWithContext(ctx, http.MethodPut, url, bytes.NewReader(body))
	if err != nil {
		return nil, fmt.Errorf("qdrant: create collection: %w", err)
	}
	req.Header.Set("Content-Type", "application/json")
	resp, err := qs.httpClient.Do(req)
	if err != nil {
		return nil, fmt.Errorf("qdrant: create collection: %w", err)
	}
	defer resp.Body.Close()
	// 409 means the collection already exists, which is fine.
	if (resp.StatusCode < 200 || resp.StatusCode >= 300) && resp.StatusCode != http.StatusConflict {
		respBytes, _ := io.ReadAll(resp.Body)
		return nil, fmt.Errorf("qdrant: create collection: bad status code %d: %s", resp.StatusCode, string(respBytes))
	}

	return qs, nil
}

func (qs *QdrantStore) do(ctx context.Context, method, url string, payload any, out any) error {
	var body io.Reader
	if payload != nil {
		b, err := json.Marshal(payload)
		if err != nil {
			return fmt.Errorf("marshal request: %w", err)
		}
		body = bytes.NewReader(b)
	}

	req, err := http.NewRequestWithContext(ctx, method, url, body)
	if err != nil {
		return fmt.Errorf("new request: %w", err)
	}
	req.Header.Set("Content-Type", "application/json")

	resp, err := qs.httpClient.Do(req)
	if err != nil {
		return fmt.Errorf("http request failed: %w", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode < 200 || resp.StatusCode >= 300 {
		respBytes, _ := io.ReadAll(resp.Body)
		return fmt.Errorf("bad status code %d: %s", resp.StatusCode, string(respBytes))
	}
	if out == nil {
		return nil
	}
	return json.NewDecoder(resp.Body).Decode(out)
}

// Upsert adds or updates chunks as Qdrant points.
func (qs *QdrantStore) Upsert(ctx context.Context, chunks []ragout.Chunk) error {
	if err := ctx.Err(); err != nil {
		return err
	}
	if len(chunks) == 0 {
		return nil
	}

	points := make([]map[string]any, len(chunks))
	for i, c := range chunks {
		payload := ragout.CloneMetadata(c.Metadata)
		payload[qdrantPayloadContent] = c.Content
		payload[qdrantPayloadDocumentID] = c.DocumentID
		payload[qdrantPayloadIndex] = c.Index
		points[i] = map[string]any{
			"id":      c.ID,
			"vector":  c.Embedding,
			"payload": payload,
		}
	}

	url := fmt.Sprintf("%s/collections/%s/points?wait=true", qs.baseURL, qs.collection)
	if err := qs.do(ctx, http.MethodPut, url, map[string]any{"points": points}, nil); err != nil {
		return fmt.Errorf("qdrant: upsert: %w", err)
	}
	return nil
}

// Delete removes points from the collection by ID.
func (qs *QdrantStore) Delete(ctx context.Context, ids []string) error {
	if err := ctx.Err(); err != nil {
		return err
	}
	if len(ids) == 0 {
		return nil
	}
	url := fmt.Sprintf("%s/collections/%s/points/delete?wait=true", qs.baseURL, qs.collection)
	if err := qs.do(ctx, http.MethodPost, url, map[string]any{"points": ids}, nil); err != nil {
		return fmt.Errorf("qdrant: delete: %w", err)
	}
	return nil
}

type qdrantQueryResponse struct {
	Result struct {
		Points []struct {
			ID      any            `json:"id"`
			Score   float64        `json:"score"`
			Payload map[string]any `json:"payload"`
			Vector  []float32      `json:"vector"`
		} `json:"points"`
	} `json:"result"`
}

// SearchDense searches for the topK most similar chunks to queryVector.
func (qs *QdrantStore) SearchDense(ctx context.Context, queryVector []float32, topK int, filter map[string]any) ([]ragout.ScoredChunk, error) {
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	if topK <= 0 {
		return []ragout.ScoredChunk{}, nil
	}

	payload := map[string]any{
		"query":        queryVector,
		"limit":        topK,
		"with_payload": true,
		"with_vector":  true,
	}
	if f := qdrantFilter(filter); f != nil {
		payload["filter"] = f
	}

	url := fmt.Sprintf("%s/collections/%s/points/query", qs.baseURL, qs.collection)
	var res qdrantQueryResponse
	if err := qs.do(ctx, http.MethodPost, url, payload, &res); err != nil {
		return nil, fmt.Errorf("qdrant: query: %w", err)
	}

	scored := make([]ragout.ScoredChunk, 0, len(res.Result.Points))
	for _, p := range res.Result.Points {
		id := fmt.Sprintf("%v", p.ID)
		meta := p.Payload
		if meta == nil {
			meta = map[string]any{}
		}
		chunk := ragout.Chunk{ID: id, Embedding: p.Vector}
		if content, ok := meta[qdrantPayloadContent].(string); ok {
			chunk.Content = content
			delete(meta, qdrantPayloadContent)
		}
		if docID, ok := meta[qdrantPayloadDocumentID].(string); ok {
			chunk.DocumentID = docID
			delete(meta, qdrantPayloadDocumentID)
		}
		if idx, ok := meta[qdrantPayloadIndex].(float64); ok {
			chunk.Index = int(idx)
			delete(meta, qdrantPayloadIndex)
		}
		chunk.Metadata = meta

		scored = append(scored, ragout.ScoredChunk{
			Chunk:      chunk,
			DenseScore: p.Score,
			Score:      p.Score,
		})
	}

	return scored, nil
}

// qdrantFilter translates a ragout metadata filter into a Qdrant filter payload.
// A bare value becomes a match.value condition, a slice or $in becomes match.any,
// $ne/$nin become must_not, and $gt/$gte/$lt/$lte become a range condition.
func qdrantFilter(filter map[string]any) map[string]any {
	if len(filter) == 0 {
		return nil
	}

	var must, mustNot []map[string]any

	addRange := func(key string, op string, val any) map[string]any {
		return map[string]any{"key": key, "range": map[string]any{op: val}}
	}

	for key, want := range filter {
		if ops, ok := want.(map[string]any); ok {
			for op, val := range ops {
				switch op {
				case "$eq":
					must = append(must, map[string]any{"key": key, "match": map[string]any{"value": val}})
				case "$ne":
					mustNot = append(mustNot, map[string]any{"key": key, "match": map[string]any{"value": val}})
				case "$in":
					must = append(must, map[string]any{"key": key, "match": map[string]any{"any": toAnySlice(val)}})
				case "$nin":
					mustNot = append(mustNot, map[string]any{"key": key, "match": map[string]any{"any": toAnySlice(val)}})
				case "$gt":
					must = append(must, addRange(key, "gt", val))
				case "$gte":
					must = append(must, addRange(key, "gte", val))
				case "$lt":
					must = append(must, addRange(key, "lt", val))
				case "$lte":
					must = append(must, addRange(key, "lte", val))
				}
			}
			continue
		}

		v := reflect.ValueOf(want)
		if v.IsValid() && (v.Kind() == reflect.Slice || v.Kind() == reflect.Array) {
			must = append(must, map[string]any{"key": key, "match": map[string]any{"any": toAnySlice(want)}})
			continue
		}
		must = append(must, map[string]any{"key": key, "match": map[string]any{"value": want}})
	}

	f := map[string]any{}
	if len(must) > 0 {
		f["must"] = must
	}
	if len(mustNot) > 0 {
		f["must_not"] = mustNot
	}
	if len(f) == 0 {
		return nil
	}
	return f
}

func toAnySlice(v any) []any {
	rv := reflect.ValueOf(v)
	if !rv.IsValid() || (rv.Kind() != reflect.Slice && rv.Kind() != reflect.Array) {
		return nil
	}
	out := make([]any, rv.Len())
	for i := range out {
		out[i] = rv.Index(i).Interface()
	}
	return out
}
