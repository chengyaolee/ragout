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

// Reserved Chroma metadata keys used to recover chunk fields that Chroma itself has
// no column for.
const (
	chromaMetaDocumentID = "ragout_document_id"
	chromaMetaIndex      = "ragout_index"
)

// ChromaStore implements ragout.VectorStore against a Chroma server over its v2 REST
// API (net/http only; no client SDK dependency).
type ChromaStore struct {
	baseURL      string
	tenant       string
	database     string
	collection   string
	collectionID string
	httpClient   *http.Client
}

type ChromaOption func(*ChromaStore)

// WithChromaTenant overrides the default tenant ("default_tenant").
func WithChromaTenant(tenant string) ChromaOption {
	return func(c *ChromaStore) { c.tenant = tenant }
}

// WithChromaDatabase overrides the default database ("default_database").
func WithChromaDatabase(database string) ChromaOption {
	return func(c *ChromaStore) { c.database = database }
}

// WithChromaHTTPClient overrides the HTTP client. Use it to set auth headers via a
// custom http.RoundTripper (e.g. a Chroma API token).
func WithChromaHTTPClient(client *http.Client) ChromaOption {
	return func(c *ChromaStore) {
		if client != nil {
			c.httpClient = client
		}
	}
}

// NewChromaStore connects to baseURL (e.g. "http://localhost:8000") and gets-or-creates
// the named collection.
func NewChromaStore(ctx context.Context, baseURL, collection string, opts ...ChromaOption) (*ChromaStore, error) {
	cs := &ChromaStore{
		baseURL:    baseURL,
		tenant:     "default_tenant",
		database:   "default_database",
		collection: collection,
		httpClient: &http.Client{Timeout: 30 * time.Second},
	}
	for _, opt := range opts {
		opt(cs)
	}

	id, err := cs.getOrCreateCollection(ctx)
	if err != nil {
		return nil, err
	}
	cs.collectionID = id
	return cs, nil
}

func (cs *ChromaStore) collectionsURL() string {
	return fmt.Sprintf("%s/api/v2/tenants/%s/databases/%s/collections", cs.baseURL, cs.tenant, cs.database)
}

type chromaCollection struct {
	ID string `json:"id"`
}

func (cs *ChromaStore) getOrCreateCollection(ctx context.Context) (string, error) {
	payload := map[string]any{
		"name":          cs.collection,
		"get_or_create": true,
		"metadata":      map[string]any{"hnsw:space": "cosine"},
	}
	var col chromaCollection
	if err := cs.do(ctx, http.MethodPost, cs.collectionsURL(), payload, &col); err != nil {
		return "", fmt.Errorf("chroma: get-or-create collection: %w", err)
	}
	if col.ID == "" {
		return "", fmt.Errorf("chroma: get-or-create collection: empty collection id in response")
	}
	return col.ID, nil
}

func (cs *ChromaStore) do(ctx context.Context, method, url string, payload any, out any) error {
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

	resp, err := cs.httpClient.Do(req)
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

// Upsert adds or updates chunks in the Chroma collection.
func (cs *ChromaStore) Upsert(ctx context.Context, chunks []ragout.Chunk) error {
	if err := ctx.Err(); err != nil {
		return err
	}
	if len(chunks) == 0 {
		return nil
	}

	ids := make([]string, len(chunks))
	embeddings := make([][]float32, len(chunks))
	documents := make([]string, len(chunks))
	metadatas := make([]map[string]any, len(chunks))

	for i, c := range chunks {
		ids[i] = c.ID
		embeddings[i] = c.Embedding
		documents[i] = c.Content
		meta := ragout.CloneMetadata(c.Metadata)
		meta[chromaMetaDocumentID] = c.DocumentID
		meta[chromaMetaIndex] = c.Index
		metadatas[i] = meta
	}

	payload := map[string]any{
		"ids":        ids,
		"embeddings": embeddings,
		"documents":  documents,
		"metadatas":  metadatas,
	}
	url := fmt.Sprintf("%s/%s/upsert", cs.collectionsURL(), cs.collectionID)
	if err := cs.do(ctx, http.MethodPost, url, payload, nil); err != nil {
		return fmt.Errorf("chroma: upsert: %w", err)
	}
	return nil
}

// Delete removes chunks from the collection by ID.
func (cs *ChromaStore) Delete(ctx context.Context, ids []string) error {
	if err := ctx.Err(); err != nil {
		return err
	}
	if len(ids) == 0 {
		return nil
	}
	url := fmt.Sprintf("%s/%s/delete", cs.collectionsURL(), cs.collectionID)
	if err := cs.do(ctx, http.MethodPost, url, map[string]any{"ids": ids}, nil); err != nil {
		return fmt.Errorf("chroma: delete: %w", err)
	}
	return nil
}

type chromaQueryResponse struct {
	IDs        [][]string         `json:"ids"`
	Distances  [][]float64        `json:"distances"`
	Documents  [][]string         `json:"documents"`
	Metadatas  [][]map[string]any `json:"metadatas"`
	Embeddings [][][]float32      `json:"embeddings"`
}

// SearchDense searches for the topK most similar chunks to queryVector.
func (cs *ChromaStore) SearchDense(ctx context.Context, queryVector []float32, topK int, filter map[string]any) ([]ragout.ScoredChunk, error) {
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	if topK <= 0 {
		return []ragout.ScoredChunk{}, nil
	}

	payload := map[string]any{
		"query_embeddings": [][]float32{queryVector},
		"n_results":        topK,
		"include":          []string{"documents", "metadatas", "distances", "embeddings"},
	}
	if where := chromaWhere(filter); where != nil {
		payload["where"] = where
	}

	url := fmt.Sprintf("%s/%s/query", cs.collectionsURL(), cs.collectionID)
	var res chromaQueryResponse
	if err := cs.do(ctx, http.MethodPost, url, payload, &res); err != nil {
		return nil, fmt.Errorf("chroma: query: %w", err)
	}
	if len(res.IDs) == 0 {
		return []ragout.ScoredChunk{}, nil
	}

	ids := res.IDs[0]
	scored := make([]ragout.ScoredChunk, 0, len(ids))
	for i, id := range ids {
		chunk := chromaChunk(id, i, at(res.Documents), at(res.Metadatas))
		if emb := at(res.Embeddings); i < len(emb) {
			chunk.Embedding = emb[i]
		}

		var dist float64
		if len(res.Distances) > 0 && i < len(res.Distances[0]) {
			dist = res.Distances[0][i]
		}
		score := 1 - dist
		scored = append(scored, ragout.ScoredChunk{
			Chunk:      chunk,
			DenseScore: score,
			Score:      score,
		})
	}

	return scored, nil
}

// GetChunks fetches chunks by ID. IDs not in the collection are skipped.
func (cs *ChromaStore) GetChunks(ctx context.Context, ids []string) ([]ragout.Chunk, error) {
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	if len(ids) == 0 {
		return []ragout.Chunk{}, nil
	}
	payload := map[string]any{"ids": ids, "include": []string{"documents", "metadatas"}}
	var res struct {
		IDs       []string         `json:"ids"`
		Documents []string         `json:"documents"`
		Metadatas []map[string]any `json:"metadatas"`
	}
	url := fmt.Sprintf("%s/%s/get", cs.collectionsURL(), cs.collectionID)
	if err := cs.do(ctx, http.MethodPost, url, payload, &res); err != nil {
		return nil, fmt.Errorf("chroma: get: %w", err)
	}
	out := make([]ragout.Chunk, len(res.IDs))
	for i, id := range res.IDs {
		out[i] = chromaChunk(id, i, res.Documents, res.Metadatas)
	}
	return out, nil
}

// chromaChunk decodes record i of a Chroma result, moving ragout's bookkeeping
// fields out of the metadata.
func chromaChunk(id string, i int, docs []string, metas []map[string]any) ragout.Chunk {
	chunk := ragout.Chunk{ID: id, Metadata: map[string]any{}}
	if i < len(docs) {
		chunk.Content = docs[i]
	}
	if i < len(metas) && metas[i] != nil {
		meta := metas[i]
		if docID, ok := meta[chromaMetaDocumentID].(string); ok {
			chunk.DocumentID = docID
			delete(meta, chromaMetaDocumentID)
		}
		if idx, ok := meta[chromaMetaIndex].(float64); ok {
			chunk.Index = int(idx)
			delete(meta, chromaMetaIndex)
		}
		chunk.Metadata = meta
	}
	return chunk
}

// at returns the first row of a batched (one list per query) Chroma field.
func at[T any](rows [][]T) []T {
	if len(rows) == 0 {
		return nil
	}
	return rows[0]
}

// chromaWhere translates a ragout metadata filter (matches ragout's $eq/$ne/$in/$nin/
// $gt/$gte/$lt/$lte operators, which Chroma's where clause already shares) into a
// Chroma "where" payload. A bare value becomes $eq and a slice becomes $in.
func chromaWhere(filter map[string]any) map[string]any {
	if len(filter) == 0 {
		return nil
	}

	var clauses []map[string]any
	for key, want := range filter {
		if ops, ok := want.(map[string]any); ok {
			for op, val := range ops {
				clauses = append(clauses, map[string]any{key: map[string]any{op: val}})
			}
			continue
		}
		v := reflect.ValueOf(want)
		if v.IsValid() && (v.Kind() == reflect.Slice || v.Kind() == reflect.Array) {
			items := make([]any, v.Len())
			for i := range items {
				items[i] = v.Index(i).Interface()
			}
			clauses = append(clauses, map[string]any{key: map[string]any{"$in": items}})
			continue
		}
		clauses = append(clauses, map[string]any{key: map[string]any{"$eq": want}})
	}

	if len(clauses) == 1 {
		return clauses[0]
	}
	return map[string]any{"$and": clauses}
}
