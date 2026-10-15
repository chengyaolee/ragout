package store

import (
	"context"
	"fmt"
	"net/http"
	"sync"

	"github.com/chengyaolee/ragout"
)

// sourceIDs lists the IDs of chunks from source, minus keep.
func sourceIDs(mu *sync.RWMutex, chunks map[string]ragout.Chunk, source string, keep []string) []string {
	skip := make(map[string]bool, len(keep))
	for _, id := range keep {
		skip[id] = true
	}
	mu.RLock()
	defer mu.RUnlock()
	var ids []string
	for id, c := range chunks {
		if src, _ := c.Metadata[ragout.MetadataSource].(string); src == source && !skip[id] {
			ids = append(ids, id)
		}
	}
	return ids
}

// DeleteSource removes every chunk whose metadata source is source, except keep.
func (vs *VectorStore) DeleteSource(ctx context.Context, source string, keep []string) error {
	return vs.Delete(ctx, sourceIDs(&vs.mu, vs.chunks, source, keep))
}

// DeleteSource removes every chunk whose metadata source is source, except keep.
func (idx *BM25Index) DeleteSource(ctx context.Context, source string, keep []string) error {
	return idx.Delete(ctx, sourceIDs(&idx.mu, idx.chunks, source, keep))
}

// DeleteSource removes every chunk whose metadata source is source, except keep.
// Chroma cannot exclude IDs in a delete, so the source's IDs are fetched first.
func (cs *ChromaStore) DeleteSource(ctx context.Context, source string, keep []string) error {
	if err := ctx.Err(); err != nil {
		return err
	}
	payload := map[string]any{
		"where":   map[string]any{ragout.MetadataSource: map[string]any{"$eq": source}},
		"include": []string{"metadatas"},
	}
	var res struct {
		IDs []string `json:"ids"`
	}
	url := fmt.Sprintf("%s/%s/get", cs.collectionsURL(), cs.collectionID)
	if err := cs.do(ctx, http.MethodPost, url, payload, &res); err != nil {
		return fmt.Errorf("chroma: list source: %w", err)
	}
	skip := make(map[string]bool, len(keep))
	for _, id := range keep {
		skip[id] = true
	}
	var ids []string
	for _, id := range res.IDs {
		if !skip[id] {
			ids = append(ids, id)
		}
	}
	return cs.Delete(ctx, ids)
}

// DeleteSource removes every point whose payload source is source, except keep.
func (qs *QdrantStore) DeleteSource(ctx context.Context, source string, keep []string) error {
	if err := ctx.Err(); err != nil {
		return err
	}
	filter := map[string]any{
		"must": []map[string]any{{"key": ragout.MetadataSource, "match": map[string]any{"value": source}}},
	}
	if len(keep) > 0 {
		filter["must_not"] = []map[string]any{{"has_id": keep}}
	}
	url := fmt.Sprintf("%s/collections/%s/points/delete?wait=true", qs.baseURL, qs.collection)
	if err := qs.do(ctx, http.MethodPost, url, map[string]any{"filter": filter}, nil); err != nil {
		return fmt.Errorf("qdrant: delete source: %w", err)
	}
	return nil
}
