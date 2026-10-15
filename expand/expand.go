// Package expand implements ragout.Expander: widening retrieved hits before generation.
package expand

import (
	"context"
	"strings"

	"github.com/chengyaolee/ragout"
)

// Neighbors returns an Expander that replaces each hit's Content with the hit plus up
// to n chunks on either side from the same document, fetched from store in one call.
// The hit keeps its ID, score and citation slot. A hit already included in a
// higher-ranked hit's window is dropped instead of repeated.
func Neighbors(store ragout.ChunkGetter, n int) ragout.Expander {
	return neighbors{store: store, n: n}
}

// Section returns an Expander that replaces each hit with its whole section, up to
// maxChunks on either side (parent-child retrieval). Section-aware readers emit one
// document per heading and windows never cross documents, so this is Neighbors with a
// window wide enough to cover the section.
func Section(store ragout.ChunkGetter, maxChunks int) ragout.Expander {
	return neighbors{store: store, n: maxChunks}
}

type neighbors struct {
	store ragout.ChunkGetter
	n     int
}

func (x neighbors) Expand(ctx context.Context, hits []ragout.ScoredChunk) ([]ragout.ScoredChunk, error) {
	if x.n <= 0 || len(hits) == 0 {
		return hits, nil
	}

	want := map[string]struct{}{}
	var ids []string
	for _, h := range hits {
		if h.Chunk.DocumentID == "" {
			continue
		}
		for i := max(0, h.Chunk.Index-x.n); i <= h.Chunk.Index+x.n; i++ {
			id := ragout.ChunkID(h.Chunk.DocumentID, i)
			if _, ok := want[id]; !ok && i != h.Chunk.Index {
				want[id] = struct{}{}
				ids = append(ids, id)
			}
		}
	}
	found, err := x.store.GetChunks(ctx, ids)
	if err != nil {
		return nil, err
	}
	byID := make(map[string]ragout.Chunk, len(found))
	for _, c := range found {
		byID[c.ID] = c
	}

	covered := map[string]bool{}
	out := make([]ragout.ScoredChunk, 0, len(hits))
	for _, h := range hits {
		if covered[h.Chunk.ID] {
			continue
		}
		covered[h.Chunk.ID] = true
		if h.Chunk.DocumentID == "" {
			out = append(out, h)
			continue
		}
		var parts []string
		for i := max(0, h.Chunk.Index-x.n); i <= h.Chunk.Index+x.n; i++ {
			id := ragout.ChunkID(h.Chunk.DocumentID, i)
			if i == h.Chunk.Index {
				parts = append(parts, h.Chunk.Content)
			} else if c, ok := byID[id]; ok {
				parts = append(parts, c.Content)
			} else {
				continue
			}
			covered[id] = true
		}
		// Chunker overlap is repeated at each seam; trim it if prompts get tight.
		h.Chunk.Content = strings.Join(parts, "\n")
		out = append(out, h)
	}
	return out, nil
}
