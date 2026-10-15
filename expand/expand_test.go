package expand

import (
	"context"
	"testing"

	"github.com/chengyaolee/ragout"
)

type getterFunc func(ctx context.Context, ids []string) ([]ragout.Chunk, error)

func (f getterFunc) GetChunks(ctx context.Context, ids []string) ([]ragout.Chunk, error) {
	return f(ctx, ids)
}

func TestNeighbors(t *testing.T) {
	doc := "doc"
	stored := map[string]ragout.Chunk{}
	for i, s := range []string{"c0", "c1", "c2", "c3", "c4", "c5"} {
		id := ragout.ChunkID(doc, i)
		stored[id] = ragout.Chunk{ID: id, DocumentID: doc, Index: i, Content: s}
	}
	calls := 0
	store := getterFunc(func(_ context.Context, ids []string) ([]ragout.Chunk, error) {
		calls++
		var out []ragout.Chunk
		for _, id := range ids {
			if c, ok := stored[id]; ok {
				out = append(out, c)
			}
		}
		return out, nil
	})
	hit := func(i int) ragout.ScoredChunk {
		return ragout.ScoredChunk{Chunk: stored[ragout.ChunkID(doc, i)], Score: float64(10 - i)}
	}

	// c1's window covers c2, so the lower-ranked c2 hit is dropped; c0 is the first chunk.
	got, err := Neighbors(store, 1).Expand(context.Background(), []ragout.ScoredChunk{hit(1), hit(2), hit(5)})
	if err != nil {
		t.Fatal(err)
	}
	want := []string{"c0\nc1\nc2", "c4\nc5"}
	if len(got) != len(want) || calls != 1 {
		t.Fatalf("got %d hits in %d calls, want %d in 1: %+v", len(got), calls, len(want), got)
	}
	for i := range want {
		if got[i].Chunk.Content != want[i] {
			t.Errorf("hit %d content = %q, want %q", i, got[i].Chunk.Content, want[i])
		}
	}
	if got[0].Chunk.ID != ragout.ChunkID(doc, 1) || got[0].Score != 9 {
		t.Errorf("hit keeps its own ID and score, got %+v", got[0])
	}

	// Section: a wide window returns the whole document.
	got, _ = Section(store, 50).Expand(context.Background(), []ragout.ScoredChunk{hit(3)})
	if got[0].Chunk.Content != "c0\nc1\nc2\nc3\nc4\nc5" {
		t.Errorf("Section content = %q", got[0].Chunk.Content)
	}

	// A hit with a custom ID keeps its own text; one with no document is passed through.
	custom := ragout.ScoredChunk{Chunk: ragout.Chunk{ID: "custom", DocumentID: doc, Index: 3, Content: "mine"}}
	loose := ragout.ScoredChunk{Chunk: ragout.Chunk{ID: "loose", Content: "alone"}}
	got, _ = Neighbors(store, 1).Expand(context.Background(), []ragout.ScoredChunk{custom, loose})
	if len(got) != 2 || got[0].Chunk.Content != "c2\nmine\nc4" || got[1].Chunk.Content != "alone" {
		t.Errorf("got %+v", got)
	}
}
