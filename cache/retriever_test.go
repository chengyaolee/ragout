package cache_test

import (
	"context"
	"errors"
	"testing"
	"testing/synctest"
	"time"

	"github.com/chengyaolee/ragout"
	"github.com/chengyaolee/ragout/cache"
)

type fakeRetriever struct {
	calls int
	err   error
	ans   ragout.Answer
}

func (f *fakeRetriever) Retrieve(context.Context, string, ...ragout.QueryOption) (ragout.Answer, error) {
	f.calls++
	return f.ans, f.err
}

// fakeEmbedder returns a set 2-d vector per text and counts calls.
type fakeEmbedder struct {
	calls int
	vecs  map[string][]float32
}

func (f *fakeEmbedder) EmbedBatch(_ context.Context, texts []string) ([][]float32, error) {
	f.calls++
	out := make([][]float32, len(texts))
	for i, t := range texts {
		out[i] = f.vecs[t]
	}
	return out, nil
}

func (f *fakeEmbedder) Dimension() int { return 2 }

func newCached(t *testing.T, p cache.Policy, store cache.Store) (*cache.Cached, *fakeRetriever, func(q string, opts ...ragout.QueryOption) bool) {
	t.Helper()
	inner := &fakeRetriever{ans: ragout.Answer{Question: "q", Retrieved: []ragout.ScoredChunk{{Score: 1}}}}
	if store == nil {
		store = cache.NewMemory(0)
	}
	c, err := cache.Wrap(inner, store, p)
	if err != nil {
		t.Fatal(err)
	}
	// ran reports whether the wrapped Retriever was called.
	ran := func(q string, opts ...ragout.QueryOption) bool {
		t.Helper()
		before := inner.calls
		if _, err := c.Retrieve(context.Background(), q, opts...); err != nil && !errors.Is(err, inner.err) {
			t.Fatal(err)
		}
		return inner.calls > before
	}
	return c, inner, ran
}

func TestCachedExact(t *testing.T) {
	synctest.Test(t, func(t *testing.T) {
		ctx := context.Background()
		c, inner, ran := newCached(t, cache.Policy{TTL: time.Hour}, nil)

		if !ran("How does Raft agree?") {
			t.Fatal("first query did not run")
		}
		if ran("  how does RAFT agree? ") {
			t.Fatal("same question with different case and spacing ran again")
		}
		if !ran("How does Raft agree?", ragout.WithQueryTopNRerank(1)) {
			t.Fatal("different TopNRerank was served from the cache")
		}
		if !ran("How does Raft agree?", ragout.WithQueryFilter(map[string]any{"k": "v"})) {
			t.Fatal("different filter was served from the cache")
		}

		// A caller reshaping its answer must not change what the next one gets.
		first, _ := c.Retrieve(ctx, "How does Raft agree?")
		first.Retrieved[0].Score = -1
		if again, _ := c.Retrieve(ctx, "How does Raft agree?"); again.Retrieved[0].Score != 1 {
			t.Fatal("stored answer was altered through a returned one")
		}

		time.Sleep(59 * time.Minute)
		if ran("How does Raft agree?") {
			t.Fatal("entry expired early")
		}
		time.Sleep(2 * time.Minute)
		if !ran("How does Raft agree?") {
			t.Fatal("entry outlived its ttl")
		}

		if err := c.Invalidate(ctx); err != nil {
			t.Fatal(err)
		}
		if !ran("How does Raft agree?") {
			t.Fatal("served from the cache after Invalidate")
		}

		// Degraded results and errors are not cached.
		inner.ans.TransformErr = errors.New("model down")
		ran("what is paxos")
		if !ran("what is paxos") {
			t.Fatal("a result whose transformer failed was cached")
		}
		inner.ans.TransformErr, inner.err = nil, ragout.ErrNoResults
		ran("nothing")
		if !ran("nothing") {
			t.Fatal("an error was cached")
		}
	})
}

func TestCachedTTLDefaultsToNone(t *testing.T) {
	synctest.Test(t, func(t *testing.T) {
		_, _, ran := newCached(t, cache.Policy{}, nil)
		ran("q")
		time.Sleep(10_000 * time.Hour)
		if ran("q") {
			t.Fatal("entry expired with no TTL set")
		}
	})
}

func TestCachedSemantic(t *testing.T) {
	ctx := context.Background()
	emb := &fakeEmbedder{vecs: map[string][]float32{
		"How does Raft agree?":           {1, 0},
		"How does Raft reach consensus?": {0.99, 0.14}, // cosine ~0.99
		"What is Paxos?":                 {0, 1},       // cosine 0
	}}

	for name, p := range map[string]cache.Policy{
		"no embedder":  {Mode: cache.Semantic, MinScore: 0.9},
		"no MinScore":  {Mode: cache.Semantic, Embedder: emb},
		"MinScore > 1": {Mode: cache.ExactThenSemantic, Embedder: emb, MinScore: 1.5},
	} {
		if _, err := cache.Wrap(&fakeRetriever{}, cache.NewMemory(0), p); err == nil {
			t.Fatalf("%s: invalid policy accepted", name)
		}
	}

	t.Run("semantic", func(t *testing.T) {
		c, _, ran := newCached(t, cache.Policy{Mode: cache.Semantic, Embedder: emb, MinScore: 0.95}, nil)
		ran("How does Raft agree?")
		if ran("How does Raft reach consensus?") {
			t.Fatal("paraphrase missed")
		}
		if !ran("What is Paxos?") {
			t.Fatal("unrelated question hit")
		}
		if !ran("How does Raft reach consensus?", ragout.WithQueryTopNRerank(1)) {
			t.Fatal("paraphrase with different options hit")
		}
		if err := c.Invalidate(ctx); err != nil {
			t.Fatal(err)
		}
		if !ran("How does Raft reach consensus?") {
			t.Fatal("paraphrase hit after Invalidate")
		}
	})

	t.Run("exact then semantic", func(t *testing.T) {
		_, _, ran := newCached(t, cache.Policy{Mode: cache.ExactThenSemantic, Embedder: emb, MinScore: 0.95}, nil)
		ran("How does Raft agree?")
		emb.calls = 0
		if ran("How does Raft agree?") || emb.calls != 0 {
			t.Fatalf("exact repeat: ran or embedded (%d calls); want an exact hit with no embedding", emb.calls)
		}
		if ran("How does Raft reach consensus?") {
			t.Fatal("paraphrase missed after the exact lookup")
		}
	})

	t.Run("exact never embeds", func(t *testing.T) {
		_, _, ran := newCached(t, cache.Policy{Embedder: emb, MinScore: 0.95}, nil)
		emb.calls = 0
		ran("How does Raft agree?")
		if !ran("How does Raft reach consensus?") || emb.calls != 0 {
			t.Fatal("exact mode matched a paraphrase or embedded")
		}
	})

	t.Run("evicted answer is a miss", func(t *testing.T) {
		_, _, ran := newCached(t, cache.Policy{Mode: cache.Semantic, Embedder: emb, MinScore: 0.95}, cache.NewMemory(1))
		ran("How does Raft agree?")
		ran("What is Paxos?") // evicts the Raft answer from the store
		if !ran("How does Raft reach consensus?") {
			t.Fatal("hit on a vector whose answer was evicted")
		}
	})

	t.Run("MaxIndexed drops the oldest vector", func(t *testing.T) {
		_, _, ran := newCached(t, cache.Policy{Mode: cache.Semantic, Embedder: emb, MinScore: 0.95, MaxIndexed: 1}, nil)
		ran("How does Raft agree?")
		ran("What is Paxos?")
		if !ran("How does Raft reach consensus?") {
			t.Fatal("hit on a vector past MaxIndexed")
		}
	})
}
