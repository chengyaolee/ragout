package cache

import (
	"context"
	"errors"
	"fmt"
	"slices"
	"strings"
	"sync"
	"sync/atomic"
	"time"

	"github.com/chengyaolee/ragout"
	"github.com/chengyaolee/ragout/store"
)

// Retriever is what Cached wraps and implements. *ragout.Engine satisfies it, and so
// does flow.Retriever, so a Cached drops into flow.Retrieve.
type Retriever interface {
	Retrieve(ctx context.Context, query string, opts ...ragout.QueryOption) (ragout.Answer, error)
}

// Store is the cache backend: answers by key. Memory is the in-process one; a Redis
// or SQLite Store plugs in the same way. Get must not return an entry whose ttl has
// passed; ttl 0 means it does not expire. The engine copies answers in both directions,
// so a Store may keep the one it is given.
type Store interface {
	Get(ctx context.Context, key string) (ragout.Answer, bool, error)
	Set(ctx context.Context, key string, a ragout.Answer, ttl time.Duration) error
}

// Mode is how a question is matched to an earlier one.
type Mode int

const (
	// Exact matches the same question, ignoring case and spacing. No model call.
	Exact Mode = iota
	// Semantic embeds the question and matches the nearest earlier one scoring at
	// least Policy.MinScore. Every lookup costs an embedding call.
	Semantic
	// ExactThenSemantic tries Exact, and Semantic only on a miss.
	ExactThenSemantic
)

// Policy says what is cached and for how long.
type Policy struct {
	Mode Mode
	// TTL is how long an entry is served after it was written. 0 keeps it until the
	// Store evicts it or Invalidate is called.
	TTL time.Duration

	// The fields below apply to Semantic and ExactThenSemantic.

	// Embedder embeds each question looked up. Required. The cache ships no model: pass
	// the engine's own embedder, or any other, such as a small local one. Switching
	// models means clearing Index, since vectors of different models do not compare.
	Embedder ragout.Embedder
	// Index holds the cached questions' vectors. nil uses store.NewVectorStore, a
	// brute-force in-memory scan; any ragout.VectorStore (Qdrant, Chroma) can replace it.
	Index ragout.VectorStore
	// MinScore is the lowest similarity, as Index scores it, that counts as the same
	// question. Required, in (0, 1]. There is no default, because scores do not compare
	// across models: with nomic-embed-text, paraphrases scored 0.67 to 0.92 and
	// near-misses such as "maximum" against "minimum" upload size scored up to 0.90.
	// Start high, measure false hits on your own questions, and lower it with care.
	MinScore float64
	// MaxIndexed caps the vectors this Cached keeps in Index, dropping the oldest.
	// 0 means 10,000: a lookup in the default Index scans every one.
	MaxIndexed int
}

const (
	defaultMaxIndexed = 10_000
	scopeKey          = "cache_scope" // Index metadata: which options and generation a vector belongs to
)

// Cached reuses a Retriever's results between matching queries. The engine itself
// never caches; this is an opt-in wrapper around it.
//
// Lookups and writes fail open: a Store, Index or Embedder error is a cache miss, and
// the query runs as if there were no cache.
type Cached struct {
	inner Retriever
	store Store
	p     Policy
	gen   atomic.Uint64

	mu      sync.Mutex
	indexed []indexedVec // vectors written to p.Index, oldest first
}

type indexedVec struct {
	id      string
	expires time.Time
}

// Wrap returns inner with its results kept in backend under policy p.
func Wrap(inner Retriever, backend Store, p Policy) (*Cached, error) {
	if inner == nil {
		return nil, errors.New("cache: retriever is nil")
	}
	if backend == nil {
		return nil, errors.New("cache: store is nil")
	}
	if p.TTL < 0 {
		return nil, errors.New("cache: ttl must be >= 0")
	}
	switch p.Mode {
	case Exact:
	case Semantic, ExactThenSemantic:
		if p.Embedder == nil {
			return nil, errors.New("cache: semantic matching needs an embedder")
		}
		if p.Index == nil {
			p.Index = store.NewVectorStore()
		}
		if p.MinScore <= 0 || p.MinScore > 1 {
			return nil, errors.New("cache: semantic matching needs MinScore in (0, 1]; scores differ by model, so there is no default")
		}
		if p.MaxIndexed <= 0 {
			p.MaxIndexed = defaultMaxIndexed
		}
	default:
		return nil, fmt.Errorf("cache: unknown mode %d", p.Mode)
	}
	return &Cached{inner: inner, store: backend, p: p}, nil
}

// Invalidate makes every earlier entry unreachable and removes this Cached's vectors
// from Index when the Index supports ragout.SourceDeleter. Call it after Ingest or
// DeleteSource on the wrapped engine. It only affects this process: changes made
// elsewhere, such as by another process sharing the Store, show up when entries expire.
func (c *Cached) Invalidate(ctx context.Context) error {
	old := c.gen.Add(1) - 1
	if d, ok := c.p.Index.(ragout.SourceDeleter); ok {
		return d.DeleteSource(ctx, genSource(old), nil)
	}
	return nil
}

// Retrieve returns a cached result matching the question and options, or runs the
// wrapped Retriever and caches what it returns. Errors and degraded results (a failed
// transformer or dense search) are not cached.
func (c *Cached) Retrieve(ctx context.Context, query string, opts ...ragout.QueryOption) (ragout.Answer, error) {
	scope, gen := c.scope(opts)
	text := strings.ToLower(strings.Join(strings.Fields(query), " "))
	key := ragout.DocumentID(scope, text)

	if c.p.Mode != Semantic {
		if a, ok := c.get(ctx, key); ok {
			return a, nil
		}
	}
	var vec []float32
	if c.p.Mode != Exact {
		if vecs, err := c.p.Embedder.EmbedBatch(ctx, []string{query}); err == nil && len(vecs) == 1 {
			vec = vecs[0]
			if a, ok := c.nearest(ctx, vec, scope); ok {
				return a, nil
			}
		}
	}

	a, err := c.inner.Retrieve(ctx, query, opts...)
	if err != nil || a.TransformErr != nil || a.DenseErr != nil {
		return a, err
	}
	if c.store.Set(ctx, key, clone(a), c.p.TTL) == nil && vec != nil {
		c.index(ctx, key, text, vec, scope, gen)
	}
	return a, nil
}

// scope identifies everything but the question that a cached result depends on: the
// per-query options and the invalidation generation. Keys and scopes are UUIDs, which
// every ragout store accepts as an ID.
func (c *Cached) scope(opts []ragout.QueryOption) (string, uint64) {
	var p ragout.QueryParams
	for _, opt := range opts {
		opt(&p)
	}
	gen := c.gen.Load()
	return ragout.DocumentID("ragout-cache", fmt.Sprintf("%d %d %d %#v %#v",
		gen, p.TopKRecall, p.TopNRerank, p.Filter, p.History)), gen
}

func (c *Cached) get(ctx context.Context, key string) (ragout.Answer, bool) {
	a, ok, err := c.store.Get(ctx, key)
	if err != nil || !ok {
		return ragout.Answer{}, false
	}
	return clone(a), true
}

// nearest returns the answer of the closest cached question in scope, if close enough.
func (c *Cached) nearest(ctx context.Context, vec []float32, scope string) (ragout.Answer, bool) {
	hits, err := c.p.Index.SearchDense(ctx, vec, 1, map[string]any{scopeKey: scope})
	if err != nil || len(hits) == 0 || hits[0].Score < c.p.MinScore {
		return ragout.Answer{}, false
	}
	id := hits[0].Chunk.ID
	if a, ok := c.get(ctx, id); ok {
		return a, true
	}
	// Its answer expired or was evicted from the Store.
	_ = c.p.Index.Delete(ctx, []string{id})
	return ragout.Answer{}, false
}

// index adds a question's vector, first dropping vectors past TTL or over MaxIndexed.
// Because every vector lives for the same TTL, write order is also expiry order.
func (c *Cached) index(ctx context.Context, id, text string, vec []float32, scope string, gen uint64) {
	now := time.Now()
	c.mu.Lock()
	var drop []string
	for len(c.indexed) > 0 && (len(c.indexed) >= c.p.MaxIndexed ||
		c.p.TTL > 0 && !now.Before(c.indexed[0].expires)) {
		drop = append(drop, c.indexed[0].id)
		c.indexed = c.indexed[1:]
	}
	c.indexed = append(c.indexed, indexedVec{id: id, expires: now.Add(c.p.TTL)})
	c.mu.Unlock()

	if len(drop) > 0 {
		_ = c.p.Index.Delete(ctx, drop)
	}
	_ = c.p.Index.Upsert(ctx, []ragout.Chunk{{
		ID:        id,
		Content:   text,
		Embedding: vec,
		Metadata:  map[string]any{scopeKey: scope, ragout.MetadataSource: genSource(gen)},
	}})
}

func genSource(gen uint64) string { return fmt.Sprintf("ragout-cache/%d", gen) }

// clone copies an Answer's slices so a caller cannot alter what the store holds. The
// chunks themselves are shared.
func clone(a ragout.Answer) ragout.Answer {
	a.Searches = slices.Clone(a.Searches)
	a.Retrieved = slices.Clone(a.Retrieved)
	a.Sources = slices.Clone(a.Sources)
	return a
}
