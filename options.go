package ragout

import "time"

type EngineOption func(*Engine)

func WithReader(r Reader) EngineOption            { return func(e *Engine) { e.reader = r } }
func WithChunker(c Chunker) EngineOption          { return func(e *Engine) { e.chunker = c } }
func WithEmbedder(em Embedder) EngineOption       { return func(e *Engine) { e.embedder = em } }
func WithVectorStore(vs VectorStore) EngineOption { return func(e *Engine) { e.vStore = vs } }
func WithIndexStore(is IndexStore) EngineOption   { return func(e *Engine) { e.iStore = is } }
func WithReranker(rr Reranker) EngineOption       { return func(e *Engine) { e.reranker = rr } }
func WithGenerator(g Generator) EngineOption      { return func(e *Engine) { e.generator = g } }

// WithContextualizer sets how each chunk's indexed text is enriched at ingest. The
// default is MetadataContext; nil turns enrichment off.
func WithContextualizer(c Contextualizer) EngineOption {
	return func(e *Engine) { e.contextualizer = c }
}

// WithQueryTransformer rewrites each question before retrieval (see the transform
// package). The default is none: the question is searched as written.
func WithQueryTransformer(t QueryTransformer) EngineOption {
	return func(e *Engine) { e.transformer = t }
}

// WithTransformTimeout bounds the QueryTransformer call. When it runs out, the query
// continues with the original question, as it does on a transformer error. The default
// (0) applies no extra deadline.
func WithTransformTimeout(d time.Duration) EngineOption {
	return func(e *Engine) { e.transformTimeout = d }
}

// WithFusion replaces the default RRF(WithRRFK) fusion of retrieval result lists.
func WithFusion(f FusionStrategy) EngineOption {
	return func(e *Engine) { e.fusion = f }
}

// WithExpander rewrites the final candidates before generation (e.g. expand.Neighbors).
func WithExpander(x Expander) EngineOption {
	return func(e *Engine) { e.expander = x }
}
func WithTopKRecall(k int) EngineOption {
	return func(e *Engine) {
		if k > 0 {
			e.topKRecall = k
		}
	}
}
func WithTopNRerank(n int) EngineOption {
	return func(e *Engine) {
		if n > 0 {
			e.topNRerank = n
		}
	}
}
func WithRRFK(k int) EngineOption {
	return func(e *Engine) {
		if k > 0 {
			e.rrfK = k
		}
	}
}
func WithEmbedBatchSize(size int) EngineOption {
	return func(e *Engine) {
		if size > 0 {
			e.embedBatchSize = size
		}
	}
}
func WithEmbedConcurrency(workers int) EngineOption {
	return func(e *Engine) {
		if workers > 0 {
			e.embedConcurrency = workers
		}
	}
}

// QueryParams holds per-query options.
type QueryParams struct {
	Filter     map[string]any
	TopKRecall int
	TopNRerank int
	History    []Turn // earlier turns, oldest first; only a QueryTransformer reads it
}

// QueryOption configures individual query executions.
type QueryOption func(*QueryParams)

// WithQueryFilter sets a metadata filter constraint for the query.
func WithQueryFilter(filter map[string]any) QueryOption {
	return func(p *QueryParams) {
		p.Filter = filter
	}
}

// WithHistory passes the conversation so far, oldest first. The engine keeps no state
// between queries; a QueryTransformer such as transform.Condense uses it to turn a
// follow-up into a standalone question.
func WithHistory(turns []Turn) QueryOption {
	return func(p *QueryParams) { p.History = turns }
}

// WithQueryTopKRecall overrides topKRecall for this specific query.
func WithQueryTopKRecall(k int) QueryOption {
	return func(p *QueryParams) {
		if k > 0 {
			p.TopKRecall = k
		}
	}
}

// WithQueryTopNRerank overrides topNRerank for this specific query.
func WithQueryTopNRerank(n int) QueryOption {
	return func(p *QueryParams) {
		if n > 0 {
			p.TopNRerank = n
		}
	}
}
