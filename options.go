package ragout

type EngineOption func(*Engine)

func WithReader(r Reader) EngineOption            { return func(e *Engine) { e.reader = r } }
func WithChunker(c Chunker) EngineOption          { return func(e *Engine) { e.chunker = c } }
func WithEmbedder(em Embedder) EngineOption       { return func(e *Engine) { e.embedder = em } }
func WithVectorStore(vs VectorStore) EngineOption { return func(e *Engine) { e.vStore = vs } }
func WithIndexStore(is IndexStore) EngineOption   { return func(e *Engine) { e.iStore = is } }
func WithReranker(rr Reranker) EngineOption       { return func(e *Engine) { e.reranker = rr } }
func WithGenerator(g Generator) EngineOption      { return func(e *Engine) { e.generator = g } }
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
}

// QueryOption configures individual query executions.
type QueryOption func(*QueryParams)

// WithQueryFilter sets a metadata filter constraint for the query.
func WithQueryFilter(filter map[string]any) QueryOption {
	return func(p *QueryParams) {
		p.Filter = filter
	}
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
