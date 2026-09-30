package ragout

import "errors"

var (
	ErrNilReader    = errors.New("ragout: reader is required for ingestion")
	ErrNilChunker   = errors.New("ragout: chunker is required for ingestion")
	ErrNilEmbedder  = errors.New("ragout: embedder is required for dense retrieval")
	ErrNilGenerator = errors.New("ragout: generator is required for queries")
	ErrNoStores     = errors.New("ragout: at least one storage store (vector or index) must be configured")
	ErrEmptyQuery   = errors.New("ragout: query text cannot be empty")
	ErrNoResults    = errors.New("ragout: no relevant documents found for query")
)

type Engine struct {
	reader    Reader
	chunker   Chunker
	embedder  Embedder
	vStore    VectorStore
	iStore    IndexStore
	reranker  Reranker
	generator Generator

	topKRecall       int // Number of candidates recalled from each store (default: 50)
	topNRerank       int // Number of candidates passed to generator (default: 5)
	rrfK             int // RRF smoothing denominator constant (default: 60)
	embedBatchSize   int // Embedding batch size (default: 32)
	embedConcurrency int // Concurrency worker pool size (default: 8)
}

func NewEngine(options ...EngineOption) (*Engine, error) {
	e := &Engine{
		topKRecall:       50,
		topNRerank:       5,
		rrfK:             60,
		embedBatchSize:   32,
		embedConcurrency: 8,
	}

	for _, opt := range options {
		opt(e)
	}

	// Validation: At least one store must exist
	if e.vStore == nil && e.iStore == nil {
		return nil, ErrNoStores
	}
	// If dense store is configured, embedder must be provided
	if e.vStore != nil && e.embedder == nil {
		return nil, ErrNilEmbedder
	}

	return e, nil
}
