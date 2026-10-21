package ragout

import (
	"context"
	"errors"
	"fmt"
	"time"
)

var (
	ErrNilReader    = errors.New("ragout: reader is required for ingestion")
	ErrNilChunker   = errors.New("ragout: chunker is required for ingestion")
	ErrNilEmbedder  = errors.New("ragout: embedder is required for dense retrieval")
	ErrNilGenerator = errors.New("ragout: generator is required for queries")
	ErrNoStores     = errors.New("ragout: at least one storage store (vector or index) must be configured")
	ErrEmptyQuery   = errors.New("ragout: query text cannot be empty")
	ErrNoResults    = errors.New("ragout: no relevant documents found for query")
	ErrNoSource     = errors.New("ragout: source must not be empty")
	// ErrDeleteUnsupported means a configured store does not implement SourceDeleter.
	ErrDeleteUnsupported = errors.New("ragout: a configured store cannot delete by source")
)

type Engine struct {
	reader    Reader
	chunker   Chunker
	embedder  Embedder
	vStore    VectorStore
	iStore    IndexStore
	reranker  Reranker
	generator Generator

	contextualizer   Contextualizer // situational context indexed with each chunk (default: MetadataContext)
	expander         Expander       // optional: rewrites final candidates before generation
	transformer      QueryTransformer
	transformTimeout time.Duration  // 0: the transformer runs under the query's own deadline
	replaceSource    bool           // Ingest removes the source's previous chunks after writing
	fusion           FusionStrategy // default: WeightedRRF(rrfK, DefaultRewriteWeight)

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
		contextualizer:   MetadataContext(),
	}

	for _, opt := range options {
		opt(e)
	}

	if e.fusion == nil {
		e.fusion = WeightedRRF(e.rrfK, DefaultRewriteWeight)
	}

	// Validation: At least one store must exist
	if e.vStore == nil && e.iStore == nil {
		return nil, ErrNoStores
	}
	// If dense store is configured, embedder must be provided
	if e.vStore != nil && e.embedder == nil {
		return nil, ErrNilEmbedder
	}
	if e.replaceSource {
		if _, err := e.sourceDeleters(); err != nil {
			return nil, err
		}
	}

	return e, nil
}

// DeleteSource removes every chunk previously ingested from source, from every
// configured store. Each store must implement SourceDeleter.
func (e *Engine) DeleteSource(ctx context.Context, source string) error {
	return e.deleteSource(ctx, source, nil)
}

// deleteSource removes source's chunks except keep. Every store is checked before any
// is touched, so an unsupported store never leaves the others half-deleted.
func (e *Engine) deleteSource(ctx context.Context, source string, keep []string) error {
	if source == "" {
		return ErrNoSource
	}
	deleters, err := e.sourceDeleters()
	if err != nil {
		return err
	}
	var errs []error
	for _, d := range deleters {
		if err := d.DeleteSource(ctx, source, keep); err != nil {
			errs = append(errs, err)
		}
	}
	return errors.Join(errs...)
}

func (e *Engine) sourceDeleters() ([]SourceDeleter, error) {
	var stores []any
	if e.vStore != nil {
		stores = append(stores, e.vStore)
	}
	if e.iStore != nil {
		stores = append(stores, e.iStore)
	}
	out := make([]SourceDeleter, 0, len(stores))
	for _, s := range stores {
		d, ok := s.(SourceDeleter)
		if !ok {
			return nil, fmt.Errorf("%w: %T", ErrDeleteUnsupported, s)
		}
		out = append(out, d)
	}
	return out, nil
}
