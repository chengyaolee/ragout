package ragout

import (
	"context"
	"errors"
	"io"
	"iter"
)

var ErrEmptyDocument = errors.New("ragout: Document has no content.")
var ErrEmptyChunk = errors.New("ragout: Chunk content is empty.")
var ErrEmptyEmbeddings = errors.New("ragout: Embedding returned is empty.")
var ErrDimensionMismatch = errors.New("ragout: Vector dimension does not match index.")
var ErrNilCallback = errors.New("ragout: Stream callback is nil.")

type Document struct {
	ID string
	Content string
	Metadata map[string]any
}

type Chunk struct {
	ID string
	DocumentID string
	Index int
	Content string
	Embedding []float32 `json:"embedding,omitempty"`
	Metadata map[string]any
}

type ScoredChunk struct {
	Chunk Chunk
	DenseScore float32
	SparseScore float32
	Score float32
}

type StreamCallback func(token string) error

type Reader interface {
	Read(ctx context.Context, r io.Reader, metadata map[string]any) ([]Document, error)
}

type Chunker interface {
	Chunk(ctx context.Context, doc Document) ([]Chunk, error)
}

type Embedder interface {
	Embed(ctx context.Context, text string) ([][]float32, error)
	Dimension() int
}

type VectorStore interface {
	Upsert(ctx context.Context, chunks []Chunk) error
    SearchDense(ctx context.Context, vector []float32, topK int, filter map[string]any) ([]ScoredChunk, error)
}

type IndexStore interface {
	Index(ctx context.Context, chunks []Chunk) error
	SearchSparse(ctx context.Context, query string, topK int, filter map[string]any) ([]ScoredChunk, error)
}

type Reranker interface {
	Rerank(ctx context.Context, query string, candidates []ScoredChunk, topN int) ([]ScoredChunk, error)
}

type Generator interface {
	Generate(ctx context.Context, queryL string, context []ScoredChunk) (string, error)
	GenerateStream(ctx context.Context, query string, context []ScoredChunk, cb StreamCallback) error
	GenerateIter(ctx context.Context, query string, context []ScoredChunk) iter.Seq2[string, error]
}

func CloneMetadata (src map[string]any) map[string]any{
	if src == nil {
		return make(map[string]any)
	}
	clonedMetadata :=make(map[string]any, len(src))
	for k, v := range src {
		clonedMetadata[k] = v
	}
	return clonedMetadata
}