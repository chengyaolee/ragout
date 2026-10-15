package ragout

import (
	"context"
	"errors"
	"io"
	"iter"
	"strconv"

	"github.com/google/uuid"
)

var ErrEmptyDocument = errors.New("ragout: Document has no content.")
var ErrEmptyChunk = errors.New("ragout: Chunk content is empty.")
var ErrEmptyEmbeddings = errors.New("ragout: Embedding returned is empty.")
var ErrCountMismatch = errors.New("ragout: Embedding count does not match input.")
var ErrDimensionMismatch = errors.New("ragout: Vector dimension does not match index.")
var ErrNilCallback = errors.New("ragout: Stream callback is nil.")
var ErrDocumentTooLarge = errors.New("ragout: Document exceeds the reader's size limit.")
var ErrUnsupportedFormat = errors.New("ragout: No reader for this file format.")

// Metadata keys with a defined meaning.
const (
	// MetadataSource is set by Ingest to the document's source name.
	MetadataSource = "source"
	// MetadataTitle is the document title, when a reader can find one (e.g. Markdown
	// frontmatter or an HTML <title>).
	MetadataTitle = "title"
	// MetadataSection is the heading path of the section a document was split from,
	// e.g. "Financials > Q3". Section-aware readers emit one Document per section.
	MetadataSection = "section"
)

type Document struct {
	ID       string
	Content  string
	Metadata map[string]any
}

type Chunk struct {
	ID         string
	DocumentID string
	Index      int
	Content    string
	// Context is optional situational text from a Contextualizer. It is indexed
	// (embedded and BM25-tokenized) with Content but never shown to the generator,
	// so citations and prompts keep the original chunk text.
	Context   string    `json:"context,omitempty"`
	Embedding []float32 `json:"embedding,omitempty"`
	Metadata  map[string]any
}

// IndexText is the text embedders and the BM25 index see for this chunk.
func (c Chunk) IndexText() string {
	if c.Context == "" {
		return c.Content
	}
	return c.Context + "\n\n" + c.Content
}

type ScoredChunk struct {
	Chunk       Chunk
	DenseScore  float64
	SparseScore float64
	Score       float64
}

// Answer is a generated response together with the chunks behind it.
type Answer struct {
	Text      string
	Sources   []ScoredChunk // chunks placed in the prompt; Sources[i] is cited as [i+1]
	Retrieved []ScoredChunk // every candidate handed to the generator, best first
	Question  string        // the standalone question reranked and answered (see QueryTransformer)
	Searches  []Search      // searches requested, the question first; one whose index is not configured is skipped
	// TransformErr is the QueryTransformer's error or timeout, if any. The query still
	// ran, with whatever the transformer returned (at least the original question).
	TransformErr error
}

// Turn is one message of a conversation, passed per query with WithHistory.
type Turn struct {
	Role    string // "user" or "assistant"
	Content string
}

// SearchTarget says which index a search runs against.
type SearchTarget int

const (
	SearchAll    SearchTarget = iota // every configured index
	SearchDense                      // vector store only
	SearchSparse                     // BM25 index only
)

// Search is one retrieval request.
type Search struct {
	Text   string
	Target SearchTarget
}

// Rewrite is what a QueryTransformer produces.
type Rewrite struct {
	// Question is the standalone question the reranker and generator use, and that
	// is always searched on every index. Empty keeps the original question.
	Question string
	// Searches run in addition to the question.
	Searches []Search
}

type StreamCallback func(token string) error

type Reader interface {
	Read(ctx context.Context, r io.Reader, metadata map[string]any) ([]Document, error)
}

type Chunker interface {
	Chunk(ctx context.Context, doc Document) ([]Chunk, error)
}

type Embedder interface {
	// EmbedBatch takes a slice of strings and returns their float32 vectors.
	EmbedBatch(ctx context.Context, texts []string) ([][]float32, error)
	Dimension() int
}

type VectorStore interface {
	Upsert(ctx context.Context, chunks []Chunk) error
	SearchDense(ctx context.Context, vector []float32, topK int, filter map[string]any) ([]ScoredChunk, error)
	Delete(ctx context.Context, ids []string) error
}

type IndexStore interface {
	Index(ctx context.Context, chunks []Chunk) error
	SearchSparse(ctx context.Context, query string, topK int, filter map[string]any) ([]ScoredChunk, error)
	Delete(ctx context.Context, ids []string) error
}

type Reranker interface {
	Rerank(ctx context.Context, query string, candidates []ScoredChunk, topN int) ([]ScoredChunk, error)
}

// Completer runs a one-shot, non-streaming LLM call. prefix is the stable part shared
// across many calls (e.g. a whole document) and prompt is the part that varies; keep
// prefix first and identical so providers can cache it. Either may be empty.
type Completer interface {
	Complete(ctx context.Context, prefix, prompt string) (string, error)
}

// Contextualizer returns one situational-context string per chunk of doc, in order.
// The engine defaults to MetadataContext.
type Contextualizer interface {
	Contextualize(ctx context.Context, doc Document, chunks []Chunk) ([]string, error)
}

// QueryTransformer rewrites a question before retrieval, e.g. condensing a follow-up
// with its history or adding rephrasings. If it returns an error, the engine records
// it on the query.transform span and continues with whatever Rewrite was returned
// (the zero value keeps the original question), unless the context was cancelled.
type QueryTransformer interface {
	Transform(ctx context.Context, question string, history []Turn) (Rewrite, error)
}

// FusionStrategy blends ranked candidate lists (best first) into one ranking of at
// most topK. Each search contributes a dense and/or a sparse list, in search order.
type FusionStrategy interface {
	Fuse(lists [][]ScoredChunk, topK int) []ScoredChunk
}

// HistoryGenerator is an optional Generator capability: answering with earlier turns
// of the conversation in the prompt. The engine calls it instead of GenerateIter only
// when the query passed WithHistory; generators decide how many turns to include.
type HistoryGenerator interface {
	GenerateIterWithHistory(ctx context.Context, question string, history []Turn, candidates []ScoredChunk) (sources []ScoredChunk, tokens iter.Seq2[string, error])
}

// Expander rewrites the final candidates before generation, e.g. widening each hit
// with its neighboring chunks. It must keep the candidates' order.
type Expander interface {
	Expand(ctx context.Context, candidates []ScoredChunk) ([]ScoredChunk, error)
}

// ChunkGetter is implemented by stores that can fetch chunks by ID. IDs that are not
// stored are skipped, not reported as errors.
type ChunkGetter interface {
	GetChunks(ctx context.Context, ids []string) ([]Chunk, error)
}

type Generator interface {
	// GenerateIter builds the prompt and returns the chunks placed in it in citation order
	// (sources[i] is cited as [i+1]), plus a lazy token stream.
	GenerateIter(ctx context.Context, query string, candidates []ScoredChunk) (sources []ScoredChunk, tokens iter.Seq2[string, error])
}

// DocumentID derives a stable ID from a document's source and content, so re-ingesting
// the same file overwrites its chunks instead of duplicating them.
func DocumentID(source, content string) string {
	return uuid.NewSHA1(uuid.NameSpaceURL, []byte(source+"\x00"+content)).String()
}

// ChunkID derives a stable UUID for the chunk at index within a document.
func ChunkID(docID string, index int) string {
	return uuid.NewSHA1(uuid.NameSpaceURL, []byte(docID+"#"+strconv.Itoa(index))).String()
}

func CloneMetadata(src map[string]any) map[string]any {
	if src == nil {
		return make(map[string]any)
	}
	clonedMetadata := make(map[string]any, len(src))
	for k, v := range src {
		clonedMetadata[k] = v
	}
	return clonedMetadata
}
