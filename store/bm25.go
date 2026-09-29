package store

import (
	"context"
	"math"
	"sort"
	"strings"
	"sync"
	"unicode"

	"github.com/chengyaolee/ragout"
)

// Posting represents a chunk entry in the inverted index for a term.
type Posting struct {
	ChunkID       string
	TermFrequency int
}

// Tokenizer defines a function that splits text into linguistic terms.
type Tokenizer func(text string) []string

// Tokenize is the default tokenizer: lowercases and strips punctuation.
func Tokenize(text string) []string {
	return strings.FieldsFunc(strings.ToLower(text), func(r rune) bool {
		return !unicode.IsLetter(r) && !unicode.IsDigit(r)
	})
}

// BM25Index is a thread-safe in-memory inverted index using the Okapi BM25 ranking function.
type BM25Index struct {
	mu       sync.RWMutex
	tokenize Tokenizer
	k1       float64
	b        float64

	// Corpus statistics
	totalDocuments        int
	totalTokens           int
	averageDocumentLength float64

	// Lookups
	documentLengths map[string]int          // chunkID -> token count
	chunks          map[string]ragout.Chunk // chunkID -> Chunk
	index           map[string][]Posting    // term -> postings
}

// BM25Option allows customizing BM25Index parameters.
type BM25Option func(*BM25Index)

// WithK1 configures the term frequency saturation parameter k1 (default: 1.5).
func WithK1(k1 float64) BM25Option {
	return func(b *BM25Index) {
		b.k1 = k1
	}
}

// WithB configures the document length normalization parameter b (default: 0.75).
func WithB(b float64) BM25Option {
	return func(idx *BM25Index) {
		idx.b = b
	}
}

// WithTokenizer configures a custom tokenizer function.
func WithTokenizer(t Tokenizer) BM25Option {
	return func(b *BM25Index) {
		if t != nil {
			b.tokenize = t
		}
	}
}

// NewBM25Index creates an initialized BM25Index.
func NewBM25Index(options ...BM25Option) *BM25Index {
	idx := &BM25Index{
		tokenize:        Tokenize,
		k1:              1.5,
		b:               0.75,
		documentLengths: make(map[string]int),
		chunks:          make(map[string]ragout.Chunk),
		index:           make(map[string][]Posting),
	}

	for _, opt := range options {
		opt(idx)
	}

	return idx
}

// Index adds or updates chunks in the BM25 inverted index.
func (idx *BM25Index) Index(ctx context.Context, chunks []ragout.Chunk) error {
	if err := ctx.Err(); err != nil {
		return err
	}

	idx.mu.Lock()
	defer idx.mu.Unlock()

	for _, chunk := range chunks {
		if err := ctx.Err(); err != nil {
			return err
		}
		if chunk.ID == "" {
			continue
		}

		// Handle updates: remove old statistics and postings if chunk already exists
		if oldLen, exists := idx.documentLengths[chunk.ID]; exists {
			idx.totalTokens -= oldLen
			for term, postings := range idx.index {
				newPostings := make([]Posting, 0, len(postings))
				for _, p := range postings {
					if p.ChunkID != chunk.ID {
						newPostings = append(newPostings, p)
					}
				}
				if len(newPostings) == 0 {
					delete(idx.index, term)
				} else {
					idx.index[term] = newPostings
				}
			}
		} else {
			idx.totalDocuments++
		}

		tokens := idx.tokenize(chunk.Content)
		docLen := len(tokens)
		idx.documentLengths[chunk.ID] = docLen
		idx.totalTokens += docLen

		// Calculate term frequency for this chunk
		tf := make(map[string]int)
		for _, token := range tokens {
			tf[token]++
		}

		// Append postings
		for term, count := range tf {
			idx.index[term] = append(idx.index[term], Posting{
				ChunkID:       chunk.ID,
				TermFrequency: count,
			})
		}

		idx.chunks[chunk.ID] = ragout.Chunk{
			ID:         chunk.ID,
			DocumentID: chunk.DocumentID,
			Index:      chunk.Index,
			Content:    chunk.Content,
			Embedding:  chunk.Embedding,
			Metadata:   ragout.CloneMetadata(chunk.Metadata),
		}
	}

	if idx.totalDocuments > 0 {
		idx.averageDocumentLength = float64(idx.totalTokens) / float64(idx.totalDocuments)
	} else {
		idx.averageDocumentLength = 0
	}

	return nil
}

// SearchSparse searches the inverted index using Okapi BM25 scoring.
func (idx *BM25Index) SearchSparse(ctx context.Context, query string, topK int, filter map[string]any) ([]ragout.ScoredChunk, error) {
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	if topK <= 0 {
		return []ragout.ScoredChunk{}, nil
	}

	idx.mu.RLock()
	defer idx.mu.RUnlock()

	if idx.totalDocuments == 0 || idx.averageDocumentLength == 0 {
		return []ragout.ScoredChunk{}, nil
	}

	tokens := idx.tokenize(query)
	if len(tokens) == 0 {
		return []ragout.ScoredChunk{}, nil
	}

	// Accumulator: chunkID -> accumulated BM25 score
	scores := make(map[string]float64)

	for _, token := range tokens {
		postings, exists := idx.index[token]
		if !exists {
			continue // Term does not exist in corpus
		}

		// Robertson-Spärck Jones IDF with Lucene non-negative floor
		N := float64(idx.totalDocuments)
		dt := float64(len(postings))
		idf := math.Log(1.0 + (N-dt+0.5)/(dt+0.5))

		for _, posting := range postings {
			docLen := float64(idx.documentLengths[posting.ChunkID])
			tf := float64(posting.TermFrequency)

			// Length normalization factor
			lenNorm := 1.0 - idx.b + idx.b*(docLen/idx.averageDocumentLength)

			// BM25 term score
			termScore := idf * (tf * (idx.k1 + 1.0)) / (tf + idx.k1*lenNorm)
			scores[posting.ChunkID] += termScore
		}
	}

	scored := make([]ragout.ScoredChunk, 0, len(scores))
	for id, score := range scores {
		chunk := idx.chunks[id]
		if !matches(chunk.Metadata, filter) {
			continue
		}
		scored = append(scored, ragout.ScoredChunk{
			Chunk:       chunk,
			SparseScore: score,
			Score:       score,
		})
	}

	// Sort results descending by score
	sort.Slice(scored, func(i, j int) bool {
		return scored[i].Score > scored[j].Score
	})

	if topK > len(scored) {
		topK = len(scored)
	}

	return scored[:topK], nil
}
