// Demo runs a hybrid query with no API key and no server.
// The embedder is deterministic and local. The generator quotes the top retrieved chunk.
// A real service swaps those two for embedder.NewOpenAIEmbedder and generator.NewOpenAIGenerator.
package main

import (
	"context"
	"fmt"
	"iter"
	"log"
	"strings"

	"github.com/chengyaolee/ragout"
	"github.com/chengyaolee/ragout/chunker"
	"github.com/chengyaolee/ragout/embedder"
	"github.com/chengyaolee/ragout/reader"
	"github.com/chengyaolee/ragout/store"
)

func main() {
	ctx := context.Background()

	chunks, err := chunker.NewCharacterChunker(400, 0, nil)
	if err != nil {
		log.Fatal(err)
	}
	emb, err := embedder.NewMockEmbedder(8, 0)
	if err != nil {
		log.Fatal(err)
	}

	engine, err := ragout.NewEngine(
		ragout.WithReader(reader.NewTextReader(1<<20)),
		ragout.WithChunker(chunks),
		ragout.WithEmbedder(emb),
		ragout.WithVectorStore(store.NewVectorStore()),
		ragout.WithIndexStore(store.NewBM25Index()),
		ragout.WithGenerator(quoteGenerator{}),
	)
	if err != nil {
		log.Fatal(err)
	}

	notes := []struct{ name, body string }{
		{"channels.txt", "Channels prevent race conditions by transferring ownership of values between goroutines."},
		{"slices.txt", "A slice is a descriptor for a contiguous segment of an underlying array."},
		{"errors.txt", "Wrap errors with fmt.Errorf and %w so callers can use errors.Is and errors.As."},
	}
	for _, note := range notes {
		if err := engine.Ingest(ctx, note.name, strings.NewReader(note.body), nil); err != nil {
			log.Fatal(err)
		}
	}

	const question = "How do channels prevent race conditions?"
	answer, err := engine.Query(ctx, question)
	if err != nil {
		log.Fatal(err)
	}

	fmt.Println(question)
	fmt.Println(answer.Text)
	for i, src := range answer.Sources {
		source, _ := src.Chunk.Metadata[ragout.MetadataSource].(string)
		fmt.Printf("[%d] %s\n", i+1, source)
	}
}

// quoteGenerator stands in for a chat model. It cites the best retrieved chunk
// and streams that chunk's text.
type quoteGenerator struct{}

func (quoteGenerator) GenerateIter(_ context.Context, _ string, candidates []ragout.ScoredChunk) ([]ragout.ScoredChunk, iter.Seq2[string, error]) {
	if len(candidates) == 0 {
		return nil, func(yield func(string, error) bool) {
			yield("No matching notes.", nil)
		}
	}
	top := candidates[:1]
	return top, func(yield func(string, error) bool) {
		yield(top[0].Chunk.Content, nil)
	}
}
