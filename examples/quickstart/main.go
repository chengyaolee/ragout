package main

import (
	"context"
	"fmt"
	"log"
	"os"

	"github.com/chengyaolee/ragout"
	"github.com/chengyaolee/ragout/chunker"
	"github.com/chengyaolee/ragout/embedder"
	"github.com/chengyaolee/ragout/generator"
	"github.com/chengyaolee/ragout/reader"
	"github.com/chengyaolee/ragout/store"
)

func main() {
	apiKey := os.Getenv("OPENAI_API_KEY")
	if apiKey == "" {
		log.Fatal("OPENAI_API_KEY is required")
	}
	if len(os.Args) != 2 {
		log.Fatal("usage: quickstart <file>")
	}

	ctx := context.Background()

	chunks, err := chunker.NewCharacterChunker(800, 100, nil)
	if err != nil {
		log.Fatal(err)
	}

	engine, err := ragout.NewEngine(
		ragout.WithReader(reader.NewByExtension(10<<20)),
		ragout.WithChunker(chunks),
		ragout.WithEmbedder(embedder.NewOpenAIEmbedder(apiKey)),
		ragout.WithVectorStore(store.NewVectorStore()),
		ragout.WithIndexStore(store.NewBM25Index()),
		ragout.WithGenerator(generator.NewOpenAIGenerator(apiKey)),
	)
	if err != nil {
		log.Fatal(err)
	}

	f, err := os.Open(os.Args[1])
	if err != nil {
		log.Fatal(err)
	}
	defer f.Close()

	if err := engine.Ingest(ctx, os.Args[1], f, nil); err != nil {
		log.Fatal(err)
	}

	answer, err := engine.Query(ctx, "What is this document about?")
	if err != nil {
		log.Fatal(err)
	}

	fmt.Println(answer.Text)
	for i, src := range answer.Sources {
		source, _ := src.Chunk.Metadata[ragout.MetadataSource].(string)
		fmt.Printf("[%d] %s\n", i+1, source)
	}
}
