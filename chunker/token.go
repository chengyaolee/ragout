package chunker

import (
	"context"
	"fmt"
	"github.com/chengyaolee/ragout"
	"github.com/google/uuid"
	"github.com/pkoukk/tiktoken-go"
)

type TokenChunker struct {
	chunkSize int
	chunkOverlap int
	encoder *tiktoken.Tiktoken
}

func NewTokenChunker(chunkSize int, chunkOverlap int, encodingName string) (*TokenChunker, error) {
	if chunkSize <= 0 {
		return nil, fmt.Errorf("chunkSize must be greater than 0")
	}
	if chunkOverlap < 0 {
		return nil, fmt.Errorf("chunkOverlap must be greater than 0")
	}
	if chunkOverlap >= chunkSize {
		return nil, fmt.Errorf("chunkOverlap must be less than chunkSize")
	}
	encoder, err := tiktoken.GetEncoding(encodingName) // e.g. "cl100k_base" for OpenAI
	if err != nil {
		return nil, fmt.Errorf("failed to get encoding, check encoding name: %w", err)
	}
	return &TokenChunker{
		chunkSize: chunkSize,
		chunkOverlap: chunkOverlap,
		encoder: encoder,
	}, nil
}

func (tc *TokenChunker) Chunk(ctx context.Context, doc ragout.Document) ([]ragout.Chunk, error) {
	if ctx.Err() != nil {
		return nil, ctx.Err()
	}
	if doc.Content == "" {
		return nil, fmt.Errorf("document content is empty")
	}

	tokens := tc.encoder.Encode(doc.Content, nil, nil)

	// Step ensures overlap
	step := tc.chunkSize - tc.chunkOverlap
	
	chunks := []ragout.Chunk{}

	for i := 0; i< len(tokens); i+= step {
		end := min(i + tc.chunkSize, len(tokens))
		chunk := tokens[i:end]
		chunks = append(chunks, ragout.Chunk{
			ID: uuid.New().String(),
			DocumentID: doc.ID,
			Index: i,
			Content: tc.encoder.Decode(chunk),
			Metadata: ragout.CloneMetadata(doc.Metadata),
		})
		if end == len(tokens) {
			break
		}
	}
	return chunks, nil
}


