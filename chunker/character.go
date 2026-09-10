package chunker

import (
	"context"
	"fmt"
	"strings"
	"github.com/chengyaolee/ragout"
	"github.com/google/uuid"
)

type CharacterChunker struct {
	chunkSize int
	chunkOverlap int
	separators []string
}

func NewCharacterChunker(chunkSize int, chunkOverlap int, separators []string) (*CharacterChunker, error) {
	if chunkSize <= 0 {
		return nil, fmt.Errorf("chunkSize must be greater than 0")
	}
	if chunkOverlap < 0 {
		return nil, fmt.Errorf("chunkOverlap must be greater than or equal to 0")
	}
	if len(separators) == 0 || separators == nil {
		separators = []string{"\n\n", "\n", " ", ""}
	}
	return &CharacterChunker{chunkSize, chunkOverlap, separators}, nil
}

func (c *CharacterChunker) Chunk(ctx context.Context, doc ragout.Document) ([]ragout.Chunk, error) {
	if ctx.Err() != nil {
		return nil, ctx.Err()
	}
	if c.chunkOverlap >= c.chunkSize {
		return nil, fmt.Errorf("chunkOverlap must be < chunkSize")
	}
	if len(doc.Content) == 0 {
		return nil, ragout.ErrEmptyDocument
	}
	

	texts, err := c.splitText(ctx, doc.Content, c.separators)
	if err != nil {
		return nil, err
	}
	chunks := make([]ragout.Chunk, 0, len(texts))
	for idx, text := range texts {
		chunk, err := c.stringToChunk(ctx, text, doc, idx)
		if err != nil {
			return nil, err
		}
		chunks = append(chunks, chunk)
	}
	return chunks, nil
}

func (c *CharacterChunker) splitText(ctx context.Context, text string, separators []string) ([]string, error) {
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	
	var outputChunks []string

	separator := separators[len(separators)-1]
	newSeparators := []string{}

	// Find first separator that appears in the text
	for i, sep := range separators {
		if sep == "" || strings.Contains(text, sep) {
			separator = sep
			newSeparators = separators[i+1:]
			break
		}
	}
	
	var splits []string
	splits = strings.Split(text, separator)

	goodSplits := []string{}

	for _, split := range splits {
		if len(split) <= c.chunkSize {
			goodSplits = append(goodSplits, split)
		} else {
			// If split too large, merge current good splits
			if len(goodSplits) > 0 {
				mergedSplits := c.merge(ctx, goodSplits, separator)
				outputChunks = append(outputChunks, mergedSplits...)
				goodSplits = []string{}
			}
			if len(newSeparators) != 0 {
				recursiveSplits, err := c.splitText(ctx, split, newSeparators)
				if err != nil {
					return nil, err
				}
				outputChunks = append(outputChunks, recursiveSplits...)
			} else {
				// No more separators
				outputChunks = append(outputChunks, split[:c.chunkSize])
			}
		}
	}
	if len(goodSplits) > 0 {
		mergedSplits := c.merge(ctx, goodSplits, separator)
		outputChunks = append(outputChunks, mergedSplits...)
	}

	return outputChunks, nil
}


func (c *CharacterChunker) merge(ctx context.Context, splits []string, separator string) []string {
	chunks := []string{}
	currentChunk := []string{}
	currentLength := 0
	separatorLength := 0

	for _, split := range splits {
		if len(currentChunk) == 0 {
			separatorLength = 0
		} else {
			separatorLength = len(separator)
		}
		candidateLength := currentLength + separatorLength + len(split)
		// If adding split to current chunk + separator doesnt exceed size, add it in
		if candidateLength <= c.chunkSize {
			currentChunk = append(currentChunk, split)
			currentLength = candidateLength
		} else {
			// If exceed, combine current chunk + separators and commit
			if len(currentChunk) > 0 {
				chunks = append(chunks, strings.Join(currentChunk, separator))
			}
			
			// Preserve overlap
			for len(currentChunk) > 0 && currentLength > c.chunkOverlap {
				removed := currentChunk[0]
				currentChunk = currentChunk[1:]
				if len(currentChunk) > 0 {
					currentLength -= len(removed) + len(separator)
				} else {
					currentLength = 0
				}
			}

			currentChunk = append(currentChunk, split)
			// Recalculate currentLength after appending new split
			currentLength = 0
			for _, s := range currentChunk {
				currentLength += len(s)
			}
			if len(currentChunk) > 1 {
				currentLength += len(separator) * (len(currentChunk) - 1)
			}
		}
	}
	if len(currentChunk) > 0 {
		chunks = append(chunks, strings.Join(currentChunk, separator))
	}
	return chunks
}

func (c *CharacterChunker) stringToChunk(ctx context.Context, text string, document ragout.Document, index int) (ragout.Chunk, error) {
	return ragout.Chunk{
		ID: uuid.New().String(),
		DocumentID: document.ID,
		Index: index,
		Content: text,
		Metadata: ragout.CloneMetadata(document.Metadata),
	}, nil
}