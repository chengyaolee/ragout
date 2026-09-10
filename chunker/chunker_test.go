package chunker_test

import (
	"context"
	"errors"
	"slices"
	"strings"
	"testing"

	"github.com/chengyaolee/ragout"
	"github.com/chengyaolee/ragout/chunker"
	"github.com/pkoukk/tiktoken-go"
)

const encoding = "cl100k_base"

func encode(t *testing.T, text string) []int {
	t.Helper()
	enc, err := tiktoken.GetEncoding(encoding)
	if err != nil {
		t.Fatalf("GetEncoding(%q): %v", encoding, err)
	}
	return enc.Encode(text, nil, nil)
}

func smallText(t *testing.T, minTokens, maxTokens int) string {
	t.Helper()
	var b strings.Builder
	for i := 0; i < 64; i++ {
		b.WriteString("word ")
		text := strings.TrimSpace(b.String())
		n := len(encode(t, text))
		if n >= minTokens && n < maxTokens {
			return text
		}
	}
	t.Fatalf("could not build text with %d <= tokens < %d", minTokens, maxTokens)
	return ""
}

func TestTokenChunker_ExactBounds(t *testing.T) {
	const chunkSize, overlap = 16, 4
	text := strings.Repeat("lorem ipsum dolor sit amet. ", 30)
	tokens := encode(t, text)
	if len(tokens) <= chunkSize {
		t.Fatalf("need more than %d tokens, got %d", chunkSize, len(tokens))
	}

	c, err := chunker.NewTokenChunker(chunkSize, overlap, encoding)
	if err != nil {
		t.Fatalf("NewTokenChunker: %v", err)
	}
	chunks, err := c.Chunk(context.Background(), ragout.Document{ID: "doc-1", Content: text})
	if err != nil {
		t.Fatalf("Chunk: %v", err)
	}
	if len(chunks) < 2 {
		t.Fatalf("expected multiple chunks for %d tokens, got %d", len(tokens), len(chunks))
	}

	for i, ch := range chunks {
		got := encode(t, ch.Content)
		n := len(got)
		if n > chunkSize {
			t.Errorf("chunk %d has %d tokens, exceeds chunkSize %d\ncontent: %q", i, n, chunkSize, ch.Content)
		}
		if i < len(chunks)-1 && n != chunkSize {
			t.Errorf("chunk %d has %d tokens, want exactly %d", i, n, chunkSize)
		}
		if n == 0 {
			t.Errorf("chunk %d is empty", i)
		}
	}

	first := encode(t, chunks[0].Content)
	last := encode(t, chunks[len(chunks)-1].Content)
	if !slices.Equal(first, tokens[:len(first)]) {
		t.Error("first chunk does not start at the beginning of the document")
	}
	if !slices.Equal(last, tokens[len(tokens)-len(last):]) {
		t.Error("last chunk does not end at the end of the document")
	}
}

func TestTokenChunker_Overlap(t *testing.T) {
	const chunkSize, overlap = 16, 4
	text := strings.Repeat("the quick brown fox jumps over the lazy dog. ", 20)

	c, err := chunker.NewTokenChunker(chunkSize, overlap, encoding)
	if err != nil {
		t.Fatalf("NewTokenChunker: %v", err)
	}
	chunks, err := c.Chunk(context.Background(), ragout.Document{ID: "doc-1", Content: text})
	if err != nil {
		t.Fatalf("Chunk: %v", err)
	}
	if len(chunks) < 2 {
		t.Fatalf("need at least 2 chunks to test overlap, got %d", len(chunks))
	}

	for i := 0; i < len(chunks)-1; i++ {
		cur := encode(t, chunks[i].Content)
		next := encode(t, chunks[i+1].Content)
		if len(cur) < overlap {
			t.Fatalf("chunk %d has %d tokens, want at least overlap %d", i, len(cur), overlap)
		}
		o := min(overlap, len(next))
		tail := cur[len(cur)-overlap : len(cur)-overlap+o]
		head := next[:o]
		if !slices.Equal(tail, head) {
			t.Errorf("chunk %d tail tokens %v do not match chunk %d head tokens %v", i, tail, i+1, head)
		}
	}
}

func TestTokenChunker_SmallText(t *testing.T) {
	const chunkSize, overlap = 16, 8
	text := smallText(t, overlap+1, chunkSize)
	tokens := encode(t, text)
	if len(tokens) >= chunkSize {
		t.Fatalf("precondition failed: %d tokens must be < chunkSize %d", len(tokens), chunkSize)
	}

	c, err := chunker.NewTokenChunker(chunkSize, overlap, encoding)
	if err != nil {
		t.Fatalf("NewTokenChunker: %v", err)
	}
	chunks, err := c.Chunk(context.Background(), ragout.Document{ID: "doc-1", Content: text})
	if err != nil {
		t.Fatalf("Chunk: %v", err)
	}
	if len(chunks) != 1 {
		t.Fatalf("expected 1 chunk, got %d", len(chunks))
	}
	if got := encode(t, chunks[0].Content); !slices.Equal(got, tokens) {
		t.Errorf("chunk tokens %v do not match input tokens %v", got, tokens)
	}
}

func TestCharacterChunker_NaturalBreaks(t *testing.T) {
	paragraphs := []string{
		"The quick brown fox jumps over the lazy dog.",
		"Pack my box with five dozen liquor jugs.",
		"How vexingly quick daft zebras jump.",
	}
	text := strings.Join(paragraphs, "\n\n")

	// Each paragraph fits; two plus the separator do not.
	chunkSize := 0
	for _, p := range paragraphs {
		if len(p)+1 > chunkSize {
			chunkSize = len(p) + 1
		}
	}
	for i := 0; i < len(paragraphs)-1; i++ {
		joined := len(paragraphs[i]) + len("\n\n") + len(paragraphs[i+1])
		if joined <= chunkSize {
			t.Fatalf("precondition failed: paragraphs %d+%d fit in chunkSize %d", i, i+1, chunkSize)
		}
	}

	c, err := chunker.NewCharacterChunker(chunkSize, 0, nil)
	if err != nil {
		t.Fatalf("NewCharacterChunker: %v", err)
	}
	chunks, err := c.Chunk(context.Background(), ragout.Document{ID: "doc-1", Content: text})
	if err != nil {
		t.Fatalf("Chunk: %v", err)
	}
	if len(chunks) < 2 {
		t.Fatalf("expected a split across paragraphs, got %d chunk(s)", len(chunks))
	}

	for i, ch := range chunks {
		if ch.Content == "" {
			t.Errorf("chunk %d is empty", i)
			continue
		}
		if !strings.Contains(text, ch.Content) {
			t.Errorf("chunk %d is not a contiguous substring:\n%q", i, ch.Content)
		}
		for _, part := range strings.Split(ch.Content, "\n\n") {
			part = strings.TrimSpace(part)
			if part == "" {
				continue
			}
			if !slices.Contains(paragraphs, part) {
				t.Errorf("chunk %d sliced a paragraph or word: %q", i, part)
			}
		}
	}
}

func TestChunker_ContextCanceled(t *testing.T) {
	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	doc := ragout.Document{ID: "doc-1", Content: "hello world from a canceled context"}

	t.Run("TokenChunker", func(t *testing.T) {
		c, err := chunker.NewTokenChunker(8, 2, encoding)
		if err != nil {
			t.Fatalf("NewTokenChunker: %v", err)
		}
		chunks, err := c.Chunk(ctx, doc)
		assertCanceled(t, chunks, err)
	})
	t.Run("CharacterChunker", func(t *testing.T) {
		c, err := chunker.NewCharacterChunker(32, 0, nil)
		if err != nil {
			t.Fatalf("NewCharacterChunker: %v", err)
		}
		chunks, err := c.Chunk(ctx, doc)
		assertCanceled(t, chunks, err)
	})
}

func assertCanceled(t *testing.T, chunks []ragout.Chunk, err error) {
	t.Helper()
	if !errors.Is(err, context.Canceled) {
		t.Errorf("expected context.Canceled, got %v", err)
	}
	if len(chunks) != 0 {
		t.Errorf("expected no chunks on cancel, got %d", len(chunks))
	}
}
