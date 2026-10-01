package reader

import (
	"bufio"
	"context"
	"io"
	"strings"

	"github.com/chengyaolee/ragout"
)

// MarkdownReader reads markdown documents with optional frontmatter parsing.
type MarkdownReader struct {
	maxBytes int64
}

// NewMarkdownReader creates an initialized MarkdownReader with a maximum byte limit.
func NewMarkdownReader(maxBytes int64) *MarkdownReader {
	if maxBytes <= 0 {
		maxBytes = 10 * 1024 * 1024 // Default 10MB
	}
	return &MarkdownReader{maxBytes: maxBytes}
}

// Read reads markdown content from r and attaches extracted frontmatter to metadata.
func (mr *MarkdownReader) Read(ctx context.Context, r io.Reader, metadata map[string]any) ([]ragout.Document, error) {
	if err := ctx.Err(); err != nil {
		return nil, err
	}

	b, err := readLimited(r, mr.maxBytes)
	if err != nil {
		return nil, err
	}
	if err := ctx.Err(); err != nil {
		return nil, err
	}

	content := string(b)
	meta := ragout.CloneMetadata(metadata)

	// Extract YAML frontmatter if present (between leading --- and ---)
	if strings.HasPrefix(strings.TrimSpace(content), "---") {
		trimmed := strings.TrimSpace(content)
		parts := strings.SplitN(trimmed, "---", 3)
		if len(parts) >= 3 {
			frontmatter := parts[1]
			content = strings.TrimSpace(parts[2])

			scanner := bufio.NewScanner(strings.NewReader(frontmatter))
			for scanner.Scan() {
				line := strings.TrimSpace(scanner.Text())
				if line == "" || strings.HasPrefix(line, "#") {
					continue
				}
				if colonIdx := strings.Index(line, ":"); colonIdx != -1 {
					key := strings.TrimSpace(line[:colonIdx])
					val := strings.TrimSpace(line[colonIdx+1:])
					val = strings.Trim(val, `"'`)
					if key != "" {
						meta[key] = val
					}
				}
			}
		}
	}

	text := strings.TrimSpace(content)
	if text == "" {
		return nil, ragout.ErrEmptyDocument
	}

	return []ragout.Document{{
		Content:  text,
		Metadata: meta,
	}}, nil
}
