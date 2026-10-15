package reader

import (
	"bufio"
	"context"
	"io"
	"regexp"
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

	return splitMarkdown(content, meta)
}

var atxHeading = regexp.MustCompile(`^ {0,3}(#{1,6})[ \t]+(.*?)(?:[ \t]+#+)?[ \t]*$`)

// splitMarkdown emits one document per ATX heading section, skipping fenced code.
// Setext headings (underlined with === or ---) are treated as body text.
func splitMarkdown(content string, meta map[string]any) ([]ragout.Document, error) {
	s := newSections(meta)
	fence := ""
	for _, line := range strings.Split(content, "\n") {
		trimmed := strings.TrimSpace(line)
		switch {
		case fence != "":
			if strings.HasPrefix(trimmed, fence) {
				fence = ""
			}
		case strings.HasPrefix(trimmed, "```"), strings.HasPrefix(trimmed, "~~~"):
			fence = trimmed[:3]
		default:
			if m := atxHeading.FindStringSubmatch(line); m != nil {
				s.heading(len(m[1]), m[2], line)
				continue
			}
		}
		s.text(line + "\n")
	}
	return s.done()
}
