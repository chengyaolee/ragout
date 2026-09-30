package reader

import (
	"context"
	"html"
	"io"
	"regexp"
	"strings"

	"github.com/chengyaolee/ragout"
	"github.com/google/uuid"
)

var (
	scriptRegex = regexp.MustCompile(`(?is)<script.*?</script>`)
	styleRegex  = regexp.MustCompile(`(?is)<style.*?</style>`)
	tagRegex    = regexp.MustCompile(`<[^>]+>`)
	spaceRegex  = regexp.MustCompile(`\s{2,}`)
)

// HTMLReader reads HTML streams, strips formatting tags and script blocks, and returns clean text.
type HTMLReader struct {
	maxBytes int64
}

// NewHTMLReader creates an initialized HTMLReader with a maximum byte limit.
func NewHTMLReader(maxBytes int64) *HTMLReader {
	if maxBytes <= 0 {
		maxBytes = 10 * 1024 * 1024 // Default 10MB
	}
	return &HTMLReader{maxBytes: maxBytes}
}

// Read strips HTML tags and non-content elements, emitting clean text documents.
func (hr *HTMLReader) Read(ctx context.Context, r io.Reader, metadata map[string]any) ([]ragout.Document, error) {
	if err := ctx.Err(); err != nil {
		return nil, err
	}

	b, err := io.ReadAll(io.LimitReader(r, hr.maxBytes))
	if err != nil {
		return nil, err
	}
	if err := ctx.Err(); err != nil {
		return nil, err
	}

	raw := string(b)
	// 1. Remove scripts and styles
	clean := scriptRegex.ReplaceAllString(raw, " ")
	clean = styleRegex.ReplaceAllString(clean, " ")
	// 2. Strip all HTML tags
	clean = tagRegex.ReplaceAllString(clean, " ")
	// 3. Unescape HTML entities (e.g. &amp;, &lt;)
	clean = html.UnescapeString(clean)
	// 4. Collapse excess whitespace
	clean = spaceRegex.ReplaceAllString(clean, " ")
	clean = strings.TrimSpace(clean)

	if clean == "" {
		return nil, ragout.ErrEmptyDocument
	}

	return []ragout.Document{{
		ID:       uuid.NewString(),
		Content:  clean,
		Metadata: ragout.CloneMetadata(metadata),
	}}, nil
}
