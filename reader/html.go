package reader

import (
	"context"
	"html"
	"io"
	"regexp"
	"strings"

	"github.com/chengyaolee/ragout"
)

var (
	scriptRegex  = regexp.MustCompile(`(?is)<script.*?</script>`)
	styleRegex   = regexp.MustCompile(`(?is)<style.*?</style>`)
	tagRegex     = regexp.MustCompile(`<[^>]+>`)
	spaceRegex   = regexp.MustCompile(`\s{2,}`)
	headRegex    = regexp.MustCompile(`(?is)<head\b.*?</head\s*>`)
	titleRegex   = regexp.MustCompile(`(?is)<title[^>]*>(.*?)</title>`)
	headingRegex = regexp.MustCompile(`(?is)<h([1-6])\b[^>]*>(.*?)</h[1-6]\s*>`)
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

	b, err := readLimited(r, hr.maxBytes)
	if err != nil {
		return nil, err
	}
	if err := ctx.Err(); err != nil {
		return nil, err
	}

	// Remove scripts and styles before anything else so their text never leaks.
	raw := scriptRegex.ReplaceAllString(string(b), " ")
	raw = styleRegex.ReplaceAllString(raw, " ")
	title := titleRegex.FindStringSubmatch(raw)
	raw = headRegex.ReplaceAllString(raw, " ")

	meta := ragout.CloneMetadata(metadata)
	if _, ok := meta[ragout.MetadataTitle]; !ok {
		if title != nil {
			if t := htmlText(title[1]); t != "" {
				meta[ragout.MetadataTitle] = t
			}
		}
	}

	// One document per <h1>-<h6> section.
	s := newSections(meta)
	locs := headingRegex.FindAllStringSubmatchIndex(raw, -1)
	end := len(raw)
	if len(locs) > 0 {
		end = locs[0][0]
	}
	s.text(htmlText(raw[:end]))
	for i, loc := range locs {
		title := htmlText(raw[loc[4]:loc[5]])
		s.heading(int(raw[loc[2]]-'0'), title, title)
		end := len(raw)
		if i+1 < len(locs) {
			end = locs[i+1][0]
		}
		s.text(htmlText(raw[loc[1]:end]))
	}
	return s.done()
}

// htmlText strips tags, unescapes entities and collapses whitespace.
func htmlText(s string) string {
	s = tagRegex.ReplaceAllString(s, " ")
	s = html.UnescapeString(s)
	s = spaceRegex.ReplaceAllString(s, " ")
	return strings.TrimSpace(s)
}
