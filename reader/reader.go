package reader

import (
	"context"
	"fmt"
	"io"
	"path/filepath"
	"strings"

	"github.com/chengyaolee/ragout"
)

// readLimited reads at most max bytes from r. If the input is larger, it returns
// ragout.ErrDocumentTooLarge instead of silently truncating the document.
func readLimited(r io.Reader, max int64) ([]byte, error) {
	b, err := io.ReadAll(io.LimitReader(r, max+1))
	if err != nil {
		return nil, err
	}
	if int64(len(b)) > max {
		return nil, ragout.ErrDocumentTooLarge
	}
	return b, nil
}

// ByExtension dispatches Read to a registered Reader based on the file extension of
// metadata[ragout.MetadataSource]. Register additional formats by assigning into the map,
// e.g. byExt[".csv"] = myCSVReader.
type ByExtension map[string]ragout.Reader

// NewByExtension returns a ByExtension dispatcher covering the readers built into ragout:
// .txt, .md/.markdown, .html/.htm, .pdf, and .docx. maxBytes is shared by every reader.
func NewByExtension(maxBytes int64) ByExtension {
	return ByExtension{
		".txt":      NewTextReader(maxBytes),
		".md":       NewMarkdownReader(maxBytes),
		".markdown": NewMarkdownReader(maxBytes),
		".html":     NewHTMLReader(maxBytes),
		".htm":      NewHTMLReader(maxBytes),
		".pdf":      NewPDFReader(maxBytes),
		".docx":     NewDocxReader(maxBytes),
	}
}

func (b ByExtension) Read(ctx context.Context, r io.Reader, metadata map[string]any) ([]ragout.Document, error) {
	source, _ := metadata[ragout.MetadataSource].(string)
	ext := strings.ToLower(filepath.Ext(source))
	rd, ok := b[ext]
	if !ok {
		return nil, fmt.Errorf("%w: %q", ragout.ErrUnsupportedFormat, ext)
	}
	return rd.Read(ctx, r, metadata)
}
