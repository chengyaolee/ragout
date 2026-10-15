package ragout

import (
	"context"
	"path/filepath"
	"strings"
)

// Breadcrumb joins a document's title and section path, e.g. "Annual Report > Financials > Q3".
// Without a title it falls back to the source file name minus its extension.
func Breadcrumb(meta map[string]any) string {
	title, _ := meta[MetadataTitle].(string)
	if title == "" {
		if src, _ := meta[MetadataSource].(string); src != "" {
			title = strings.TrimSuffix(filepath.Base(src), filepath.Ext(src))
		}
	}
	section, _ := meta[MetadataSection].(string)
	switch {
	case title == "":
		return section
	case section == "":
		return title
	}
	return title + " > " + section
}

// MetadataContext is the default Contextualizer. It gives every chunk its document's
// Breadcrumb and makes no model call.
func MetadataContext() Contextualizer { return metadataContext{} }

type metadataContext struct{}

func (metadataContext) Contextualize(_ context.Context, doc Document, chunks []Chunk) ([]string, error) {
	crumb := Breadcrumb(doc.Metadata)
	out := make([]string, len(chunks))
	for i := range out {
		out[i] = crumb
	}
	return out, nil
}
