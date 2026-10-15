package reader

import (
	"strings"

	"github.com/chengyaolee/ragout"
)

// sections splits a document into one ragout.Document per heading, recording the
// heading path in metadata[ragout.MetadataSection]. Text before the first heading has
// no section. A heading with no body of its own emits nothing: it still appears in the
// path of the sections under it.
type sections struct {
	meta    map[string]any
	path    []heading
	body    strings.Builder
	hasBody bool
	docs    []ragout.Document
	bare    []string // heading-only sections, used if the document has nothing else
}

type heading struct {
	level int
	text  string
}

func newSections(meta map[string]any) *sections {
	return &sections{meta: meta}
}

// heading starts a new section. raw is the heading as it should appear in the content.
func (s *sections) heading(level int, text, raw string) {
	s.flush()
	for len(s.path) > 0 && s.path[len(s.path)-1].level >= level {
		s.path = s.path[:len(s.path)-1]
	}
	s.path = append(s.path, heading{level, strings.TrimSpace(text)})
	s.body.WriteString(raw)
	s.body.WriteString("\n")
}

func (s *sections) text(t string) {
	if strings.TrimSpace(t) != "" {
		s.hasBody = true
	}
	s.body.WriteString(t)
}

func (s *sections) flush() {
	content := strings.TrimSpace(s.body.String())
	if s.hasBody && content != "" {
		meta := ragout.CloneMetadata(s.meta)
		if len(s.path) > 0 {
			names := make([]string, len(s.path))
			for i, h := range s.path {
				names[i] = h.text
			}
			meta[ragout.MetadataSection] = strings.Join(names, " > ")
		}
		s.docs = append(s.docs, ragout.Document{Content: content, Metadata: meta})
	} else if content != "" {
		s.bare = append(s.bare, content)
	}
	s.body.Reset()
	s.hasBody = false
}

func (s *sections) done() ([]ragout.Document, error) {
	s.flush()
	if len(s.docs) == 0 && len(s.bare) > 0 {
		return []ragout.Document{{Content: strings.Join(s.bare, "\n"), Metadata: ragout.CloneMetadata(s.meta)}}, nil
	}
	if len(s.docs) == 0 {
		return nil, ragout.ErrEmptyDocument
	}
	return s.docs, nil
}
