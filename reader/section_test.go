package reader

import (
	"archive/zip"
	"bytes"
	"context"
	"strings"
	"testing"

	"github.com/chengyaolee/ragout"
)

type wantDoc struct{ section, contains string }

func checkSections(t *testing.T, docs []ragout.Document, want []wantDoc) {
	t.Helper()
	if len(docs) != len(want) {
		for _, d := range docs {
			t.Logf("section=%q content=%q", d.Metadata[ragout.MetadataSection], d.Content)
		}
		t.Fatalf("got %d docs, want %d", len(docs), len(want))
	}
	for i, w := range want {
		got, _ := docs[i].Metadata[ragout.MetadataSection].(string)
		if got != w.section || !strings.Contains(docs[i].Content, w.contains) {
			t.Errorf("doc %d: section=%q content=%q, want section %q containing %q", i, got, docs[i].Content, w.section, w.contains)
		}
	}
}

func TestMarkdownSections(t *testing.T) {
	md := "---\ntitle: Report\n---\nIntro text.\n# Finance\n## Q3 ##\nRevenue grew.\n```\n# not a heading\n```\n# Hiring\nWe hired.\n"
	docs, err := NewMarkdownReader(0).Read(context.Background(), strings.NewReader(md), nil)
	if err != nil {
		t.Fatal(err)
	}
	// "# Finance" has no body of its own, so it only shows up in its child's path.
	checkSections(t, docs, []wantDoc{
		{"", "Intro text."},
		{"Finance > Q3", "# not a heading"},
		{"Hiring", "We hired."},
	})
	if docs[1].Metadata[ragout.MetadataTitle] != "Report" {
		t.Errorf("title = %v, want frontmatter title", docs[1].Metadata[ragout.MetadataTitle])
	}
}

func TestHTMLSections(t *testing.T) {
	page := `<html><head><title>Acme &amp; Co</title><style>h1{}</style></head><body>
<p>Lead.</p><h1 class="x">Products</h1><p>Widgets.</p><h2>Pricing</h2><p>Cheap.</p></body></html>`
	docs, err := NewHTMLReader(0).Read(context.Background(), strings.NewReader(page), nil)
	if err != nil {
		t.Fatal(err)
	}
	checkSections(t, docs, []wantDoc{
		{"", "Lead."},
		{"Products", "Widgets."},
		{"Products > Pricing", "Cheap."},
	})
	if docs[0].Metadata[ragout.MetadataTitle] != "Acme & Co" {
		t.Errorf("title = %v", docs[0].Metadata[ragout.MetadataTitle])
	}
	if strings.Contains(docs[0].Content, "Acme") {
		t.Errorf("<head> text leaked into content: %q", docs[0].Content)
	}
}

func TestDocxSections(t *testing.T) {
	p := func(style, text string) string {
		pPr := ""
		if style != "" {
			pPr = `<w:pPr><w:pStyle w:val="` + style + `"/></w:pPr>`
		}
		return `<w:p>` + pPr + `<w:r><w:t>` + text + `</w:t></w:r></w:p>`
	}
	xml := `<w:document xmlns:w="w"><w:body>` + p("", "Preface.") + p("Heading1", "Scope") + p("", "In scope.") +
		p("heading 2", "Limits") + p("", "Out of scope.") + `</w:body></w:document>`

	var buf bytes.Buffer
	zw := zip.NewWriter(&buf)
	f, _ := zw.Create("word/document.xml")
	f.Write([]byte(xml))
	zw.Close()

	docs, err := NewDocxReader(0).Read(context.Background(), &buf, nil)
	if err != nil {
		t.Fatal(err)
	}
	checkSections(t, docs, []wantDoc{
		{"", "Preface."},
		{"Scope", "In scope."},
		{"Scope > Limits", "Out of scope."},
	})
}

func TestHeadingOnlyDocumentIsKept(t *testing.T) {
	docs, err := NewMarkdownReader(0).Read(context.Background(), strings.NewReader("# Just a title\n"), nil)
	if err != nil || len(docs) != 1 || docs[0].Content != "# Just a title" {
		t.Fatalf("got %+v, %v", docs, err)
	}
}
