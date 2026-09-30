package reader_test

import (
	"context"
	"strings"
	"testing"

	"github.com/chengyaolee/ragout/reader"
)

func TestMarkdownReader_WithFrontmatter(t *testing.T) {
	md := `---
title: "Ragout Guide"
author: "Antigravity"
version: 2
---
# Welcome to Ragout
This is the content body.
`
	r := reader.NewMarkdownReader(0)
	docs, err := r.Read(context.Background(), strings.NewReader(md), map[string]any{"source": "test"})
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if len(docs) != 1 {
		t.Fatalf("expected 1 doc, got %d", len(docs))
	}
	if docs[0].Metadata["title"] != "Ragout Guide" {
		t.Errorf("expected title 'Ragout Guide', got %v", docs[0].Metadata["title"])
	}
	if docs[0].Metadata["author"] != "Antigravity" {
		t.Errorf("expected author 'Antigravity', got %v", docs[0].Metadata["author"])
	}
	if docs[0].Metadata["source"] != "test" {
		t.Errorf("expected original metadata preserved, got %v", docs[0].Metadata["source"])
	}
	if !strings.HasPrefix(docs[0].Content, "# Welcome") {
		t.Errorf("unexpected content: %q", docs[0].Content)
	}
}

func TestHTMLReader_StripTags(t *testing.T) {
	htmlContent := `
<!DOCTYPE html>
<html>
<head>
<style>body { color: red; }</style>
<script>console.log("secret");</script>
</head>
<body>
<h1>Title Here</h1>
<p>This is a &lt;paragraph&gt; with <strong>bold</strong> text.</p>
</body>
</html>
`
	hr := reader.NewHTMLReader(0)
	docs, err := hr.Read(context.Background(), strings.NewReader(htmlContent), nil)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if len(docs) != 1 {
		t.Fatalf("expected 1 doc, got %d", len(docs))
	}
	if strings.Contains(docs[0].Content, "console.log") {
		t.Errorf("script content not stripped: %s", docs[0].Content)
	}
	if strings.Contains(docs[0].Content, "color: red") {
		t.Errorf("style content not stripped: %s", docs[0].Content)
	}
	if strings.Contains(docs[0].Content, "<h1>") || strings.Contains(docs[0].Content, "<strong>") {
		t.Errorf("HTML tags not stripped: %s", docs[0].Content)
	}
	if !strings.Contains(docs[0].Content, "<paragraph>") {
		t.Errorf("HTML entities unescaping failed: %s", docs[0].Content)
	}
}
