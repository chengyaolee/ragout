package store

import (
	"bytes"
	"context"
	"encoding/gob"
	"io"
	"os"

	"github.com/chengyaolee/ragout"
)

// Save gob-encodes every chunk in the store to w. Pair with Load to restore state
// across a process restart. Metadata values beyond the basic Go types (string, int,
// float64, bool, and slices/maps of them) must be registered with gob.Register before
// Save or Load is called, or gob returns an encoding error naming the type.
func (vs *VectorStore) Save(w io.Writer) error {
	vs.mu.RLock()
	chunks := make([]ragout.Chunk, 0, len(vs.chunks))
	for _, c := range vs.chunks {
		chunks = append(chunks, c)
	}
	vs.mu.RUnlock()
	return gob.NewEncoder(w).Encode(chunks)
}

// Load replaces the store's contents with chunks decoded from r.
func (vs *VectorStore) Load(r io.Reader) error {
	var chunks []ragout.Chunk
	if err := gob.NewDecoder(r).Decode(&chunks); err != nil {
		return err
	}

	vs.mu.Lock()
	defer vs.mu.Unlock()
	vs.chunks = make(map[string]ragout.Chunk, len(chunks))
	for _, c := range chunks {
		vs.chunks[c.ID] = c
	}
	return nil
}

// Save gob-encodes every chunk in the index to w. See VectorStore.Save for the
// metadata type caveat.
func (idx *BM25Index) Save(w io.Writer) error {
	idx.mu.RLock()
	chunks := make([]ragout.Chunk, 0, len(idx.chunks))
	for _, c := range idx.chunks {
		chunks = append(chunks, c)
	}
	idx.mu.RUnlock()
	return gob.NewEncoder(w).Encode(chunks)
}

// Load rebuilds the inverted index and BM25 statistics from chunks decoded from r.
func (idx *BM25Index) Load(r io.Reader) error {
	var chunks []ragout.Chunk
	if err := gob.NewDecoder(r).Decode(&chunks); err != nil {
		return err
	}

	idx.mu.Lock()
	idx.documentLengths = make(map[string]int)
	idx.chunks = make(map[string]ragout.Chunk)
	idx.index = make(map[string][]Posting)
	idx.totalDocuments = 0
	idx.totalTokens = 0
	idx.averageDocumentLength = 0
	idx.mu.Unlock()

	return idx.Index(context.Background(), chunks)
}

// Persistable is implemented by stores that can snapshot themselves to and from a stream.
type Persistable interface {
	Save(w io.Writer) error
	Load(r io.Reader) error
}

// SaveFile snapshots s to path, writing through a temp file and renaming it into place
// so a crash mid-write can't corrupt an existing snapshot.
func SaveFile(path string, s Persistable) error {
	var buf bytes.Buffer
	if err := s.Save(&buf); err != nil {
		return err
	}
	tmp := path + ".tmp"
	if err := os.WriteFile(tmp, buf.Bytes(), 0o644); err != nil {
		return err
	}
	return os.Rename(tmp, path)
}

// LoadFile restores s from a snapshot written by SaveFile. A missing file is not an
// error: s is left as-is, which is typically empty on first run.
func LoadFile(path string, s Persistable) error {
	f, err := os.Open(path)
	if err != nil {
		if os.IsNotExist(err) {
			return nil
		}
		return err
	}
	defer f.Close()
	return s.Load(f)
}
