// Package cache is an opt-in cache for retrieval, wrapped around a ragout.Engine (or
// any Retriever) from outside. The engine itself never caches.
package cache

import (
	"container/list"
	"context"
	"sync"
	"time"

	"github.com/chengyaolee/ragout"
)

// Memory is an in-process Store. Once it holds maxEntries, the oldest write is dropped
// first. maxEntries <= 0 means no limit, which with no TTL lets it grow without bound.
type Memory struct {
	max int

	mu      sync.Mutex
	entries map[string]*list.Element
	order   *list.List // *entry, oldest write first
}

type entry struct {
	key     string
	answer  ragout.Answer
	expires time.Time // zero: never
}

func NewMemory(maxEntries int) *Memory {
	return &Memory{max: maxEntries, entries: make(map[string]*list.Element), order: list.New()}
}

func (m *Memory) Get(_ context.Context, key string) (ragout.Answer, bool, error) {
	m.mu.Lock()
	defer m.mu.Unlock()
	el, ok := m.entries[key]
	if !ok {
		return ragout.Answer{}, false, nil
	}
	if ent := el.Value.(*entry); ent.expired(time.Now()) {
		m.drop(el)
		return ragout.Answer{}, false, nil
	}
	return el.Value.(*entry).answer, true, nil
}

func (m *Memory) Set(_ context.Context, key string, a ragout.Answer, ttl time.Duration) error {
	now := time.Now()
	var expires time.Time
	if ttl > 0 {
		expires = now.Add(ttl)
	}
	m.mu.Lock()
	defer m.mu.Unlock()
	if el, ok := m.entries[key]; ok {
		m.drop(el)
	}
	m.entries[key] = m.order.PushBack(&entry{key: key, answer: a, expires: expires})
	for el := m.order.Front(); el != nil; el = m.order.Front() {
		if (m.max <= 0 || m.order.Len() <= m.max) && !el.Value.(*entry).expired(now) {
			break
		}
		m.drop(el)
	}
	return nil
}

func (e *entry) expired(now time.Time) bool {
	return !e.expires.IsZero() && !now.Before(e.expires)
}

func (m *Memory) drop(el *list.Element) {
	m.order.Remove(el)
	delete(m.entries, el.Value.(*entry).key)
}
