package store

import "testing"

func TestChromaWhere(t *testing.T) {
	if got := chromaWhere(nil); got != nil {
		t.Fatalf("chromaWhere(nil) = %v, want nil", got)
	}

	got := chromaWhere(map[string]any{"topic": "ai"})
	want := map[string]any{"topic": map[string]any{"$eq": "ai"}}
	if !deepEqualAny(got, want) {
		t.Fatalf("bare value: got %v, want %v", got, want)
	}

	got = chromaWhere(map[string]any{"topic": []string{"ai", "golang"}})
	topicClause, _ := got["topic"].(map[string]any)
	in, _ := topicClause["$in"].([]any)
	if len(in) != 2 {
		t.Fatalf("slice value: got %v, want a 2-item $in", got)
	}

	got = chromaWhere(map[string]any{"score": map[string]any{"$gte": 0.5}})
	want = map[string]any{"score": map[string]any{"$gte": 0.5}}
	if !deepEqualAny(got, want) {
		t.Fatalf("operator passthrough: got %v, want %v", got, want)
	}

	got = chromaWhere(map[string]any{"a": "x", "b": "y"})
	and, ok := got["$and"].([]map[string]any)
	if !ok || len(and) != 2 {
		t.Fatalf("multi-key filter should $and its clauses, got %v", got)
	}
}

func TestQdrantFilter(t *testing.T) {
	if got := qdrantFilter(nil); got != nil {
		t.Fatalf("qdrantFilter(nil) = %v, want nil", got)
	}

	got := qdrantFilter(map[string]any{"topic": "ai"})
	must, _ := got["must"].([]map[string]any)
	if len(must) != 1 || must[0]["key"] != "topic" {
		t.Fatalf("bare value: got %v", got)
	}

	got = qdrantFilter(map[string]any{"topic": map[string]any{"$ne": "ai"}})
	mustNot, _ := got["must_not"].([]map[string]any)
	if len(mustNot) != 1 || mustNot[0]["key"] != "topic" {
		t.Fatalf("$ne: got %v, want a must_not clause", got)
	}

	got = qdrantFilter(map[string]any{"score": map[string]any{"$gte": 0.5}})
	must, _ = got["must"].([]map[string]any)
	if len(must) != 1 {
		t.Fatalf("$gte: got %v", got)
	}
	rng, _ := must[0]["range"].(map[string]any)
	if rng["gte"] != 0.5 {
		t.Fatalf("$gte range clause = %v, want gte=0.5", rng)
	}

	got = qdrantFilter(map[string]any{"topic": []string{"ai", "golang"}})
	must, _ = got["must"].([]map[string]any)
	match, _ := must[0]["match"].(map[string]any)
	any, _ := match["any"].([]any)
	if len(any) != 2 {
		t.Fatalf("slice value: got %v, want a 2-item match.any", got)
	}
}

// deepEqualAny avoids importing reflect.DeepEqual's exact-type requirements for map
// literals built with different concrete value types in the tests above.
func deepEqualAny(a, b map[string]any) bool {
	if len(a) != len(b) {
		return false
	}
	for k, av := range a {
		bv, ok := b[k]
		if !ok {
			return false
		}
		am, aok := av.(map[string]any)
		bm, bok := bv.(map[string]any)
		if aok != bok {
			return false
		}
		if aok {
			if !deepEqualAny(am, bm) {
				return false
			}
			continue
		}
		if av != bv {
			return false
		}
	}
	return true
}
