package store

import "reflect"

// matches checks if a chunk's metadata satisfies all key-value constraints in the filter.
func matches(meta, filter map[string]any) bool {
	if len(filter) == 0 {
		return true
	}
	if len(meta) == 0 {
		return false
	}
	for k, want := range filter {
		got, ok := meta[k]
		if !ok || !reflect.DeepEqual(got, want) {
			return false
		}
	}
	return true
}
