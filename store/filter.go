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
		if !ok {
			return false
		}
		if !matchValue(got, want) {
			return false
		}
	}
	return true
}

func matchValue(got, want any) bool {
	if wantMap, ok := want.(map[string]any); ok {
		return matchOperators(got, wantMap)
	}

	wantVal := reflect.ValueOf(want)
	if wantVal.IsValid() && (wantVal.Kind() == reflect.Slice || wantVal.Kind() == reflect.Array) {
		for i := 0; i < wantVal.Len(); i++ {
			if reflect.DeepEqual(got, wantVal.Index(i).Interface()) {
				return true
			}
		}
		return false
	}

	return reflect.DeepEqual(got, want)
}

func matchOperators(got any, ops map[string]any) bool {
	for op, expected := range ops {
		switch op {
		case "$eq":
			if !reflect.DeepEqual(got, expected) {
				return false
			}
		case "$ne":
			if reflect.DeepEqual(got, expected) {
				return false
			}
		case "$in":
			expVal := reflect.ValueOf(expected)
			if !expVal.IsValid() || (expVal.Kind() != reflect.Slice && expVal.Kind() != reflect.Array) {
				return false
			}
			found := false
			for i := 0; i < expVal.Len(); i++ {
				if reflect.DeepEqual(got, expVal.Index(i).Interface()) {
					found = true
					break
				}
			}
			if !found {
				return false
			}
		case "$nin":
			expVal := reflect.ValueOf(expected)
			if expVal.IsValid() && (expVal.Kind() == reflect.Slice || expVal.Kind() == reflect.Array) {
				for i := 0; i < expVal.Len(); i++ {
					if reflect.DeepEqual(got, expVal.Index(i).Interface()) {
						return false
					}
				}
			}
		case "$gt", "$gte", "$lt", "$lte":
			gNum, ok1 := toFloat64(got)
			eNum, ok2 := toFloat64(expected)
			if !ok1 || !ok2 {
				return false
			}
			switch op {
			case "$gt":
				if !(gNum > eNum) {
					return false
				}
			case "$gte":
				if !(gNum >= eNum) {
					return false
				}
			case "$lt":
				if !(gNum < eNum) {
					return false
				}
			case "$lte":
				if !(gNum <= eNum) {
					return false
				}
			}
		default:
			if !reflect.DeepEqual(got, expected) {
				return false
			}
		}
	}
	return true
}

func toFloat64(v any) (float64, bool) {
	switch n := v.(type) {
	case int:
		return float64(n), true
	case int32:
		return float64(n), true
	case int64:
		return float64(n), true
	case float32:
		return float64(n), true
	case float64:
		return n, true
	case uint:
		return float64(n), true
	case uint32:
		return float64(n), true
	case uint64:
		return float64(n), true
	default:
		return 0, false
	}
}
