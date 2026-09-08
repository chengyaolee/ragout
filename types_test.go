package ragout_test

import (
	"testing"
	"github.com/chengyaolee/ragout"
)

func TestCloneMetadata_DeepIsolation(t *testing.T) {
	// Random metadata
	metadata := map[string]any{
		"rag_id": "rag-1",
	}
	clonedMetadata := ragout.CloneMetadata(metadata)
	
	// Mutate cloned metadata
	clonedMetadata["rag_id"] = "rag-2"
	clonedMetadata["user_id"] = "user-1"
	
	// Assert that the original map was not modified
	if metadata["rag_id"] != "rag-1" {
		t.Errorf("metadata['rag_id'] should remain as 'rag-1', but got %v", metadata["rag_id"])
	}
	if _, exists := metadata["user_id"]; exists {
		t.Errorf("'user_id' key should not exist in original map")
	}
}

func TestCloneMetadata_NilSafety(t *testing.T) {
	clonedMetadata := ragout.CloneMetadata(nil)
	if clonedMetadata == nil {
		t.Errorf("clonedMetadata should not be nil")
	}
	
	// Test writing of key value pair
	clonedMetadata["key"] = "value"
	if _, exists := clonedMetadata["key"]; !exists {
		t.Errorf("'key' key should exist in clonedMetadata, but got %v", clonedMetadata["key"])
	}
}