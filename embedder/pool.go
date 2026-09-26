package embedder

import (
	"context"
	"fmt"

	"github.com/chengyaolee/ragout"
	"golang.org/x/sync/errgroup"
)


func EmbedConcurrent(ctx context.Context, embedder Embedder, chunks []ragout.Chunk, maxWorkers int, batchSize int)  error {
	if maxWorkers <= 0 {
		return fmt.Errorf("maxWorkers must be greater than 0")
	}
	if batchSize <= 0 {
		return fmt.Errorf("batchSize must be greater than 0")
	}
	if len(chunks) == 0 {
		return nil
	}
	if err:= ctx.Err(); err != nil {
		return err
	}

	// Create a new error group and add the context to it
	g, ctx := errgroup.WithContext(ctx)
	batches := make(chan []ragout.Chunk)
	
	// Add a goroutine to send batches to the channel
	g.Go(func() error {
		defer close(batches) // Close the channel when the goroutine exits
		for i:= 0; i < len(chunks); i += batchSize {
			end := min(i+batchSize, len(chunks))
			select{
			case <-ctx.Done(): // If the context is done, return the error
				return ctx.Err()
			case batches <- chunks[i:end]: // Send the batch to the channel
			}
		}
		return nil
	})
	
	// Add a goroutine to embed the batches
	// Each worker has multiple batches to embed
	for range maxWorkers {
		g.Go(func() error {
			for batch := range batches { // Receive a batch from the channel
				texts := make([]string, len(batch)) // Extract texts from batch
				for j, chunk := range batch {
					texts[j] = chunk.Content
				}
				embeddings, err := embedder.EmbedBatch(ctx, texts)
				if err != nil {
					return err
				}
				if len(embeddings) != len(batch) {
					return ragout.ErrCountMismatch
				}
				dim := embedder.Dimension()
				for j, embedding := range embeddings {
					if len(embedding) == 0 {
						return ragout.ErrEmptyEmbeddings
					}
					if len(embedding) != dim {
						return ragout.ErrDimensionMismatch
					}
					// Attach embedding to chunk
					batch[j].Embedding = embedding
				}
			}
			return nil
		})
	}
	return g.Wait()
}
