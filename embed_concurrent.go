package ragout

import (
	"context"
	"fmt"

	"golang.org/x/sync/errgroup"
)

// EmbedConcurrent embeds chunks with a bounded worker pool.
// Workers write embeddings into the chunk slice they receive.
func EmbedConcurrent(ctx context.Context, em Embedder, chunks []Chunk, maxWorkers int, batchSize int) error {
	return embedConcurrent(ctx, em, chunks, maxWorkers, batchSize)
}

func embedConcurrent(ctx context.Context, em Embedder, chunks []Chunk, maxWorkers int, batchSize int) error {
	if maxWorkers <= 0 {
		return fmt.Errorf("maxWorkers must be greater than 0")
	}
	if batchSize <= 0 {
		return fmt.Errorf("batchSize must be greater than 0")
	}
	if len(chunks) == 0 {
		return nil
	}
	if err := ctx.Err(); err != nil {
		return err
	}

	g, ctx := errgroup.WithContext(ctx)
	batches := make(chan []Chunk)

	g.Go(func() error {
		defer close(batches)
		for i := 0; i < len(chunks); i += batchSize {
			end := min(i+batchSize, len(chunks))
			select {
			case <-ctx.Done():
				return ctx.Err()
			case batches <- chunks[i:end]:
			}
		}
		return nil
	})

	for range maxWorkers {
		g.Go(func() error {
			for batch := range batches {
				texts := make([]string, len(batch))
				for j, chunk := range batch {
					texts[j] = chunk.Content
				}
				embeddings, err := em.EmbedBatch(ctx, texts)
				if err != nil {
					return err
				}
				if len(embeddings) != len(batch) {
					return ErrCountMismatch
				}
				dim := em.Dimension()
				for j, embedding := range embeddings {
					if len(embedding) == 0 {
						return ErrEmptyEmbeddings
					}
					if len(embedding) != dim {
						return ErrDimensionMismatch
					}
					batch[j].Embedding = embedding
				}
			}
			return nil
		})
	}
	return g.Wait()
}
