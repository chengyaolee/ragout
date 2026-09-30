package ragout

import (
	"errors"

	"github.com/prometheus/client_golang/prometheus"
)

var (
	QueryDuration = prometheus.NewHistogramVec(
		prometheus.HistogramOpts{
			Name:    "ragout_query_duration_seconds",
			Help:    "Latency distribution of RAG query stages.",
			Buckets: []float64{0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0},
		},
		[]string{"stage"}, // "embed", "dense", "sparse", "fusion", "rerank", "generation"
	)
	IngestedChunksTotal = prometheus.NewCounter(
		prometheus.CounterOpts{
			Name: "ragout_ingested_chunks_total",
			Help: "Total number of semantic chunks ingested into storage.",
		},
	)
)

// RegisterMetrics registers ragout metrics into the given Prometheus registerer.
// If a metric is already registered, the AlreadyRegisteredError is safely ignored.
func RegisterMetrics(reg prometheus.Registerer) error {
	if reg == nil {
		reg = prometheus.DefaultRegisterer
	}
	collectors := []prometheus.Collector{QueryDuration, IngestedChunksTotal}
	for _, c := range collectors {
		if err := reg.Register(c); err != nil {
			var alreadyRegistered prometheus.AlreadyRegisteredError
			if !errors.As(err, &alreadyRegistered) {
				return err
			}
		}
	}
	return nil
}

func init() {
	_ = RegisterMetrics(prometheus.DefaultRegisterer)
}
