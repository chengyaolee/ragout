package ragout

import (
	"context"

	"go.opentelemetry.io/otel"
	"go.opentelemetry.io/otel/attribute"
	"go.opentelemetry.io/otel/codes"
	"go.opentelemetry.io/otel/trace"
)

var tracer = otel.Tracer("github.com/chengyaolee/ragout")

func (e *Engine) traceQuery(ctx context.Context, query string) (context.Context, func(err error)) {
	ctx, span := tracer.Start(ctx, "ragout.Engine.Query")
	span.SetAttributes(
		attribute.String("ragout.query", query),
		attribute.Int("ragout.top_k_recall", e.topKRecall),
		attribute.Int("ragout.top_n_rerank", e.topNRerank),
	)
	return ctx, func(err error) {
		if err != nil {
			span.RecordError(err)
			span.SetStatus(codes.Error, err.Error())
		}
		span.End()
	}
}

func startSpan(ctx context.Context, name string, attrs ...attribute.KeyValue) (context.Context, trace.Span) {
	ctx, span := tracer.Start(ctx, name)
	if len(attrs) > 0 {
		span.SetAttributes(attrs...)
	}
	return ctx, span
}

func endSpan(span trace.Span, err error) {
	if err != nil {
		span.RecordError(err)
		span.SetStatus(codes.Error, err.Error())
	}
	span.End()
}
