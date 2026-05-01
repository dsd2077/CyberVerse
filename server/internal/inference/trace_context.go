package inference

import (
	"context"
	"time"
)

type traceContextKey struct{}

// TraceContext carries per-turn identifiers across internal gRPC calls.
type TraceContext struct {
	SessionID   string
	QuestionID  string
	ReplyID     string
	TurnSeq     uint64
	UserFinalAt time.Time
}

// WithTraceContext attaches per-turn identifiers to ctx.
func WithTraceContext(ctx context.Context, trace TraceContext) context.Context {
	return context.WithValue(ctx, traceContextKey{}, trace)
}

// TraceContextFromContext returns per-turn identifiers attached to ctx.
func TraceContextFromContext(ctx context.Context) (TraceContext, bool) {
	trace, ok := ctx.Value(traceContextKey{}).(TraceContext)
	return trace, ok
}
