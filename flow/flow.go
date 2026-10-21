// Package flow runs retrieval as a workflow: a graph of named steps that share one
// State, with edges that can loop. Every step is a plain interface, so any of them can
// be replaced, removed, or joined by steps of your own; CRAG builds a ready-made
// corrective-RAG loop out of the built-in ones.
//
// Steps only read and write State. The flow owns the wiring: after a step runs, the
// next one is chosen by its router if it has one, else by its fixed edge, else it is
// the step added after it. Replacing a step therefore keeps its wiring.
package flow

import (
	"context"
	"errors"
	"fmt"
	"slices"
	"time"

	"github.com/chengyaolee/ragout"
	"go.opentelemetry.io/otel"
	"go.opentelemetry.io/otel/attribute"
	"go.opentelemetry.io/otel/codes"
)

// End is the step name that finishes a run.
const End = "end"

// ErrMaxSteps means a run took more steps than the flow allows, usually a loop
// whose exit condition never holds.
var ErrMaxSteps = errors.New("flow: too many steps")

// Step is one node of a flow.
type Step interface {
	Run(ctx context.Context, s *State) error
}

// StepFunc adapts a function to Step.
type StepFunc func(ctx context.Context, s *State) error

func (f StepFunc) Run(ctx context.Context, s *State) error { return f(ctx, s) }

// Router picks the step to run next from the state, or End.
type Router func(s *State) string

// Flow is an ordered set of named steps plus the edges between them. Build it with
// Add, change it with InsertBefore, InsertAfter, Replace and Remove, and wire it with
// Edge and Route. A Flow is not safe to modify while it runs; Clone it instead.
type Flow struct {
	order    []string
	steps    map[string]Step
	edges    map[string]string
	routes   map[string]Router
	maxSteps int
}

// New returns an empty flow that allows 32 steps per run.
func New() *Flow {
	return &Flow{
		steps:    map[string]Step{},
		edges:    map[string]string{},
		routes:   map[string]Router{},
		maxSteps: 32,
	}
}

// Add appends a step. Adding a name that exists replaces that step in place.
func (f *Flow) Add(name string, s Step) *Flow {
	if _, ok := f.steps[name]; !ok {
		f.order = append(f.order, name)
	}
	f.steps[name] = s
	return f
}

// InsertBefore adds a step just before anchor in the default order.
func (f *Flow) InsertBefore(anchor, name string, s Step) error { return f.insert(anchor, 0, name, s) }

// InsertAfter adds a step just after anchor in the default order.
func (f *Flow) InsertAfter(anchor, name string, s Step) error { return f.insert(anchor, 1, name, s) }

func (f *Flow) insert(anchor string, offset int, name string, s Step) error {
	i := slices.Index(f.order, anchor)
	if i < 0 {
		return fmt.Errorf("flow: no step %q", anchor)
	}
	if _, ok := f.steps[name]; ok {
		return fmt.Errorf("flow: step %q already exists", name)
	}
	f.order = slices.Insert(f.order, i+offset, name)
	f.steps[name] = s
	return nil
}

// Replace swaps the step behind name, keeping its position, edge and router.
func (f *Flow) Replace(name string, s Step) error {
	if _, ok := f.steps[name]; !ok {
		return fmt.Errorf("flow: no step %q", name)
	}
	f.steps[name] = s
	return nil
}

// Remove deletes a step and its own edge and router. Edges or routers elsewhere that
// still point at it fail when the run reaches them.
func (f *Flow) Remove(name string) error {
	i := slices.Index(f.order, name)
	if i < 0 {
		return fmt.Errorf("flow: no step %q", name)
	}
	f.order = slices.Delete(f.order, i, i+1)
	delete(f.steps, name)
	delete(f.edges, name)
	delete(f.routes, name)
	return nil
}

// Edge always continues from one step to another (or End).
func (f *Flow) Edge(from, to string) *Flow {
	f.edges[from] = to
	return f
}

// Route chooses the step after from by calling r. It takes precedence over Edge.
func (f *Flow) Route(from string, r Router) *Flow {
	f.routes[from] = r
	return f
}

// MaxSteps sets how many steps one run may take. Values <= 0 are ignored.
func (f *Flow) MaxSteps(n int) *Flow {
	if n > 0 {
		f.maxSteps = n
	}
	return f
}

// Steps lists step names in default order.
func (f *Flow) Steps() []string { return slices.Clone(f.order) }

// Clone returns an independent copy, so a preset can be customized without changing it.
func (f *Flow) Clone() *Flow {
	c := New()
	c.order = slices.Clone(f.order)
	for k, v := range f.steps {
		c.steps[k] = v
	}
	for k, v := range f.edges {
		c.edges[k] = v
	}
	for k, v := range f.routes {
		c.routes[k] = v
	}
	c.maxSteps = f.maxSteps
	return c
}

// RunOption configures one run.
type RunOption func(*State)

// WithHistory passes the earlier turns of a conversation, oldest first.
func WithHistory(turns []ragout.Turn) RunOption { return func(s *State) { s.History = turns } }

// WithTokens streams generated tokens to fn as Generate produces them. An error from fn
// stops the run.
func WithTokens(fn func(token string) error) RunOption { return func(s *State) { s.onToken = fn } }

// WithValue seeds State.Values, for custom steps that need configuration per run.
func WithValue(key string, v any) RunOption { return func(s *State) { s.Values[key] = v } }

var tracer = otel.Tracer("github.com/chengyaolee/ragout/flow")

// Run executes the flow from its first step until a step routes to End, the last step
// finishes, a step fails, or MaxSteps is exceeded. The returned State is never nil and
// holds everything gathered so far, including on error.
func (f *Flow) Run(ctx context.Context, question string, opts ...RunOption) (*State, error) {
	s := &State{Question: question, Standalone: question, Values: map[string]any{}}
	for _, opt := range opts {
		opt(s)
	}
	ctx, span := tracer.Start(ctx, "flow.Run")
	defer span.End()

	err := f.run(ctx, s)
	if err != nil {
		span.RecordError(err)
		span.SetStatus(codes.Error, err.Error())
	}
	span.SetAttributes(attribute.Int("flow.steps", len(s.Trace)), attribute.Bool("flow.abstained", s.Abstained))
	return s, err
}

func (f *Flow) run(ctx context.Context, s *State) error {
	if len(f.order) == 0 {
		return errors.New("flow: no steps")
	}
	name := f.order[0]
	for n := 0; name != End; n++ {
		if n >= f.maxSteps {
			return fmt.Errorf("%w (%d)", ErrMaxSteps, f.maxSteps)
		}
		if err := ctx.Err(); err != nil {
			return err
		}
		step, ok := f.steps[name]
		if !ok {
			return fmt.Errorf("flow: no step %q", name)
		}

		start := time.Now()
		stepCtx, span := tracer.Start(ctx, "flow.step")
		span.SetAttributes(attribute.String("flow.step", name))
		err := step.Run(stepCtx, s)
		next := f.next(name, s)
		ev := Event{Step: name, Next: next, Duration: time.Since(start), Err: err}
		s.Trace = append(s.Trace, ev)
		if err != nil {
			span.RecordError(err)
			span.SetStatus(codes.Error, err.Error())
			span.End()
			return fmt.Errorf("flow: step %q: %w", name, err)
		}
		span.SetAttributes(attribute.String("flow.next", next))
		span.End()
		name = next
	}
	return nil
}

// next resolves the step after name: router, then edge, then default order.
func (f *Flow) next(name string, s *State) string {
	if r, ok := f.routes[name]; ok {
		return r(s)
	}
	if to, ok := f.edges[name]; ok {
		return to
	}
	if i := slices.Index(f.order, name); i >= 0 && i+1 < len(f.order) {
		return f.order[i+1]
	}
	return End
}
