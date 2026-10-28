package telemetrytest

import (
	"context"
	"fmt"
	"slices"
	"testing"

	"go.opentelemetry.io/otel/codes"
	sdktrace "go.opentelemetry.io/otel/sdk/trace"
	"go.opentelemetry.io/otel/sdk/trace/tracetest"
	"go.opentelemetry.io/otel/trace"
)

// Span is a finished span, reduced to what the tests compare.
type Span struct {
	Name     string
	TraceID  string
	SpanID   string
	ParentID string         // empty for a root span
	Attrs    map[string]any // string, int64, bool or float64 values by key
	Error    bool           // the status is Error
	Status   string         // the status description
	Events   []string       // event names
	texts    []string
}

// Texts returns every string the span carries: its name, attribute keys and
// values, status description and events. Tests use it to prove that nothing
// sensitive reached a span.
func (s Span) Texts() []string { return slices.Clone(s.texts) }

// Spans records the finished spans of a tracer in memory.
type Spans struct {
	exp *tracetest.InMemoryExporter
	tp  *sdktrace.TracerProvider
}

// NewSpans returns a recorder whose spans are exported as soon as they end.
func NewSpans(t testing.TB) *Spans {
	t.Helper()
	exp := tracetest.NewInMemoryExporter()
	tp := sdktrace.NewTracerProvider(sdktrace.WithSyncer(exp))
	t.Cleanup(func() { _ = tp.Shutdown(context.Background()) })
	return &Spans{exp: exp, tp: tp}
}

// Tracer returns a tracer that records into s.
func (s *Spans) Tracer() trace.Tracer { return s.tp.Tracer("telemetrytest") }

// Finished returns the spans that have ended, in the order they ended.
func (s *Spans) Finished() []Span {
	stubs := s.exp.GetSpans()
	out := make([]Span, len(stubs))
	for i, st := range stubs {
		sp := Span{
			Name:    st.Name,
			TraceID: st.SpanContext.TraceID().String(),
			SpanID:  st.SpanContext.SpanID().String(),
			Attrs:   make(map[string]any, len(st.Attributes)),
			Error:   st.Status.Code == codes.Error,
			Status:  st.Status.Description,
			texts:   []string{st.Name, st.Status.Description},
		}
		if st.Parent.IsValid() {
			sp.ParentID = st.Parent.SpanID().String()
		}
		for _, kv := range st.Attributes {
			v := kv.Value.AsInterface()
			sp.Attrs[string(kv.Key)] = v
			sp.texts = append(sp.texts, string(kv.Key), fmt.Sprint(v))
		}
		for _, ev := range st.Events {
			sp.Events = append(sp.Events, ev.Name)
			sp.texts = append(sp.texts, ev.Name)
			for _, kv := range ev.Attributes {
				sp.texts = append(sp.texts, string(kv.Key), fmt.Sprint(kv.Value.AsInterface()))
			}
		}
		out[i] = sp
	}
	return out
}

// Named returns the finished spans with the given name, in the order they ended.
func (s *Spans) Named(name string) []Span {
	var out []Span
	for _, sp := range s.Finished() {
		if sp.Name == name {
			out = append(out, sp)
		}
	}
	return out
}

// One returns the only finished span with the given name and fails the test
// when there is not exactly one.
func (s *Spans) One(t testing.TB, name string) Span {
	t.Helper()
	got := s.Named(name)
	if len(got) != 1 {
		t.Fatalf("got %d spans named %q, want 1; finished spans: %v", len(got), name, names(s.Finished()))
	}
	return got[0]
}

func names(spans []Span) []string {
	out := make([]string, len(spans))
	for i, sp := range spans {
		out[i] = sp.Name
	}
	return out
}
