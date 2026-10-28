package executor_test

import (
	"context"
	"io"
	"testing"
	"time"

	"go.opentelemetry.io/otel/trace"

	"github.com/omorsi45/routeforge/internal/executor"
	"github.com/omorsi45/routeforge/internal/provider"
	"github.com/omorsi45/routeforge/internal/reliability"
	"github.com/omorsi45/routeforge/internal/router"
	"github.com/omorsi45/routeforge/internal/telemetry/telemetrytest"
)

func withTracer(spans *telemetrytest.Spans) func(*executor.Options) {
	return func(o *executor.Options) { o.Tracer = spans.Tracer() }
}

// tracedRoute is route with the decision id and policy version the span
// attributes carry.
func tracedRoute(chosen string, fallbacks ...string) *router.Decision {
	d := route(chosen, fallbacks...)
	d.ID, d.PolicyVersion = "dec_x", "pv-9"
	return d
}

func TestAttemptSpanForACompletion(t *testing.T) {
	spans := telemetrytest.NewSpans(t)
	r := newRig("a")
	var seen context.Context
	r.p["a"].complete = func(ctx context.Context, _ provider.Request) (*provider.Response, error) {
		seen = ctx
		return &provider.Response{Content: "hi"}, nil
	}

	// The attempt is a child of the span in the context.
	ctx, parent := spans.Tracer().Start(context.Background(), "parent")
	if _, err := r.exec(withTracer(spans)).Complete(ctx, tracedRoute("a"), provider.Request{}, nil); err != nil {
		t.Fatal(err)
	}
	parent.End()

	attempt := spans.One(t, "provider.attempt")
	if attempt.ParentID != spans.One(t, "parent").SpanID {
		t.Errorf("parent = %s, want the span of the context", attempt.ParentID)
	}
	want := map[string]any{
		"routeforge.model":          "a",
		"routeforge.attempt":        int64(1),
		"routeforge.decision_id":    "dec_x",
		"routeforge.policy_version": "pv-9",
		"routeforge.outcome":        "ok",
	}
	for k, v := range want {
		if got := attempt.Attrs[k]; got != v {
			t.Errorf("attribute %s = %v, want %v", k, got, v)
		}
	}
	if len(attempt.Attrs) != len(want) || attempt.Error {
		t.Errorf("attributes = %v (Error %v), want exactly %v and no error", attempt.Attrs, attempt.Error, want)
	}

	// The provider is called under the attempt's span, so whatever it starts
	// nests inside it.
	if got := trace.SpanFromContext(seen).SpanContext().SpanID().String(); got != attempt.SpanID {
		t.Errorf("the provider saw span %s, want the attempt span %s", got, attempt.SpanID)
	}
}

func TestAttemptSpanWithoutAPolicy(t *testing.T) {
	spans := telemetrytest.NewSpans(t)
	r := newRig("a")
	r.p["a"].complete = seq(ok("hi"))
	d := tracedRoute("a")
	d.PolicyVersion = ""

	if _, err := r.exec(withTracer(spans)).Complete(context.Background(), d, provider.Request{}, nil); err != nil {
		t.Fatal(err)
	}
	if got := spans.One(t, "provider.attempt").Attrs["routeforge.policy_version"]; got != "none" {
		t.Errorf("policy_version = %v, want none", got)
	}
}

func TestAttemptSpansForRetriesAndFailover(t *testing.T) {
	spans := telemetrytest.NewSpans(t)
	r := newRig("a", "b")
	r.p["a"].complete = seq(fail(provErr(provider.KindUnavailable, "down")))
	r.p["b"].complete = seq(fail(provErr(provider.KindTimeout, "slow")), ok("fine"))

	if _, err := r.exec(withTracer(spans)).Complete(context.Background(), tracedRoute("a", "b"), provider.Request{}, nil); err != nil {
		t.Fatal(err)
	}

	type attempt struct {
		model     string
		n         int64
		outcome   string
		errorKind any
		isErr     bool
	}
	var got []attempt
	for _, sp := range spans.Named("provider.attempt") {
		got = append(got, attempt{
			model:     sp.Attrs["routeforge.model"].(string),
			n:         sp.Attrs["routeforge.attempt"].(int64),
			outcome:   sp.Attrs["routeforge.outcome"].(string),
			errorKind: sp.Attrs["routeforge.error_kind"],
			isErr:     sp.Error,
		})
	}
	want := []attempt{
		{"a", 1, "error", "unavailable", true},
		{"b", 2, "error", "timeout", true},
		{"b", 3, "ok", nil, false},
	}
	if len(got) != len(want) {
		t.Fatalf("attempt spans = %+v, want %+v", got, want)
	}
	for i := range want {
		if got[i] != want[i] {
			t.Errorf("attempt span %d = %+v, want %+v", i+1, got[i], want[i])
		}
	}
	// The status description is the error kind and nothing the provider said.
	for _, sp := range spans.Named("provider.attempt")[:2] {
		if sp.Status != sp.Attrs["routeforge.error_kind"] {
			t.Errorf("status description %q, want the error kind %v", sp.Status, sp.Attrs["routeforge.error_kind"])
		}
	}
}

func TestSkippedTargetMakesNoAttemptSpan(t *testing.T) {
	spans := telemetrytest.NewSpans(t)
	r := newRig("a", "b")
	r.p["b"].complete = seq(ok("from b"))
	breakers := reliability.NewBreakers(reliability.BreakerConfig{
		ConsecutiveFailures: 1, ErrorRate: 1, MinRequests: 100, Window: time.Minute, OpenFor: time.Hour, HalfOpenProbes: 1,
	}, nil)
	ticket, _ := breakers.Allow("a")
	ticket.Done(false) // opens a's breaker

	if _, err := r.exec(withTracer(spans), func(o *executor.Options) { o.Breakers = breakers }).
		Complete(context.Background(), tracedRoute("a", "b"), provider.Request{}, nil); err != nil {
		t.Fatal(err)
	}
	attempts := spans.Named("provider.attempt")
	if len(attempts) != 1 || attempts[0].Attrs["routeforge.model"] != "b" || attempts[0].Attrs["routeforge.attempt"] != int64(1) {
		t.Errorf("attempt spans = %+v, want one for b numbered 1: a skipped target is not an attempt", attempts)
	}
}

func TestStreamAttemptSpanEndsWithTheStream(t *testing.T) {
	spans := telemetrytest.NewSpans(t)
	p := &stubProvider{stream: streamOf(&stubStream{steps: []step{delta("a"), delta("b")}})}
	exec := executor.NewWithOptions(executor.Options{
		Targets: map[string]executor.Target{"m": {Model: "m", Upstream: "up", Provider: p}},
		Tracer:  spans.Tracer(),
	})

	s, err := exec.Stream(context.Background(), tracedRoute("m"), provider.Request{}, nil)
	if err != nil {
		t.Fatal(err)
	}
	// The attempt is still open: the stream is being read.
	if got := spans.Named("provider.attempt"); len(got) != 0 {
		t.Fatalf("attempt span ended before the stream did: %+v", got)
	}
	if _, err := drain(s); err != io.EOF {
		t.Fatalf("drain error = %v, want io.EOF", err)
	}
	attempt := spans.One(t, "provider.attempt")
	if attempt.Attrs["routeforge.outcome"] != "ok" || attempt.Error {
		t.Errorf("attempt = %v (Error %v), want outcome ok", attempt.Attrs, attempt.Error)
	}
}

func TestStreamAttemptSpanOutcomes(t *testing.T) {
	cut := provErr(provider.KindUnavailable, "connection lost")
	tests := []struct {
		name      string
		steps     []step
		consume   func(t *testing.T, s *executor.Stream)
		outcome   string
		errorKind any
		isErr     bool
	}{
		{
			name:  "cut off after output",
			steps: []step{delta("a"), {err: cut}},
			consume: func(t *testing.T, s *executor.Stream) {
				if _, err := drain(s); err == nil || err == io.EOF {
					t.Fatalf("drain error = %v, want the failure", err)
				}
			},
			outcome: "partial", errorKind: "unavailable", isErr: true,
		},
		{
			// The caller closing early is not the model failing.
			name:  "closed before the end",
			steps: []step{delta("a"), delta("b")},
			consume: func(t *testing.T, s *executor.Stream) {
				if err := s.Close(); err != nil {
					t.Fatal(err)
				}
			},
			outcome: "canceled",
		},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			spans := telemetrytest.NewSpans(t)
			p := &stubProvider{stream: streamOf(&stubStream{steps: tc.steps})}
			exec := executor.NewWithOptions(executor.Options{
				Targets: map[string]executor.Target{"m": {Model: "m", Upstream: "up", Provider: p}},
				Tracer:  spans.Tracer(),
			})
			s, err := exec.Stream(context.Background(), tracedRoute("m"), provider.Request{}, nil)
			if err != nil {
				t.Fatal(err)
			}
			tc.consume(t, s)

			attempt := spans.One(t, "provider.attempt")
			if attempt.Attrs["routeforge.outcome"] != tc.outcome || attempt.Attrs["routeforge.error_kind"] != tc.errorKind || attempt.Error != tc.isErr {
				t.Errorf("attempt = %v (Error %v), want outcome %s, error kind %v, Error %v", attempt.Attrs, attempt.Error, tc.outcome, tc.errorKind, tc.isErr)
			}
		})
	}
}
