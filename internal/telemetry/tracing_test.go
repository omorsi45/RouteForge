package telemetry_test

import (
	"bytes"
	"context"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync"
	"testing"

	"go.opentelemetry.io/otel/trace/noop"

	"github.com/omorsi45/routeforge/internal/telemetry"
)

const endpointVar = "OTEL_EXPORTER_OTLP_ENDPOINT"

// env is a getenv over a fixed map.
func env(vars map[string]string) func(string) string {
	return func(k string) string { return vars[k] }
}

// collector is a local OTLP/HTTP endpoint that records what it is sent.
type collector struct {
	srv  *httptest.Server
	mu   sync.Mutex
	reqs []received
}

type received struct {
	method, path, contentType string
	body                      []byte
}

func newCollector(t *testing.T) *collector {
	t.Helper()
	c := &collector{}
	c.srv = httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		body, _ := io.ReadAll(r.Body)
		c.mu.Lock()
		c.reqs = append(c.reqs, received{r.Method, r.URL.Path, r.Header.Get("Content-Type"), body})
		c.mu.Unlock()
		w.WriteHeader(http.StatusOK)
	}))
	t.Cleanup(c.srv.Close)
	return c
}

func (c *collector) requests() []received {
	c.mu.Lock()
	defer c.mu.Unlock()
	return append([]received(nil), c.reqs...)
}

func TestNoopWithoutEndpoint(t *testing.T) {
	for name, vars := range map[string]map[string]string{
		"unset": nil,
		"empty": {endpointVar: ""},
		// Only the variable the gateway names turns tracing on.
		"another variable": {"OTEL_EXPORTER_OTLP_TRACES_ENDPOINT": "http://127.0.0.1:1"},
	} {
		t.Run(name, func(t *testing.T) {
			ctx := context.Background()
			tp, shutdown, err := telemetry.NewTracerProvider(ctx, "routeforge", env(vars))
			if err != nil {
				t.Fatal(err)
			}
			if _, ok := tp.(noop.TracerProvider); !ok {
				t.Errorf("provider = %T, want the noop provider", tp)
			}
			_, span := tp.Tracer("t").Start(ctx, "probe")
			if span.SpanContext().IsValid() || span.IsRecording() {
				t.Error("a span of the noop provider is valid or recording")
			}
			span.End()
			if err := shutdown(ctx); err != nil {
				t.Errorf("shutdown error = %v, want nil", err)
			}
		})
	}
}

func TestExporterWithEndpoint(t *testing.T) {
	c := newCollector(t)
	ctx := context.Background()

	tp, shutdown, err := telemetry.NewTracerProvider(ctx, "routeforge", env(map[string]string{endpointVar: c.srv.URL}))
	if err != nil {
		t.Fatal(err)
	}
	if _, ok := tp.(noop.TracerProvider); ok {
		t.Fatal("got the noop provider with an endpoint set")
	}
	_, span := tp.Tracer("t").Start(ctx, "probe-span")
	span.End()

	// Spans leave in batches; shutting down flushes the batch.
	if got := len(c.requests()); got != 0 {
		t.Fatalf("collector saw %d requests before the flush, want 0", got)
	}
	if err := shutdown(ctx); err != nil {
		t.Fatalf("shutdown: %v", err)
	}

	reqs := c.requests()
	if len(reqs) == 0 {
		t.Fatal("collector saw no request after the flush")
	}
	for _, r := range reqs {
		if r.method != http.MethodPost || r.path != "/v1/traces" {
			t.Errorf("request = %s %s, want POST /v1/traces", r.method, r.path)
		}
	}
	body := reqs[0].body
	for _, want := range []string{"probe-span", "routeforge"} {
		if !bytes.Contains(body, []byte(want)) {
			t.Errorf("the exported batch lacks %q (the span name and the service name are in the payload)", want)
		}
	}
}

func TestEndpointKeepsItsBasePath(t *testing.T) {
	c := newCollector(t)
	ctx := context.Background()

	// A trailing slash and a prefix: the signal path is appended to the base.
	tp, shutdown, err := telemetry.NewTracerProvider(ctx, "routeforge", env(map[string]string{endpointVar: c.srv.URL + "/otlp/"}))
	if err != nil {
		t.Fatal(err)
	}
	_, span := tp.Tracer("t").Start(ctx, "probe")
	span.End()
	if err := shutdown(ctx); err != nil {
		t.Fatal(err)
	}
	reqs := c.requests()
	if len(reqs) == 0 || reqs[0].path != "/otlp/v1/traces" {
		t.Errorf("requests = %+v, want one to /otlp/v1/traces", reqs)
	}
}

func TestInvalidEndpointIsAnError(t *testing.T) {
	// The error names the variable and never echoes its value, which may carry
	// credentials.
	for _, value := range []string{"://nope", "localhost:4318", "ftp://collector:21", "http://", "collector.internal"} {
		t.Run(value, func(t *testing.T) {
			tp, shutdown, err := telemetry.NewTracerProvider(context.Background(), "routeforge", env(map[string]string{endpointVar: value}))
			if err == nil {
				_ = shutdown(context.Background())
				t.Fatalf("NewTracerProvider(%q) succeeded with %T, want an error", value, tp)
			}
			if !strings.Contains(err.Error(), endpointVar) {
				t.Errorf("error %q does not name %s", err, endpointVar)
			}
			if strings.Contains(err.Error(), value) {
				t.Errorf("error %q echoes the endpoint value", err)
			}
		})
	}
}
