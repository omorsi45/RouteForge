package telemetry

import (
	"context"
	"errors"
	"fmt"
	"net/url"
	"path"

	"go.opentelemetry.io/otel/attribute"
	"go.opentelemetry.io/otel/exporters/otlp/otlptrace/otlptracehttp"
	"go.opentelemetry.io/otel/sdk/resource"
	sdktrace "go.opentelemetry.io/otel/sdk/trace"
	"go.opentelemetry.io/otel/trace"
	"go.opentelemetry.io/otel/trace/noop"
)

// tracesEndpointVar is the environment variable that turns tracing on: the base
// URL of an OTLP/HTTP collector.
const tracesEndpointVar = "OTEL_EXPORTER_OTLP_ENDPOINT"

// tracesPath is the path of the OTLP/HTTP traces signal, appended to the base
// URL.
const tracesPath = "/v1/traces"

// NewTracerProvider returns the provider the gateway's spans come from, and a
// function that flushes the spans still waiting and shuts the provider down.
//
// With OTEL_EXPORTER_OTLP_ENDPOINT set (read through getenv), spans go to that
// OTLP/HTTP collector in batches, the way the OpenTelemetry specification has it
// for a base URL: the traces path is appended to the path of the URL. Without
// it the provider is a no-op and the returned function does nothing. Other
// OTEL_EXPORTER_OTLP_* variables (headers, timeout, compression) are read by
// the exporter itself. The provider is not installed as the global one; callers
// take a Tracer from it and hand it to the components.
func NewTracerProvider(ctx context.Context, serviceName string, getenv func(string) string) (trace.TracerProvider, func(context.Context) error, error) {
	base := getenv(tracesEndpointVar)
	if base == "" {
		return noop.NewTracerProvider(), func(context.Context) error { return nil }, nil
	}
	endpoint, err := tracesURL(base)
	if err != nil {
		return nil, nil, err
	}
	exp, err := otlptracehttp.New(ctx, otlptracehttp.WithEndpointURL(endpoint))
	if err != nil {
		return nil, nil, fmt.Errorf("create the OTLP trace exporter: %w", err)
	}
	// Merge over the default resource, which has the SDK's own attributes, so
	// the service name is the gateway's whatever the environment says.
	res, err := resource.Merge(resource.Default(), resource.NewSchemaless(attribute.String("service.name", serviceName)))
	if err != nil {
		_ = exp.Shutdown(ctx)
		return nil, nil, fmt.Errorf("create the trace resource: %w", err)
	}
	tp := sdktrace.NewTracerProvider(sdktrace.WithBatcher(exp), sdktrace.WithResource(res))
	return tp, tp.Shutdown, nil
}

// tracesURL returns the URL spans are posted to for the collector base URL. The
// errors name the variable and never repeat its value, which may carry
// credentials.
func tracesURL(base string) (string, error) {
	u, err := url.Parse(base)
	if err != nil || (u.Scheme != "http" && u.Scheme != "https") || u.Host == "" {
		return "", errors.New(tracesEndpointVar + " must be an http or https URL")
	}
	u.User = nil
	u.Path = path.Join("/", u.Path, tracesPath)
	u.RawPath, u.RawQuery, u.Fragment = "", "", ""
	return u.String(), nil
}
