package provider_test

import (
	"context"
	"errors"
	"fmt"
	"io"
	"net"
	"net/http"
	"syscall"
	"testing"
	"time"

	"github.com/omorsi45/routeforge/internal/provider"
)

func TestKindFromStatus(t *testing.T) {
	cases := map[int]provider.ErrorKind{
		400: provider.KindBadRequest, 404: provider.KindBadRequest, 413: provider.KindBadRequest, 422: provider.KindBadRequest,
		401: provider.KindAuth, 403: provider.KindAuth,
		408: provider.KindTimeout, 504: provider.KindTimeout,
		429: provider.KindRateLimited,
		500: provider.KindUnavailable, 502: provider.KindUnavailable, 503: provider.KindUnavailable, 529: provider.KindUnavailable,
		418: provider.KindBadRequest, 599: provider.KindUnavailable, 302: provider.KindInternal,
	}
	for status, want := range cases {
		if got := provider.KindFromStatus(status); got != want {
			t.Errorf("KindFromStatus(%d) = %q, want %q", status, got, want)
		}
	}
}

func TestRetryable(t *testing.T) {
	cases := map[provider.ErrorKind]bool{
		provider.KindRateLimited:   true,
		provider.KindUnavailable:   true,
		provider.KindTimeout:       true,
		provider.KindBadRequest:    false,
		provider.KindAuth:          false,
		provider.KindContextLength: false,
		provider.KindInternal:      false,
	}
	for kind, want := range cases {
		if !kind.Valid() {
			t.Errorf("%q should be valid", kind)
		}
		if got := kind.Retryable(); got != want {
			t.Errorf("%q.Retryable() = %v, want %v", kind, got, want)
		}
	}
	if provider.ErrorKind("bogus").Valid() {
		t.Error("bogus kind should not be valid")
	}
}

func TestKindOf(t *testing.T) {
	wrapped := fmt.Errorf("x: %w", &provider.Error{Kind: provider.KindAuth})
	cases := []struct {
		name string
		err  error
		want provider.ErrorKind
	}{
		{"nil", nil, ""},
		{"wrapped provider error", wrapped, provider.KindAuth},
		{"deadline", context.DeadlineExceeded, provider.KindTimeout},
		{"unexpected eof", io.ErrUnexpectedEOF, provider.KindUnavailable},
		{"conn refused", &net.OpError{Op: "dial", Err: syscall.ECONNREFUSED}, provider.KindUnavailable},
		{"plain", errors.New("x"), provider.KindInternal},
	}
	for _, tc := range cases {
		if got := provider.KindOf(tc.err); got != tc.want {
			t.Errorf("%s: KindOf = %q, want %q", tc.name, got, tc.want)
		}
	}
}

func TestParseRetryAfter(t *testing.T) {
	now := time.Date(2026, 10, 26, 12, 0, 0, 0, time.UTC)
	cases := []struct {
		name    string
		headers map[string]string
		want    time.Duration
	}{
		{"seconds", map[string]string{"Retry-After": "7"}, 7 * time.Second},
		{"http date", map[string]string{"Retry-After": "Mon, 26 Oct 2026 12:00:30 GMT"}, 30 * time.Second},
		{"past date", map[string]string{"Retry-After": "Mon, 26 Oct 2026 11:59:00 GMT"}, 0},
		{"ms wins", map[string]string{"retry-after-ms": "1500", "Retry-After": "7"}, 1500 * time.Millisecond},
		{"garbage", map[string]string{"Retry-After": "soon"}, 0},
		{"absent", map[string]string{}, 0},
		{"negative", map[string]string{"Retry-After": "-3"}, 0},
	}
	for _, tc := range cases {
		h := http.Header{}
		for k, v := range tc.headers {
			h.Set(k, v)
		}
		if got := provider.ParseRetryAfter(h, now); got != tc.want {
			t.Errorf("%s: got %v, want %v", tc.name, got, tc.want)
		}
	}
}

func TestRedact(t *testing.T) {
	got := provider.Redact("Incorrect API key provided: test-key-aaaa.", "test-key-aaaa", "")
	if got != "Incorrect API key provided: [REDACTED]." {
		t.Fatal(got)
	}
}

func TestTransportError(t *testing.T) {
	t.Run("canceled", func(t *testing.T) {
		ctx, cancel := context.WithCancel(context.Background())
		cancel()
		err := provider.TransportError(ctx, "p", errors.New("boom"))
		if !errors.Is(err, context.Canceled) {
			t.Fatalf("want context.Canceled, got %v", err)
		}
		var pe *provider.Error
		if errors.As(err, &pe) {
			t.Fatalf("canceled must not be a *Error: %v", err)
		}
	})
	t.Run("deadline", func(t *testing.T) {
		ctx, cancel := context.WithDeadline(context.Background(), time.Now().Add(-time.Second))
		defer cancel()
		err := provider.TransportError(ctx, "p", errors.New("boom"))
		var pe *provider.Error
		if !errors.As(err, &pe) || pe.Kind != provider.KindTimeout || pe.Provider != "p" {
			t.Fatalf("got %#v", err)
		}
	})
	t.Run("transport", func(t *testing.T) {
		err := provider.TransportError(context.Background(), "p", io.ErrUnexpectedEOF)
		var pe *provider.Error
		if !errors.As(err, &pe) || pe.Kind != provider.KindUnavailable {
			t.Fatalf("got %#v", err)
		}
		if !errors.Is(err, io.ErrUnexpectedEOF) {
			t.Fatal("cause not preserved")
		}
	})
}

func TestErrorString(t *testing.T) {
	e := &provider.Error{Kind: provider.KindRateLimited, Provider: "openai", Status: 429, Message: "slow down"}
	if e.Error() != "openai: rate_limited: slow down (status 429)" {
		t.Fatal(e.Error())
	}
	e = &provider.Error{Kind: provider.KindAuth, Provider: "openai", Message: "bad key"}
	if e.Error() != "openai: auth: bad key" {
		t.Fatal(e.Error())
	}
}
