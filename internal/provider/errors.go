package provider

import (
	"context"
	"errors"
	"fmt"
	"io"
	"math"
	"net"
	"net/http"
	"strconv"
	"strings"
	"syscall"
	"time"
)

// ErrorKind classifies a provider failure.
type ErrorKind string

// Error kinds.
const (
	KindRateLimited   ErrorKind = "rate_limited"
	KindUnavailable   ErrorKind = "unavailable"
	KindTimeout       ErrorKind = "timeout"
	KindBadRequest    ErrorKind = "bad_request"
	KindAuth          ErrorKind = "auth"
	KindContextLength ErrorKind = "context_length"
	KindInternal      ErrorKind = "internal"
)

// Valid reports whether k is one of the defined kinds.
func (k ErrorKind) Valid() bool {
	switch k {
	case KindRateLimited, KindUnavailable, KindTimeout, KindBadRequest,
		KindAuth, KindContextLength, KindInternal:
		return true
	}
	return false
}

// Retryable reports whether a failure of this kind may succeed on retry.
func (k ErrorKind) Retryable() bool {
	switch k {
	case KindRateLimited, KindUnavailable, KindTimeout:
		return true
	}
	return false
}

// Error is a classified provider failure.
type Error struct {
	Kind       ErrorKind
	Provider   string
	Status     int           // upstream HTTP status, 0 if none
	RetryAfter time.Duration // 0 if none
	Message    string        // credentials already redacted
	Err        error
}

// Error implements error.
func (e *Error) Error() string {
	s := fmt.Sprintf("%s: %s: %s", e.Provider, e.Kind, e.Message)
	if e.Status != 0 {
		s += fmt.Sprintf(" (status %d)", e.Status)
	}
	return s
}

// Unwrap returns the underlying cause.
func (e *Error) Unwrap() error { return e.Err }

// KindOf classifies err. A nil error yields the empty kind.
func KindOf(err error) ErrorKind {
	if err == nil {
		return ""
	}
	var pe *Error
	if errors.As(err, &pe) {
		return pe.Kind
	}
	if errors.Is(err, context.DeadlineExceeded) {
		return KindTimeout
	}
	var ne net.Error
	if errors.Is(err, io.ErrUnexpectedEOF) || errors.Is(err, io.EOF) ||
		errors.As(err, &ne) ||
		errors.Is(err, syscall.ECONNREFUSED) || errors.Is(err, syscall.ECONNRESET) {
		return KindUnavailable
	}
	return KindInternal
}

// KindFromStatus maps an upstream HTTP status to an ErrorKind.
func KindFromStatus(status int) ErrorKind {
	switch {
	case status == http.StatusUnauthorized, status == http.StatusForbidden:
		return KindAuth
	case status == http.StatusRequestTimeout, status == http.StatusGatewayTimeout:
		return KindTimeout
	case status == http.StatusTooManyRequests:
		return KindRateLimited
	case status >= 400 && status < 500:
		return KindBadRequest
	case status >= 500 && status < 600:
		return KindUnavailable
	}
	return KindInternal
}

// ParseRetryAfter reads retry-after-ms, then Retry-After (seconds or HTTP
// date), and returns a non-negative delay. Unparseable or absent values give 0.
func ParseRetryAfter(h http.Header, now time.Time) time.Duration {
	if v := strings.TrimSpace(h.Get("retry-after-ms")); v != "" {
		if ms, err := strconv.ParseFloat(v, 64); err == nil && ms >= 0 && !math.IsInf(ms, 0) && !math.IsNaN(ms) {
			return capDuration(ms * float64(time.Millisecond))
		}
	}
	v := strings.TrimSpace(h.Get("Retry-After"))
	if v == "" {
		return 0
	}
	if secs, err := strconv.ParseInt(v, 10, 64); err == nil {
		if secs < 0 {
			return 0
		}
		return capDuration(float64(secs) * float64(time.Second))
	}
	if t, err := http.ParseTime(v); err == nil {
		if d := t.Sub(now); d > 0 {
			return d
		}
	}
	return 0
}

// capDuration converts a nanosecond count, clamping to avoid overflow.
func capDuration(ns float64) time.Duration {
	if ns >= float64(math.MaxInt64) {
		return time.Duration(math.MaxInt64)
	}
	return time.Duration(ns)
}

// Redact replaces every non-empty secret in s with [REDACTED].
func Redact(s string, secrets ...string) string {
	for _, secret := range secrets {
		if secret == "" {
			continue
		}
		s = strings.ReplaceAll(s, secret, "[REDACTED]")
	}
	return s
}

// TransportError converts a transport-level failure into a provider error.
// Caller cancellation is returned as context.Canceled, not as an *Error.
func TransportError(ctx context.Context, providerName string, err error) error {
	switch ctx.Err() {
	case context.Canceled:
		return fmt.Errorf("%s: %w", providerName, context.Canceled)
	case context.DeadlineExceeded:
		return &Error{Kind: KindTimeout, Provider: providerName, Message: "deadline exceeded", Err: err}
	}
	kind := KindOf(err)
	if kind == KindInternal {
		kind = KindUnavailable
	}
	return &Error{Kind: kind, Provider: providerName, Message: "transport error", Err: err}
}
