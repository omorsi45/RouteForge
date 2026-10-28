package api

import (
	"encoding/json"
	"fmt"
	"net/http"
	"net/http/httptest"
	"reflect"
	"strings"
	"testing"
	"time"

	"github.com/omorsi45/routeforge/internal/provider"
	"github.com/omorsi45/routeforge/internal/router"
)

func TestWriteErrorEnvelope(t *testing.T) {
	rec := httptest.NewRecorder()
	writeError(rec, &Error{Status: 429, Code: CodeRateLimited, Message: "slow down", RetryAfter: 1500 * time.Millisecond})

	if rec.Code != 429 {
		t.Fatalf("status = %d, want 429", rec.Code)
	}
	if got := rec.Header().Get("Content-Type"); got != "application/json" {
		t.Errorf("Content-Type = %q, want application/json", got)
	}
	if got := rec.Header().Get("Retry-After"); got != "2" {
		t.Errorf("Retry-After = %q, want 2", got)
	}
	var got map[string]any
	if err := json.Unmarshal(rec.Body.Bytes(), &got); err != nil {
		t.Fatalf("body is not JSON: %v", err)
	}
	want := map[string]any{"error": map[string]any{
		"message": "slow down",
		"type":    "rate_limit_error",
		"code":    "rate_limited",
	}}
	if !reflect.DeepEqual(got, want) {
		t.Errorf("body = %v, want %v", got, want)
	}
}

func TestWriteErrorRetryAfter(t *testing.T) {
	cases := []struct {
		name string
		d    time.Duration
		want string
	}{
		{"zero omits header", 0, ""},
		{"sub-second rounds up to minimum", 10 * time.Millisecond, "1"},
		{"exact seconds", 3 * time.Second, "3"},
		{"fraction rounds up", 3*time.Second + time.Nanosecond, "4"},
		{"max duration does not overflow", time.Duration(1<<63 - 1), "9223372037"},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			rec := httptest.NewRecorder()
			writeError(rec, &Error{Status: 503, Code: CodeOverloaded, Message: "busy", RetryAfter: tc.d})
			if got := rec.Header().Get("Retry-After"); got != tc.want {
				t.Errorf("Retry-After = %q, want %q", got, tc.want)
			}
		})
	}
}

func TestErrorMethod(t *testing.T) {
	e := &Error{Status: 400, Code: CodeInvalidRequest, Message: "bad"}
	if got, want := e.Error(), "invalid_request: bad"; got != want {
		t.Errorf("Error() = %q, want %q", got, want)
	}
}

func TestErrorType(t *testing.T) {
	cases := []struct {
		status int
		want   string
	}{
		{400, "invalid_request_error"},
		{413, "invalid_request_error"},
		{401, "authentication_error"},
		{403, "permission_error"},
		{429, "rate_limit_error"},
		{502, "server_error"},
		{503, "server_error"},
		{504, "server_error"},
	}
	for _, tc := range cases {
		t.Run(fmt.Sprint(tc.status), func(t *testing.T) {
			if got := errorType(tc.status); got != tc.want {
				t.Errorf("errorType(%d) = %q, want %q", tc.status, got, tc.want)
			}
		})
	}
}

func TestFromRouterError(t *testing.T) {
	cases := []struct {
		name       string
		err        error
		wantStatus int
		wantCode   string
		wantMsg    string
		wantRetry  time.Duration
	}{
		{"unknown model", fmt.Errorf("%w: %q", router.ErrUnknownModel, "nope"), http.StatusBadRequest, CodeInvalidRequest, `unknown model: "nope"`, 0},
		{"model not allowed", fmt.Errorf("%w: x", router.ErrModelNotAllowed), http.StatusForbidden, CodeModelNotAllowed, "model not allowed: x", 0},
		{"strategy unavailable", fmt.Errorf("%w: x", router.ErrStrategyUnavailable), http.StatusBadRequest, CodeInvalidRequest, "strategy not available: x", 0},
		{"no eligible model", fmt.Errorf("%w: x", router.ErrNoEligibleModel), http.StatusServiceUnavailable, CodeNoEligibleModel, "no eligible model: x", time.Second},
		{"overloaded", fmt.Errorf("%w: x", router.ErrOverloaded), http.StatusServiceUnavailable, CodeOverloaded, "overloaded: x", time.Second},
		{"other", fmt.Errorf("disk on fire"), http.StatusBadGateway, CodeUpstreamError, "routing failed", 0},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			got := fromRouterError(tc.err)
			want := &Error{Status: tc.wantStatus, Code: tc.wantCode, Message: tc.wantMsg, RetryAfter: tc.wantRetry}
			if *got != *want {
				t.Errorf("fromRouterError() = %+v, want %+v", *got, *want)
			}
		})
	}
}

func TestInFlightFullError(t *testing.T) {
	rec := httptest.NewRecorder()
	writeError(rec, errInFlightFull())

	if rec.Code != http.StatusServiceUnavailable {
		t.Errorf("status = %d, want 503", rec.Code)
	}
	if got := rec.Header().Get("Retry-After"); got != "1" {
		t.Errorf("Retry-After = %q, want 1", got)
	}
	var got struct {
		Error struct {
			Message string `json:"message"`
			Type    string `json:"type"`
			Code    string `json:"code"`
		} `json:"error"`
	}
	if err := json.Unmarshal(rec.Body.Bytes(), &got); err != nil {
		t.Fatalf("body is not JSON: %v", err)
	}
	if got.Error.Code != CodeOverloaded || got.Error.Type != "server_error" || got.Error.Message == "" {
		t.Errorf("error = %+v, want code %q, type server_error and a message", got.Error, CodeOverloaded)
	}
}

func TestFromUpstreamError(t *testing.T) {
	const secret = "secret upstream detail"
	cases := []struct {
		name       string
		err        error
		wantStatus int
		wantCode   string
		wantMsg    string // exact when non-empty
		wantRetry  time.Duration
	}{
		{"rate limited with retry after", &provider.Error{Kind: provider.KindRateLimited, RetryAfter: 3 * time.Second}, 429, CodeRateLimited, "", 3 * time.Second},
		{"rate limited without retry after", &provider.Error{Kind: provider.KindRateLimited}, 429, CodeRateLimited, "", time.Second},
		{"context length", &provider.Error{Kind: provider.KindContextLength, Message: "too long"}, 400, CodeContextLengthExceeded, "too long", 0},
		{"bad request", &provider.Error{Kind: provider.KindBadRequest, Message: "bad temp"}, 400, CodeInvalidRequest, "upstream rejected the request: bad temp", 0},
		{"timeout", &provider.Error{Kind: provider.KindTimeout, Message: secret}, 504, CodeDeadlineExceeded, "", 0},
		{"unavailable", &provider.Error{Kind: provider.KindUnavailable, Message: secret}, 502, CodeUpstreamError, "upstream provider error (unavailable)", 0},
		{"auth", &provider.Error{Kind: provider.KindAuth, Message: secret}, 502, CodeUpstreamError, "upstream provider error (auth)", 0},
		{"internal", &provider.Error{Kind: provider.KindInternal, Message: secret}, 502, CodeUpstreamError, "upstream provider error (internal)", 0},
		{"wrapped", fmt.Errorf("attempt 2: %w", &provider.Error{Kind: provider.KindContextLength, Message: "too long"}), 400, CodeContextLengthExceeded, "too long", 0},
		{"non-provider error", fmt.Errorf("%s", secret), 502, CodeUpstreamError, "upstream provider error (internal)", 0},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			got := fromUpstreamError(tc.err)
			if got.Status != tc.wantStatus || got.Code != tc.wantCode || got.RetryAfter != tc.wantRetry {
				t.Errorf("fromUpstreamError() = %+v, want status %d code %q retry %v", *got, tc.wantStatus, tc.wantCode, tc.wantRetry)
			}
			if tc.wantMsg != "" && got.Message != tc.wantMsg {
				t.Errorf("Message = %q, want %q", got.Message, tc.wantMsg)
			}
			if got.Message == "" {
				t.Error("Message is empty")
			}
			if got.Status >= 500 && strings.Contains(got.Message, secret) {
				t.Errorf("Message %q leaks upstream text", got.Message)
			}
		})
	}
}

func TestStreamErrorEvent(t *testing.T) {
	got := string(streamErrorEvent(&Error{Status: 502, Code: CodeUpstreamStreamInterrupted, Message: "upstream stream interrupted"}))
	want := `data: {"error":{"message":"upstream stream interrupted","type":"server_error","code":"upstream_stream_interrupted"}}` + "\n\n"
	if got != want {
		t.Errorf("streamErrorEvent() = %q, want %q", got, want)
	}
}
