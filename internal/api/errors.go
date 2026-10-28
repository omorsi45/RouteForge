// Package api implements the OpenAI-compatible HTTP surface of the gateway.
package api

import (
	"encoding/json"
	"errors"
	"fmt"
	"net/http"
	"strconv"
	"time"

	"github.com/omorsi45/routeforge/internal/provider"
	"github.com/omorsi45/routeforge/internal/router"
)

// Error codes returned in the error envelope.
const (
	CodeInvalidRequest            = "invalid_request"
	CodeUnsupportedParameter      = "unsupported_parameter"
	CodeContextLengthExceeded     = "context_length_exceeded"
	CodeInvalidAPIKey             = "invalid_api_key"
	CodeStrategyOverrideForbidden = "strategy_override_forbidden"
	CodeModelNotAllowed           = "model_not_allowed"
	CodeRequestTooLarge           = "request_too_large"
	CodeRateLimited               = "rate_limited"
	CodeBudgetExceeded            = "budget_exceeded"
	CodeUpstreamError             = "upstream_error"
	CodeNoEligibleModel           = "no_eligible_model"
	CodeOverloaded                = "overloaded"
	CodeAccountingUnavailable     = "accounting_unavailable"
	CodeDeadlineExceeded          = "deadline_exceeded"
	CodeUpstreamStreamInterrupted = "upstream_stream_interrupted"
	CodeNotFound                  = "not_found"
	CodeInternalError             = "internal_error"
	CodeNoPreviousPolicy          = "no_previous_policy"
)

// Error is an API failure that maps onto the OpenAI error envelope.
type Error struct {
	Status     int
	Code       string
	Message    string
	RetryAfter time.Duration // written as Retry-After seconds, rounded up, minimum 1
}

// Error implements error.
func (e *Error) Error() string { return e.Code + ": " + e.Message }

// errorType returns the OpenAI error type for an HTTP status.
func errorType(status int) string {
	switch {
	case status == http.StatusUnauthorized:
		return "authentication_error"
	case status == http.StatusForbidden:
		return "permission_error"
	case status == http.StatusTooManyRequests:
		return "rate_limit_error"
	case status >= 500:
		return "server_error"
	}
	return "invalid_request_error"
}

type envelope struct {
	Error envelopeBody `json:"error"`
	// Decision is set only by the explain endpoint, for a 503 that still has
	// a routing decision to show.
	Decision *router.Decision `json:"decision,omitempty"`
}

type envelopeBody struct {
	Message string `json:"message"`
	Type    string `json:"type"`
	Code    string `json:"code"`
}

func (e *Error) envelope() ([]byte, error) { return e.envelopeWith(nil) }

// envelopeWith renders e, plus d when it is not nil.
func (e *Error) envelopeWith(d *router.Decision) ([]byte, error) {
	return json.Marshal(envelope{
		Error: envelopeBody{
			Message: e.Message,
			Type:    errorType(e.Status),
			Code:    e.Code,
		},
		Decision: d,
	})
}

// retryAfterSeconds renders d as whole seconds, rounded up, minimum 1.
func retryAfterSeconds(d time.Duration) string {
	secs := int64(d / time.Second)
	if d%time.Second != 0 {
		secs++
	}
	return strconv.FormatInt(max(secs, 1), 10)
}

// writeError writes e as a JSON error response.
func writeError(w http.ResponseWriter, e *Error) { writeErrorWith(w, e, nil) }

// writeErrorWith writes e as a JSON error response that also carries d when d
// is not nil.
func writeErrorWith(w http.ResponseWriter, e *Error, d *router.Decision) {
	body, err := e.envelopeWith(d)
	if err != nil {
		// Unreachable: the envelope holds only strings. Fail closed anyway.
		http.Error(w, "internal error", http.StatusInternalServerError)
		return
	}
	w.Header().Set("Content-Type", "application/json")
	if e.RetryAfter > 0 {
		w.Header().Set("Retry-After", retryAfterSeconds(e.RetryAfter))
	}
	w.WriteHeader(e.Status)
	// A failed write means the client is gone; there is nobody to report to.
	_, _ = w.Write(body)
}

// streamErrorEvent renders e as a terminal SSE error event.
func streamErrorEvent(e *Error) []byte {
	body, err := e.envelope()
	if err != nil {
		// Unreachable: the envelope holds only strings.
		body = []byte(`{"error":{"message":"internal error","type":"server_error","code":"upstream_stream_interrupted"}}`)
	}
	out := make([]byte, 0, len("data: ")+len(body)+2)
	out = append(out, "data: "...)
	out = append(out, body...)
	return append(out, '\n', '\n')
}

// errInFlightFull is the response to a request that arrives while the gateway
// is already serving its maximum number of requests. It is the gateway's own
// overload, unlike a router ErrOverloaded, where the models' queues are full.
func errInFlightFull() *Error {
	return &Error{
		Status:     http.StatusServiceUnavailable,
		Code:       CodeOverloaded,
		Message:    "the gateway is serving its maximum number of requests",
		RetryAfter: time.Second,
	}
}

// fromRouterError maps a router failure to an API error.
func fromRouterError(err error) *Error {
	switch {
	case errors.Is(err, router.ErrUnknownModel), errors.Is(err, router.ErrStrategyUnavailable):
		return &Error{Status: http.StatusBadRequest, Code: CodeInvalidRequest, Message: err.Error()}
	case errors.Is(err, router.ErrModelNotAllowed):
		return &Error{Status: http.StatusForbidden, Code: CodeModelNotAllowed, Message: err.Error()}
	case errors.Is(err, router.ErrNoEligibleModel):
		return &Error{Status: http.StatusServiceUnavailable, Code: CodeNoEligibleModel, Message: err.Error(), RetryAfter: time.Second}
	case errors.Is(err, router.ErrOverloaded):
		return &Error{Status: http.StatusServiceUnavailable, Code: CodeOverloaded, Message: err.Error(), RetryAfter: time.Second}
	}
	return &Error{Status: http.StatusBadGateway, Code: CodeUpstreamError, Message: "routing failed"}
}

// fromUpstreamError maps a provider failure to an API error. Upstream text
// reaches the client only for 4xx kinds the client can act on; 5xx responses
// carry the kind alone.
func fromUpstreamError(err error) *Error {
	kind := provider.KindOf(err)
	var pe *provider.Error
	if !errors.As(err, &pe) {
		pe = &provider.Error{Kind: kind}
	}
	switch kind {
	case provider.KindRateLimited:
		retry := pe.RetryAfter
		if retry <= 0 {
			retry = time.Second
		}
		return &Error{Status: http.StatusTooManyRequests, Code: CodeRateLimited, Message: "upstream provider rate limited the request", RetryAfter: retry}
	case provider.KindContextLength:
		return &Error{Status: http.StatusBadRequest, Code: CodeContextLengthExceeded, Message: pe.Message}
	case provider.KindBadRequest:
		return &Error{Status: http.StatusBadRequest, Code: CodeInvalidRequest, Message: "upstream rejected the request: " + pe.Message}
	case provider.KindTimeout:
		return &Error{Status: http.StatusGatewayTimeout, Code: CodeDeadlineExceeded, Message: "request deadline exceeded"}
	}
	return &Error{Status: http.StatusBadGateway, Code: CodeUpstreamError, Message: fmt.Sprintf("upstream provider error (%s)", kind)}
}
