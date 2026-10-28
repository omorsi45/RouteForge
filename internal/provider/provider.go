// Package provider defines the contracts between the gateway core and
// upstream model providers, plus shared error and SSE helpers.
package provider

import "context"

// Message roles.
const (
	RoleSystem    = "system"
	RoleUser      = "user"
	RoleAssistant = "assistant"
)

// Message is one chat message. Array text parts are already joined with "\n".
type Message struct {
	Role    string
	Content string
}

// Request is a provider-neutral completion request.
type Request struct {
	Model       string // upstream model id, set by the executor per attempt
	Messages    []Message
	MaxTokens   int // always > 0 from the API; adapters treat <= 0 as unset
	Temperature *float64
	TopP        *float64
	Stop        []string
	User        string
}

// Usage reports token counts.
type Usage struct {
	PromptTokens     int
	CompletionTokens int
}

// Response is a complete (non-streaming) result.
type Response struct {
	Content      string
	FinishReason string
	Usage        *Usage // nil means the upstream did not report usage
}

// StreamEvent is one incremental streaming result. At least one field is set.
type StreamEvent struct {
	Delta        string
	FinishReason string
	Usage        *Usage
}

// Stream yields events from a streaming completion.
//
// Events arrive in order. A clean end is io.EOF. An upstream failure is a
// *Error. Caller cancellation returns an error satisfying
// errors.Is(err, context.Canceled). Deadline expiry is *Error{Kind:
// KindTimeout}. Close is idempotent and safe after io.EOF. Close may be called
// concurrently with Recv.
type Stream interface {
	Recv() (StreamEvent, error)
	Close() error
}

// Provider is an upstream model provider adapter.
type Provider interface {
	Complete(ctx context.Context, req Request) (*Response, error)
	Stream(ctx context.Context, req Request) (Stream, error)
}
