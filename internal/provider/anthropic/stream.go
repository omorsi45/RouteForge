package anthropic

import (
	"context"
	"encoding/json"
	"errors"
	"io"
	"sync"

	"github.com/omorsi45/routeforge/internal/provider"
)

// envelope is the superset of fields the stream events carry that this
// adapter reads. The event type comes from the data's "type" field.
type envelope struct {
	Type    string `json:"type"`
	Message *struct {
		Usage *usage `json:"usage"`
	} `json:"message"`
	Delta json.RawMessage `json:"delta"`
	Usage *usage          `json:"usage"`
	Error *apiError       `json:"error"`
}

// stream reads an Anthropic SSE completion. Any terminal outcome (clean end
// or failure) is stored and returned by every later Recv. It is meant for a
// single receiving goroutine; Close may be called from any goroutine.
type stream struct {
	ctx    context.Context
	name   string
	apiKey string
	body   io.ReadCloser
	sse    *provider.SSEReader

	// Accumulated across events. Output usage is cumulative: later values
	// replace earlier ones.
	input, output int
	haveUsage     bool
	finish        string

	err error // terminal result of Recv: io.EOF or a failure

	closeOnce sync.Once
	closeErr  error
}

var _ provider.Stream = (*stream)(nil)

// Recv returns the next event. A clean end requires message_stop and is
// io.EOF; end of input before it is a retryable interruption.
func (s *stream) Recv() (provider.StreamEvent, error) {
	if s.err != nil {
		return provider.StreamEvent{}, s.err
	}
	for {
		ev, err := s.sse.Next()
		if err != nil {
			s.err = s.readError(err)
			return provider.StreamEvent{}, s.err
		}
		var env envelope
		if err := provider.DecodeJSON([]byte(ev.Data), &env); err != nil {
			s.err = s.undecodable(err)
			return provider.StreamEvent{}, s.err
		}
		out, emit, err := s.handle(&env)
		if err != nil {
			s.err = err
			return provider.StreamEvent{}, s.err
		}
		if emit {
			return out, nil
		}
	}
}

// handle applies one event. emit reports whether out is an event to return.
func (s *stream) handle(env *envelope) (out provider.StreamEvent, emit bool, err error) {
	switch env.Type {
	case "message_start":
		if env.Message != nil && env.Message.Usage != nil {
			s.input, s.output, s.haveUsage = env.Message.Usage.prompt(), env.Message.Usage.OutputTokens, true
		}
	case "content_block_delta":
		var d struct {
			Type string `json:"type"`
			Text string `json:"text"`
		}
		if err := decodeDelta(env.Delta, &d); err != nil {
			return out, false, s.undecodable(err)
		}
		if d.Type == "text_delta" && d.Text != "" {
			return provider.StreamEvent{Delta: d.Text}, true, nil
		}
	case "message_delta":
		var d struct {
			StopReason string `json:"stop_reason"`
		}
		if err := decodeDelta(env.Delta, &d); err != nil {
			return out, false, s.undecodable(err)
		}
		if d.StopReason != "" {
			s.finish = mapStop(d.StopReason)
		}
		if env.Usage != nil {
			s.output = env.Usage.OutputTokens // cumulative
			if env.Usage.InputTokens != nil {
				s.input = env.Usage.prompt()
			}
		}
	case "message_stop":
		s.err = io.EOF
		out.FinishReason = s.finish
		if out.FinishReason == "" {
			out.FinishReason = "stop"
		}
		if s.haveUsage {
			out.Usage = &provider.Usage{PromptTokens: s.input, CompletionTokens: s.output}
		}
		return out, true, nil
	case "error":
		return out, false, s.eventError(env.Error)
	} // ping, content_block_start, content_block_stop and unknown types are skipped
	return out, false, nil
}

// Close releases the response body. It is idempotent.
func (s *stream) Close() error {
	s.closeOnce.Do(func() { s.closeErr = s.body.Close() })
	return s.closeErr
}

// decodeDelta decodes an event's delta object. An absent delta is empty.
func decodeDelta(raw json.RawMessage, v any) error {
	if len(raw) == 0 {
		return nil
	}
	return provider.DecodeJSON(raw, v)
}

// readError converts an SSEReader error into a provider error.
func (s *stream) readError(err error) error {
	switch {
	case errors.Is(err, io.EOF):
		return &provider.Error{Kind: provider.KindUnavailable, Provider: s.name, Message: "stream ended before message_stop"}
	case errors.Is(err, provider.ErrSSETooLarge):
		return &provider.Error{Kind: provider.KindInternal, Provider: s.name, Message: "stream event exceeds size limit", Err: err}
	}
	return provider.TransportError(s.ctx, s.name, err)
}

// undecodable reports an event that is not valid JSON of the expected shape.
// That is an upstream fault (for example a stream cut mid-line), so it is
// retryable like any other interruption.
func (s *stream) undecodable(err error) error {
	return &provider.Error{Kind: provider.KindUnavailable, Provider: s.name, Message: "undecodable stream event", Err: err}
}

// eventError converts an in-stream error event, using the same kind table as
// HTTP errors. Without a recognized type the failure is unavailable.
func (s *stream) eventError(e *apiError) error {
	kind, msg := provider.KindUnavailable, ""
	if e != nil {
		if k, ok := kindForType(e.Type, e.Message); ok {
			kind = k
		}
		msg = provider.Redact(e.Message, s.apiKey)
	}
	if msg == "" {
		msg = "upstream stream error"
	}
	return &provider.Error{Kind: kind, Provider: s.name, Message: msg}
}
