package openai

import (
	"context"
	"errors"
	"io"
	"strings"
	"sync"

	"github.com/omorsi45/routeforge/internal/provider"
)

type wireChunk struct {
	Error   *wireError `json:"error"`
	Choices []struct {
		Delta struct {
			Content *string `json:"content"`
		} `json:"delta"`
		FinishReason *string `json:"finish_reason"`
	} `json:"choices"`
	Usage *wireUsage `json:"usage"`
}

// stream reads an OpenAI-style SSE completion. Any terminal outcome (clean
// end or failure) is stored and returned by every later Recv. It is meant for
// a single receiving goroutine; Close may be called from any goroutine.
type stream struct {
	ctx    context.Context
	name   string
	apiKey string
	body   io.ReadCloser
	sse    *provider.SSEReader

	err error // terminal result of Recv: io.EOF or a failure

	closeOnce sync.Once
	closeErr  error
}

var _ provider.Stream = (*stream)(nil)

// Recv returns the next event. A clean end requires "data: [DONE]" and is
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
		data := strings.TrimSpace(ev.Data)
		if data == "[DONE]" {
			s.err = io.EOF
			return provider.StreamEvent{}, s.err
		}
		out, err := s.parseChunk(data)
		if err != nil {
			s.err = err
			return provider.StreamEvent{}, s.err
		}
		if out != (provider.StreamEvent{}) {
			return out, nil
		}
	}
}

// Close releases the response body. It is idempotent.
func (s *stream) Close() error {
	s.closeOnce.Do(func() { s.closeErr = s.body.Close() })
	return s.closeErr
}

// readError converts an SSEReader error into a provider error.
func (s *stream) readError(err error) error {
	switch {
	case errors.Is(err, io.EOF):
		return &provider.Error{Kind: provider.KindUnavailable, Provider: s.name, Message: "stream ended before [DONE]"}
	case errors.Is(err, provider.ErrSSETooLarge):
		return &provider.Error{Kind: provider.KindInternal, Provider: s.name, Message: "stream event exceeds size limit", Err: err}
	}
	return provider.TransportError(s.ctx, s.name, err)
}

// parseChunk converts one data payload into an event. An empty event means
// the chunk carried nothing to report.
func (s *stream) parseChunk(data string) (provider.StreamEvent, error) {
	var c wireChunk
	if err := provider.DecodeJSON([]byte(data), &c); err != nil {
		// An undecodable chunk is an upstream fault (for example a stream cut
		// mid-line), so it is retryable like any other interruption.
		return provider.StreamEvent{}, &provider.Error{Kind: provider.KindUnavailable, Provider: s.name, Message: "undecodable stream chunk", Err: err}
	}
	if c.Error != nil {
		kind := provider.KindUnavailable
		if isContextLength(c.Error.Code) {
			kind = provider.KindContextLength
		}
		msg := provider.Redact(c.Error.Message, s.apiKey)
		if msg == "" {
			msg = "upstream stream error"
		}
		return provider.StreamEvent{}, &provider.Error{Kind: kind, Provider: s.name, Message: msg}
	}
	var out provider.StreamEvent
	if len(c.Choices) > 0 {
		ch := c.Choices[0]
		if ch.Delta.Content != nil {
			out.Delta = *ch.Delta.Content
		}
		if ch.FinishReason != nil && *ch.FinishReason != "" {
			out.FinishReason = normalizeFinish(ch.FinishReason)
		}
	}
	out.Usage = c.Usage.toUsage()
	return out, nil
}
