// Package anthropic adapts the Anthropic Messages API to the provider
// contracts.
package anthropic

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"io"
	"net/http"
	"strings"
	"time"

	"github.com/omorsi45/routeforge/internal/provider"
)

// APIVersion is the anthropic-version header value sent with every request.
const APIVersion = "2023-06-01"

const (
	maxErrorBodyBytes = 64 << 10
	defaultMaxTokens  = 4096 // max_tokens is required by the API
)

// Options configures a Provider.
type Options struct {
	Name    string       // provider name used in errors
	BaseURL string       // requests go to TrimRight(BaseURL, "/") + "/v1/messages"
	APIKey  string       // sent as x-api-key; empty omits the header
	Client  *http.Client // nil: http.DefaultClient
}

// Provider talks to an Anthropic Messages endpoint.
type Provider struct {
	name   string
	url    string
	apiKey string
	client *http.Client
}

var _ provider.Provider = (*Provider)(nil)

// New returns a Provider for o.
func New(o Options) *Provider {
	p := &Provider{
		name:   o.Name,
		url:    strings.TrimRight(o.BaseURL, "/") + "/v1/messages",
		apiKey: o.APIKey,
		client: o.Client,
	}
	if p.client == nil {
		p.client = http.DefaultClient
	}
	return p
}

type wireMessage struct {
	Role    string `json:"role"`
	Content string `json:"content"`
}

type wireMetadata struct {
	UserID string `json:"user_id"`
}

type wireRequest struct {
	Model         string        `json:"model"`
	MaxTokens     int           `json:"max_tokens"`
	System        string        `json:"system,omitempty"`
	Messages      []wireMessage `json:"messages"`
	Temperature   *float64      `json:"temperature,omitempty"`
	TopP          *float64      `json:"top_p,omitempty"`
	StopSequences []string      `json:"stop_sequences,omitempty"`
	Metadata      *wireMetadata `json:"metadata,omitempty"`
	Stream        bool          `json:"stream"`
}

type apiError struct {
	Type    string `json:"type"`
	Message string `json:"message"`
}

// usage holds token counts. InputTokens is a pointer so a stream
// message_delta that omits it leaves the earlier input count alone.
type usage struct {
	InputTokens              *int `json:"input_tokens"`
	CacheCreationInputTokens int  `json:"cache_creation_input_tokens"`
	CacheReadInputTokens     int  `json:"cache_read_input_tokens"`
	OutputTokens             int  `json:"output_tokens"`
}

// prompt returns the input tokens including both cache counters.
func (u *usage) prompt() int {
	n := u.CacheCreationInputTokens + u.CacheReadInputTokens
	if u.InputTokens != nil {
		n += *u.InputTokens
	}
	return n
}

type wireResponse struct {
	Content []struct {
		Type string `json:"type"`
		Text string `json:"text"`
	} `json:"content"`
	StopReason string `json:"stop_reason"`
	Usage      *usage `json:"usage"`
}

// Complete performs a non-streaming completion.
func (p *Provider) Complete(ctx context.Context, req provider.Request) (*provider.Response, error) {
	resp, err := p.do(ctx, req, false)
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()

	body, err := provider.ReadAllLimited(resp.Body, provider.MaxResponseBytes)
	if err != nil {
		if errors.Is(err, provider.ErrResponseTooLarge) {
			return nil, p.internal("response body exceeds size limit", err)
		}
		return nil, provider.TransportError(ctx, p.name, err)
	}
	var wr wireResponse
	if err := decodeJSON(body, &wr); err != nil {
		return nil, p.internal("undecodable response", err)
	}
	var text strings.Builder
	for _, block := range wr.Content {
		if block.Type == "text" {
			text.WriteString(block.Text)
		}
	}
	out := &provider.Response{Content: text.String(), FinishReason: mapStop(wr.StopReason)}
	if wr.Usage != nil {
		out.Usage = &provider.Usage{PromptTokens: wr.Usage.prompt(), CompletionTokens: wr.Usage.OutputTokens}
	}
	return out, nil
}

// Stream starts a streaming completion. A non-2xx status is returned as an
// error before any event is read.
func (p *Provider) Stream(ctx context.Context, req provider.Request) (provider.Stream, error) {
	resp, err := p.do(ctx, req, true)
	if err != nil {
		return nil, err
	}
	return &stream{
		ctx:    ctx,
		name:   p.name,
		apiKey: p.apiKey,
		body:   resp.Body,
		sse:    provider.NewSSEReader(resp.Body, provider.MaxSSEBytes),
	}, nil
}

// do sends the request and returns the response for a 2xx status. Any other
// outcome is returned as an error with the body already closed.
func (p *Provider) do(ctx context.Context, req provider.Request, streaming bool) (*http.Response, error) {
	wr, err := p.buildBody(req, streaming)
	if err != nil {
		return nil, err
	}
	payload, err := json.Marshal(wr)
	if err != nil {
		return nil, p.internal("encode request", err)
	}
	hreq, err := http.NewRequestWithContext(ctx, http.MethodPost, p.url, bytes.NewReader(payload))
	if err != nil {
		return nil, p.internal("build request", err)
	}
	hreq.Header.Set("Content-Type", "application/json")
	hreq.Header.Set("anthropic-version", APIVersion)
	if streaming {
		hreq.Header.Set("Accept", "text/event-stream")
	} else {
		hreq.Header.Set("Accept", "application/json")
	}
	if p.apiKey != "" {
		hreq.Header.Set("x-api-key", p.apiKey)
	}

	resp, err := p.client.Do(hreq)
	if err != nil {
		return nil, provider.TransportError(ctx, p.name, err)
	}
	if resp.StatusCode >= 200 && resp.StatusCode < 300 {
		return resp, nil
	}
	defer resp.Body.Close()
	return nil, p.statusError(ctx, resp)
}

// buildBody moves system messages into the top-level system field. It fails
// without a request when no user or assistant message remains.
func (p *Provider) buildBody(req provider.Request, streaming bool) (*wireRequest, error) {
	var system []string
	msgs := make([]wireMessage, 0, len(req.Messages))
	for _, m := range req.Messages {
		if m.Role == provider.RoleSystem {
			system = append(system, m.Content)
			continue
		}
		msgs = append(msgs, wireMessage{Role: m.Role, Content: m.Content})
	}
	if len(msgs) == 0 {
		return nil, &provider.Error{
			Kind:     provider.KindBadRequest,
			Provider: p.name,
			Message:  "anthropic: at least one user or assistant message is required",
		}
	}
	wr := &wireRequest{
		Model:         req.Model,
		MaxTokens:     req.MaxTokens,
		System:        strings.Join(system, "\n\n"),
		Messages:      msgs,
		Temperature:   req.Temperature,
		TopP:          req.TopP,
		StopSequences: req.Stop,
		Stream:        streaming,
	}
	if wr.MaxTokens <= 0 {
		wr.MaxTokens = defaultMaxTokens
	}
	if req.User != "" {
		wr.Metadata = &wireMetadata{UserID: req.User}
	}
	return wr, nil
}

// statusError maps a non-2xx response to a classified error. The error type
// in the body wins over the status; an unknown type or unparsable body falls
// back to the status.
func (p *Provider) statusError(ctx context.Context, resp *http.Response) error {
	kind := provider.KindFromStatus(resp.StatusCode)
	msg := http.StatusText(resp.StatusCode)

	raw, err := provider.ReadAllLimited(resp.Body, maxErrorBodyBytes)
	if err != nil && ctx.Err() != nil {
		return provider.TransportError(ctx, p.name, err)
	}
	if err == nil {
		var env struct {
			Error *apiError `json:"error"`
		}
		if decodeJSON(raw, &env) == nil && env.Error != nil {
			if k, ok := kindForType(env.Error.Type, env.Error.Message); ok {
				kind = k
			}
			if env.Error.Message != "" {
				msg = provider.Redact(env.Error.Message, p.apiKey)
			}
		}
	}
	return &provider.Error{
		Kind:       kind,
		Provider:   p.name,
		Status:     resp.StatusCode,
		RetryAfter: provider.ParseRetryAfter(resp.Header, time.Now()),
		Message:    msg,
	}
}

// kindForType maps an Anthropic error type to a kind. It reports false for
// an unknown type.
func kindForType(typ, message string) (provider.ErrorKind, bool) {
	switch typ {
	case "invalid_request_error":
		if strings.Contains(strings.ToLower(message), "prompt is too long") {
			return provider.KindContextLength, true
		}
		return provider.KindBadRequest, true
	case "authentication_error", "permission_error":
		return provider.KindAuth, true
	case "not_found_error", "request_too_large":
		return provider.KindBadRequest, true
	case "rate_limit_error":
		return provider.KindRateLimited, true
	case "api_error", "overloaded_error":
		return provider.KindUnavailable, true
	case "timeout_error":
		return provider.KindTimeout, true
	}
	return "", false
}

func (p *Provider) internal(msg string, err error) *provider.Error {
	return &provider.Error{Kind: provider.KindInternal, Provider: p.name, Message: msg, Err: err}
}

// mapStop converts an Anthropic stop_reason to a gateway finish reason.
func mapStop(reason string) string {
	switch reason {
	case "max_tokens":
		return "length"
	case "refusal":
		return "content_filter"
	}
	return "stop" // end_turn, stop_sequence, absent and unknown reasons
}

// decodeJSON decodes exactly one JSON value from b, rejecting trailing data.
func decodeJSON(b []byte, v any) error {
	dec := json.NewDecoder(bytes.NewReader(b))
	if err := dec.Decode(v); err != nil {
		return err
	}
	if _, err := dec.Token(); err != io.EOF {
		return errors.New("trailing data after JSON value")
	}
	return nil
}
