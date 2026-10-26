// Package openai adapts OpenAI-compatible chat completion endpoints (OpenAI,
// Ollama, vLLM and similar) to the provider contracts.
package openai

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

const (
	maxErrorBodyBytes = 64 << 10
	codeContextLength = "context_length_exceeded"
)

// Options configures a Provider.
type Options struct {
	Name           string       // provider name used in errors
	BaseURL        string       // e.g. https://api.openai.com/v1; requests go to TrimRight(BaseURL, "/") + "/chat/completions"
	APIKey         string       // empty: no Authorization header (Ollama, vLLM)
	MaxTokensParam string       // "max_completion_tokens" or "max_tokens"; empty means "max_tokens"
	Client         *http.Client // nil: http.DefaultClient
}

// Provider talks to an OpenAI-compatible chat completions endpoint.
type Provider struct {
	name      string
	url       string
	apiKey    string
	maxTokens string
	client    *http.Client
}

var _ provider.Provider = (*Provider)(nil)

// New returns a Provider for o.
func New(o Options) *Provider {
	p := &Provider{
		name:      o.Name,
		url:       strings.TrimRight(o.BaseURL, "/") + "/chat/completions",
		apiKey:    o.APIKey,
		maxTokens: o.MaxTokensParam,
		client:    o.Client,
	}
	if p.maxTokens == "" {
		p.maxTokens = "max_tokens"
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

type wireError struct {
	Message string `json:"message"`
	Type    string `json:"type"`
	Code    any    `json:"code"`
}

type wireUsage struct {
	PromptTokens     int `json:"prompt_tokens"`
	CompletionTokens int `json:"completion_tokens"`
}

func (u *wireUsage) toUsage() *provider.Usage {
	if u == nil {
		return nil
	}
	return &provider.Usage{PromptTokens: u.PromptTokens, CompletionTokens: u.CompletionTokens}
}

type wireResponse struct {
	Choices []struct {
		Message struct {
			Content *string `json:"content"`
		} `json:"message"`
		FinishReason *string `json:"finish_reason"`
	} `json:"choices"`
	Usage *wireUsage `json:"usage"`
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
	if len(wr.Choices) == 0 {
		return nil, p.internal("response has no choices", nil)
	}
	choice := wr.Choices[0]
	out := &provider.Response{
		FinishReason: normalizeFinish(choice.FinishReason),
		Usage:        wr.Usage.toUsage(),
	}
	if choice.Message.Content != nil {
		out.Content = *choice.Message.Content
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
	payload, err := json.Marshal(p.buildBody(req, streaming))
	if err != nil {
		return nil, p.internal("encode request", err)
	}
	hreq, err := http.NewRequestWithContext(ctx, http.MethodPost, p.url, bytes.NewReader(payload))
	if err != nil {
		return nil, p.internal("build request", err)
	}
	hreq.Header.Set("Content-Type", "application/json")
	if streaming {
		hreq.Header.Set("Accept", "text/event-stream")
	} else {
		hreq.Header.Set("Accept", "application/json")
	}
	if p.apiKey != "" {
		hreq.Header.Set("Authorization", "Bearer "+p.apiKey)
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

func (p *Provider) buildBody(req provider.Request, streaming bool) map[string]any {
	msgs := make([]wireMessage, len(req.Messages))
	for i, m := range req.Messages {
		msgs[i] = wireMessage{Role: m.Role, Content: m.Content}
	}
	body := map[string]any{
		"model":    req.Model,
		"messages": msgs,
		"stream":   streaming,
	}
	if req.MaxTokens > 0 {
		body[p.maxTokens] = req.MaxTokens
	}
	if req.Temperature != nil {
		body["temperature"] = *req.Temperature
	}
	if req.TopP != nil {
		body["top_p"] = *req.TopP
	}
	if len(req.Stop) > 0 {
		body["stop"] = req.Stop
	}
	if req.User != "" {
		body["user"] = req.User
	}
	if streaming {
		body["stream_options"] = map[string]any{"include_usage": true}
	}
	return body
}

// statusError maps a non-2xx response to a classified error.
func (p *Provider) statusError(ctx context.Context, resp *http.Response) error {
	kind := provider.KindFromStatus(resp.StatusCode)
	msg := http.StatusText(resp.StatusCode)

	raw, err := provider.ReadAllLimited(resp.Body, maxErrorBodyBytes)
	if err != nil && ctx.Err() != nil {
		return provider.TransportError(ctx, p.name, err)
	}
	if err == nil {
		var env struct {
			Error *wireError `json:"error"`
		}
		if decodeJSON(raw, &env) == nil && env.Error != nil && env.Error.Message != "" {
			msg = provider.Redact(env.Error.Message, p.apiKey)
			if isContextLength(env.Error.Code) {
				kind = provider.KindContextLength
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

func (p *Provider) internal(msg string, err error) *provider.Error {
	return &provider.Error{Kind: provider.KindInternal, Provider: p.name, Message: msg, Err: err}
}

func isContextLength(code any) bool {
	s, ok := code.(string)
	return ok && s == codeContextLength
}

// normalizeFinish keeps the finish reasons the gateway exposes and maps the
// rest (and absence) to "stop".
func normalizeFinish(fr *string) string {
	if fr != nil {
		switch *fr {
		case "stop", "length", "content_filter":
			return *fr
		}
	}
	return "stop"
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
