package api

import (
	"bytes"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http"
	"strconv"
	"strings"

	"github.com/omorsi45/routeforge/internal/provider"
)

// maxStopSequences is the OpenAI limit on stop entries.
const maxStopSequences = 4

// ChatRequest is a validated chat completions request.
type ChatRequest struct {
	Model        string
	Messages     []provider.Message
	MaxTokens    *int // max_completion_tokens if set, else max_tokens, else nil
	Temperature  *float64
	TopP         *float64
	Stop         []string
	Stream       bool
	IncludeUsage bool // stream_options.include_usage
	User         string
}

// wireRequest mirrors the request JSON. Fields that change semantics are kept
// raw so "set" can be judged without committing to a type.
type wireRequest struct {
	Model               string          `json:"model"`
	Messages            []wireMessage   `json:"messages"`
	MaxTokens           *int            `json:"max_tokens"`
	MaxCompletionTokens *int            `json:"max_completion_tokens"`
	Temperature         *float64        `json:"temperature"`
	TopP                *float64        `json:"top_p"`
	Stop                json.RawMessage `json:"stop"`
	Stream              bool            `json:"stream"`
	StreamOptions       *struct {
		IncludeUsage bool `json:"include_usage"`
	} `json:"stream_options"`
	User string `json:"user"`
	N    *int   `json:"n"`

	Tools          json.RawMessage `json:"tools"`
	ToolChoice     json.RawMessage `json:"tool_choice"`
	Functions      json.RawMessage `json:"functions"`
	FunctionCall   json.RawMessage `json:"function_call"`
	Logprobs       json.RawMessage `json:"logprobs"`
	TopLogprobs    json.RawMessage `json:"top_logprobs"`
	Audio          json.RawMessage `json:"audio"`
	Modalities     json.RawMessage `json:"modalities"`
	ResponseFormat json.RawMessage `json:"response_format"`
}

type wireMessage struct {
	Role      string          `json:"role"`
	Content   json.RawMessage `json:"content"`
	ToolCalls json.RawMessage `json:"tool_calls"`
}

// wirePart is one element of an array content value. Text is a pointer so a
// missing or null text can be told apart from an empty one.
type wirePart struct {
	Type string  `json:"type"`
	Text *string `json:"text"`
}

func invalid(format string, args ...any) *Error {
	return &Error{Status: http.StatusBadRequest, Code: CodeInvalidRequest, Message: fmt.Sprintf(format, args...)}
}

func unsupported(format string, args ...any) *Error {
	return &Error{Status: http.StatusBadRequest, Code: CodeUnsupportedParameter, Message: fmt.Sprintf(format, args...)}
}

// isSet reports whether a raw JSON value is present and not null, false, 0,
// "", [] or {}.
func isSet(raw json.RawMessage) bool {
	if len(raw) == 0 {
		return false
	}
	var buf bytes.Buffer
	if err := json.Compact(&buf, raw); err != nil {
		// The decoder already validated raw; treat anything odd as set.
		return true
	}
	switch s := buf.String(); s {
	case "null", "false", `""`, "[]", "{}":
		return false
	default:
		f, err := strconv.ParseFloat(s, 64)
		return err != nil || f != 0
	}
}

// decodeChatRequest reads one chat completions body and validates it.
func decodeChatRequest(body io.Reader) (*ChatRequest, *Error) {
	data, err := io.ReadAll(body)
	if err != nil {
		var mbe *http.MaxBytesError
		if errors.As(err, &mbe) {
			return nil, &Error{Status: http.StatusRequestEntityTooLarge, Code: CodeRequestTooLarge, Message: "request body is too large"}
		}
		return nil, invalid("failed to read request body")
	}
	var w wireRequest
	if err := provider.DecodeJSON(data, &w); err != nil {
		return nil, decodeError(err)
	}
	if e := checkUnsupported(&w); e != nil {
		return nil, e
	}
	if w.Model == "" {
		return nil, invalid("model is required")
	}
	if len(w.Messages) == 0 {
		return nil, invalid("messages must be a non-empty array")
	}
	msgs := make([]provider.Message, len(w.Messages))
	for i, m := range w.Messages {
		pm, e := convertMessage(i, m)
		if e != nil {
			return nil, e
		}
		msgs[i] = pm
	}
	req := &ChatRequest{
		Model:       w.Model,
		Messages:    msgs,
		Temperature: w.Temperature,
		TopP:        w.TopP,
		Stream:      w.Stream,
		User:        w.User,
	}
	if w.StreamOptions != nil {
		req.IncludeUsage = w.StreamOptions.IncludeUsage
	}
	if e := checkRanges(&w); e != nil {
		return nil, e
	}
	stop, e := parseStop(w.Stop)
	if e != nil {
		return nil, e
	}
	req.Stop = stop
	req.MaxTokens = w.MaxCompletionTokens
	if req.MaxTokens == nil {
		req.MaxTokens = w.MaxTokens
	}
	return req, nil
}

// decodeError maps a JSON decode failure to an API error.
func decodeError(err error) *Error {
	var ute *json.UnmarshalTypeError
	if errors.As(err, &ute) && ute.Field != "" {
		return invalid("field %q has the wrong type: expected %s, got %s", ute.Field, ute.Type, ute.Value)
	}
	var se *json.SyntaxError
	if errors.As(err, &se) {
		return invalid("request body is not valid JSON")
	}
	return invalid("request body must contain a single JSON object")
}

// checkUnsupported rejects top-level fields the gateway does not implement.
func checkUnsupported(w *wireRequest) *Error {
	for _, f := range []struct {
		name string
		raw  json.RawMessage
	}{
		{"tools", w.Tools},
		{"tool_choice", w.ToolChoice},
		{"functions", w.Functions},
		{"function_call", w.FunctionCall},
		{"logprobs", w.Logprobs},
		{"top_logprobs", w.TopLogprobs},
		{"audio", w.Audio},
	} {
		if isSet(f.raw) {
			return unsupported("parameter %q is not supported", f.name)
		}
	}
	if w.N != nil && *w.N > 1 {
		return unsupported("parameter %q is not supported above 1", "n")
	}
	if isSet(w.Modalities) {
		var mods []string
		if err := json.Unmarshal(w.Modalities, &mods); err != nil {
			return invalid("field %q must be an array of strings", "modalities")
		}
		for _, m := range mods {
			if m != "text" {
				return unsupported("modality %q in %q is not supported", m, "modalities")
			}
		}
	}
	if isSet(w.ResponseFormat) {
		var rf struct {
			Type string `json:"type"`
		}
		if err := json.Unmarshal(w.ResponseFormat, &rf); err != nil {
			return invalid("field %q must be an object", "response_format")
		}
		if rf.Type != "text" {
			return unsupported("%q type %q is not supported", "response_format", rf.Type)
		}
	}
	return nil
}

// convertMessage validates message i and converts it to the provider shape.
func convertMessage(i int, m wireMessage) (provider.Message, *Error) {
	var role string
	switch m.Role {
	case provider.RoleSystem, "developer":
		role = provider.RoleSystem
	case provider.RoleUser, provider.RoleAssistant:
		role = m.Role
	case "tool", "function":
		return provider.Message{}, unsupported("messages[%d].role %q is not supported", i, m.Role)
	default:
		return provider.Message{}, invalid("messages[%d].role %q is not a valid role", i, m.Role)
	}
	if isSet(m.ToolCalls) {
		return provider.Message{}, unsupported("messages[%d].tool_calls is not supported", i)
	}
	content, e := parseContent(i, m.Content)
	if e != nil {
		return provider.Message{}, e
	}
	return provider.Message{Role: role, Content: content}, nil
}

// parseContent accepts a string or an array of text parts.
func parseContent(i int, raw json.RawMessage) (string, *Error) {
	trimmed := bytes.TrimSpace(raw)
	if len(trimmed) == 0 || bytes.Equal(trimmed, []byte("null")) {
		return "", invalid("messages[%d].content is required", i)
	}
	switch trimmed[0] {
	case '"':
		var s string
		if err := json.Unmarshal(trimmed, &s); err != nil {
			return "", invalid("messages[%d].content is not a valid string", i)
		}
		return s, nil
	case '[':
		var parts []wirePart
		if err := json.Unmarshal(trimmed, &parts); err != nil {
			return "", invalid("messages[%d].content must be an array of content parts", i)
		}
		texts := make([]string, len(parts))
		for j, p := range parts {
			if p.Type != "text" {
				return "", unsupported("messages[%d].content[%d].type %q is not supported", i, j, p.Type)
			}
			if p.Text == nil {
				return "", invalid("messages[%d].content[%d].text must be a string", i, j)
			}
			texts[j] = *p.Text
		}
		return strings.Join(texts, "\n"), nil
	}
	return "", invalid("messages[%d].content must be a string or an array of text parts", i)
}

// checkRanges validates the numeric fields.
func checkRanges(w *wireRequest) *Error {
	if w.N != nil && *w.N < 1 {
		return invalid("n must be at least 1")
	}
	if w.MaxTokens != nil && *w.MaxTokens < 1 {
		return invalid("max_tokens must be at least 1")
	}
	if w.MaxCompletionTokens != nil && *w.MaxCompletionTokens < 1 {
		return invalid("max_completion_tokens must be at least 1")
	}
	if w.Temperature != nil && (*w.Temperature < 0 || *w.Temperature > 2) {
		return invalid("temperature must be between 0 and 2")
	}
	if w.TopP != nil && (*w.TopP < 0 || *w.TopP > 1) {
		return invalid("top_p must be between 0 and 1")
	}
	return nil
}

// parseStop accepts a string or an array of at most four strings.
func parseStop(raw json.RawMessage) ([]string, *Error) {
	trimmed := bytes.TrimSpace(raw)
	if len(trimmed) == 0 || bytes.Equal(trimmed, []byte("null")) {
		return nil, nil
	}
	if trimmed[0] == '"' {
		var s string
		if err := json.Unmarshal(trimmed, &s); err != nil {
			return nil, invalid("stop must be a string or an array of strings")
		}
		return []string{s}, nil
	}
	var list []string
	if trimmed[0] != '[' || json.Unmarshal(trimmed, &list) != nil {
		return nil, invalid("stop must be a string or an array of strings")
	}
	if len(list) > maxStopSequences {
		return nil, invalid("stop accepts at most %d sequences", maxStopSequences)
	}
	if len(list) == 0 {
		return nil, nil
	}
	return list, nil
}
