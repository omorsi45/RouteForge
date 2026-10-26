package api

import (
	"io"
	"net/http"
	"reflect"
	"strings"
	"testing"

	"github.com/omorsi45/routeforge/internal/provider"
)

// base is a minimal valid request body without its closing brace.
const base = `{"model":"m","messages":[{"role":"user","content":"hi"}]`

func intp(v int) *int { return &v }

func TestDecodeChatRequestValid(t *testing.T) {
	hi := []provider.Message{{Role: provider.RoleUser, Content: "hi"}}
	cases := []struct {
		name string
		body string
		want ChatRequest
	}{
		{"minimal", base + `}`, ChatRequest{Model: "m", Messages: hi}},
		{
			"array content joined",
			`{"model":"m","messages":[{"role":"user","content":[{"type":"text","text":"a"},{"type":"text","text":"b"}]}]}`,
			ChatRequest{Model: "m", Messages: []provider.Message{{Role: provider.RoleUser, Content: "a\nb"}}},
		},
		{
			"developer becomes system",
			`{"model":"m","messages":[{"role":"developer","content":"sys"},{"role":"user","content":"hi"}]}`,
			ChatRequest{Model: "m", Messages: []provider.Message{{Role: provider.RoleSystem, Content: "sys"}, {Role: provider.RoleUser, Content: "hi"}}},
		},
		{
			"max_completion_tokens wins",
			base + `,"max_tokens":10,"max_completion_tokens":20}`,
			ChatRequest{Model: "m", Messages: hi, MaxTokens: intp(20)},
		},
		{
			"max_tokens alone",
			base + `,"max_tokens":10}`,
			ChatRequest{Model: "m", Messages: hi, MaxTokens: intp(10)},
		},
		{
			"stop string stream usage user",
			base + `,"stop":"END","stream":true,"stream_options":{"include_usage":true},"user":"u"}`,
			ChatRequest{Model: "m", Messages: hi, Stop: []string{"END"}, Stream: true, IncludeUsage: true, User: "u"},
		},
		{
			"stop array",
			base + `,"stop":["a","b","c","d"]}`,
			ChatRequest{Model: "m", Messages: hi, Stop: []string{"a", "b", "c", "d"}},
		},
		{
			"unset unsupported fields and unknown fields accepted",
			base + `,"tools":[],"tool_choice":null,"logprobs":false,"n":1,"modalities":["text"],"response_format":{"type":"text"},"seed":4,"foo":"bar"}`,
			ChatRequest{Model: "m", Messages: hi},
		},
		{
			"zero and empty unsupported fields count as unset",
			base + `,"functions":[],"function_call":"","top_logprobs":0,"audio":{},"modalities":[],"response_format":{}}`,
			ChatRequest{Model: "m", Messages: hi},
		},
		{
			"whitespace inside empty values is still unset",
			base + `,"tools":[ ],"audio":{ },"top_logprobs":0.0}`,
			ChatRequest{Model: "m", Messages: hi},
		},
		{
			"assistant tool_calls null is unset",
			`{"model":"m","messages":[{"role":"assistant","content":"x","tool_calls":null}]}`,
			ChatRequest{Model: "m", Messages: []provider.Message{{Role: provider.RoleAssistant, Content: "x"}}},
		},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			got, derr := decodeChatRequest(strings.NewReader(tc.body))
			if derr != nil {
				t.Fatalf("decodeChatRequest() error: %+v", derr)
			}
			if !reflect.DeepEqual(*got, tc.want) {
				t.Errorf("decodeChatRequest() = %+v, want %+v", *got, tc.want)
			}
		})
	}
}

func TestDecodeChatRequestSamplingParams(t *testing.T) {
	got, derr := decodeChatRequest(strings.NewReader(base + `,"temperature":2,"top_p":0}`))
	if derr != nil {
		t.Fatalf("decodeChatRequest() error: %+v", derr)
	}
	if got.Temperature == nil || *got.Temperature != 2 {
		t.Errorf("Temperature = %v, want 2", got.Temperature)
	}
	if got.TopP == nil || *got.TopP != 0 {
		t.Errorf("TopP = %v, want 0 (explicit zero must be kept)", got.TopP)
	}
}

func TestDecodeChatRequestRejects(t *testing.T) {
	cases := []struct {
		name            string
		body            string
		wantStatus      int
		wantCode        string
		wantMsgContains string
	}{
		{"tools", base + `,"tools":[{"type":"function"}]}`, 400, "unsupported_parameter", "tools"},
		{"tool_choice", base + `,"tool_choice":"auto"}`, 400, "unsupported_parameter", "tool_choice"},
		{"functions", base + `,"functions":[{"name":"f"}]}`, 400, "unsupported_parameter", "functions"},
		{"function_call", base + `,"function_call":"auto"}`, 400, "unsupported_parameter", "function_call"},
		{"n2", base + `,"n":2}`, 400, "unsupported_parameter", "n"},
		{"logprobs", base + `,"logprobs":true}`, 400, "unsupported_parameter", "logprobs"},
		{"top_logprobs", base + `,"top_logprobs":3}`, 400, "unsupported_parameter", "top_logprobs"},
		{"audio", base + `,"audio":{"voice":"alloy","format":"wav"}}`, 400, "unsupported_parameter", "audio"},
		{"modalities", base + `,"modalities":["text","audio"]}`, 400, "unsupported_parameter", "modalities"},
		{"json mode", base + `,"response_format":{"type":"json_object"}}`, 400, "unsupported_parameter", "response_format"},
		{"image part", `{"model":"m","messages":[{"role":"user","content":[{"type":"image_url","image_url":{"url":"x"}}]}]}`, 400, "unsupported_parameter", "image_url"},
		{"tool role", `{"model":"m","messages":[{"role":"tool","content":"x","tool_call_id":"1"}]}`, 400, "unsupported_parameter", "tool"},
		{"function role", `{"model":"m","messages":[{"role":"function","content":"x","name":"f"}]}`, 400, "unsupported_parameter", "function"},
		{"tool_calls", `{"model":"m","messages":[{"role":"assistant","content":null,"tool_calls":[{"id":"1"}]}]}`, 400, "unsupported_parameter", "tool_calls"},
		{"unknown role", `{"model":"m","messages":[{"role":"robot","content":"x"}]}`, 400, "invalid_request", "role"},
		{"missing role", `{"model":"m","messages":[{"content":"x"}]}`, 400, "invalid_request", "role"},
		{"no model", `{"messages":[{"role":"user","content":"hi"}]}`, 400, "invalid_request", "model"},
		{"empty model", `{"model":"","messages":[{"role":"user","content":"hi"}]}`, 400, "invalid_request", "model"},
		{"model type", `{"model":5,"messages":[{"role":"user","content":"hi"}]}`, 400, "invalid_request", "model"},
		{"stream type", base + `,"stream":"yes"}`, 400, "invalid_request", "stream"},
		{"nested type", `{"model":"m","messages":[{"role":7,"content":"hi"}]}`, 400, "invalid_request", ".role"},
		{"no messages", `{"model":"m"}`, 400, "invalid_request", "messages"},
		{"empty messages", `{"model":"m","messages":[]}`, 400, "invalid_request", "messages"},
		{"missing content", `{"model":"m","messages":[{"role":"user"}]}`, 400, "invalid_request", "content"},
		{"null content", `{"model":"m","messages":[{"role":"user","content":null}]}`, 400, "invalid_request", "content"},
		{"number content", `{"model":"m","messages":[{"role":"user","content":5}]}`, 400, "invalid_request", "content"},
		{"object content", `{"model":"m","messages":[{"role":"user","content":{"type":"text"}}]}`, 400, "invalid_request", "content"},
		{"n0", base + `,"n":0}`, 400, "invalid_request", "n"},
		{"max_tokens 0", base + `,"max_tokens":0}`, 400, "invalid_request", "max_tokens"},
		{"max_completion_tokens negative", base + `,"max_completion_tokens":-1}`, 400, "invalid_request", "max_completion_tokens"},
		{"temperature high", base + `,"temperature":2.5}`, 400, "invalid_request", "temperature"},
		{"temperature negative", base + `,"temperature":-0.1}`, 400, "invalid_request", "temperature"},
		{"top_p high", base + `,"top_p":1.5}`, 400, "invalid_request", "top_p"},
		{"top_p negative", base + `,"top_p":-0.5}`, 400, "invalid_request", "top_p"},
		{"stop 5", base + `,"stop":["a","b","c","d","e"]}`, 400, "invalid_request", "stop"},
		{"stop number", base + `,"stop":5}`, 400, "invalid_request", "stop"},
		{"stop mixed array", base + `,"stop":["a",1]}`, 400, "invalid_request", "stop"},
		{"trailing data", base + `} {}`, 400, "invalid_request", "single JSON object"},
		{"empty body", ``, 400, "invalid_request", "single JSON object"},
		{"malformed", `{"model":`, 400, "invalid_request", "JSON"},
		{"array body", `[]`, 400, "invalid_request", ""},
		{
			// Unsupported top-level fields win over later structural errors.
			"unsupported before model", `{"tools":[{"type":"function"}],"messages":[]}`, 400, "unsupported_parameter", "tools",
		},
		{
			// A message's unsupported role is reported before its missing content.
			"role before content", `{"model":"m","messages":[{"role":"tool"}]}`, 400, "unsupported_parameter", "tool",
		},
		{
			// A message's tool_calls is reported before its content problem.
			"tool_calls before content", `{"model":"m","messages":[{"role":"assistant","tool_calls":[{"id":"1"}]}]}`, 400, "unsupported_parameter", "tool_calls",
		},
		{
			// Message problems win over numeric range problems.
			"message before range", `{"model":"m","messages":[{"role":"robot","content":"x"}],"temperature":9}`, 400, "invalid_request", "role",
		},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			got, derr := decodeChatRequest(strings.NewReader(tc.body))
			if derr == nil {
				t.Fatalf("decodeChatRequest() = %+v, want error", got)
			}
			if derr.Status != tc.wantStatus || derr.Code != tc.wantCode {
				t.Errorf("error = %d %q (%s), want %d %q", derr.Status, derr.Code, derr.Message, tc.wantStatus, tc.wantCode)
			}
			if !strings.Contains(derr.Message, tc.wantMsgContains) {
				t.Errorf("Message = %q, want it to contain %q", derr.Message, tc.wantMsgContains)
			}
		})
	}
}

func TestDecodeChatRequestTooLarge(t *testing.T) {
	big := base + `,"user":"` + strings.Repeat("x", 1024) + `"}`
	body := http.MaxBytesReader(nil, io.NopCloser(strings.NewReader(big)), 64)
	got, derr := decodeChatRequest(body)
	if derr == nil {
		t.Fatalf("decodeChatRequest() = %+v, want error", got)
	}
	if derr.Status != http.StatusRequestEntityTooLarge || derr.Code != CodeRequestTooLarge {
		t.Errorf("error = %d %q, want 413 %q", derr.Status, derr.Code, CodeRequestTooLarge)
	}
}
