package bench

import (
	"bufio"
	"bytes"
	"cmp"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http"
	"net/url"
	"strconv"
	"strings"
	"time"

	"github.com/omorsi45/routeforge/internal/config"
)

// maxResponseBytes bounds how much of one response the client reads. A reply
// to a chat request is far smaller; the bound only protects the benchmark from
// a gateway that never stops.
const maxResponseBytes = 16 << 20

// maxSSELineBytes bounds one line of an event stream.
const maxSSELineBytes = 1 << 20

// ErrDecisionNotFound means the gateway has no stored decision with that id.
var ErrDecisionNotFound = errors.New("decision not found")

// Client sends requests to a running gateway as one tenant. It is safe for
// concurrent use.
type Client struct {
	baseURL string
	key     string
	http    *http.Client
}

// NewClient returns a client for the gateway at baseURL that presents key as
// its API key. It keeps many connections alive, so that a run measures the
// gateway and not the cost of connecting. It sets no overall timeout: callers
// bound a request with its context.
func NewClient(baseURL, key string) *Client {
	transport := http.DefaultTransport.(*http.Transport).Clone()
	transport.MaxIdleConns = 0 // no limit
	transport.MaxIdleConnsPerHost = 4096
	return &Client{
		baseURL: strings.TrimRight(baseURL, "/"),
		key:     key,
		http:    &http.Client{Transport: transport},
	}
}

// Close drops the connections the client holds idle, including any it dialed and
// never used. A gateway that is told to stop waits for those connections to go
// away, so a benchmark closes them before it stops the gateway. The client can
// be used again afterwards.
func (c *Client) Close() { c.http.CloseIdleConnections() }

// ChatOutcome is what came back for one chat request. It holds no prompt or
// completion text.
type ChatOutcome struct {
	Status     int           // HTTP status
	Model      string        // X-RouteForge-Model: the model that served the request
	Reason     string        // X-RouteForge-Reason: why the router chose
	DecisionID string        // X-RouteForge-Decision-Id
	Duration   time.Duration // from sending the request to the end of the response
	CostUSD    *float64      // routeforge.cost_usd; nil when the response has none
	// ErrorCode is the error code of a response that is not 200, or of a stream
	// that did not finish: the code of its terminal error event, or
	// stream_truncated when it just stopped.
	ErrorCode string
	Streamed  bool // the request asked for a stream
	// Partial is set when a stream, after its first byte, ended with an error
	// event instead of [DONE].
	Partial bool
}

// ChatRequest describes one chat completion request.
type ChatRequest struct {
	Prompt    string
	Model     string // "" asks the router (routeforge-auto)
	Stream    bool
	MaxTokens int // 0 sends no cap
	Headers   map[string]string
}

type chatMessage struct {
	Role    string `json:"role"`
	Content string `json:"content"`
}

type chatBody struct {
	Model     string        `json:"model"`
	Messages  []chatMessage `json:"messages"`
	Stream    bool          `json:"stream,omitempty"`
	MaxTokens int           `json:"max_tokens,omitempty"`
}

// Chat sends one non-streaming chat completion request to the router
// (model routeforge-auto) with a single user message, plus the given headers.
// A response with any HTTP status is an outcome; only a request that gets no
// readable response is an error.
func (c *Client) Chat(ctx context.Context, prompt string, headers map[string]string) (ChatOutcome, error) {
	return c.Do(ctx, ChatRequest{Prompt: prompt, Headers: headers})
}

// Do sends r. A stream is read to its end, and its text is dropped. A response
// with any HTTP status is an outcome, and so is a stream that breaks off; only a
// request that gets no response, or a read that fails for another reason than
// the stream ending early, is an error.
func (c *Client) Do(ctx context.Context, r ChatRequest) (ChatOutcome, error) {
	body, err := json.Marshal(chatBody{
		Model:     cmp.Or(r.Model, config.AutoModel),
		Messages:  []chatMessage{{Role: "user", Content: r.Prompt}},
		Stream:    r.Stream,
		MaxTokens: r.MaxTokens,
	})
	if err != nil {
		return ChatOutcome{}, fmt.Errorf("encode chat request: %w", err)
	}
	req, err := http.NewRequestWithContext(ctx, http.MethodPost, c.baseURL+"/v1/chat/completions", bytes.NewReader(body))
	if err != nil {
		return ChatOutcome{}, fmt.Errorf("build chat request: %w", err)
	}
	req.Header.Set("Authorization", "Bearer "+c.key)
	req.Header.Set("Content-Type", "application/json")
	for k, v := range r.Headers {
		req.Header.Set(k, v)
	}

	start := time.Now()
	resp, err := c.http.Do(req)
	if err != nil {
		return ChatOutcome{}, fmt.Errorf("send chat request: %w", err)
	}
	defer resp.Body.Close()
	out := ChatOutcome{
		Status:     resp.StatusCode,
		DecisionID: resp.Header.Get("X-RouteForge-Decision-Id"),
		Streamed:   r.Stream,
	}

	if resp.StatusCode == http.StatusOK && r.Stream {
		out.Model = resp.Header.Get("X-RouteForge-Model")
		out.Reason = resp.Header.Get("X-RouteForge-Reason")
		if err := readStream(ctx, resp.Body, &out); err != nil {
			return ChatOutcome{}, err
		}
		out.Duration = time.Since(start)
		return out, nil
	}

	data, err := io.ReadAll(io.LimitReader(resp.Body, maxResponseBytes))
	if err != nil {
		return ChatOutcome{}, fmt.Errorf("read chat response: %w", err)
	}
	out.Duration = time.Since(start)
	if resp.StatusCode != http.StatusOK {
		out.ErrorCode = errorCode(resp.StatusCode, data)
		return out, nil
	}
	// Only the routeforge extension is kept: the completion text is read past
	// and dropped.
	var ok struct {
		RouteForge struct {
			DecisionID string   `json:"decision_id"`
			CostUSD    *float64 `json:"cost_usd"`
		} `json:"routeforge"`
	}
	if err := json.Unmarshal(data, &ok); err != nil {
		return ChatOutcome{}, fmt.Errorf("decode chat response (status 200): %w", err)
	}
	out.Model = resp.Header.Get("X-RouteForge-Model")
	out.Reason = resp.Header.Get("X-RouteForge-Reason")
	if out.DecisionID == "" {
		out.DecisionID = ok.RouteForge.DecisionID
	}
	out.CostUSD = ok.RouteForge.CostUSD
	return out, nil
}

// readStream reads the event stream of a 200 response to its end and records in
// out how it ended: with [DONE], with an error event (Partial) or with neither
// (stream_truncated). The content of the chunks is read past and dropped. A
// stream cut off by the end of ctx is an error, not an outcome.
func readStream(ctx context.Context, body io.Reader, out *ChatOutcome) error {
	sc := bufio.NewScanner(body)
	sc.Buffer(make([]byte, 0, 64<<10), maxSSELineBytes)
	ended := false // [DONE] or an error event was read
	for !ended && sc.Scan() {
		payload, ok := bytes.CutPrefix(sc.Bytes(), []byte("data: "))
		if !ok {
			continue
		}
		switch {
		case string(payload) == "[DONE]":
			ended = true
		case bytes.HasPrefix(payload, []byte(`{"error"`)):
			// The gateway's terminal error event is an error envelope. A content
			// chunk starts with its id instead, so only these are decoded.
			out.Partial, out.ErrorCode = true, errorEventCode(payload)
			ended = true
		}
	}
	if !ended {
		if err := ctx.Err(); err != nil {
			return fmt.Errorf("read stream: %w", err)
		}
		if err := sc.Err(); errors.Is(err, bufio.ErrTooLong) {
			return fmt.Errorf("read stream: %w", err)
		}
		out.ErrorCode = "stream_truncated"
		return nil
	}
	// Let the connection be reused: whatever follows the end is dropped.
	_, _ = io.Copy(io.Discard, io.LimitReader(body, maxResponseBytes))
	return nil
}

// errorEventCode is the code of a stream's terminal error event.
func errorEventCode(payload []byte) string {
	var ev struct {
		Error struct {
			Code string `json:"code"`
		} `json:"error"`
	}
	if err := json.Unmarshal(payload, &ev); err != nil || ev.Error.Code == "" {
		return "stream_error"
	}
	return ev.Error.Code
}

// errorCode is the code of the gateway's error envelope, or http_<status> for
// a body that is not one (a proxy's page, an empty reply).
func errorCode(status int, body []byte) string {
	var env struct {
		Error struct {
			Code string `json:"code"`
		} `json:"error"`
	}
	if err := json.Unmarshal(body, &env); err == nil && env.Error.Code != "" {
		return env.Error.Code
	}
	return "http_" + strconv.Itoa(status)
}

// AttemptRecord is one upstream call of a stored decision.
type AttemptRecord struct {
	Model      string  `json:"model"`
	Outcome    string  `json:"outcome"`
	ErrorKind  string  `json:"error_kind"`
	DurationMS float64 `json:"duration_ms"`
}

// DecisionRecord is the part of a stored routing decision that the benchmarks
// read.
type DecisionRecord struct {
	ID          string          `json:"id"`
	Chosen      string          `json:"chosen"`
	DecidedInUS int64           `json:"decided_in_us"`
	Attempts    []AttemptRecord `json:"attempts"`
}

// Decision reads the stored decision id of the client's tenant. It returns
// ErrDecisionNotFound when the gateway has none, which is also what it says
// about another tenant's decision.
func (c *Client) Decision(ctx context.Context, id string) (DecisionRecord, error) {
	var d DecisionRecord
	status, code, err := c.getJSON(ctx, "/v1/routing/decisions/"+url.PathEscape(id), &d)
	switch {
	case err != nil:
		return DecisionRecord{}, fmt.Errorf("read decision %s: %w", id, err)
	case status == http.StatusNotFound:
		return DecisionRecord{}, fmt.Errorf("read decision %s: %w", id, ErrDecisionNotFound)
	case status != http.StatusOK:
		return DecisionRecord{}, fmt.Errorf("read decision %s: status %d (%s)", id, status, code)
	}
	return d, nil
}

// AccountingState is the answer of GET /admin/accounting: what a tenant spent,
// holds in reservations and overshot its budget by on a UTC day, and what its
// usage records add up to. All amounts are nano-USD.
type AccountingState struct {
	Tenant         string `json:"tenant"`
	Day            string `json:"day"`
	SpentNanos     int64  `json:"spent_nanos"`
	ReservedNanos  int64  `json:"reserved_nanos"`
	OvershootNanos int64  `json:"overshoot_nanos"`
	UsageRecords   int    `json:"usage_records"`
	UsageCostNanos int64  `json:"usage_cost_nanos"`
}

// Accounting reads the books of tenant on day (yyyymmdd, UTC). The client's key
// must be an admin key.
func (c *Client) Accounting(ctx context.Context, tenant, day string) (AccountingState, error) {
	query := url.Values{"tenant": {tenant}, "day": {day}}.Encode()
	var st AccountingState
	status, code, err := c.getJSON(ctx, "/admin/accounting?"+query, &st)
	switch {
	case err != nil:
		return AccountingState{}, fmt.Errorf("read accounting of %s: %w", tenant, err)
	case status != http.StatusOK:
		return AccountingState{}, fmt.Errorf("read accounting of %s: status %d (%s)", tenant, status, code)
	}
	return st, nil
}

// getJSON GETs path with the client's key and decodes a 200 response into v. For
// another status it returns the status and the error code of the body, and
// leaves v alone.
func (c *Client) getJSON(ctx context.Context, path string, v any) (status int, code string, err error) {
	req, err := http.NewRequestWithContext(ctx, http.MethodGet, c.baseURL+path, nil)
	if err != nil {
		return 0, "", fmt.Errorf("build request: %w", err)
	}
	req.Header.Set("Authorization", "Bearer "+c.key)
	resp, err := c.http.Do(req)
	if err != nil {
		return 0, "", fmt.Errorf("send request: %w", err)
	}
	defer resp.Body.Close()
	data, err := io.ReadAll(io.LimitReader(resp.Body, maxResponseBytes))
	if err != nil {
		return 0, "", fmt.Errorf("read response: %w", err)
	}
	if resp.StatusCode != http.StatusOK {
		return resp.StatusCode, errorCode(resp.StatusCode, data), nil
	}
	if err := json.Unmarshal(data, v); err != nil {
		return 0, "", fmt.Errorf("decode response: %w", err)
	}
	return resp.StatusCode, "", nil
}
