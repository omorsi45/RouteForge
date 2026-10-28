package bench

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"strconv"
	"strings"
	"time"

	"github.com/omorsi45/routeforge/internal/config"
)

// maxResponseBytes bounds how much of one response the client reads. A reply
// to a chat request is far smaller; the bound only protects the benchmark from
// a gateway that never stops.
const maxResponseBytes = 16 << 20

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

// ChatOutcome is what came back for one chat request. It holds no prompt or
// completion text.
type ChatOutcome struct {
	Status     int           // HTTP status
	Model      string        // X-RouteForge-Model: the model that served the request
	Reason     string        // X-RouteForge-Reason: why the router chose
	DecisionID string        // X-RouteForge-Decision-Id
	Duration   time.Duration // from sending the request to the end of the response
	CostUSD    *float64      // routeforge.cost_usd; nil when the response has none
	ErrorCode  string        // the error code of a response that is not 200
}

type chatMessage struct {
	Role    string `json:"role"`
	Content string `json:"content"`
}

type chatBody struct {
	Model    string        `json:"model"`
	Messages []chatMessage `json:"messages"`
}

// Chat sends one non-streaming chat completion request to the router
// (model routeforge-auto) with a single user message, plus the given headers.
// A response with any HTTP status is an outcome; only a request that gets no
// readable response is an error.
func (c *Client) Chat(ctx context.Context, prompt string, headers map[string]string) (ChatOutcome, error) {
	body, err := json.Marshal(chatBody{
		Model:    config.AutoModel,
		Messages: []chatMessage{{Role: "user", Content: prompt}},
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
	for k, v := range headers {
		req.Header.Set(k, v)
	}

	start := time.Now()
	resp, err := c.http.Do(req)
	if err != nil {
		return ChatOutcome{}, fmt.Errorf("send chat request: %w", err)
	}
	defer resp.Body.Close()
	data, err := io.ReadAll(io.LimitReader(resp.Body, maxResponseBytes))
	if err != nil {
		return ChatOutcome{}, fmt.Errorf("read chat response: %w", err)
	}
	out := ChatOutcome{
		Status:     resp.StatusCode,
		Duration:   time.Since(start),
		DecisionID: resp.Header.Get("X-RouteForge-Decision-Id"),
	}

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
