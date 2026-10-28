package api

import (
	"cmp"
	"context"
	"errors"
	"net/http"
	"slices"
	"strconv"
	"time"

	"github.com/omorsi45/routeforge/internal/accounting"
	"github.com/omorsi45/routeforge/internal/config"
	"github.com/omorsi45/routeforge/internal/executor"
	"github.com/omorsi45/routeforge/internal/policy"
	"github.com/omorsi45/routeforge/internal/provider"
	"github.com/omorsi45/routeforge/internal/router"
)

// writeGrace is how long after the request deadline the connection stays
// writable, so a 504 or a terminal error event can still reach the client.
const writeGrace = 5 * time.Second

// chatState holds the per-request values shared by the JSON and SSE paths.
type chatState struct {
	reqID, completionID string
	created             int64
	decision            *router.Decision
	includeUsage        bool
	meter               *accounting.Meter // nil without accounting
	deadlineCtx         context.Context   // carries the request deadline
	clientCtx           context.Context   // r.Context(); canceled when the client leaves
}

// execMeter is the meter to give the executor. It is a nil interface, not a
// nil *Meter in one, when there is no accounting.
func (st *chatState) execMeter() executor.Meter {
	if st.meter == nil {
		return nil
	}
	return st.meter
}

// failure returns the error to send for a failed upstream call, or nil when
// the client has already left and nothing should be written. The deadline is
// checked first: it is the only cancellation the gateway itself causes.
func (st *chatState) failure(err error) *Error {
	if errors.Is(st.deadlineCtx.Err(), context.DeadlineExceeded) {
		return &Error{Status: http.StatusGatewayTimeout, Code: CodeDeadlineExceeded, Message: "request deadline exceeded"}
	}
	if st.clientCtx.Err() != nil {
		return nil
	}
	return fromUpstreamError(err)
}

type chatChoice struct {
	Index   int `json:"index"`
	Message struct {
		Role    string `json:"role"`
		Content string `json:"content"`
	} `json:"message"`
	FinishReason string `json:"finish_reason"`
}

type chatUsage struct {
	PromptTokens     int `json:"prompt_tokens"`
	CompletionTokens int `json:"completion_tokens"`
	TotalTokens      int `json:"total_tokens"`
}

type routeforgeExt struct {
	DecisionID string `json:"decision_id"`
	Reason     string `json:"reason"`
	Attempts   int    `json:"attempts"`
	// CostUSD and CostSource are set on non-streaming responses of a gateway
	// with accounting: the cost of every attempt, and whether it came from
	// reported or estimated usage. A stream has no place for them.
	CostUSD    *float64 `json:"cost_usd,omitempty"`
	CostSource string   `json:"cost_source,omitempty"`
}

type chatResponse struct {
	ID         string        `json:"id"`
	Object     string        `json:"object"`
	Created    int64         `json:"created"`
	Model      string        `json:"model"`
	Choices    []chatChoice  `json:"choices"`
	Usage      *chatUsage    `json:"usage,omitempty"`
	RouteForge routeforgeExt `json:"routeforge"`
}

// chatCall is a chat request that passed authentication, header and body
// validation. The caller owns cancel, which ends the request context and also
// frees the request's slot of the in-flight limit, so it must be called once
// the response is done, a stream included.
type chatCall struct {
	req    *ChatRequest
	preq   provider.Request // what the provider is asked
	rreq   *router.Request  // what the router is asked
	ctx    context.Context  // carries the request deadline
	cancel context.CancelFunc
}

// acquireInFlight takes a slot of the global in-flight limit. The returned
// release is never nil when aerr is nil. Without a limit it frees nothing.
func (s *Server) acquireInFlight() (release func(), aerr *Error) {
	if s.inflight == nil {
		return func() {}, nil
	}
	release, ok := s.inflight.TryAcquire()
	if !ok {
		return nil, errInFlightFull()
	}
	return release, nil
}

// readChat authenticates r, validates its headers and body and builds the
// provider and router requests. Chat and explain both start here, so they
// apply the same rules and share the in-flight limit. The feature text is
// computed once and shared by the router (scoring, sizes) and, later,
// accounting and replay keys.
func (s *Server) readChat(w http.ResponseWriter, r *http.Request) (*chatCall, *Error) {
	// The slot comes first, so a gateway at its limit refuses before it spends
	// anything on the request, and it covers reading the body.
	release, aerr := s.acquireInFlight()
	if aerr != nil {
		return nil, aerr
	}
	ok := false
	defer func() {
		if !ok {
			release()
		}
	}()

	tenant, aerr := s.authenticate(r)
	if aerr != nil {
		return nil, aerr
	}
	profile, strategy, aerr := routingHeaders(r, tenant)
	if aerr != nil {
		return nil, aerr
	}
	budget, aerr := s.requestBudget(r, tenant)
	if aerr != nil {
		return nil, aerr
	}

	// The context deadline runs on the real clock: the context package has no
	// other.
	deadline := time.Now().Add(budget)
	ctx, cancel := context.WithDeadline(r.Context(), deadline)
	defer func() {
		if !ok {
			cancel()
		}
	}()
	rc := http.NewResponseController(w)
	s.setConnDeadline(r, rc.SetReadDeadline, "read", deadline)
	s.setConnDeadline(r, rc.SetWriteDeadline, "write", deadline.Add(writeGrace))

	maxBody := s.cfg.Server.MaxBodyBytes
	if r.ContentLength > maxBody {
		return nil, &Error{Status: http.StatusRequestEntityTooLarge, Code: CodeRequestTooLarge, Message: "request body is too large"}
	}
	req, aerr := decodeChatRequest(http.MaxBytesReader(w, r.Body, maxBody))
	// The read deadline only guards the body read. Once the body is in,
	// net/http's background read would hit it at the request deadline and
	// cancel r.Context(), making a gateway deadline look like a client leaving.
	s.setConnDeadline(r, rc.SetReadDeadline, "read", deadline.Add(writeGrace))
	if aerr != nil {
		return nil, aerr
	}

	maxTokens := tenant.MaxOutputTokens
	if req.MaxTokens != nil {
		maxTokens = min(maxTokens, *req.MaxTokens)
	}
	text := policy.FeatureText(req.Messages)
	ok = true
	return &chatCall{
		req: req,
		preq: provider.Request{
			Messages:    req.Messages,
			MaxTokens:   maxTokens,
			Temperature: req.Temperature,
			TopP:        req.TopP,
			Stop:        req.Stop,
			User:        req.User,
		},
		rreq: &router.Request{
			Tenant:      tenant,
			Model:       req.Model,
			Strategy:    strategy,
			Profile:     profile,
			Messages:    req.Messages,
			MaxTokens:   maxTokens,
			Deadline:    deadline,
			PromptBytes: len(text),
			Text:        text,
		},
		ctx: ctx,
		cancel: func() {
			cancel()
			release()
		},
	}, nil
}

func (s *Server) handleChat(w http.ResponseWriter, r *http.Request) {
	info := infoFrom(r.Context())
	call, aerr := s.readChat(w, r)
	if aerr != nil {
		writeError(w, aerr)
		return
	}
	defer call.cancel()

	// Admission follows the body check and precedes routing, and nothing is
	// reserved until a decision has been made: a request refused for its rate
	// or by the router costs no budget.
	if s.acct != nil {
		if err := s.acct.Admit(call.ctx, call.rreq.Tenant, call.rreq.PromptBytes, call.rreq.MaxTokens); err != nil {
			aerr := accountingError(err)
			if aerr == nil {
				aerr = errAccountingUnavailable()
			}
			writeError(w, aerr)
			return
		}
	}

	d, err := s.router.Decide(call.ctx, call.rreq)
	if err != nil {
		writeError(w, fromRouterError(err))
		return
	}
	info.decisionID = d.ID
	info.model = d.Chosen
	h := w.Header()
	h.Set("X-RouteForge-Decision-Id", d.ID)
	h.Set("X-RouteForge-Policy-Version", cmp.Or(d.PolicyVersion, "none"))
	h.Set("X-RouteForge-Reason", d.Reason)

	st := &chatState{
		reqID:        info.requestID,
		completionID: "chatcmpl-" + randHex(12),
		created:      s.now().Unix(),
		decision:     d,
		includeUsage: call.req.IncludeUsage,
		deadlineCtx:  call.ctx,
		clientCtx:    r.Context(),
	}
	if s.acct != nil {
		st.meter = s.acct.NewMeter(call.rreq.Tenant, info.requestID, d.ID, call.rreq.PromptBytes, call.rreq.MaxTokens)
	}
	if call.req.Stream {
		s.serveStream(w, r, st, call.preq)
		return
	}
	s.serveJSON(w, r, st, call.preq)
}

// serveJSON runs the non-streaming call and writes the completion.
func (s *Server) serveJSON(w http.ResponseWriter, r *http.Request, st *chatState, preq provider.Request) {
	res, err := s.exec.Complete(st.deadlineCtx, st.decision, preq, st.execMeter())
	// The executor has recorded the attempt, so the decision is final. Store
	// it before replying so a client that has the response can read it back.
	s.storeDecision(r, st.decision)
	if err != nil {
		if aerr := st.failure(err); aerr != nil {
			writeError(w, aerr)
		}
		return
	}
	resp := chatResponse{
		ID:      st.completionID,
		Object:  "chat.completion",
		Created: st.created,
		Model:   res.Model,
		Choices: []chatChoice{{FinishReason: res.Response.FinishReason}},
		RouteForge: routeforgeExt{
			DecisionID: st.decision.ID,
			Reason:     st.decision.Reason,
			Attempts:   len(st.decision.Attempts),
		},
	}
	resp.Choices[0].Message.Role = provider.RoleAssistant
	resp.Choices[0].Message.Content = res.Response.Content
	if st.meter != nil {
		// Complete has settled every attempt by the time it returns.
		cost := st.meter.CostUSD()
		resp.RouteForge.CostUSD, resp.RouteForge.CostSource = &cost, st.meter.CostSource()
	}
	if u := res.Response.Usage; u != nil {
		resp.Usage = &chatUsage{
			PromptTokens:     u.PromptTokens,
			CompletionTokens: u.CompletionTokens,
			TotalTokens:      u.PromptTokens + u.CompletionTokens,
		}
	}
	w.Header().Set("X-RouteForge-Model", res.Model)
	s.writeJSON(w, r, http.StatusOK, resp)
}

// routingHeaders validates X-RouteForge-Profile and X-RouteForge-Strategy.
func routingHeaders(r *http.Request, tenant *config.Tenant) (profile, strategy string, aerr *Error) {
	profile = cmp.Or(r.Header.Get("X-RouteForge-Profile"), tenant.Profile)
	if !slices.Contains(config.Profiles, profile) {
		return "", "", invalid("X-RouteForge-Profile %q is not a valid profile", profile)
	}
	strategy = r.Header.Get("X-RouteForge-Strategy")
	if strategy == "" {
		return profile, "", nil
	}
	if !router.ValidStrategy(strategy) {
		return "", "", invalid("X-RouteForge-Strategy %q is not a valid strategy", strategy)
	}
	if !tenant.AllowStrategyOverride {
		return "", "", &Error{Status: http.StatusForbidden, Code: CodeStrategyOverrideForbidden, Message: "this API key may not override the routing strategy"}
	}
	return profile, strategy, nil
}

// requestBudget returns the time allowed for the request: the smaller of the
// tenant and server limits, lowered by X-RouteForge-Deadline-Ms. The header is
// compared in milliseconds before converting so a huge value clamps instead of
// overflowing time.Duration.
func (s *Server) requestBudget(r *http.Request, tenant *config.Tenant) (time.Duration, *Error) {
	limit := min(tenant.MaxDeadline, s.cfg.Server.DefaultDeadline)
	v := r.Header.Get("X-RouteForge-Deadline-Ms")
	if v == "" {
		return limit, nil
	}
	ms, err := strconv.ParseInt(v, 10, 64)
	if err != nil || ms <= 0 {
		return 0, invalid("X-RouteForge-Deadline-Ms must be a positive integer number of milliseconds")
	}
	if ms < limit.Milliseconds() {
		return time.Duration(ms) * time.Millisecond, nil
	}
	return limit, nil
}

// setConnDeadline applies a connection deadline. A writer without deadline
// support (a test recorder) is not an error.
func (s *Server) setConnDeadline(r *http.Request, set func(time.Time) error, kind string, t time.Time) {
	if err := set(t); err != nil && !errors.Is(err, http.ErrNotSupported) {
		s.logger.WarnContext(r.Context(), "set connection deadline failed", "request_id", infoFrom(r.Context()).requestID, "kind", kind, "error", err)
	}
}
