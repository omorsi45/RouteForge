package bench

import (
	"context"
	"encoding/json"
	"fmt"
	"io"
	"math"
	"net/http"
	"net/http/httptest"
	"strconv"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/omorsi45/routeforge/internal/auth"
	"github.com/omorsi45/routeforge/internal/config"
)

const (
	faultsBenchKey = "test-key-bench"
	faultsOverKey  = "test-key-bench-over"
	faultsAdminKey = "test-admin-key"
)

func TestFaultInvariantChecks(t *testing.T) {
	// Good books: every invariant holds.
	bench := AccountingState{Tenant: "bench", SpentNanos: 7000, UsageCostNanos: 7000}
	over := AccountingState{Tenant: "bench-over", SpentNanos: 5000, UsageCostNanos: 5000, OvershootNanos: 2000}
	const benchBudget, overBudget = 10000, 3000

	good := checkInvariants(bench, over, benchBudget, overBudget)
	want := Invariants{
		SpentEqualsUsage: true, ReservedReturnsToZero: true, SpendWithinBudget: true,
		OvershootEqualsExcess: true, OvershootPositive: true, InvariantsHold: true,
	}
	if good != want {
		t.Fatalf("good books: %+v, want %+v", good, want)
	}

	for _, c := range []struct {
		name   string
		mutate func(bench, over *AccountingState, benchBudget, overBudget *int64)
		broken func(i Invariants) bool // true when the check that must fail failed
	}{
		{"(a) bench spend differs from its usage", func(b, _ *AccountingState, _, _ *int64) { b.UsageCostNanos = 6999 },
			func(i Invariants) bool {
				return !i.SpentEqualsUsage && i.ReservedReturnsToZero && i.SpendWithinBudget && i.OvershootEqualsExcess && i.OvershootPositive
			}},
		{"(a) over spend differs from its usage", func(_, o *AccountingState, _, _ *int64) { o.UsageCostNanos = 5001 },
			func(i Invariants) bool { return !i.SpentEqualsUsage }},
		{"(b) bench holds a reservation", func(b, _ *AccountingState, _, _ *int64) { b.ReservedNanos = 1 },
			func(i Invariants) bool { return !i.ReservedReturnsToZero && i.SpentEqualsUsage && i.SpendWithinBudget }},
		{"(b) over holds a reservation", func(_, o *AccountingState, _, _ *int64) { o.ReservedNanos = 40 },
			func(i Invariants) bool { return !i.ReservedReturnsToZero }},
		{"(c) bench spend is over its budget", func(_, _ *AccountingState, bb, _ *int64) { *bb = 6999 },
			func(i Invariants) bool { return !i.SpendWithinBudget && i.SpentEqualsUsage && i.ReservedReturnsToZero }},
		{"(d) overshoot is not the excess", func(_, o *AccountingState, _, _ *int64) { o.OvershootNanos = 1999 },
			func(i Invariants) bool { return !i.OvershootEqualsExcess && i.OvershootPositive }},
		{"(d) the over tenant never went over", func(_, o *AccountingState, _, ob *int64) {
			*ob = 5000 // spent exactly its budget: the books agree (excess 0) but the fault never fired
			o.OvershootNanos = 0
		}, func(i Invariants) bool { return i.OvershootEqualsExcess && !i.OvershootPositive }},
	} {
		t.Run(c.name, func(t *testing.T) {
			b, o, bb, ob := bench, over, int64(benchBudget), int64(overBudget)
			c.mutate(&b, &o, &bb, &ob)
			got := checkInvariants(b, o, bb, ob)
			if !c.broken(got) {
				t.Errorf("invariants = %+v, want only the named check to fail", got)
			}
			if got.InvariantsHold {
				t.Errorf("invariants_hold is true with a failed check: %+v", got)
			}
		})
	}
}

func TestFaultInvariantSpendEqualToBudgetIsWithin(t *testing.T) {
	bench := AccountingState{SpentNanos: 100, UsageCostNanos: 100}
	over := AccountingState{SpentNanos: 20, UsageCostNanos: 20, OvershootNanos: 5}
	if got := checkInvariants(bench, over, 100, 15); !got.SpendWithinBudget || !got.InvariantsHold {
		t.Errorf("spending exactly the budget is within it: %+v", got)
	}
}

// faultsStub is a gateway for the fault benchmark. Requests are answered by
// arrival number n (1, 2, ...), so a run with one request in flight is
// deterministic. See benchBehavior and overBehavior.
type faultsStub struct {
	*httptest.Server
	mu          sync.Mutex
	n           map[string]int // chat requests seen per key
	headers     map[string][]http.Header
	bodies      map[string][]stubChatBody
	accounting  map[string]AccountingState
	accountingQ []string // the query of every accounting request
	adminStatus int
	delay       time.Duration
	inflight    int
	maxInflight int
}

type stubChatBody struct {
	Model     string `json:"model"`
	Stream    bool   `json:"stream"`
	MaxTokens *int   `json:"max_tokens"`
}

// behavior is how the stub answers one request.
type behavior struct {
	status  int    // 200 unless set
	code    string // error code of a non-200
	partial bool   // a stream that ends with an error event
	noID    bool   // no decision id
}

// benchBehavior answers the bench tenant's request n.
func benchBehavior(n int, stream bool) behavior {
	switch {
	case n%10 == 0:
		return behavior{status: 503, code: "no_eligible_model", noID: true}
	case n%5 == 0:
		return behavior{status: 504, code: "deadline_exceeded"}
	case stream && n%6 == 0:
		return behavior{partial: true}
	}
	return behavior{status: 200}
}

// overBehavior answers the over tenant's request n: the first 6 are served and
// the budget refuses the rest.
func overBehavior(n int) behavior {
	if n > 6 {
		return behavior{status: 503, code: "no_eligible_model", noID: true}
	}
	return behavior{status: 200}
}

func newFaultsStub(t *testing.T) *faultsStub {
	t.Helper()
	s := &faultsStub{
		n:       map[string]int{},
		headers: map[string][]http.Header{},
		bodies:  map[string][]stubChatBody{},
		accounting: map[string]AccountingState{
			"bench":      {Tenant: "bench", SpentNanos: 7000, UsageCostNanos: 7000, UsageRecords: 30},
			"bench-over": {Tenant: "bench-over", SpentNanos: 5000, UsageCostNanos: 5000, OvershootNanos: 2000, UsageRecords: 6},
		},
		adminStatus: http.StatusOK,
	}
	mux := http.NewServeMux()
	mux.HandleFunc("POST /v1/chat/completions", s.chat)
	mux.HandleFunc("GET /v1/routing/decisions/{id}", s.decision)
	mux.HandleFunc("GET /admin/accounting", s.admin)
	s.Server = httptest.NewServer(mux)
	t.Cleanup(s.Close)
	return s
}

func (s *faultsStub) chat(w http.ResponseWriter, r *http.Request) {
	key := strings.TrimPrefix(r.Header.Get("Authorization"), "Bearer ")
	var body stubChatBody
	raw, _ := io.ReadAll(r.Body)
	_ = json.Unmarshal(raw, &body)

	s.mu.Lock()
	s.n[key]++
	n := s.n[key]
	s.headers[key] = append(s.headers[key], r.Header.Clone())
	s.bodies[key] = append(s.bodies[key], body)
	s.inflight++
	s.maxInflight = max(s.maxInflight, s.inflight)
	s.mu.Unlock()
	defer func() {
		s.mu.Lock()
		s.inflight--
		s.mu.Unlock()
	}()
	time.Sleep(s.delay)

	b := behavior{status: 200}
	switch key {
	case faultsBenchKey:
		b = benchBehavior(n, body.Stream)
	case faultsOverKey:
		b = overBehavior(n)
	}
	if !b.noID {
		w.Header().Set("X-RouteForge-Decision-Id", fmt.Sprintf("dec_%s_%d", key, n))
	}
	if b.status != 0 && b.status != 200 {
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(b.status)
		_, _ = fmt.Fprintf(w, `{"error":{"message":"m","type":"t","code":%q}}`, b.code)
		return
	}
	if body.Stream {
		w.Header().Set("Content-Type", "text/event-stream")
		_, _ = io.WriteString(w, "data: "+contentFrame+"\n\n")
		if b.partial {
			_, _ = io.WriteString(w, `data: {"error":{"message":"upstream stream interrupted","type":"server_error","code":"upstream_stream_interrupted"}}`+"\n\n")
			return
		}
		_, _ = io.WriteString(w, "data: [DONE]\n\n")
		return
	}
	_, _ = io.WriteString(w, `{"routeforge":{"decision_id":"x","cost_usd":0.001}}`)
}

// attemptShape says how the attempts of request n went: "failover" (an
// unavailable first attempt, then another model), "retry" (a rate limit, then the
// same model again) or "single". The two multi-attempt shapes are not equally
// frequent, so a count that mixes them up is noticed.
func attemptShape(n int) string {
	switch {
	case n%3 == 0:
		return "failover"
	case n%7 == 1:
		return "retry"
	}
	return "single"
}

// decision answers with the attempts of request n of any key (see attemptShape).
func (s *faultsStub) decision(w http.ResponseWriter, r *http.Request) {
	id := r.PathValue("id")
	n, err := strconv.Atoi(id[strings.LastIndex(id, "_")+1:])
	if err != nil {
		http.Error(w, "bad id", http.StatusBadRequest)
		return
	}
	attempts := `[{"model":"m1","outcome":"ok","duration_ms":4}]`
	switch attemptShape(n) {
	case "failover":
		attempts = `[{"model":"m1","outcome":"error","error_kind":"unavailable","duration_ms":1},{"model":"m2","outcome":"ok","duration_ms":4}]`
	case "retry":
		attempts = `[{"model":"m1","outcome":"error","error_kind":"rate_limited","duration_ms":1},{"model":"m1","outcome":"ok","duration_ms":4}]`
	}
	_, _ = fmt.Fprintf(w, `{"id":%q,"chosen":"m1","decided_in_us":100,"attempts":%s}`, id, attempts)
}

func (s *faultsStub) admin(w http.ResponseWriter, r *http.Request) {
	s.mu.Lock()
	s.accountingQ = append(s.accountingQ, r.URL.RawQuery+"|"+r.Header.Get("Authorization"))
	state, ok := s.accounting[r.URL.Query().Get("tenant")]
	status := s.adminStatus
	s.mu.Unlock()
	if status != http.StatusOK {
		w.WriteHeader(status)
		_, _ = io.WriteString(w, `{"error":{"message":"m","type":"t","code":"invalid_api_key"}}`)
		return
	}
	if !ok {
		state = AccountingState{Tenant: r.URL.Query().Get("tenant")}
	}
	state.Day = r.URL.Query().Get("day")
	_ = json.NewEncoder(w).Encode(map[string]any{
		"tenant": state.Tenant, "day": state.Day, "spent_nanos": state.SpentNanos, "reserved_nanos": state.ReservedNanos,
		"overshoot_nanos": state.OvershootNanos, "usage_records": state.UsageRecords, "usage_cost_nanos": state.UsageCostNanos,
	})
}

func faultsOptions(s *faultsStub, requests, overRequests, concurrency int) FaultsOptions {
	return FaultsOptions{
		Bench:            NewClient(s.URL, faultsBenchKey),
		Over:             NewClient(s.URL, faultsOverKey),
		Admin:            NewClient(s.URL, faultsAdminKey),
		BenchTenant:      "bench",
		OverTenant:       "bench-over",
		OverModel:        "m-over",
		BenchBudgetNanos: 10000,
		OverBudgetNanos:  3000,
		Requests:         requests,
		OverRequests:     overRequests,
		Concurrency:      concurrency,
		DeadlineMS:       2000,
		Seed:             1,
		Now:              func() time.Time { return time.Date(2026, 10, 28, 12, 0, 0, 0, time.UTC) },
	}
}

func TestRunFaultsCountsWhatHappened(t *testing.T) {
	s := newFaultsStub(t)
	const total = 40
	res, err := RunFaults(context.Background(), faultsOptions(s, total, 10, 1))
	if err != nil {
		t.Fatal(err)
	}

	// Expected counts, from the stub's rules applied to each request in turn.
	var ok, refused, timedOut, partial, streams, retries, failovers, withRetries int
	kinds := map[string]int{}
	fetched := 0
	for i := 0; i < total; i++ {
		n, stream := i+1, i%2 == 1
		if stream {
			streams++
		}
		b := benchBehavior(n, stream)
		switch {
		case b.partial:
			partial++
		case b.status == 503:
			refused++
		case b.status == 504:
			timedOut++
		default:
			ok++
		}
		if b.noID {
			continue
		}
		fetched++
		switch attemptShape(n) {
		case "failover":
			retries, failovers, withRetries = retries+1, failovers+1, withRetries+1
			kinds["unavailable"]++
		case "retry":
			retries, withRetries = retries+1, withRetries+1
			kinds["rate_limited"]++
		}
	}

	b := res.Bench
	if b.Tenant != "bench" || b.Requests != total || b.StreamRequests != streams || streams != total/2 {
		t.Errorf("requests = %d (%d streaming), want %d with half streaming; tenant %q", b.Requests, b.StreamRequests, total, b.Tenant)
	}
	if b.Succeeded != ok || math.Abs(b.SuccessRate-float64(ok)/total) > 1e-12 {
		t.Errorf("succeeded = %d, rate = %v, want %d of %d", b.Succeeded, b.SuccessRate, ok, total)
	}
	wantStatuses := map[string]int{"200": ok + partial, "503": refused, "504": timedOut}
	if fmt.Sprint(b.Statuses) != fmt.Sprint(wantStatuses) {
		t.Errorf("statuses = %v, want %v (a partial stream is a 200)", b.Statuses, wantStatuses)
	}
	wantCodes := map[string]int{"no_eligible_model": refused, "deadline_exceeded": timedOut, "upstream_stream_interrupted": partial}
	if fmt.Sprint(b.ErrorCodes) != fmt.Sprint(wantCodes) {
		t.Errorf("error codes = %v, want %v", b.ErrorCodes, wantCodes)
	}
	if b.PartialStreams != partial || partial == 0 {
		t.Errorf("partial streams = %d, want %d (and the stub makes some)", b.PartialStreams, partial)
	}
	if b.DecisionsFetched != fetched || b.DecisionsMissing != 0 {
		t.Errorf("decisions fetched = %d, missing = %d, want %d, 0", b.DecisionsFetched, b.DecisionsMissing, fetched)
	}
	if b.Retries != retries || b.Failovers != failovers || b.RequestsWithRetries != withRetries {
		t.Errorf("retries = %d, failovers = %d, requests with retries = %d, want %d, %d, %d",
			b.Retries, b.Failovers, b.RequestsWithRetries, retries, failovers, withRetries)
	}
	if fmt.Sprint(b.AttemptErrorKinds) != fmt.Sprint(kinds) {
		t.Errorf("attempt error kinds = %v, want %v", b.AttemptErrorKinds, kinds)
	}
	if failovers == 0 || retries <= failovers {
		t.Fatalf("the stub should produce failovers and plain retries (retries %d, failovers %d)", retries, failovers)
	}

	o := res.BenchOver
	if o.Tenant != "bench-over" || o.Requests != 10 || o.Succeeded != 6 || o.Statuses["503"] != 4 || o.ErrorCodes["no_eligible_model"] != 4 {
		t.Errorf("over phase = %+v, want 10 requests, 6 served and 4 refused", o)
	}
	if res.SuccessRate != b.SuccessRate {
		t.Errorf("top-level success rate = %v, want the bench tenant's %v", res.SuccessRate, b.SuccessRate)
	}
}

func TestRunFaultsSendsTheRequestsTheBriefAsks(t *testing.T) {
	s := newFaultsStub(t)
	if _, err := RunFaults(context.Background(), faultsOptions(s, 40, 10, 1)); err != nil {
		t.Fatal(err)
	}

	// Every request carries the 2000 ms deadline.
	for key, list := range s.headers {
		for i, h := range list {
			if h.Get("X-RouteForge-Deadline-Ms") != "2000" {
				t.Errorf("%s request %d: deadline header = %q, want 2000", key, i, h.Get("X-RouteForge-Deadline-Ms"))
			}
			if h.Get("Authorization") != "Bearer "+key {
				t.Errorf("%s request %d: Authorization = %q", key, i, h.Get("Authorization"))
			}
		}
	}
	if len(s.headers) != 2 {
		t.Errorf("chat requests came with %d keys, want the bench key and the over key", len(s.headers))
	}

	// The bench tenant asks the router, half of its requests stream, and the
	// strategies rotate so that every model sees traffic.
	strategies := map[string]int{}
	var streams int
	for i, h := range s.headers[faultsBenchKey] {
		strategies[h.Get("X-RouteForge-Strategy")]++
		if h.Get("X-RouteForge-Strategy") == "quality" && h.Get("X-RouteForge-Profile") != "balanced" {
			t.Errorf("request %d asks for quality without the balanced profile", i)
		}
		if body := s.bodies[faultsBenchKey][i]; body.Model != "routeforge-auto" || body.MaxTokens != nil {
			t.Errorf("bench request %d body = %+v, want the router and no token cap", i, body)
		} else if body.Stream {
			streams++
		}
	}
	for _, name := range []string{"quality", "cheapest", "strongest", "static", "latency"} {
		if strategies[name] != 8 {
			t.Errorf("strategy %s sent %d times, want 8 of 40 (they rotate)", name, strategies[name])
		}
	}
	if streams != 20 {
		t.Errorf("%d bench requests stream, want 20 of 40", streams)
	}

	// The over tenant is pinned to its one model with a small token cap.
	overStreams := 0
	for i, body := range s.bodies[faultsOverKey] {
		if body.Model != "m-over" || body.MaxTokens == nil || *body.MaxTokens != 16 {
			t.Errorf("over request %d body = %+v, want model m-over and max_tokens 16", i, body)
		}
		if body.Stream {
			overStreams++
		}
	}
	if overStreams != 5 {
		t.Errorf("%d over requests stream, want 5 of 10", overStreams)
	}
}

func TestRunFaultsReadsTheBooksAfterTheRun(t *testing.T) {
	s := newFaultsStub(t)
	res, err := RunFaults(context.Background(), faultsOptions(s, 10, 10, 2))
	if err != nil {
		t.Fatal(err)
	}

	// One read per tenant, with the admin key, for the day the run began on.
	want := []string{
		"day=20261028&tenant=bench|Bearer " + faultsAdminKey,
		"day=20261028&tenant=bench-over|Bearer " + faultsAdminKey,
	}
	if fmt.Sprint(s.accountingQ) != fmt.Sprint(want) {
		t.Fatalf("accounting reads = %q, want %q", s.accountingQ, want)
	}

	acc := res.Accounting
	if acc["bench"].SpentNanos != 7000 || acc["bench"].BudgetNanos != 10000 || acc["bench-over"].OvershootNanos != 2000 || acc["bench-over"].BudgetNanos != 3000 {
		t.Errorf("accounting = %+v", acc)
	}
	if !res.Invariants.InvariantsHold {
		t.Errorf("invariants = %+v, want them to hold on good books", res.Invariants)
	}
}

func TestRunFaultsReportsViolatedInvariants(t *testing.T) {
	s := newFaultsStub(t)
	s.accounting["bench"] = AccountingState{Tenant: "bench", SpentNanos: 7000, UsageCostNanos: 7000, ReservedNanos: 12}

	res, err := RunFaults(context.Background(), faultsOptions(s, 10, 10, 2))
	if err != nil {
		t.Fatalf("violated invariants are a result, not an error: %v", err)
	}
	if res.Invariants.ReservedReturnsToZero || res.Invariants.InvariantsHold {
		t.Errorf("invariants = %+v, want (b) and the verdict false", res.Invariants)
	}
	if !res.Invariants.SpentEqualsUsage || !res.Invariants.OvershootPositive {
		t.Errorf("invariants = %+v, the others still hold", res.Invariants)
	}
}

func TestRunFaultsRefusesARunAcrossUTCMidnight(t *testing.T) {
	s := newFaultsStub(t)
	o := faultsOptions(s, 5, 5, 1)
	calls := 0
	o.Now = func() time.Time {
		calls++
		if calls == 1 {
			return time.Date(2026, 10, 28, 23, 59, 59, 0, time.UTC)
		}
		return time.Date(2026, 10, 29, 0, 0, 1, 0, time.UTC)
	}
	_, err := RunFaults(context.Background(), o)
	if err == nil || !strings.Contains(err.Error(), "midnight") {
		t.Fatalf("err = %v, want a refusal that names midnight", err)
	}
}

func TestRunFaultsAccountingFailureIsAnError(t *testing.T) {
	s := newFaultsStub(t)
	s.adminStatus = http.StatusUnauthorized
	_, err := RunFaults(context.Background(), faultsOptions(s, 5, 5, 1))
	if err == nil || !strings.Contains(err.Error(), "401") {
		t.Fatalf("err = %v, want the accounting failure", err)
	}
}

func TestRunFaultsHonorsTheConcurrencyLimit(t *testing.T) {
	s := newFaultsStub(t)
	s.delay = 10 * time.Millisecond
	if _, err := RunFaults(context.Background(), faultsOptions(s, 60, 20, 4)); err != nil {
		t.Fatal(err)
	}
	if s.maxInflight > 4 || s.maxInflight < 2 {
		t.Errorf("up to %d requests in flight with 4 workers, want between 2 and 4", s.maxInflight)
	}
}

func TestRunFaultsCountsTransportFailures(t *testing.T) {
	s := newFaultsStub(t)
	o := faultsOptions(s, 6, 3, 1)
	// The chat endpoint is unreachable but the books can be read: send the
	// requests to a closed port.
	dead := httptest.NewServer(http.NotFoundHandler())
	deadURL := dead.URL
	dead.Close()
	o.Bench, o.Over = NewClient(deadURL, faultsBenchKey), NewClient(deadURL, faultsOverKey)

	res, err := RunFaults(context.Background(), o)
	if err != nil {
		t.Fatal(err)
	}
	if res.Bench.Succeeded != 0 || res.Bench.ErrorCodes["transport_error"] != 6 || res.SuccessRate != 0 {
		t.Errorf("bench phase = %+v, want every request counted as a transport error", res.Bench)
	}
	if len(res.Bench.Statuses) != 0 {
		t.Errorf("statuses = %v, want none without a response", res.Bench.Statuses)
	}
}

func TestRunFaultsStopsWhenTheContextEnds(t *testing.T) {
	s := newFaultsStub(t)
	s.delay = 100 * time.Millisecond
	ctx, cancel := context.WithTimeout(context.Background(), 150*time.Millisecond)
	defer cancel()
	if _, err := RunFaults(ctx, faultsOptions(s, 100, 10, 1)); err == nil {
		t.Fatal("want an error when the context ends mid run")
	}
}

func TestRunFaultsRejectsBadOptions(t *testing.T) {
	s := newFaultsStub(t)
	for _, c := range []struct {
		name   string
		mutate func(o *FaultsOptions)
	}{
		{"no bench client", func(o *FaultsOptions) { o.Bench = nil }},
		{"no over client", func(o *FaultsOptions) { o.Over = nil }},
		{"no admin client", func(o *FaultsOptions) { o.Admin = nil }},
		{"no tenant name", func(o *FaultsOptions) { o.BenchTenant = "" }},
		{"no over tenant name", func(o *FaultsOptions) { o.OverTenant = "" }},
		{"no over model", func(o *FaultsOptions) { o.OverModel = "" }},
		{"negative budget", func(o *FaultsOptions) { o.OverBudgetNanos = -1 }},
		{"negative requests", func(o *FaultsOptions) { o.Requests = -1 }},
	} {
		t.Run(c.name, func(t *testing.T) {
			o := faultsOptions(s, 4, 4, 1)
			c.mutate(&o)
			if _, err := RunFaults(context.Background(), o); err == nil {
				t.Fatal("want an error")
			}
		})
	}
}

func TestFaultsOptionDefaults(t *testing.T) {
	o := FaultsOptions{}.withDefaults()
	if o.Requests != 2000 || o.OverRequests != 200 || o.Concurrency != 32 || o.DeadlineMS != 2000 {
		t.Errorf("defaults = %d requests, %d over requests, %d concurrency, %d ms deadline; want 2000, 200, 32, 2000",
			o.Requests, o.OverRequests, o.Concurrency, o.DeadlineMS)
	}
	if o.Now == nil || o.RequestTimeout <= 0 {
		t.Errorf("defaults leave the clock or the request timeout unset: %+v", o)
	}
}

func TestFaultsResultsAreJSONSafe(t *testing.T) {
	s := newFaultsStub(t)
	res, err := RunFaults(context.Background(), faultsOptions(s, 10, 10, 2))
	if err != nil {
		t.Fatal(err)
	}
	data, err := json.Marshal(res)
	if err != nil {
		t.Fatal(err)
	}
	var generic map[string]any
	if err := json.Unmarshal(data, &generic); err != nil {
		t.Fatal(err)
	}
	for _, key := range []string{"bench", "bench_over", "accounting", "invariants", "success_rate"} {
		if _, ok := generic[key]; !ok {
			t.Errorf("results lack %q: %s", key, data)
		}
	}
	inv, _ := generic["invariants"].(map[string]any)
	for _, key := range []string{
		"a_spent_equals_usage", "b_reserved_returns_to_zero", "c_spend_within_budget",
		"d_overshoot_equals_excess", "d_overshoot_positive", "invariants_hold",
	} {
		if _, ok := inv[key].(bool); !ok {
			t.Errorf("invariants lack the boolean %q: %v", key, inv)
		}
	}
	for _, leaked := range []string{"COMPLETION", "CHUNK-TEXT"} {
		if strings.Contains(string(data), leaked) {
			t.Errorf("results contain %q", leaked)
		}
	}
}

// faultsConfigYAML is a config with the two fault tenants. %s is the tenants'
// extra settings.
func faultsConfigYAML(benchExtra, overExtra string) string {
	return fmt.Sprintf(`tenants:
  - name: bench
    api_key_sha256:
      - %s
%s
  - name: bench-over
    api_key_sha256:
      - %s
%s
providers:
  - name: fake-1
    type: fake
models:
  - name: m1
    provider: fake-1
    input_per_mtok: 1
    output_per_mtok: 2
  - name: m-over
    provider: fake-1
    input_per_mtok: 1
    output_per_mtok: 2
`, auth.HashKey(faultsBenchKey), benchExtra, auth.HashKey(faultsOverKey), overExtra)
}

func TestFaultsSetupFromConfig(t *testing.T) {
	cfg, err := config.Parse([]byte(faultsConfigYAML(
		"    daily_budget_usd: 100",
		"    daily_budget_usd: 0.003\n    allowed_models: [m-over]")))
	if err != nil {
		t.Fatal(err)
	}
	got, err := FaultsSetup(cfg)
	if err != nil {
		t.Fatal(err)
	}
	want := FaultsSetupInfo{BenchBudgetNanos: 100_000_000_000, OverBudgetNanos: 3_000_000, OverModel: "m-over"}
	if got != want {
		t.Errorf("setup = %+v, want %+v", got, want)
	}
}

func TestFaultsSetupRejectsConfigsItCannotCheck(t *testing.T) {
	for _, c := range []struct {
		name, bench, over, want string
	}{
		{"bench has no budget", "    rpm: 5", "    daily_budget_usd: 0.003\n    allowed_models: [m-over]", "bench"},
		{"over has no budget", "    daily_budget_usd: 100", "    allowed_models: [m-over]", "bench-over"},
		{"over has no allowed model", "    daily_budget_usd: 100", "    daily_budget_usd: 0.003", "allowed_models"},
		{"over has two allowed models", "    daily_budget_usd: 100", "    daily_budget_usd: 0.003\n    allowed_models: [m1, m-over]", "allowed_models"},
	} {
		t.Run(c.name, func(t *testing.T) {
			cfg, err := config.Parse([]byte(faultsConfigYAML(c.bench, c.over)))
			if err != nil {
				t.Fatal(err)
			}
			if _, err := FaultsSetup(cfg); err == nil || !strings.Contains(err.Error(), c.want) {
				t.Fatalf("err = %v, want one that mentions %q", err, c.want)
			}
		})
	}
	t.Run("missing tenant", func(t *testing.T) {
		cfg, err := config.Parse([]byte(strings.Replace(faultsConfigYAML("    daily_budget_usd: 1", "    daily_budget_usd: 1\n    allowed_models: [m1]"), "name: bench-over", "name: other", 1)))
		if err != nil {
			t.Fatal(err)
		}
		if _, err := FaultsSetup(cfg); err == nil || !strings.Contains(err.Error(), "bench-over") {
			t.Fatalf("err = %v, want one that names the missing tenant", err)
		}
	})
}
