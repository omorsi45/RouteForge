package main

import (
	"bufio"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"log/slog"
	"net/http"
	"net/http/httptest"
	"slices"
	"strconv"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/omorsi45/routeforge/internal/accounting"
	"github.com/omorsi45/routeforge/internal/api"
	"github.com/omorsi45/routeforge/internal/auth"
	"github.com/omorsi45/routeforge/internal/config"
)

// These tests check the four accounting invariants (listed under "Accounting"
// in docs/operations.md) against the real gateway, built by buildGateway
// exactly as serve builds it, under fault injection. Fake providers fail
// calls, answer 429, cut streams short and leave usage out, and the client
// abandons some streams after their first frame. Every request has returned,
// and every handler with it, when the books are read, and each invariant must
// hold whatever interleaving the faults fell in.
//
// The providers sleep 1ms per token, so a 16 token stream lasts about 16ms: long
// enough for a client that abandons it after the first frame to leave it
// mid-flight, short enough that a run takes well under a second.

const (
	invRequests    = 300 // requests per run
	invConcurrency = 16  // requests in flight at once
	invBackoff     = 20 * time.Millisecond
	invClientLimit = 30 * time.Second
	invIdleLimit   = 30 * time.Second
)

// invTenant is a tenant of the harness. Each invariant has its own, so a budget
// that one run uses up cannot disturb another.
type invTenant struct {
	name      string
	key       string
	budgetUSD float64
	maxTokens int      // max_output_tokens, the cap every reservation is priced at
	models    []string // the models its requests rotate over
	abandon   bool     // whether the client leaves every 7th stream after its first frame
}

var (
	invMixed = []string{"m-unavailable", "m-limited", "m-steady"}

	// invOpen has a budget it cannot reach: every attempt reserves and settles,
	// and no request is refused.
	invOpen = invTenant{name: "inv-open", key: "test-key-invariants-open", budgetUSD: 100, maxTokens: 64, models: invMixed, abandon: true}
	// invCapped runs out of budget early in the run (invariant c). The output
	// cap of 64 tokens is four times what the providers produce, so even a
	// charge estimated from delivered bytes stays under its reservation: the
	// providers honor the limits the gateway reserved for.
	invCapped = invTenant{name: "inv-capped", key: "test-key-invariants-capped", budgetUSD: 0.003, maxTokens: 64, models: invMixed, abandon: true}
	// invOver is billed for 50 tokens more than it can reserve (invariant d).
	// A reservation is about 53,000 nano-USD, the budget 120,000, and a
	// reported response costs about 138,000: the first response to settle
	// pushes spend past the budget, whatever else happens. Nothing else is
	// injected, so every attempt that is admitted is such a response.
	invOver = invTenant{name: "inv-over", key: "test-key-invariants-over", budgetUSD: 0.00012, maxTokens: 16, models: []string{"m-over"}}
	// invCut has its stream cut after three words every time.
	invCut = invTenant{name: "inv-cut", key: "test-key-invariants-cut", budgetUSD: 1, maxTokens: 64, models: []string{"m-cut"}}
)

// invConfigYAML is the harness configuration; %s is the tenants. Breakers are
// set out of reach and retries are plentiful and quick so that the injected
// faults reach accounting instead of being absorbed by the reliability layer.
// All prices are explicit, so no price file is read.
const invConfigYAML = `
reliability:
  backoff_base: 1ms
  backoff_cap: 5ms
  retry_budget_min: 1000
  breaker:
    consecutive_failures: 100000
    min_requests: 100000
tenants:
%s
providers:
  - name: p-unavailable
    type: fake
    fake:
      seed: 101
      per_token: 1ms
      error_rate: 0.3
      error_kind: unavailable
      disconnect_rate: 0.2
      disconnect_after: 3
      omit_usage_rate: 0.3
  - name: p-limited
    type: fake
    fake:
      seed: 102
      per_token: 1ms
      error_rate: 0.3
      error_kind: rate_limited
      retry_after: 20ms
      disconnect_rate: 0.2
      disconnect_after: 3
      omit_usage_rate: 0.3
  - name: p-steady
    type: fake
    fake:
      seed: 103
      per_token: 1ms
      disconnect_rate: 0.2
      disconnect_after: 3
      omit_usage_rate: 0.3
  - name: p-over
    type: fake
    fake:
      seed: 104
      per_token: 1ms
      over_report_tokens: 50
  - name: p-cut
    type: fake
    fake:
      seed: 105
      disconnect_rate: 1
      disconnect_after: 3
models:
  - name: m-unavailable
    provider: p-unavailable
    input_per_mtok: 1
    output_per_mtok: 2
  - name: m-limited
    provider: p-limited
    input_per_mtok: 3
    output_per_mtok: 6
  - name: m-steady
    provider: p-steady
    input_per_mtok: 0.5
    output_per_mtok: 1.5
  - name: m-over
    provider: p-over
    input_per_mtok: 1
    output_per_mtok: 2
  - name: m-cut
    provider: p-cut
    input_per_mtok: 1
    output_per_mtok: 2
`

func invConfig(t *testing.T) *config.Config {
	t.Helper()
	var tenants strings.Builder
	for _, tn := range []invTenant{invOpen, invCapped, invOver, invCut} {
		fmt.Fprintf(&tenants, "  - name: %s\n    api_key_sha256: [%s]\n    max_output_tokens: %d\n    daily_budget_usd: %s\n",
			tn.name, auth.HashKey(tn.key), tn.maxTokens, strconv.FormatFloat(tn.budgetUSD, 'f', -1, 64))
	}
	cfg, err := config.Parse([]byte(fmt.Sprintf(invConfigYAML, tenants.String())))
	if err != nil {
		t.Fatalf("harness config: %v", err)
	}
	return cfg
}

// budgetNanos is the tenant's daily budget in nano-USD.
func (tn invTenant) budgetNanos(t *testing.T) int64 {
	t.Helper()
	n, err := accounting.USDToNanos(tn.budgetUSD)
	if err != nil {
		t.Fatal(err)
	}
	return n
}

// invHarness is the real gateway behind an httptest.Server.
type invHarness struct {
	t     *testing.T
	gw    *gateway
	ts    *httptest.Server
	logs  *syncBuffer
	began time.Time
}

func newInvHarness(t *testing.T) *invHarness {
	t.Helper()
	logs := &syncBuffer{}
	gw, err := buildGateway(invConfig(t), func(string) string { return "" }, slog.New(slog.NewJSONHandler(logs, nil)))
	if err != nil {
		t.Fatalf("buildGateway: %v", err)
	}
	ts := httptest.NewServer(gw.Handler())
	t.Cleanup(ts.Close)
	return &invHarness{t: t, gw: gw, ts: ts, logs: logs, began: time.Now()}
}

// invStats is what the clients of a run saw.
type invStats struct {
	mu        sync.Mutex
	statuses  map[int]int    // HTTP status counts
	codes     map[string]int // error codes of the responses that carried one
	abandoned int            // streams the client left after the first frame
}

func (s *invStats) record(status int, code string) {
	s.mu.Lock()
	defer s.mu.Unlock()
	s.statuses[status]++
	if code != "" {
		s.codes[code]++
	}
}

func (s *invStats) abandon() {
	s.mu.Lock()
	defer s.mu.Unlock()
	s.abandoned++
}

func invErrorCode(raw []byte) string {
	var env struct {
		Error struct {
			Code string `json:"code"`
		} `json:"error"`
	}
	_ = json.Unmarshal(raw, &env) // a body without an envelope has no code
	return env.Error.Code
}

func (h *invHarness) chatRequest(ctx context.Context, key, body string) (*http.Request, error) {
	req, err := http.NewRequestWithContext(ctx, http.MethodPost, h.ts.URL+"/v1/chat/completions", strings.NewReader(body))
	if err != nil {
		return nil, err
	}
	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("Authorization", "Bearer "+key)
	return req, nil
}

// load sends n requests of tn, invConcurrency at a time. Request i goes to
// models[i % len(models)]; even ones stream, and so does every 7th, which a
// tenant that abandons leaves after the first frame. It returns once every
// client has returned, which does not mean the gateway has finished with them:
// call idle for that.
func (h *invHarness) load(tn invTenant, n int) *invStats {
	stats := &invStats{statuses: map[int]int{}, codes: map[string]int{}}
	jobs := make(chan int)
	var wg sync.WaitGroup
	for range invConcurrency {
		wg.Add(1)
		go func() {
			defer wg.Done()
			for i := range jobs {
				h.send(tn, i, stats)
			}
		}()
	}
	for i := range n {
		jobs <- i
	}
	close(jobs)
	wg.Wait()
	return stats
}

// send is one request of a load. It reports failures with Errorf, because it
// runs off the test goroutine.
func (h *invHarness) send(tn invTenant, i int, stats *invStats) {
	leaves := tn.abandon && i%7 == 0
	stream := leaves || i%2 == 0
	body := fmt.Sprintf(`{"model":%q,"stream":%t,"messages":[{"role":"user","content":"invariant request %d"}]}`,
		tn.models[i%len(tn.models)], stream, i)

	ctx, cancel := context.WithTimeout(context.Background(), invClientLimit)
	defer cancel()
	req, err := h.chatRequest(ctx, tn.key, body)
	if err != nil {
		h.t.Errorf("request %d: %v", i, err)
		return
	}
	resp, err := e2eClient.Do(req)
	if err != nil {
		h.t.Errorf("request %d: %v", i, err)
		return
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		raw, err := io.ReadAll(resp.Body)
		if err != nil {
			h.t.Errorf("request %d: reading the error: %v", i, err)
		}
		stats.record(resp.StatusCode, invErrorCode(raw))
		// A client that has been refused backs off. The gateway answers a
		// refusal at once, and without this pause the clients would use up
		// the run's requests in a few milliseconds, while the calls already
		// admitted were still holding their reservations and before the
		// budget had been spent.
		time.Sleep(invBackoff)
		return
	}
	stats.record(resp.StatusCode, "")

	if !stream {
		if _, err := io.Copy(io.Discard, resp.Body); err != nil {
			h.t.Errorf("request %d: reading the response: %v", i, err)
		}
		return
	}
	rd := bufio.NewReader(resp.Body)
	for {
		line, err := rd.ReadString('\n')
		if err != nil {
			if err != io.EOF {
				h.t.Errorf("request %d: reading the stream: %v", i, err)
			}
			return
		}
		if leaves && strings.HasPrefix(line, "data: ") {
			stats.abandon()
			cancel()
			return
		}
	}
}

// idle waits until the gateway has finished every request, a request whose
// client left included, and returns the budget days the run can have touched:
// a run that crosses midnight UTC is split over two.
func (h *invHarness) idle() []string {
	h.t.Helper()
	done := make(chan struct{})
	go func() {
		h.ts.Close() // returns when every handler has
		close(done)
	}()
	select {
	case <-done:
	case <-time.After(invIdleLimit):
		h.t.Fatalf("a request was still being served %s after its client had returned", invIdleLimit)
	}
	days := []string{accounting.DayKey(h.began), accounting.DayKey(time.Now())}
	return slices.Compact(days)
}

// dayBooks is the budget and usage state of one tenant-day.
type dayBooks struct {
	day    string
	budget accounting.BudgetState
	usage  accounting.UsageTotals
}

func (h *invHarness) books(tenant string, days []string) []dayBooks {
	h.t.Helper()
	out := make([]dayBooks, len(days))
	for i, day := range days {
		b, u, err := h.gw.accountant.State(context.Background(), tenant, day)
		if err != nil {
			h.t.Fatalf("State(%s, %s): %v", tenant, day, err)
		}
		out[i] = dayBooks{day: day, budget: b, usage: u}
	}
	return out
}

// records returns the usage records of a tenant.
func (h *invHarness) records(tenant string) []accounting.UsageRecord {
	var out []accounting.UsageRecord
	for _, r := range h.gw.usage.Recent() {
		if r.Tenant == tenant {
			out = append(out, r)
		}
	}
	return out
}

// requireSpendEqualsUsage checks invariant (a): what the budget says was spent
// is what the usage records add up to, to the nano-USD.
func requireSpendEqualsUsage(t *testing.T, tenant string, books []dayBooks) {
	t.Helper()
	var spent int64
	var records int
	for _, b := range books {
		if b.budget.Spent != b.usage.CostNanos {
			t.Errorf("%s on %s: Spent = %d nano-USD, usage records add up to %d (%d records)",
				tenant, b.day, b.budget.Spent, b.usage.CostNanos, b.usage.Records)
		}
		spent += b.budget.Spent
		records += b.usage.Records
	}
	if spent == 0 || records == 0 {
		t.Errorf("%s: spent %d nano-USD over %d usage records; the run charged nothing, so the check proved nothing", tenant, spent, records)
	}
}

// requireReservedZero checks invariant (b): every reservation has been returned.
func requireReservedZero(t *testing.T, tenant string, books []dayBooks) {
	t.Helper()
	for _, b := range books {
		if b.budget.Reserved != 0 {
			t.Errorf("%s on %s: Reserved = %d nano-USD after every request returned, want 0", tenant, b.day, b.budget.Reserved)
		}
	}
}

// requireFaultsExercised fails when a run of the mixed providers did not reach
// the paths the invariants are about: successes, failures, streams cut short,
// streams the client left, usage the provider sent and usage it left out. A
// run that missed one would pass the invariants without testing them.
func requireFaultsExercised(t *testing.T, h *invHarness, tn invTenant, stats *invStats) {
	t.Helper()
	outcomes := map[string]int{}
	sources := map[string]int{}
	for _, r := range h.records(tn.name) {
		outcomes[r.Outcome]++
		sources[r.UsageSource]++
	}
	t.Logf("%s: HTTP statuses %v, error codes %v, %d streams abandoned; attempts by outcome %v, by usage source %v",
		tn.name, stats.statuses, stats.codes, stats.abandoned, outcomes, sources)
	for _, o := range []string{accounting.OutcomeOK, accounting.OutcomeError, accounting.OutcomePartial, accounting.OutcomeCanceled} {
		if outcomes[o] == 0 {
			t.Errorf("no attempt ended %q; outcomes %v", o, outcomes)
		}
	}
	for _, s := range []string{accounting.SourceReported, accounting.SourceEstimated} {
		if sources[s] == 0 {
			t.Errorf("no attempt was billed from %s usage; sources %v", s, sources)
		}
	}
	if stats.abandoned == 0 {
		t.Error("no client left a stream")
	}
}

func TestInvariantSpendEqualsUsage(t *testing.T) {
	h := newInvHarness(t)
	stats := h.load(invOpen, invRequests)
	days := h.idle()

	requireFaultsExercised(t, h, invOpen, stats)
	requireSpendEqualsUsage(t, invOpen.name, h.books(invOpen.name, days))
}

func TestInvariantReservedReturnsToZero(t *testing.T) {
	h := newInvHarness(t)
	stats := h.load(invOpen, invRequests)
	days := h.idle()

	requireFaultsExercised(t, h, invOpen, stats)
	requireReservedZero(t, invOpen.name, h.books(invOpen.name, days))

	// The access log marks the clients that left: status 499 and client_gone
	// go together, and some of the abandoned streams were seen leaving.
	gone := 0
	for _, raw := range strings.Split(strings.TrimSpace(h.logs.String()), "\n") {
		if raw == "" {
			continue
		}
		var line struct {
			Msg        string `json:"msg"`
			Status     int    `json:"status"`
			ClientGone bool   `json:"client_gone"`
		}
		if err := json.Unmarshal([]byte(raw), &line); err != nil {
			t.Fatalf("log line is not JSON: %v; line %q", err, raw)
		}
		if line.Msg != "request" {
			continue
		}
		if (line.Status == 499) != line.ClientGone {
			t.Errorf("status %d with client_gone %t: the two must go together", line.Status, line.ClientGone)
		}
		if line.ClientGone {
			gone++
		}
	}
	if gone == 0 {
		t.Errorf("no request was logged with client_gone, though %d streams were abandoned", stats.abandoned)
	}
}

func TestInvariantSpendWithinBudget(t *testing.T) {
	h := newInvHarness(t)
	stats := h.load(invCapped, invRequests)
	days := h.idle()
	budget := invCapped.budgetNanos(t)

	// A request is refused for its budget either by the router, which sees
	// that the remaining budget cannot hold the reservation, or at the
	// reservation itself. Either way, the budget has run out.
	refused := stats.codes[api.CodeBudgetExceeded] + stats.codes[api.CodeNoEligibleModel]
	if refused == 0 {
		t.Fatalf("no request was refused for its budget, so it did not run out; status counts %v, codes %v", stats.statuses, stats.codes)
	}

	books := h.books(invCapped.name, days)
	t.Logf("%s: HTTP statuses %v, error codes %v; books %+v, budget %d", invCapped.name, stats.statuses, stats.codes, books, budget)
	var spent int64
	for _, b := range books {
		if b.budget.Spent > budget {
			t.Errorf("Spent = %d on %s, over the budget of %d", b.budget.Spent, b.day, budget)
		}
		if b.budget.Overshoot != 0 {
			t.Errorf("Overshoot = %d on %s, want 0 for providers that honor the limits", b.budget.Overshoot, b.day)
		}
		spent += b.budget.Spent
	}
	// The budget was spent, not merely reserved away: refusals that hold back
	// a budget nobody used would make Spent <= budget true for no reason.
	if spent*10 < budget*8 {
		t.Errorf("spent %d of a budget of %d: under 80%%, so the budget did not run out through spending", spent, budget)
	}
	requireSpendEqualsUsage(t, invCapped.name, books)
	requireReservedZero(t, invCapped.name, books)
}

func TestInvariantOvershootEqualsExcess(t *testing.T) {
	h := newInvHarness(t)
	stats := h.load(invOver, invRequests)
	days := h.idle()
	budget := invOver.budgetNanos(t)

	books := h.books(invOver.name, days)
	t.Logf("%s: HTTP statuses %v, error codes %v; books %+v, budget %d", invOver.name, stats.statuses, stats.codes, books, budget)
	var overshoot int64
	for _, b := range books {
		if want := max(0, b.budget.Spent-budget); b.budget.Overshoot != want {
			t.Errorf("on %s: Overshoot = %d, want Spent - budget = %d (Spent %d, budget %d)",
				b.day, b.budget.Overshoot, want, b.budget.Spent, budget)
		}
		overshoot += b.budget.Overshoot
	}
	if overshoot == 0 {
		t.Fatalf("spend never passed the budget of %d, though each response costs more: the over-report did not reach accounting", budget)
	}
	requireSpendEqualsUsage(t, invOver.name, books)
	requireReservedZero(t, invOver.name, books)
}

func TestInvariantPartialStreamSettles(t *testing.T) {
	h := newInvHarness(t)
	const prompt = "partial stream probe"
	body := fmt.Sprintf(`{"model":"m-cut","stream":true,"messages":[{"role":"user","content":%q}]}`, prompt)

	ctx, cancel := context.WithTimeout(context.Background(), invClientLimit)
	defer cancel()
	req, err := h.chatRequest(ctx, invCut.key, body)
	if err != nil {
		t.Fatal(err)
	}
	resp, err := e2eClient.Do(req)
	if err != nil {
		t.Fatal(err)
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		raw, _ := io.ReadAll(resp.Body)
		t.Fatalf("status = %d, want 200; body %s", resp.StatusCode, raw)
	}

	// Read the stream to its end: three words, then the error event.
	var delivered strings.Builder
	chunks, done, errCode := 0, false, ""
	sc := bufio.NewScanner(resp.Body)
	for sc.Scan() {
		payload, ok := strings.CutPrefix(sc.Text(), "data: ")
		if !ok {
			continue
		}
		if payload == "[DONE]" {
			done = true
			continue
		}
		var frame struct {
			Error *struct {
				Code string `json:"code"`
			} `json:"error"`
			Choices []struct {
				Delta struct {
					Content string `json:"content"`
				} `json:"delta"`
			} `json:"choices"`
		}
		if err := json.Unmarshal([]byte(payload), &frame); err != nil {
			t.Fatalf("frame %q is not JSON: %v", payload, err)
		}
		if frame.Error != nil {
			errCode = frame.Error.Code
		}
		for _, c := range frame.Choices {
			if c.Delta.Content != "" {
				delivered.WriteString(c.Delta.Content)
				chunks++
			}
		}
	}
	if err := sc.Err(); err != nil {
		t.Fatalf("reading the stream: %v", err)
	}
	if chunks != 3 || done || errCode != "upstream_stream_interrupted" {
		t.Fatalf("stream = %d content chunks, [DONE] %t, error %q; want 3 chunks cut by upstream_stream_interrupted", chunks, done, errCode)
	}
	days := h.idle()

	recs := h.records(invCut.name)
	if len(recs) != 1 {
		t.Fatalf("got %d usage records, want exactly one for the one attempt: %+v", len(recs), recs)
	}
	rec := recs[0]
	// No usage arrived, so the attempt is billed ceil(prompt_bytes/4) prompt
	// tokens and ceil(delivered_bytes/4) completion tokens. The model costs
	// 1000 nano-USD per input token and 2000 per output token.
	wantPrompt := (len(prompt) + 3) / 4
	wantCompletion := (delivered.Len() + 3) / 4
	wantCost := int64(wantPrompt)*1000 + int64(wantCompletion)*2000
	if !rec.Partial || rec.Outcome != accounting.OutcomePartial {
		t.Errorf("record Outcome = %q, Partial = %t; want a partial record", rec.Outcome, rec.Partial)
	}
	if rec.PromptTokens != wantPrompt || rec.CompletionTokens != wantCompletion {
		t.Errorf("record bills %d prompt and %d completion tokens, want %d and %d (%d bytes delivered)",
			rec.PromptTokens, rec.CompletionTokens, wantPrompt, wantCompletion, delivered.Len())
	}
	if rec.UsageSource != accounting.SourceEstimated || rec.CostSource != accounting.SourceEstimated {
		t.Errorf("record sources = %q / %q, want estimated for both", rec.UsageSource, rec.CostSource)
	}
	if rec.CostNanos != wantCost {
		t.Errorf("record CostNanos = %d, want %d", rec.CostNanos, wantCost)
	}
	if rec.Model != "m-cut" || rec.Attempt != 1 {
		t.Errorf("record = model %q attempt %d, want m-cut attempt 1", rec.Model, rec.Attempt)
	}

	books := h.books(invCut.name, days)
	requireReservedZero(t, invCut.name, books)
	requireSpendEqualsUsage(t, invCut.name, books)
	var spent int64
	for _, b := range books {
		spent += b.budget.Spent
	}
	if spent != wantCost {
		t.Errorf("Spent = %d, want the partial charge %d", spent, wantCost)
	}
}
