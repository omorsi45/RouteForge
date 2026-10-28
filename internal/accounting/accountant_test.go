package accounting_test

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"log/slog"
	"regexp"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/omorsi45/routeforge/internal/accounting"
	"github.com/omorsi45/routeforge/internal/config"
	"github.com/omorsi45/routeforge/internal/executor"
	"github.com/omorsi45/routeforge/internal/provider"
	"github.com/omorsi45/routeforge/internal/router"
)

// The meter tests send 400 prompt bytes with an output cap of 100 tokens to
// m-fast, priced at 1000 nano-USD per input token and 2000 per output token, so
// a reservation is 400*1000 + 100*2000 = 600,000 nano-USD.
const (
	promptBytes = 400
	maxTokens   = 100
	reservation = 600_000
	testDay     = "20261027"
)

var errBackend = errors.New("backend unreachable")

// clock is a settable time source.
type clock struct {
	mu sync.Mutex
	t  time.Time
}

func (c *clock) Now() time.Time {
	c.mu.Lock()
	defer c.mu.Unlock()
	return c.t
}

func (c *clock) Advance(d time.Duration) {
	c.mu.Lock()
	defer c.mu.Unlock()
	c.t = c.t.Add(d)
}

// flakyBudgets wraps a budget store, counts its calls and can be taken down.
type flakyBudgets struct {
	accounting.BudgetStore

	mu                    sync.Mutex
	down                  bool
	failSettle            bool
	reserves, settles     int
	settleCtxErr          error
	lastReserveLimit      int64
	lastSettleReservedArg int64
}

func (b *flakyBudgets) isDown() bool {
	b.mu.Lock()
	defer b.mu.Unlock()
	return b.down
}

func (b *flakyBudgets) Reserve(ctx context.Context, k accounting.BudgetKey, amount, limit int64) (bool, error) {
	b.mu.Lock()
	b.reserves++
	b.lastReserveLimit = limit
	down := b.down
	b.mu.Unlock()
	if down {
		return false, errBackend
	}
	return b.BudgetStore.Reserve(ctx, k, amount, limit)
}

func (b *flakyBudgets) Settle(ctx context.Context, k accounting.BudgetKey, reserved, actual, limit int64) (int64, error) {
	b.mu.Lock()
	b.settles++
	b.settleCtxErr = ctx.Err()
	b.lastSettleReservedArg = reserved
	down := b.down || b.failSettle
	b.mu.Unlock()
	if down {
		return 0, errBackend
	}
	return b.BudgetStore.Settle(ctx, k, reserved, actual, limit)
}

func (b *flakyBudgets) State(ctx context.Context, k accounting.BudgetKey) (accounting.BudgetState, error) {
	if b.isDown() {
		return accounting.BudgetState{}, errBackend
	}
	return b.BudgetStore.State(ctx, k)
}

func (b *flakyBudgets) Ping(ctx context.Context) error {
	if b.isDown() {
		return errBackend
	}
	return b.BudgetStore.Ping(ctx)
}

func (b *flakyBudgets) setDown(v bool) {
	b.mu.Lock()
	defer b.mu.Unlock()
	b.down = v
}

func (b *flakyBudgets) counts() (reserves, settles int) {
	b.mu.Lock()
	defer b.mu.Unlock()
	return b.reserves, b.settles
}

// flakyLimiter wraps a limiter, counts its calls and can be taken down.
type flakyLimiter struct {
	accounting.RateLimiter
	mu    sync.Mutex
	down  bool
	calls int
}

func (l *flakyLimiter) Allow(ctx context.Context, tenant, kind string, n, perMinute int64) (accounting.RateDecision, error) {
	l.mu.Lock()
	l.calls++
	down := l.down
	l.mu.Unlock()
	if down {
		return accounting.RateDecision{}, errBackend
	}
	return l.RateLimiter.Allow(ctx, tenant, kind, n, perMinute)
}

func (l *flakyLimiter) setDown(v bool) {
	l.mu.Lock()
	defer l.mu.Unlock()
	l.down = v
}

func (l *flakyLimiter) callCount() int {
	l.mu.Lock()
	defer l.mu.Unlock()
	return l.calls
}

// flakyUsage wraps the memory usage store; it can report full or fail writes.
type flakyUsage struct {
	*accounting.MemoryUsage
	mu       sync.Mutex
	full     bool
	writeErr error
}

func (u *flakyUsage) Full() bool {
	u.mu.Lock()
	defer u.mu.Unlock()
	return u.full
}

func (u *flakyUsage) Write(ctx context.Context, r accounting.UsageRecord) error {
	u.mu.Lock()
	err := u.writeErr
	u.mu.Unlock()
	if err != nil {
		return err
	}
	return u.MemoryUsage.Write(ctx, r)
}

func (u *flakyUsage) set(full bool, writeErr error) {
	u.mu.Lock()
	defer u.mu.Unlock()
	u.full, u.writeErr = full, writeErr
}

// rig is an Accountant over memory stores wrapped to be counted and broken.
type rig struct {
	clk     *clock
	budgets *flakyBudgets
	limiter *flakyLimiter
	usage   *flakyUsage
	logs    *bytes.Buffer
	acct    *accounting.Accountant
}

func newRig(t *testing.T, mode string, tenants ...config.Tenant) *rig {
	t.Helper()
	price, err := accounting.NewPrice("m-fast", 1, 2)
	if err != nil {
		t.Fatal(err)
	}
	r := &rig{
		clk:     &clock{t: time.Date(2026, 10, 27, 22, 0, 0, 0, time.UTC)},
		budgets: &flakyBudgets{BudgetStore: accounting.NewMemoryBudgets()},
		usage:   &flakyUsage{MemoryUsage: accounting.NewMemoryUsage(100)},
		logs:    &bytes.Buffer{},
	}
	r.limiter = &flakyLimiter{RateLimiter: accounting.NewMemoryLimiter(r.clk.Now)}
	var (
		idMu sync.Mutex
		ids  int
	)
	r.acct = accounting.NewAccountant(accounting.AccountantOptions{
		Tenants:  tenants,
		Prices:   map[string]accounting.Price{"m-fast": price},
		Budgets:  r.budgets,
		Limiter:  r.limiter,
		Usage:    r.usage,
		FailMode: mode,
		Now:      r.clk.Now,
		Logger:   slog.New(slog.NewJSONHandler(r.logs, nil)),
		NewID: func() string {
			idMu.Lock()
			defer idMu.Unlock()
			ids++
			return fmt.Sprintf("usage-%d", ids)
		},
	})
	return r
}

// logLines decodes the JSON log lines written so far.
func (r *rig) logLines(t *testing.T) []map[string]any {
	t.Helper()
	var out []map[string]any
	for _, line := range strings.Split(strings.TrimSpace(r.logs.String()), "\n") {
		if line == "" {
			continue
		}
		var m map[string]any
		if err := json.Unmarshal([]byte(line), &m); err != nil {
			t.Fatalf("log line %q: %v", line, err)
		}
		out = append(out, m)
	}
	return out
}

func (r *rig) meter(ten *config.Tenant) *accounting.Meter {
	return r.acct.NewMeter(ten, "req_1", "dec_1", promptBytes, maxTokens)
}

func (r *rig) wantBudget(t *testing.T, tenant string, want accounting.BudgetState) {
	t.Helper()
	got, err := r.budgets.BudgetStore.State(context.Background(), accounting.BudgetKey{Tenant: tenant, Day: testDay})
	if err != nil || got != want {
		t.Fatalf("budget state = %+v, %v; want %+v", got, err, want)
	}
}

func (r *rig) records() []accounting.UsageRecord { return r.usage.Recent() }

// begin starts an attempt on m-fast and fails the test if it is refused.
func begin(t *testing.T, m *accounting.Meter) accounting.Charge {
	t.Helper()
	c, err := m.Begin(context.Background(), "m-fast")
	if err != nil {
		t.Fatalf("Begin: %v", err)
	}
	return c
}

func settleOK(prompt, completion int) accounting.Settlement {
	return accounting.Settlement{
		Outcome:        accounting.OutcomeOK,
		Usage:          &provider.Usage{PromptTokens: prompt, CompletionTokens: completion},
		DeliveredBytes: completion * 4,
	}
}

func TestAdmitRPMThenTPM(t *testing.T) {
	ctx := context.Background()

	t.Run("requests per minute", func(t *testing.T) {
		ten := config.Tenant{Name: "t", RPM: 2, TPM: 100}
		r := newRig(t, config.FailModeClosed, ten)
		// 40 bytes are 10 tokens, plus 20 output tokens: 30 tokens a request.
		for i := range 2 {
			if err := r.acct.Admit(ctx, &ten, 40, 20); err != nil {
				t.Fatalf("request %d: %v", i+1, err)
			}
		}
		err := r.acct.Admit(ctx, &ten, 40, 20)
		var rl *accounting.RateLimitedError
		if !errors.As(err, &rl) {
			t.Fatalf("err = %v, want a *RateLimitedError", err)
		}
		if rl.Kind != "rpm" || rl.RetryAfter != 30*time.Second {
			t.Errorf("limit = %s retry after %v, want rpm 30s", rl.Kind, rl.RetryAfter)
		}
		// The bucket refills with the clock.
		r.clk.Advance(30 * time.Second)
		if err := r.acct.Admit(ctx, &ten, 40, 20); err != nil {
			t.Fatalf("after 30s: %v", err)
		}
	})

	t.Run("tokens per minute", func(t *testing.T) {
		ten := config.Tenant{Name: "t", RPM: 10, TPM: 50}
		r := newRig(t, config.FailModeClosed, ten)
		if err := r.acct.Admit(ctx, &ten, 40, 20); err != nil { // 30 of 50 tokens
			t.Fatal(err)
		}
		err := r.acct.Admit(ctx, &ten, 40, 20)
		var rl *accounting.RateLimitedError
		if !errors.As(err, &rl) {
			t.Fatalf("err = %v, want a *RateLimitedError", err)
		}
		// 20 tokens are left and 30 are needed: 10 tokens at 50 a minute.
		if rl.Kind != "tpm" || rl.RetryAfter != 12*time.Second {
			t.Errorf("limit = %s retry after %v, want tpm 12s", rl.Kind, rl.RetryAfter)
		}
	})

	t.Run("a request that can never fit", func(t *testing.T) {
		ten := config.Tenant{Name: "t", TPM: 50}
		r := newRig(t, config.FailModeClosed, ten)
		err := r.acct.Admit(ctx, &ten, 400, 100) // 200 tokens against a bucket of 50
		var rl *accounting.RateLimitedError
		if !errors.As(err, &rl) || rl.Kind != "tpm" || rl.RetryAfter != time.Minute {
			t.Fatalf("err = %v, want tpm limited for a minute", err)
		}
	})

	t.Run("rpm is checked before tpm", func(t *testing.T) {
		ten := config.Tenant{Name: "t", RPM: 1, TPM: 30}
		r := newRig(t, config.FailModeClosed, ten)
		if err := r.acct.Admit(ctx, &ten, 40, 20); err != nil {
			t.Fatal(err)
		}
		// Both buckets are empty now; the request is refused for rpm, and the
		// tpm bucket is left alone.
		err := r.acct.Admit(ctx, &ten, 40, 20)
		var rl *accounting.RateLimitedError
		if !errors.As(err, &rl) || rl.Kind != "rpm" {
			t.Fatalf("err = %v, want rpm", err)
		}
		r.clk.Advance(time.Minute)
		if err := r.acct.Admit(ctx, &ten, 40, 20); err != nil {
			t.Fatalf("after a minute: %v", err)
		}
	})

	t.Run("a tenant with no limits never reaches the limiter", func(t *testing.T) {
		ten := config.Tenant{Name: "t"}
		r := newRig(t, config.FailModeClosed, ten)
		for range 100 {
			if err := r.acct.Admit(ctx, &ten, 4000, 4000); err != nil {
				t.Fatal(err)
			}
		}
		if n := r.limiter.callCount(); n != 0 {
			t.Errorf("limiter called %d times, want 0", n)
		}
	})

	t.Run("only the limits that are set are checked", func(t *testing.T) {
		ten := config.Tenant{Name: "t", RPM: 5}
		r := newRig(t, config.FailModeClosed, ten)
		if err := r.acct.Admit(ctx, &ten, 40, 20); err != nil {
			t.Fatal(err)
		}
		if n := r.limiter.callCount(); n != 1 {
			t.Errorf("limiter called %d times, want 1 (rpm only)", n)
		}
	})
}

func TestAdmitUsageQueueFull(t *testing.T) {
	for _, mode := range []string{config.FailModeClosed, config.FailModeOpen} {
		t.Run(mode, func(t *testing.T) {
			ten := config.Tenant{Name: "t", RPM: 5}
			r := newRig(t, mode, ten)
			r.usage.set(true, nil)
			err := r.acct.Admit(context.Background(), &ten, 40, 20)
			if !errors.Is(err, accounting.ErrAccountingUnavailable) {
				t.Fatalf("err = %v, want ErrAccountingUnavailable", err)
			}
			if n := r.limiter.callCount(); n != 0 {
				t.Errorf("limiter called %d times, want 0: a request that cannot be billed must not use up rate", n)
			}
			// Room again: the same request goes through.
			r.usage.set(false, nil)
			if err := r.acct.Admit(context.Background(), &ten, 40, 20); err != nil {
				t.Fatalf("after the queue drained: %v", err)
			}
		})
	}
}

func TestAdmitBackendDown(t *testing.T) {
	ctx := context.Background()
	ten := config.Tenant{Name: "t", RPM: 5}

	t.Run("closed mode refuses", func(t *testing.T) {
		r := newRig(t, config.FailModeClosed, ten)
		r.limiter.setDown(true)
		err := r.acct.Admit(ctx, &ten, 40, 20)
		if !errors.Is(err, accounting.ErrAccountingUnavailable) {
			t.Fatalf("err = %v, want ErrAccountingUnavailable", err)
		}
	})

	t.Run("an unknown fail mode is closed", func(t *testing.T) {
		r := newRig(t, "sometimes", ten)
		r.limiter.setDown(true)
		if err := r.acct.Admit(ctx, &ten, 40, 20); !errors.Is(err, accounting.ErrAccountingUnavailable) {
			t.Fatalf("err = %v, want ErrAccountingUnavailable", err)
		}
	})

	t.Run("open mode lets the request through and logs once a minute", func(t *testing.T) {
		r := newRig(t, config.FailModeOpen, ten)
		r.limiter.setDown(true)
		for range 3 {
			if err := r.acct.Admit(ctx, &ten, 40, 20); err != nil {
				t.Fatalf("Admit in open mode: %v", err)
			}
		}
		lines := r.logLines(t)
		if len(lines) != 1 {
			t.Fatalf("%d log lines, want 1: %s", len(lines), r.logs)
		}
		if lines[0]["level"] != "ERROR" || lines[0]["backend"] != "limiter" {
			t.Errorf("log line = %v, want an error naming the limiter", lines[0])
		}

		r.clk.Advance(59 * time.Second)
		_ = r.acct.Admit(ctx, &ten, 40, 20)
		if n := len(r.logLines(t)); n != 1 {
			t.Errorf("%d log lines after 59s, want still 1", n)
		}
		r.clk.Advance(2 * time.Second)
		_ = r.acct.Admit(ctx, &ten, 40, 20)
		if n := len(r.logLines(t)); n != 2 {
			t.Errorf("%d log lines after 61s, want 2", n)
		}
	})

	t.Run("closed mode logs the cause once a minute too", func(t *testing.T) {
		r := newRig(t, config.FailModeClosed, ten)
		r.limiter.setDown(true)
		for range 5 {
			_ = r.acct.Admit(ctx, &ten, 40, 20)
		}
		if lines := r.logLines(t); len(lines) != 1 || lines[0]["backend"] != "limiter" {
			t.Fatalf("log lines = %v, want one naming the limiter", lines)
		}
	})
}

func TestMeterSettlesReported(t *testing.T) {
	ten := config.Tenant{Name: "t", DailyBudgetUSD: 1}
	r := newRig(t, config.FailModeClosed, ten)
	m := r.meter(&ten)

	c := begin(t, m)
	r.wantBudget(t, "t", accounting.BudgetState{Reserved: reservation})
	if got := r.budgets.lastReserveLimit; got != 1_000_000_000 {
		t.Errorf("reserved against a limit of %d, want 1,000,000,000 nano-USD", got)
	}

	c.End(settleOK(90, 40))
	// 90 * 1000 + 40 * 2000.
	const cost = 170_000
	r.wantBudget(t, "t", accounting.BudgetState{Spent: cost})

	want := accounting.UsageRecord{
		ID:               "usage-1",
		RequestID:        "req_1",
		DecisionID:       "dec_1",
		Tenant:           "t",
		Day:              testDay,
		Model:            "m-fast",
		Attempt:          1,
		Outcome:          accounting.OutcomeOK,
		PromptTokens:     90,
		CompletionTokens: 40,
		UsageSource:      accounting.SourceReported,
		CostNanos:        cost,
		CostSource:       accounting.SourceReported,
		CreatedAt:        r.clk.Now(),
	}
	if got := r.records(); len(got) != 1 || got[0] != want {
		t.Fatalf("usage records = %+v, want [%+v]", got, want)
	}
	if got := m.CostUSD(); got != accounting.NanosToUSD(cost) {
		t.Errorf("CostUSD = %v, want %v", got, accounting.NanosToUSD(cost))
	}
	if got := m.CostSource(); got != accounting.SourceReported {
		t.Errorf("CostSource = %q, want reported", got)
	}
}

func TestMeterSettlesEstimatedWhenUsageMissing(t *testing.T) {
	ten := config.Tenant{Name: "t", DailyBudgetUSD: 1}
	r := newRig(t, config.FailModeClosed, ten)
	m := r.meter(&ten)
	begin(t, m).End(accounting.Settlement{Outcome: accounting.OutcomeOK, DeliveredBytes: 41})

	// Prompt: 400 bytes are 100 tokens. Completion: 41 bytes are 11 tokens.
	const cost = 100*1000 + 11*2000
	r.wantBudget(t, "t", accounting.BudgetState{Spent: cost})
	got := r.records()
	if len(got) != 1 {
		t.Fatalf("%d usage records, want 1", len(got))
	}
	rec := got[0]
	if rec.PromptTokens != 100 || rec.CompletionTokens != 11 || rec.CostNanos != cost {
		t.Errorf("record = %+v, want 100 + 11 tokens costing %d", rec, cost)
	}
	if rec.UsageSource != accounting.SourceEstimated || rec.CostSource != accounting.SourceEstimated {
		t.Errorf("sources = %q / %q, want estimated", rec.UsageSource, rec.CostSource)
	}
	if m.CostSource() != accounting.SourceEstimated {
		t.Errorf("CostSource = %q, want estimated", m.CostSource())
	}
}

func TestMeterFailedAttemptChargesZero(t *testing.T) {
	for _, outcome := range []string{accounting.OutcomeError, accounting.OutcomeCanceled} {
		t.Run(outcome, func(t *testing.T) {
			ten := config.Tenant{Name: "t", DailyBudgetUSD: 1}
			r := newRig(t, config.FailModeClosed, ten)
			m := r.meter(&ten)
			begin(t, m).End(accounting.Settlement{Outcome: outcome})

			r.wantBudget(t, "t", accounting.BudgetState{})
			got := r.records()
			if len(got) != 1 {
				t.Fatalf("%d usage records, want 1", len(got))
			}
			rec := got[0]
			if rec.Outcome != outcome || rec.Partial || rec.PromptTokens != 0 || rec.CompletionTokens != 0 || rec.CostNanos != 0 {
				t.Errorf("record = %+v, want an empty %s attempt", rec, outcome)
			}
			if rec.UsageSource != accounting.SourceEstimated || rec.CostSource != accounting.SourceEstimated {
				t.Errorf("sources = %q / %q, want estimated", rec.UsageSource, rec.CostSource)
			}
			if m.CostUSD() != 0 || m.CostSource() != accounting.SourceEstimated {
				t.Errorf("meter = %v %s, want 0 estimated", m.CostUSD(), m.CostSource())
			}
		})
	}
}

func TestMeterPartialStreamEstimate(t *testing.T) {
	ten := config.Tenant{Name: "t", DailyBudgetUSD: 1}

	t.Run("delivered bytes are estimated", func(t *testing.T) {
		r := newRig(t, config.FailModeClosed, ten)
		m := r.meter(&ten)
		begin(t, m).End(accounting.Settlement{Outcome: accounting.OutcomePartial, DeliveredBytes: 10})

		// 400 prompt bytes are 100 tokens and 10 delivered bytes are 3.
		const cost = 100*1000 + 3*2000
		r.wantBudget(t, "t", accounting.BudgetState{Spent: cost})
		got := r.records()
		if len(got) != 1 || !got[0].Partial || got[0].Outcome != accounting.OutcomePartial {
			t.Fatalf("records = %+v, want one partial record", got)
		}
		if got[0].PromptTokens != 100 || got[0].CompletionTokens != 3 || got[0].CostNanos != cost {
			t.Errorf("record = %+v", got[0])
		}
		if got[0].UsageSource != accounting.SourceEstimated || m.CostSource() != accounting.SourceEstimated {
			t.Errorf("sources = %q / %q, want estimated", got[0].UsageSource, m.CostSource())
		}
	})

	t.Run("reported usage still wins", func(t *testing.T) {
		r := newRig(t, config.FailModeClosed, ten)
		m := r.meter(&ten)
		begin(t, m).End(accounting.Settlement{
			Outcome:        accounting.OutcomePartial,
			Usage:          &provider.Usage{PromptTokens: 80, CompletionTokens: 5},
			DeliveredBytes: 10,
		})
		const cost = 80*1000 + 5*2000
		got := r.records()
		if len(got) != 1 || !got[0].Partial || got[0].CostNanos != cost || got[0].UsageSource != accounting.SourceReported {
			t.Fatalf("records = %+v, want one partial record costing %d from reported usage", got, cost)
		}
	})

	t.Run("a stream closed after output is charged but not partial", func(t *testing.T) {
		r := newRig(t, config.FailModeClosed, ten)
		m := r.meter(&ten)
		begin(t, m).End(accounting.Settlement{Outcome: accounting.OutcomeCanceled, DeliveredBytes: 10})
		got := r.records()
		if len(got) != 1 || got[0].Partial || got[0].CostNanos != 100*1000+3*2000 {
			t.Fatalf("records = %+v, want one charged record that is not partial", got)
		}
	})
}

func TestMeterBudgetExceededOnFirstAttempt(t *testing.T) {
	// 500,000 nano-USD are less than the 600,000 reserved.
	ten := config.Tenant{Name: "t", DailyBudgetUSD: 0.0005}
	r := newRig(t, config.FailModeClosed, ten)
	m := r.meter(&ten)

	c, err := m.Begin(context.Background(), "m-fast")
	if c != nil {
		t.Errorf("Begin returned a charge %v with its error", c)
	}
	var be *accounting.BudgetExceededError
	if !errors.As(err, &be) {
		t.Fatalf("err = %v, want a *BudgetExceededError", err)
	}
	// 22:00 UTC: two hours to the next UTC midnight.
	if be.RetryAfter != 2*time.Hour {
		t.Errorf("RetryAfter = %v, want 2h", be.RetryAfter)
	}
	r.wantBudget(t, "t", accounting.BudgetState{})
	if got := r.records(); len(got) != 0 {
		t.Errorf("%d usage records for an attempt that never ran", len(got))
	}
}

func TestBudgetExceededRetryAfterIsWholeSecondsToMidnight(t *testing.T) {
	ten := config.Tenant{Name: "t", DailyBudgetUSD: 0.0005}
	for _, tc := range []struct {
		name string
		at   time.Time
		want time.Duration
	}{
		{"mid evening", time.Date(2026, 10, 27, 22, 0, 0, 0, time.UTC), 2 * time.Hour},
		{"a fraction of a second is rounded up", time.Date(2026, 10, 27, 23, 59, 59, 400_000_000, time.UTC), time.Second},
		{"the last nanosecond is at least a second", time.Date(2026, 10, 27, 23, 59, 59, 999_999_999, time.UTC), time.Second},
		{"just after midnight", time.Date(2026, 10, 28, 0, 0, 0, 0, time.UTC), 24 * time.Hour},
		{"one and a half seconds", time.Date(2026, 10, 27, 23, 59, 58, 500_000_000, time.UTC), 2 * time.Second},
		{"a local evening is still the UTC day", time.Date(2026, 10, 27, 20, 0, 0, 0, time.FixedZone("west", -5*3600)), 23 * time.Hour},
	} {
		t.Run(tc.name, func(t *testing.T) {
			r := newRig(t, config.FailModeClosed, ten)
			r.clk.t = tc.at
			_, err := r.meter(&ten).Begin(context.Background(), "m-fast")
			var be *accounting.BudgetExceededError
			if !errors.As(err, &be) {
				t.Fatalf("err = %v, want a *BudgetExceededError", err)
			}
			if be.RetryAfter != tc.want {
				t.Errorf("RetryAfter = %v, want %v", be.RetryAfter, tc.want)
			}
		})
	}
}

func TestMeterNeverClipsAnActualCharge(t *testing.T) {
	// The budget holds one reservation; the call then costs more than reserved.
	ten := config.Tenant{Name: "t", DailyBudgetUSD: 0.0007}
	r := newRig(t, config.FailModeClosed, ten)
	m := r.meter(&ten)
	c := begin(t, m)
	c.End(settleOK(900, 0)) // 900,000 nano-USD against a 700,000 budget

	r.wantBudget(t, "t", accounting.BudgetState{Spent: 900_000, Overshoot: 200_000})
	if got := m.CostUSD(); got != accounting.NanosToUSD(900_000) {
		t.Errorf("CostUSD = %v, want the whole 900,000 nano-USD", got)
	}
	// Over its budget, the tenant is refused.
	_, err := r.meter(&ten).Begin(context.Background(), "m-fast")
	var be *accounting.BudgetExceededError
	if !errors.As(err, &be) {
		t.Fatalf("err = %v, want a *BudgetExceededError", err)
	}
}

func TestMeterUnlimitedTenantReservesNothing(t *testing.T) {
	ten := config.Tenant{Name: "t"} // daily_budget_usd 0: unlimited
	r := newRig(t, config.FailModeClosed, ten)
	r.budgets.setDown(true) // an unlimited tenant must not depend on the budget store to start

	m := r.meter(&ten)
	c, err := m.Begin(context.Background(), "m-fast")
	if err != nil {
		t.Fatalf("Begin: %v", err)
	}
	if reserves, _ := r.budgets.counts(); reserves != 0 {
		t.Errorf("Reserve called %d times for an unlimited tenant", reserves)
	}

	r.budgets.setDown(false)
	c.End(settleOK(90, 40))
	// The meter still bills it: spend is recorded and never counts as overshoot.
	r.wantBudget(t, "t", accounting.BudgetState{Spent: 170_000})
	if got := r.records(); len(got) != 1 || got[0].CostNanos != 170_000 {
		t.Fatalf("records = %+v, want one costing 170,000", got)
	}
	if m.CostUSD() != accounting.NanosToUSD(170_000) {
		t.Errorf("CostUSD = %v", m.CostUSD())
	}
}

func TestMeterBackendDown(t *testing.T) {
	ten := config.Tenant{Name: "t", DailyBudgetUSD: 1}

	t.Run("closed mode refuses the attempt", func(t *testing.T) {
		r := newRig(t, config.FailModeClosed, ten)
		r.budgets.setDown(true)
		c, err := r.meter(&ten).Begin(context.Background(), "m-fast")
		if c != nil || !errors.Is(err, accounting.ErrAccountingUnavailable) {
			t.Fatalf("Begin = %v, %v; want ErrAccountingUnavailable", c, err)
		}
		if got := r.records(); len(got) != 0 {
			t.Errorf("%d usage records for an attempt that never ran", len(got))
		}
	})

	t.Run("open mode goes on without a reservation", func(t *testing.T) {
		r := newRig(t, config.FailModeOpen, ten)
		r.budgets.setDown(true)
		m := r.meter(&ten)
		c, err := m.Begin(context.Background(), "m-fast")
		if err != nil || c == nil {
			t.Fatalf("Begin = %v, %v; want a charge", c, err)
		}
		c.End(settleOK(90, 40)) // the failing Settle must not panic or fail anything

		got := r.records()
		if len(got) != 1 || got[0].CostNanos != 170_000 {
			t.Fatalf("records = %+v, want the bill written even though the budget is down", got)
		}
		if m.CostUSD() != accounting.NanosToUSD(170_000) {
			t.Errorf("CostUSD = %v", m.CostUSD())
		}
		// One log line for the reservation and the settlement together: they
		// name the same backend within the same minute.
		lines := r.logLines(t)
		if len(lines) != 1 || lines[0]["backend"] != "budget" {
			t.Errorf("log lines = %v, want one naming the budget", lines)
		}
		if _, settles := r.budgets.counts(); settles != 1 {
			t.Errorf("Settle called %d times, want 1", settles)
		}
		if got := r.budgets.lastSettleReservedArg; got != 0 {
			t.Errorf("settled against a reservation of %d, want 0: none was made", got)
		}
	})

	t.Run("a failing settlement never fails the request", func(t *testing.T) {
		for _, mode := range []string{config.FailModeClosed, config.FailModeOpen} {
			r := newRig(t, mode, ten)
			m := r.meter(&ten)
			c := begin(t, m)
			r.budgets.mu.Lock()
			r.budgets.failSettle = true
			r.budgets.mu.Unlock()
			c.End(settleOK(90, 40))
			if got := r.records(); len(got) != 1 {
				t.Errorf("%s: %d usage records, want the bill written", mode, len(got))
			}
			lines := r.logLines(t)
			if len(lines) != 1 || lines[0]["backend"] != "budget" {
				t.Errorf("%s: log lines = %v, want one naming the budget", mode, lines)
			}
		}
	})

	t.Run("a failing usage write is logged and does not fail", func(t *testing.T) {
		r := newRig(t, config.FailModeClosed, ten)
		r.usage.set(false, errBackend)
		begin(t, r.meter(&ten)).End(settleOK(90, 40))
		r.wantBudget(t, "t", accounting.BudgetState{Spent: 170_000})
		lines := r.logLines(t)
		if len(lines) != 1 || lines[0]["backend"] != "usage" {
			t.Errorf("log lines = %v, want one naming the usage store", lines)
		}
	})
}

func TestMeterSettlesAfterTheRequestContextEnds(t *testing.T) {
	ten := config.Tenant{Name: "t", DailyBudgetUSD: 1}
	r := newRig(t, config.FailModeClosed, ten)
	ctx, cancel := context.WithCancel(context.Background())
	c, err := r.meter(&ten).Begin(ctx, "m-fast")
	if err != nil {
		t.Fatal(err)
	}
	cancel() // the client left, or the deadline passed
	c.End(settleOK(90, 40))

	r.budgets.mu.Lock()
	settleErr := r.budgets.settleCtxErr
	r.budgets.mu.Unlock()
	if settleErr != nil {
		t.Errorf("Settle ran with a done context: %v", settleErr)
	}
	r.wantBudget(t, "t", accounting.BudgetState{Spent: 170_000})
}

func TestMeterChargesTheDayOfTheAttemptStart(t *testing.T) {
	ten := config.Tenant{Name: "t", DailyBudgetUSD: 1}
	r := newRig(t, config.FailModeClosed, ten)
	c := begin(t, r.meter(&ten))
	r.clk.Advance(3 * time.Hour) // the call runs across midnight UTC
	c.End(settleOK(90, 40))

	r.wantBudget(t, "t", accounting.BudgetState{Spent: 170_000})
	tomorrow, _ := r.budgets.BudgetStore.State(context.Background(), accounting.BudgetKey{Tenant: "t", Day: "20261028"})
	if tomorrow != (accounting.BudgetState{}) {
		t.Errorf("the next day's budget = %+v, want untouched", tomorrow)
	}
	if got := r.records(); len(got) != 1 || got[0].Day != testDay {
		t.Errorf("records = %+v, want the day the attempt started", got)
	}
}

// A charge is settled once: a second End must not bill the attempt again.
func TestChargeSettlesOnce(t *testing.T) {
	ten := config.Tenant{Name: "t", DailyBudgetUSD: 1}
	r := newRig(t, config.FailModeClosed, ten)
	m := r.meter(&ten)
	c := begin(t, m)
	c.End(settleOK(90, 40))
	c.End(settleOK(90, 40))

	r.wantBudget(t, "t", accounting.BudgetState{Spent: 170_000})
	if got := r.records(); len(got) != 1 {
		t.Errorf("%d usage records after two Ends, want 1", len(got))
	}
	if got := m.CostUSD(); got != accounting.NanosToUSD(170_000) {
		t.Errorf("CostUSD = %v, want one settlement's", got)
	}
}

// A budget that is not a whole number of nano-USD cannot get past config
// validation. If one reaches the accountant anyway it fails closed whatever
// the fail mode, instead of turning the tenant unlimited.
func TestInvalidBudgetFailsClosed(t *testing.T) {
	bad := config.Tenant{Name: "bad", DailyBudgetUSD: 0.0000000001}
	for _, mode := range []string{config.FailModeClosed, config.FailModeOpen} {
		t.Run(mode, func(t *testing.T) {
			r := newRig(t, mode, bad)
			if _, err := r.meter(&bad).Begin(context.Background(), "m-fast"); !errors.Is(err, accounting.ErrAccountingUnavailable) {
				t.Errorf("Begin err = %v, want ErrAccountingUnavailable", err)
			}
			if _, _, err := r.acct.RemainingNanos(context.Background(), "bad"); !errors.Is(err, accounting.ErrAccountingUnavailable) {
				t.Errorf("RemainingNanos err = %v, want ErrAccountingUnavailable", err)
			}
		})
	}
}

func TestMeterAttemptsAreNumbered(t *testing.T) {
	ten := config.Tenant{Name: "t", DailyBudgetUSD: 1}
	r := newRig(t, config.FailModeClosed, ten)
	m := r.meter(&ten)
	begin(t, m).End(accounting.Settlement{Outcome: accounting.OutcomeError})
	begin(t, m).End(settleOK(90, 40))

	got := r.records()
	if len(got) != 2 || got[0].Attempt != 1 || got[1].Attempt != 2 || got[0].ID == got[1].ID {
		t.Fatalf("records = %+v, want attempts 1 and 2 with distinct ids", got)
	}
}

func TestCostSourceMixed(t *testing.T) {
	ten := config.Tenant{Name: "t", DailyBudgetUSD: 1}

	t.Run("every attempt reported", func(t *testing.T) {
		r := newRig(t, config.FailModeClosed, ten)
		m := r.meter(&ten)
		begin(t, m).End(settleOK(90, 40))
		begin(t, m).End(settleOK(10, 5))
		if m.CostSource() != accounting.SourceReported {
			t.Errorf("CostSource = %q, want reported", m.CostSource())
		}
		// 170,000 + (10 * 1000 + 5 * 2000).
		if got, want := m.CostUSD(), accounting.NanosToUSD(190_000); got != want {
			t.Errorf("CostUSD = %v, want %v (the sum over attempts)", got, want)
		}
	})

	t.Run("a failed attempt makes it estimated", func(t *testing.T) {
		r := newRig(t, config.FailModeClosed, ten)
		m := r.meter(&ten)
		begin(t, m).End(accounting.Settlement{Outcome: accounting.OutcomeError})
		begin(t, m).End(settleOK(90, 40))
		if m.CostSource() != accounting.SourceEstimated {
			t.Errorf("CostSource = %q, want estimated", m.CostSource())
		}
		if got, want := m.CostUSD(), accounting.NanosToUSD(170_000); got != want {
			t.Errorf("CostUSD = %v, want %v", got, want)
		}
	})

	t.Run("one estimated attempt among reported ones", func(t *testing.T) {
		r := newRig(t, config.FailModeClosed, ten)
		m := r.meter(&ten)
		begin(t, m).End(settleOK(90, 40))
		begin(t, m).End(accounting.Settlement{Outcome: accounting.OutcomeOK, DeliveredBytes: 8})
		if m.CostSource() != accounting.SourceEstimated {
			t.Errorf("CostSource = %q, want estimated", m.CostSource())
		}
	})

	t.Run("no attempt was charged", func(t *testing.T) {
		r := newRig(t, config.FailModeClosed, ten)
		m := r.meter(&ten)
		if m.CostUSD() != 0 || m.CostSource() != accounting.SourceEstimated {
			t.Errorf("fresh meter = %v %q, want 0 estimated", m.CostUSD(), m.CostSource())
		}
	})
}

func TestMeterUnknownModelIsFree(t *testing.T) {
	// The engine treats a model without a price as free, and so does the meter;
	// the gateway prices every configured model at startup.
	ten := config.Tenant{Name: "t", DailyBudgetUSD: 1}
	r := newRig(t, config.FailModeClosed, ten)
	c, err := r.meter(&ten).Begin(context.Background(), "m-unpriced")
	if err != nil {
		t.Fatal(err)
	}
	c.End(settleOK(90, 40))
	r.wantBudget(t, "t", accounting.BudgetState{})
}

func TestMeterConcurrentAttempts(t *testing.T) {
	ten := config.Tenant{Name: "t", DailyBudgetUSD: 1}
	r := newRig(t, config.FailModeClosed, ten)
	m := r.meter(&ten)
	var wg sync.WaitGroup
	for range 20 {
		wg.Add(1)
		go func() {
			defer wg.Done()
			c, err := m.Begin(context.Background(), "m-fast")
			if err != nil {
				t.Errorf("Begin: %v", err)
				return
			}
			c.End(settleOK(10, 5))
		}()
	}
	wg.Wait()
	if got, want := m.CostUSD(), accounting.NanosToUSD(20*(10*1000+5*2000)); got != want {
		t.Errorf("CostUSD = %v, want %v", got, want)
	}
	r.wantBudget(t, "t", accounting.BudgetState{Spent: 20 * 20_000})
}

func TestRemainingNanos(t *testing.T) {
	ctx := context.Background()
	limited := config.Tenant{Name: "limited", DailyBudgetUSD: 1}
	free := config.Tenant{Name: "free"}

	t.Run("a budget minus what is spent and reserved", func(t *testing.T) {
		r := newRig(t, config.FailModeClosed, limited, free)
		got, isLimited, err := r.acct.RemainingNanos(ctx, "limited")
		if err != nil || !isLimited || got != 1_000_000_000 {
			t.Fatalf("RemainingNanos = %d, %v, %v; want 1,000,000,000 limited", got, isLimited, err)
		}
		m := r.meter(&limited)
		c := begin(t, m)
		if got, _, _ := r.acct.RemainingNanos(ctx, "limited"); got != 1_000_000_000-reservation {
			t.Errorf("with a reservation: %d, want %d", got, 1_000_000_000-reservation)
		}
		c.End(settleOK(90, 40))
		if got, _, _ := r.acct.RemainingNanos(ctx, "limited"); got != 1_000_000_000-170_000 {
			t.Errorf("after settling: %d, want %d", got, 1_000_000_000-170_000)
		}
	})

	t.Run("it can go negative once a tenant is over budget", func(t *testing.T) {
		tiny := config.Tenant{Name: "tiny", DailyBudgetUSD: 0.0007}
		r := newRig(t, config.FailModeClosed, tiny)
		begin(t, r.meter(&tiny)).End(settleOK(900, 0)) // 900,000 against 700,000
		got, isLimited, err := r.acct.RemainingNanos(ctx, "tiny")
		if err != nil || !isLimited || got != -200_000 {
			t.Fatalf("RemainingNanos = %d, %v, %v; want -200,000 limited", got, isLimited, err)
		}
	})

	t.Run("a tenant without a budget is not limited", func(t *testing.T) {
		r := newRig(t, config.FailModeClosed, limited, free)
		if _, isLimited, err := r.acct.RemainingNanos(ctx, "free"); err != nil || isLimited {
			t.Fatalf("free tenant: limited = %v, err = %v", isLimited, err)
		}
		if _, isLimited, err := r.acct.RemainingNanos(ctx, "nobody"); err != nil || isLimited {
			t.Fatalf("unknown tenant: limited = %v, err = %v", isLimited, err)
		}
		if reserves, _ := r.budgets.counts(); reserves != 0 {
			t.Errorf("Reserve called %d times", reserves)
		}
	})

	t.Run("a down backend fails closed", func(t *testing.T) {
		r := newRig(t, config.FailModeClosed, limited)
		r.budgets.setDown(true)
		_, _, err := r.acct.RemainingNanos(ctx, "limited")
		if !errors.Is(err, accounting.ErrAccountingUnavailable) {
			t.Fatalf("err = %v, want ErrAccountingUnavailable", err)
		}
	})

	t.Run("a down backend in open mode leaves the model eligible", func(t *testing.T) {
		r := newRig(t, config.FailModeOpen, limited)
		r.budgets.setDown(true)
		got, isLimited, err := r.acct.RemainingNanos(ctx, "limited")
		if err != nil || isLimited || got != 0 {
			t.Fatalf("RemainingNanos = %d, %v, %v; want unlimited without an error", got, isLimited, err)
		}
		if lines := r.logLines(t); len(lines) != 1 || lines[0]["backend"] != "budget" {
			t.Errorf("log lines = %v, want one naming the budget", lines)
		}
	})
}

func TestState(t *testing.T) {
	ctx := context.Background()
	ten := config.Tenant{Name: "t", DailyBudgetUSD: 0.0007}
	r := newRig(t, config.FailModeClosed, ten)
	begin(t, r.meter(&ten)).End(settleOK(900, 0)) // 900,000: 200,000 over

	b, u, err := r.acct.State(ctx, "t", testDay)
	if err != nil {
		t.Fatal(err)
	}
	if b != (accounting.BudgetState{Spent: 900_000, Overshoot: 200_000}) {
		t.Errorf("budget = %+v", b)
	}
	if u != (accounting.UsageTotals{Records: 1, CostNanos: 900_000}) {
		t.Errorf("usage = %+v", u)
	}

	b, u, err = r.acct.State(ctx, "nobody", testDay)
	if err != nil || b != (accounting.BudgetState{}) || u != (accounting.UsageTotals{}) {
		t.Errorf("unknown tenant: %+v %+v %v; want zeros", b, u, err)
	}

	r.budgets.setDown(true)
	if _, _, err := r.acct.State(ctx, "t", testDay); err == nil {
		t.Error("State with the budget store down returned no error")
	}
}

func TestDefaultUsageIDs(t *testing.T) {
	price, _ := accounting.NewPrice("m-fast", 1, 2)
	usage := accounting.NewMemoryUsage(10)
	a := accounting.NewAccountant(accounting.AccountantOptions{
		Prices:  map[string]accounting.Price{"m-fast": price},
		Budgets: accounting.NewMemoryBudgets(),
		Limiter: accounting.NewMemoryLimiter(time.Now),
		Usage:   usage,
	})
	ten := config.Tenant{Name: "t"}
	for range 2 {
		c, err := a.NewMeter(&ten, "req", "dec", 40, 10).Begin(context.Background(), "m-fast")
		if err != nil {
			t.Fatal(err)
		}
		c.End(settleOK(1, 1))
	}
	re := regexp.MustCompile(`^usage_[0-9a-f]{24}$`)
	got := usage.Recent()
	if len(got) != 2 || !re.MatchString(got[0].ID) || !re.MatchString(got[1].ID) || got[0].ID == got[1].ID {
		t.Fatalf("ids = %+v, want two distinct usage_ plus 24 hex characters", got)
	}
}

// The accounting package cannot import the router, whose outcome names it
// repeats; this keeps them equal.
func TestOutcomesMatchTheRouter(t *testing.T) {
	for _, tc := range []struct{ got, want string }{
		{accounting.OutcomeOK, router.OutcomeOK},
		{accounting.OutcomeError, router.OutcomeError},
		{accounting.OutcomeCanceled, router.OutcomeCanceled},
		{accounting.OutcomePartial, router.OutcomePartial},
	} {
		if tc.got != tc.want {
			t.Errorf("accounting outcome %q differs from router outcome %q", tc.got, tc.want)
		}
	}
}

// The executor takes a Meter, and the accountant's meter must fit it.
var _ executor.Meter = (*accounting.Meter)(nil)

// Charge and Settlement are the executor's, shared with accounting, so a
// charge from one is a charge for the other.
var (
	_ executor.Charge     = accounting.Charge(nil)
	_ executor.Settlement = accounting.Settlement{}
)
