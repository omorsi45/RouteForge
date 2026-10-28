package accounting

import (
	"context"
	"crypto/rand"
	"encoding/hex"
	"errors"
	"fmt"
	"log/slog"
	"math"
	"slices"
	"sync"
	"time"

	"github.com/omorsi45/routeforge/internal/config"
	"github.com/omorsi45/routeforge/internal/provider"
	"github.com/omorsi45/routeforge/internal/telemetry"
)

// Attempt outcomes, repeated from the router package, which imports this one
// and so cannot be imported here; a test keeps them equal.
const (
	OutcomeOK       = "ok"
	OutcomeError    = "error"
	OutcomeCanceled = "canceled"
	OutcomePartial  = "partial"
)

// Names of the backends in logs and errors.
const (
	backendBudget  = "budget"
	backendLimiter = "limiter"
	backendUsage   = "usage"
)

// settleTimeout bounds the settlement of one attempt. It runs on the real
// clock because backends do real I/O, and it replaces the request context,
// which is often done by then.
const settleTimeout = 5 * time.Second

// logInterval is how often a failing backend is named in the log.
const logInterval = time.Minute

// RateLimitedError means a tenant is over its rate limit.
type RateLimitedError struct {
	Kind       string // KindRPM or KindTPM
	RetryAfter time.Duration
}

func (e *RateLimitedError) Error() string {
	return fmt.Sprintf("rate limit exceeded (%s), retry after %s", e.Kind, e.RetryAfter)
}

// BudgetExceededError means a tenant cannot reserve what an attempt needs from
// today's budget.
type BudgetExceededError struct {
	// RetryAfter is the time to the next UTC midnight, when budgets reset:
	// whole seconds, at least one.
	RetryAfter time.Duration
}

func (e *BudgetExceededError) Error() string {
	return fmt.Sprintf("daily budget exceeded, resets in %s", e.RetryAfter)
}

// ErrAccountingUnavailable means a budget, limiter or usage backend cannot be
// reached, or is too full to take more, and the fail mode is closed.
var ErrAccountingUnavailable = errors.New("accounting unavailable")

// Charge is one attempt's open charge, returned by Meter.Begin.
//
// Charge and Settlement belong to this package although the executor is their
// caller: the executor imports the router, the router imports this package, and
// so this package cannot import the executor. The executor aliases both.
type Charge interface {
	// End settles the attempt: it returns the reservation, charges the actual
	// cost and writes the usage record. It does not fail; backend errors are
	// logged. The executor calls it exactly once for every charge that Begin
	// returned, whatever happened, a panic included.
	End(s Settlement)
}

// Settlement says what an attempt did.
type Settlement struct {
	// Outcome is an Outcome* value; a stream that failed after delivering
	// output is OutcomePartial.
	Outcome string
	// Usage is what the provider reported, or nil. A stream reports the last
	// usage it sent.
	Usage *provider.Usage
	// DeliveredBytes is the size of the output handed to the caller: the
	// content of a response, or the sum of the deltas a stream returned.
	DeliveredBytes int
}

// AccountantOptions configures an Accountant.
type AccountantOptions struct {
	// Tenants are the configured tenants. RemainingNanos looks a tenant's
	// budget up here by name; Admit and NewMeter use the tenant they are given.
	Tenants []config.Tenant
	// Prices holds every configured model's price, by configured name. A model
	// without one is free.
	Prices  map[string]Price
	Budgets BudgetStore
	Limiter RateLimiter
	Usage   UsageStore
	// FailMode is config.FailModeOpen to let requests through when a backend
	// fails; any other value, the empty one included, fails closed.
	FailMode string
	Now      func() time.Time   // nil: time.Now
	Logger   *slog.Logger       // nil: logs nothing
	NewID    func() string      // usage record ids; nil: "usage_" and 24 hex characters
	Metrics  *telemetry.Metrics // nil: records no metrics
}

// Accountant admits requests and meters their attempts: it enforces rate
// limits, reserves budget before each upstream call and settles the real cost
// after it. It is safe for concurrent use.
type Accountant struct {
	tenants  map[string]*config.Tenant
	prices   map[string]Price
	budgets  BudgetStore
	limiter  RateLimiter
	usage    UsageStore
	failOpen bool
	now      func() time.Time
	logger   *slog.Logger
	newID    func() string
	metrics  *telemetry.Metrics

	logMu   sync.Mutex
	lastLog map[string]time.Time // when each backend was last named in the log
}

// NewAccountant returns an Accountant over the options. The stores are
// required.
func NewAccountant(o AccountantOptions) *Accountant {
	tenants := slices.Clone(o.Tenants)
	a := &Accountant{
		tenants:  make(map[string]*config.Tenant, len(tenants)),
		prices:   o.Prices,
		budgets:  o.Budgets,
		limiter:  o.Limiter,
		usage:    o.Usage,
		failOpen: o.FailMode == config.FailModeOpen,
		now:      o.Now,
		logger:   o.Logger,
		newID:    o.NewID,
		metrics:  o.Metrics,
		lastLog:  make(map[string]time.Time),
	}
	for i := range tenants {
		a.tenants[tenants[i].Name] = &tenants[i]
	}
	if a.now == nil {
		a.now = time.Now
	}
	if a.logger == nil {
		a.logger = slog.New(slog.DiscardHandler)
	}
	if a.newID == nil {
		a.newID = newUsageID
	}
	return a
}

// newUsageID returns "usage_" followed by 24 lowercase hex characters from
// crypto/rand.
func newUsageID() string {
	var b [12]byte
	if _, err := rand.Read(b[:]); err != nil {
		panic("accounting: crypto/rand failed: " + err.Error())
	}
	return "usage_" + hex.EncodeToString(b[:])
}

// logBackend names a failing backend in the log, at most once a minute per
// backend so that an outage does not flood it.
func (a *Accountant) logBackend(ctx context.Context, backend string, err error) {
	now := a.now()
	a.logMu.Lock()
	last, seen := a.lastLog[backend]
	if seen && now.Sub(last) < logInterval {
		a.logMu.Unlock()
		return
	}
	a.lastLog[backend] = now
	a.logMu.Unlock()

	mode := config.FailModeClosed
	if a.failOpen {
		mode = config.FailModeOpen
	}
	a.logger.ErrorContext(ctx, "accounting backend failed", "backend", backend, "fail_mode", mode, "error", err)
}

// backendFailure handles err from the named backend: it logs it, then in closed
// mode returns an error wrapping ErrAccountingUnavailable and in open mode
// nil, which tells the caller to go on without that backend.
func (a *Accountant) backendFailure(ctx context.Context, backend string, err error) error {
	a.logBackend(ctx, backend, err)
	if a.failOpen {
		return nil
	}
	return fmt.Errorf("%w: %s: %w", ErrAccountingUnavailable, backend, err)
}

// budgetLimit returns a tenant's daily budget in nano-USD. limited is false for
// a tenant without one. A budget that is not exact cannot happen after
// config.Validate; it fails closed whatever the fail mode rather than make the
// tenant unlimited.
func budgetLimit(t *config.Tenant) (limit int64, limited bool, err error) {
	if t.DailyBudgetUSD == 0 {
		return 0, false, nil
	}
	limit, err = USDToNanos(t.DailyBudgetUSD)
	if err != nil {
		return 0, false, fmt.Errorf("%w: tenant %q daily_budget_usd: %w", ErrAccountingUnavailable, t.Name, err)
	}
	return limit, true, nil
}

// Admit checks the tenant's rate limits for a request of promptBytes bytes
// that may produce up to maxTokens tokens: one request against RPM, then
// EstimateTokens(promptBytes)+maxTokens tokens against TPM, each only when the
// tenant sets it. It returns a *RateLimitedError, or an error wrapping
// ErrAccountingUnavailable when the usage store is full or, in closed mode, a
// backend fails. Call it once the request is known to be well formed and
// before routing.
func (a *Accountant) Admit(ctx context.Context, t *config.Tenant, promptBytes, maxTokens int) error {
	if a.usage.Full() {
		err := errors.New("usage queue is full")
		a.logBackend(ctx, backendUsage, err)
		return fmt.Errorf("%w: %s: %w", ErrAccountingUnavailable, backendUsage, err)
	}
	if t.RPM > 0 {
		if err := a.allow(ctx, t.Name, KindRPM, 1, int64(t.RPM)); err != nil {
			return err
		}
	}
	if t.TPM > 0 {
		tokens := int64(EstimateTokens(promptBytes) + maxTokens)
		if err := a.allow(ctx, t.Name, KindTPM, tokens, int64(t.TPM)); err != nil {
			return err
		}
	}
	return nil
}

func (a *Accountant) allow(ctx context.Context, tenant, kind string, n, perMinute int64) error {
	d, err := a.limiter.Allow(ctx, tenant, kind, n, perMinute)
	if err != nil {
		return a.backendFailure(ctx, backendLimiter, err)
	}
	if !d.OK {
		a.metrics.RateLimited(kind)
		return &RateLimitedError{Kind: kind, RetryAfter: d.RetryAfter}
	}
	return nil
}

// RemainingNanos is what the tenant may still reserve today: its budget minus
// what is spent and reserved, which is negative once it is over. limited is
// false for a tenant without a budget and for one this accountant does not
// know. It feeds router.Signals.BudgetRemainingNanos. A budget backend that
// cannot answer is an error wrapping ErrAccountingUnavailable in closed mode,
// so routing fails closed, and "no budget" in open mode.
func (a *Accountant) RemainingNanos(ctx context.Context, tenant string) (remaining int64, limited bool, err error) {
	t, ok := a.tenants[tenant]
	if !ok {
		return 0, false, nil
	}
	limit, limited, err := budgetLimit(t)
	if err != nil || !limited {
		return 0, false, err
	}
	st, err := a.budgets.State(ctx, BudgetKey{Tenant: tenant, Day: DayKey(a.now())})
	if err != nil {
		if err = a.backendFailure(ctx, backendBudget, err); err != nil {
			return 0, false, err
		}
		return 0, false, nil
	}
	return limit - st.Spent - st.Reserved, true, nil
}

// State returns what a tenant has spent, reserved and overshot on a day (a
// DayKey) and what its usage records add up to.
func (a *Accountant) State(ctx context.Context, tenant, day string) (BudgetState, UsageTotals, error) {
	b, err := a.budgets.State(ctx, BudgetKey{Tenant: tenant, Day: day})
	if err != nil {
		return BudgetState{}, UsageTotals{}, fmt.Errorf("budget state of %q on %s: %w", tenant, day, err)
	}
	u, err := a.usage.Totals(ctx, tenant, day)
	if err != nil {
		return BudgetState{}, UsageTotals{}, fmt.Errorf("usage totals of %q on %s: %w", tenant, day, err)
	}
	return b, u, nil
}

// Meter meters the attempts of one request. It implements executor.Meter.
type Meter struct {
	a                      *Accountant
	t                      *config.Tenant
	requestID, decisionID  string
	promptBytes, maxTokens int

	mu        sync.Mutex
	attempts  int   // attempts begun
	settled   int   // attempts settled
	reported  int   // settled attempts whose cost came from reported usage
	costNanos int64 // the settled cost of all attempts
}

// NewMeter returns the meter of one request of tenant t. promptBytes and
// maxTokens are the sizes Admit saw: the reservation of every attempt is priced
// from them.
func (a *Accountant) NewMeter(t *config.Tenant, requestID, decisionID string, promptBytes, maxTokens int) *Meter {
	return &Meter{a: a, t: t, requestID: requestID, decisionID: decisionID, promptBytes: promptBytes, maxTokens: maxTokens}
}

// Begin reserves what an attempt on the named model may cost, for a tenant with
// a budget, and returns the charge to settle when it ends. It returns a
// *BudgetExceededError when the budget cannot hold the reservation and an error
// wrapping ErrAccountingUnavailable when, in closed mode, the budget backend
// fails; in open mode the attempt goes on without a reservation. A tenant
// without a budget reserves nothing.
func (m *Meter) Begin(ctx context.Context, model string) (Charge, error) {
	a := m.a
	limit, limited, err := budgetLimit(m.t)
	if err != nil {
		return nil, err
	}
	price := a.prices[model]
	key := BudgetKey{Tenant: m.t.Name, Day: DayKey(a.now())}
	var held int64
	if limited {
		reserve := ReservationNanos(price, m.promptBytes, m.maxTokens)
		ok, err := a.budgets.Reserve(ctx, key, reserve, limit)
		switch {
		case err != nil:
			if err := a.backendFailure(ctx, backendBudget, err); err != nil {
				return nil, err
			}
		case !ok:
			a.metrics.BudgetRejected()
			return nil, &BudgetExceededError{RetryAfter: untilNextDay(a.now())}
		default:
			held = reserve
		}
	} else {
		limit = math.MaxInt64 // spend is recorded, and can never overshoot
	}

	m.mu.Lock()
	m.attempts++
	attempt := m.attempts
	m.mu.Unlock()
	return &charge{
		m:        m,
		ctx:      context.WithoutCancel(ctx),
		model:    model,
		attempt:  attempt,
		price:    price,
		key:      key,
		limit:    limit,
		reserved: held,
	}, nil
}

// untilNextDay is the time from now to the next UTC midnight, in whole seconds,
// rounded up, and at least one.
func untilNextDay(now time.Time) time.Duration {
	now = now.UTC()
	next := time.Date(now.Year(), now.Month(), now.Day()+1, 0, 0, 0, 0, time.UTC)
	secs := (next.Sub(now) + time.Second - 1) / time.Second
	return max(secs, 1) * time.Second
}

// CostUSD is what the request has cost so far, summed over its settled
// attempts, in USD.
func (m *Meter) CostUSD() float64 {
	m.mu.Lock()
	defer m.mu.Unlock()
	return NanosToUSD(m.costNanos)
}

// CostSource is SourceReported when every settled attempt was priced from
// usage the provider reported, and SourceEstimated otherwise. An attempt that
// failed before any output counts as estimated: its zero cost is an assumption.
func (m *Meter) CostSource() string {
	m.mu.Lock()
	defer m.mu.Unlock()
	if m.settled > 0 && m.reported == m.settled {
		return SourceReported
	}
	return SourceEstimated
}

func (m *Meter) add(cost int64, source string) {
	m.mu.Lock()
	defer m.mu.Unlock()
	m.settled++
	m.costNanos += cost
	if source == SourceReported {
		m.reported++
	}
}

// charge is the open charge of one attempt.
type charge struct {
	m        *Meter
	ctx      context.Context // the request's values without its cancellation
	model    string
	attempt  int
	price    Price
	key      BudgetKey // the day is the one the attempt started on
	limit    int64     // math.MaxInt64 for a tenant without a budget
	reserved int64     // 0 when nothing was reserved

	once sync.Once
}

// End implements Charge.
func (c *charge) End(s Settlement) {
	c.once.Do(func() { c.settle(s) })
}

func (c *charge) settle(s Settlement) {
	a := c.m.a
	prompt, completion, source := billedTokens(s, c.m.promptBytes)
	cost := CostNanos(c.price, prompt, completion)
	c.m.add(cost, source)

	// The request context is often done by now (the client left or the deadline
	// passed), but the money was spent all the same.
	ctx, cancel := context.WithTimeout(c.ctx, settleTimeout)
	defer cancel()
	overshoot, err := a.budgets.Settle(ctx, c.key, c.reserved, cost, c.limit)
	if err != nil {
		a.logBackend(ctx, backendBudget, err)
	}
	// The store keeps the running overshoot; State reports it. The metric adds
	// what this settlement contributed, so its total follows the store's.
	if err == nil && overshoot > 0 {
		a.metrics.BudgetOvershoot(c.m.t.Name, NanosToUSD(overshoot))
	}
	a.metrics.Cost(c.model, source, NanosToUSD(cost))
	a.metrics.Tokens(c.model, source, prompt, completion)
	rec := UsageRecord{
		ID:               a.newID(),
		RequestID:        c.m.requestID,
		DecisionID:       c.m.decisionID,
		Tenant:           c.m.t.Name,
		Day:              c.key.Day,
		Model:            c.model,
		Attempt:          c.attempt,
		Outcome:          s.Outcome,
		Partial:          s.Outcome == OutcomePartial,
		PromptTokens:     prompt,
		CompletionTokens: completion,
		UsageSource:      source,
		CostNanos:        cost,
		CostSource:       source,
		CreatedAt:        a.now(),
	}
	if err := a.usage.Write(ctx, rec); err != nil {
		a.logBackend(ctx, backendUsage, err)
	}
}

// billedTokens decides what an attempt is billed for: the provider's usage
// whenever it sent one; else, for an attempt that delivered output, an
// estimate from the prompt and the delivered bytes; else nothing. A call that
// fails before delivering output is billed zero tokens and its cost source is
// estimated, because the provider may have charged for it anyway.
func billedTokens(s Settlement, promptBytes int) (prompt, completion int, source string) {
	switch {
	case s.Usage != nil:
		return s.Usage.PromptTokens, s.Usage.CompletionTokens, SourceReported
	case s.Outcome == OutcomeOK || s.DeliveredBytes > 0:
		return EstimateTokens(promptBytes), EstimateTokens(s.DeliveredBytes), SourceEstimated
	default:
		return 0, 0, SourceEstimated
	}
}
