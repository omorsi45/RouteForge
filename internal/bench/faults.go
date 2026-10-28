package bench

import (
	"context"
	"errors"
	"fmt"
	"strconv"
	"time"

	"github.com/omorsi45/routeforge/internal/accounting"
	"github.com/omorsi45/routeforge/internal/config"
)

// The two tenants of the fault benchmark, as bench/configs/faults.yaml names
// them. Bench has a budget that never binds, and bench-over a small one that
// the over-reporting model exhausts.
const (
	BenchTenant = "bench"
	OverTenant  = "bench-over"
)

// The defaults of the fault benchmark.
const (
	defaultFaultRequests     = 2000
	defaultFaultOverRequests = 200
	defaultFaultConcurrency  = 32
	defaultFaultDeadlineMS   = 2000
)

// overMaxTokens is the output cap of the over tenant's requests. The over
// model reports more completion tokens than this, so every attempt it serves
// settles above what was reserved for it.
const overMaxTokens = 16

// overPrompt is the one prompt of the over tenant. It is short on purpose: the
// reservation of a request is priced from its length in bytes, and a short
// prompt makes the reservation small next to the over-reported charge.
const overPrompt = "Say hello."

// benchStrategies rotate over the bench tenant's requests, so that traffic
// reaches several models and not only the one the quality strategy likes best.
var benchStrategies = []string{"quality", "cheapest", "strongest", "static", "latency"}

// FaultsSetupInfo is what the fault benchmark takes from the gateway config.
type FaultsSetupInfo struct {
	BenchBudgetNanos int64  // the bench tenant's daily budget
	OverBudgetNanos  int64  // the over tenant's daily budget
	OverModel        string // the one model the over tenant may use
}

// FaultsSetup reads the budgets of the two fault tenants and the model of the
// over tenant from cfg. Both tenants need a budget, since two of the checks are
// made against them, and the over tenant needs exactly one allowed model.
func FaultsSetup(cfg *config.Config) (FaultsSetupInfo, error) {
	tenant := func(name string) (*config.Tenant, error) {
		for i := range cfg.Tenants {
			if cfg.Tenants[i].Name == name {
				return &cfg.Tenants[i], nil
			}
		}
		return nil, fmt.Errorf("the config has no tenant %q", name)
	}
	budget := func(t *config.Tenant) (int64, error) {
		if t.DailyBudgetUSD <= 0 {
			return 0, fmt.Errorf("tenant %q needs a daily_budget_usd", t.Name)
		}
		nanos, err := accounting.USDToNanos(t.DailyBudgetUSD)
		if err != nil {
			return 0, fmt.Errorf("tenant %q: %w", t.Name, err)
		}
		return nanos, nil
	}

	var info FaultsSetupInfo
	bench, err := tenant(BenchTenant)
	if err != nil {
		return info, err
	}
	if info.BenchBudgetNanos, err = budget(bench); err != nil {
		return info, err
	}
	over, err := tenant(OverTenant)
	if err != nil {
		return info, err
	}
	if info.OverBudgetNanos, err = budget(over); err != nil {
		return info, err
	}
	if len(over.AllowedModels) != 1 {
		return info, fmt.Errorf("tenant %q must have exactly one model in allowed_models, has %d", OverTenant, len(over.AllowedModels))
	}
	info.OverModel = over.AllowedModels[0]
	return info, nil
}

// FaultsOptions configures RunFaults. A zero count takes its default.
type FaultsOptions struct {
	Bench, Over *Client // the clients of the bench and bench-over tenants
	Admin       *Client // a client with an admin key, which reads the books

	BenchTenant, OverTenant string
	// OverModel is the model the over tenant's requests are pinned to.
	OverModel                         string
	BenchBudgetNanos, OverBudgetNanos int64

	Requests     int // requests of the bench tenant (default 2000)
	OverRequests int // requests of the over tenant (default 200)
	Concurrency  int // requests in flight at once (default 32)
	DeadlineMS   int // sent as X-RouteForge-Deadline-Ms on every request (default 2000)
	Seed         uint64

	// RequestTimeout bounds one request on the client's side (default 30 s).
	RequestTimeout time.Duration
	// Now is the clock that tells the UTC day (default time.Now).
	Now func() time.Time
}

func (o FaultsOptions) withDefaults() FaultsOptions {
	if o.Requests == 0 {
		o.Requests = defaultFaultRequests
	}
	if o.OverRequests == 0 {
		o.OverRequests = defaultFaultOverRequests
	}
	if o.Concurrency == 0 {
		o.Concurrency = defaultFaultConcurrency
	}
	if o.DeadlineMS == 0 {
		o.DeadlineMS = defaultFaultDeadlineMS
	}
	if o.RequestTimeout == 0 {
		o.RequestTimeout = 30 * time.Second
	}
	if o.Now == nil {
		o.Now = time.Now
	}
	return o
}

func (o FaultsOptions) validate() error {
	switch {
	case o.Bench == nil || o.Over == nil || o.Admin == nil:
		return errors.New("fault options need a client for the bench tenant, the over tenant and the admin key")
	case o.BenchTenant == "" || o.OverTenant == "":
		return errors.New("fault options need the names of both tenants")
	case o.OverModel == "":
		return errors.New("fault options need the model of the over tenant")
	case o.BenchBudgetNanos < 0 || o.OverBudgetNanos < 0:
		return errors.New("a budget cannot be negative")
	case o.Requests < 0 || o.OverRequests < 0 || o.Concurrency < 0 || o.DeadlineMS < 0 || o.RequestTimeout < 0:
		return errors.New("counts, concurrency and deadlines cannot be negative")
	}
	return nil
}

// Params describes the run for its result file.
func (o FaultsOptions) Params() map[string]any {
	o = o.withDefaults()
	return map[string]any{
		"requests":           o.Requests,
		"over_requests":      o.OverRequests,
		"concurrency":        o.Concurrency,
		"deadline_ms":        o.DeadlineMS,
		"over_model":         o.OverModel,
		"over_max_tokens":    overMaxTokens,
		"bench_strategies":   benchStrategies,
		"bench_budget_nanos": o.BenchBudgetNanos,
		"over_budget_nanos":  o.OverBudgetNanos,
		"seed":               o.Seed,
	}
}

// PhaseResult is what the requests of one tenant came to.
type PhaseResult struct {
	Tenant         string `json:"tenant"`
	Requests       int    `json:"requests"`
	StreamRequests int    `json:"stream_requests"`
	// Succeeded counts the requests that returned 200 and, for a stream, ended
	// with [DONE].
	Succeeded   int     `json:"succeeded"`
	SuccessRate float64 `json:"success_rate"`
	// Statuses counts the responses by HTTP status. A request that got none, a
	// connection failure, is in ErrorCodes only.
	Statuses   map[string]int `json:"statuses"`
	ErrorCodes map[string]int `json:"error_codes"`
	// PartialStreams counts the streams that ended with an error event after
	// their first byte.
	PartialStreams int `json:"partial_streams"`

	// The next fields come from the stored decisions of the requests that have
	// one. Retries counts the attempts after the first, Failovers the attempts
	// that went to another model than the one before, and AttemptErrorKinds the
	// failed attempts by the kind of error.
	DecisionsFetched    int            `json:"decisions_fetched"`
	DecisionsMissing    int            `json:"decisions_missing"`
	Retries             int            `json:"retries"`
	Failovers           int            `json:"failovers"`
	RequestsWithRetries int            `json:"requests_with_retries"`
	AttemptErrorKinds   map[string]int `json:"attempt_error_kinds"`

	ElapsedS float64 `json:"elapsed_s"`
}

// TenantAccounting is the books of a tenant after the run, with its budget.
type TenantAccounting struct {
	AccountingState
	BudgetNanos int64 `json:"budget_nanos"`
}

// Invariants are the accounting checks of spec 11, made on the books the
// gateway reports after the run. InvariantsHold is true when all of them are.
type Invariants struct {
	// SpentEqualsUsage (a): each tenant's spend equals the sum of its usage
	// records.
	SpentEqualsUsage bool `json:"a_spent_equals_usage"`
	// ReservedReturnsToZero (b): no reservation is left open.
	ReservedReturnsToZero bool `json:"b_reserved_returns_to_zero"`
	// SpendWithinBudget (c): the bench tenant spent no more than its budget.
	SpendWithinBudget bool `json:"c_spend_within_budget"`
	// OvershootEqualsExcess (d): the over tenant's overshoot is what it spent
	// beyond its budget, and OvershootPositive says it did spend beyond it: the
	// fault fired. The two together make (d).
	OvershootEqualsExcess bool `json:"d_overshoot_equals_excess"`
	OvershootPositive     bool `json:"d_overshoot_positive"`
	InvariantsHold        bool `json:"invariants_hold"`
}

// checkInvariants judges the books of the two tenants against their budgets.
func checkInvariants(bench, over AccountingState, benchBudget, overBudget int64) Invariants {
	inv := Invariants{
		SpentEqualsUsage:      bench.SpentNanos == bench.UsageCostNanos && over.SpentNanos == over.UsageCostNanos,
		ReservedReturnsToZero: bench.ReservedNanos == 0 && over.ReservedNanos == 0,
		SpendWithinBudget:     bench.SpentNanos <= benchBudget,
		OvershootEqualsExcess: over.OvershootNanos == max(0, over.SpentNanos-overBudget),
		OvershootPositive:     over.OvershootNanos > 0,
	}
	inv.InvariantsHold = inv.SpentEqualsUsage && inv.ReservedReturnsToZero && inv.SpendWithinBudget &&
		inv.OvershootEqualsExcess && inv.OvershootPositive
	return inv
}

// FaultsResults is what RunFaults measured.
type FaultsResults struct {
	// SuccessRate is the bench tenant's.
	SuccessRate float64                     `json:"success_rate"`
	Bench       PhaseResult                 `json:"bench"`
	BenchOver   PhaseResult                 `json:"bench_over"`
	Accounting  map[string]TenantAccounting `json:"accounting"` // by tenant name
	Invariants  Invariants                  `json:"invariants"`
}

// RunFaults sends traffic through a gateway whose providers fail, time out, cut
// streams short and leave usage out, then reads the tenants' books through the
// admin API and checks the accounting invariants. The bench tenant sends
// o.Requests requests to the router, half of them streaming, with the
// strategies in turn. The over tenant, after it, sends o.OverRequests requests
// pinned to the model that reports more tokens than it can have reserved. Every
// request carries the deadline.
//
// Failed requests are results, counted by status and error code; so are
// violated invariants. An error ends the run: a context that ends, a decision
// or the books that cannot be read, or a run that crossed UTC midnight, whose
// day's books would mix two runs.
func RunFaults(ctx context.Context, o FaultsOptions) (FaultsResults, error) {
	o = o.withDefaults()
	if err := o.validate(); err != nil {
		return FaultsResults{}, err
	}
	for _, c := range []*Client{o.Bench, o.Over, o.Admin} {
		defer c.Close() // the gateway is stopped next, and waits for idle connections to go
	}
	day := accounting.DayKey(o.Now())

	prompts := SyntheticPrompts(o.Requests, o.Seed)
	deadline := strconv.Itoa(o.DeadlineMS)
	bench, err := runPhase(ctx, o, phase{
		tenant: o.BenchTenant, client: o.Bench, requests: o.Requests,
		build: func(i int) ChatRequest {
			strategy := benchStrategies[i%len(benchStrategies)]
			h := map[string]string{"X-RouteForge-Deadline-Ms": deadline, "X-RouteForge-Strategy": strategy}
			if strategy == "quality" {
				h["X-RouteForge-Profile"] = "balanced"
			}
			return ChatRequest{Prompt: prompts[i], Stream: i%2 == 1, Headers: h}
		},
	})
	if err != nil {
		return FaultsResults{}, err
	}
	over, err := runPhase(ctx, o, phase{
		tenant: o.OverTenant, client: o.Over, requests: o.OverRequests,
		build: func(i int) ChatRequest {
			return ChatRequest{
				Prompt: overPrompt, Model: o.OverModel, Stream: i%2 == 1, MaxTokens: overMaxTokens,
				Headers: map[string]string{"X-RouteForge-Deadline-Ms": deadline},
			}
		},
	})
	if err != nil {
		return FaultsResults{}, err
	}

	if end := accounting.DayKey(o.Now()); end != day {
		return FaultsResults{}, fmt.Errorf("the run crossed UTC midnight (it began on %s and ended on %s): "+
			"the books of one day would mix two runs, so run it again", day, end)
	}
	benchBooks, err := o.Admin.Accounting(ctx, o.BenchTenant, day)
	if err != nil {
		return FaultsResults{}, err
	}
	overBooks, err := o.Admin.Accounting(ctx, o.OverTenant, day)
	if err != nil {
		return FaultsResults{}, err
	}
	return FaultsResults{
		SuccessRate: bench.SuccessRate,
		Bench:       bench,
		BenchOver:   over,
		Accounting: map[string]TenantAccounting{
			o.BenchTenant: {AccountingState: benchBooks, BudgetNanos: o.BenchBudgetNanos},
			o.OverTenant:  {AccountingState: overBooks, BudgetNanos: o.OverBudgetNanos},
		},
		Invariants: checkInvariants(benchBooks, overBooks, o.BenchBudgetNanos, o.OverBudgetNanos),
	}, nil
}

// phase is the traffic of one tenant.
type phase struct {
	tenant   string
	client   *Client
	requests int
	build    func(i int) ChatRequest
}

// faultRecord is what one request came to.
type faultRecord struct {
	stream     bool
	status     int    // 0 when no response came
	errCode    string // why it failed, empty if it succeeded
	partial    bool
	decisionID string

	haveDecision bool
	missing      bool
	attempts     []AttemptRecord
}

// runPhase sends the requests of one tenant, reads back their decisions and
// counts what happened.
func runPhase(ctx context.Context, o FaultsOptions, p phase) (PhaseResult, error) {
	records := make([]faultRecord, p.requests)
	start := time.Now()
	err := forEach(ctx, p.requests, o.Concurrency, func(ctx context.Context, i int) error {
		req := p.build(i)
		rctx, cancel := context.WithTimeout(ctx, o.RequestTimeout)
		defer cancel()
		out, err := p.client.Do(rctx, req)
		rec := faultRecord{stream: req.Stream, status: out.Status, partial: out.Partial, decisionID: out.DecisionID}
		switch {
		case errors.Is(err, context.DeadlineExceeded) && ctx.Err() == nil:
			rec.errCode = "client_timeout"
		case err != nil:
			rec.errCode = "transport_error"
		default:
			rec.errCode = out.ErrorCode
		}
		records[i] = rec
		return nil
	})
	elapsed := time.Since(start)
	if err != nil {
		return PhaseResult{}, fmt.Errorf("%s requests: %w", p.tenant, err)
	}

	var ids []int
	for i, r := range records {
		if r.decisionID != "" {
			ids = append(ids, i)
		}
	}
	err = forEach(ctx, len(ids), 16, func(ctx context.Context, k int) error {
		r := &records[ids[k]]
		d, err := p.client.Decision(ctx, r.decisionID)
		if errors.Is(err, ErrDecisionNotFound) {
			r.missing = true
			return nil
		}
		if err != nil {
			return err
		}
		r.haveDecision, r.attempts = true, d.Attempts
		return nil
	})
	if err != nil {
		return PhaseResult{}, fmt.Errorf("%s decisions: %w", p.tenant, err)
	}
	return summarizePhase(p.tenant, records, elapsed), nil
}

// summarizePhase counts the records of one tenant.
func summarizePhase(tenant string, records []faultRecord, elapsed time.Duration) PhaseResult {
	res := PhaseResult{
		Tenant:            tenant,
		Requests:          len(records),
		Statuses:          map[string]int{},
		ErrorCodes:        map[string]int{},
		AttemptErrorKinds: map[string]int{},
		ElapsedS:          elapsed.Seconds(),
	}
	for _, r := range records {
		if r.stream {
			res.StreamRequests++
		}
		if r.status != 0 {
			res.Statuses[strconv.Itoa(r.status)]++
		}
		if r.errCode == "" {
			res.Succeeded++
		} else {
			res.ErrorCodes[r.errCode]++
		}
		if r.partial {
			res.PartialStreams++
		}
		if r.missing {
			res.DecisionsMissing++
		}
		if !r.haveDecision {
			continue
		}
		res.DecisionsFetched++
		if extra := len(r.attempts) - 1; extra > 0 {
			res.Retries += extra
			res.RequestsWithRetries++
		}
		for i, a := range r.attempts {
			if a.ErrorKind != "" {
				res.AttemptErrorKinds[a.ErrorKind]++
			}
			if i > 0 && a.Model != r.attempts[i-1].Model {
				res.Failovers++
			}
		}
	}
	if res.Requests > 0 {
		res.SuccessRate = float64(res.Succeeded) / float64(res.Requests)
	}
	return res
}
