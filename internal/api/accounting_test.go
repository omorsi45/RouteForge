package api

import (
	"context"
	"encoding/json"
	"errors"
	"log/slog"
	"net/http"
	"net/http/httptest"
	"strconv"
	"strings"
	"sync/atomic"
	"testing"
	"time"

	"github.com/omorsi45/routeforge/internal/accounting"
	"github.com/omorsi45/routeforge/internal/auth"
	"github.com/omorsi45/routeforge/internal/config"
	"github.com/omorsi45/routeforge/internal/executor"
	"github.com/omorsi45/routeforge/internal/policy"
	"github.com/omorsi45/routeforge/internal/provider"
	"github.com/omorsi45/routeforge/internal/reliability"
	"github.com/omorsi45/routeforge/internal/router"
)

var acctNow = time.Date(2026, 10, 27, 22, 0, 0, 0, time.UTC)

const acctDay = "20261027"

// m-fast and m-strong cost 1000 nano-USD per input token and 2000 per output
// token.
const (
	acctInNanos  = 1000
	acctOutNanos = 2000
)

var errAcctDown = errors.New("accounting backend down")

// spyBudgets counts reservations and can be taken down.
type spyBudgets struct {
	accounting.BudgetStore
	down     atomic.Bool
	reserves atomic.Int64
}

func (b *spyBudgets) Reserve(ctx context.Context, k accounting.BudgetKey, amount, limit int64) (bool, error) {
	b.reserves.Add(1)
	if b.down.Load() {
		return false, errAcctDown
	}
	return b.BudgetStore.Reserve(ctx, k, amount, limit)
}

func (b *spyBudgets) Settle(ctx context.Context, k accounting.BudgetKey, reserved, actual, limit int64) (int64, error) {
	if b.down.Load() {
		return 0, errAcctDown
	}
	return b.BudgetStore.Settle(ctx, k, reserved, actual, limit)
}

func (b *spyBudgets) State(ctx context.Context, k accounting.BudgetKey) (accounting.BudgetState, error) {
	if b.down.Load() {
		return accounting.BudgetState{}, errAcctDown
	}
	return b.BudgetStore.State(ctx, k)
}

// brokenLimiter always fails.
type brokenLimiter struct{ accounting.RateLimiter }

func (brokenLimiter) Allow(context.Context, string, string, int64, int64) (accounting.RateDecision, error) {
	return accounting.RateDecision{}, errAcctDown
}

// fullUsage is a usage store that refuses more work.
type fullUsage struct{ *accounting.MemoryUsage }

func (fullUsage) Full() bool { return true }

// acctEnv is a server with an accountant over memory stores. Tweaks run before
// the accountant and the server are built.
type acctEnv struct {
	cfg      *config.Config
	prov     *stubProvider
	budgets  *spyBudgets
	limiter  accounting.RateLimiter
	usage    *accounting.MemoryUsage
	usageSt  accounting.UsageStore
	mode     string
	prices   map[string]accounting.Price
	acct     *accounting.Accountant
	opts     []func(*Options)
	useRoute bool // route with the engine, whose budget signal is the accountant's
	srv      *Server
}

func newAcctEnv(t *testing.T, tweaks ...func(*acctEnv)) *acctEnv {
	t.Helper()
	e := &acctEnv{
		cfg:     testConfig(),
		prov:    &stubProvider{complete: replyWith("ok")},
		budgets: &spyBudgets{BudgetStore: accounting.NewMemoryBudgets()},
		usage:   accounting.NewMemoryUsage(100),
		mode:    config.FailModeClosed,
		prices:  map[string]accounting.Price{},
	}
	for _, name := range []string{"m-fast", "m-strong"} {
		p, err := accounting.NewPrice(name, 1, 2)
		if err != nil {
			t.Fatal(err)
		}
		e.prices[name] = p
	}
	e.usageSt = e.usage
	e.limiter = accounting.NewMemoryLimiter(func() time.Time { return acctNow })
	for _, tw := range tweaks {
		tw(e)
	}
	e.acct = accounting.NewAccountant(accounting.AccountantOptions{
		Tenants:  e.cfg.Tenants,
		Prices:   e.prices,
		Budgets:  e.budgets,
		Limiter:  e.limiter,
		Usage:    e.usageSt,
		FailMode: e.mode,
		Now:      func() time.Time { return acctNow },
		Logger:   slog.New(slog.DiscardHandler),
	})
	opts := []func(*Options){func(o *Options) {
		o.Accountant = e.acct
		o.Now = func() time.Time { return acctNow }
	}}
	if e.useRoute {
		opts = append(opts, func(o *Options) {
			o.Router = router.NewEngine(router.EngineOptions{
				Models:  e.cfg.Models,
				Prices:  e.prices,
				Signals: router.Signals{BudgetRemainingNanos: e.acct.RemainingNanos},
			})
		})
	}
	opts = append(opts, e.opts...)
	e.srv = newServer(t, e.cfg, map[string]provider.Provider{"p-fast": e.prov, "p-strong": e.prov}, opts...)
	return e
}

func (e *acctEnv) post(t *testing.T, body string, hdr map[string]string) *httptest.ResponseRecorder {
	t.Helper()
	return do(t, e.srv.Handler(), http.MethodPost, "/v1/chat/completions", keyA, body, hdr)
}

func (e *acctEnv) budget(t *testing.T, tenant string) accounting.BudgetState {
	t.Helper()
	st, err := e.budgets.BudgetStore.State(context.Background(), accounting.BudgetKey{Tenant: tenant, Day: acctDay})
	if err != nil {
		t.Fatal(err)
	}
	return st
}

// wantRetryAfter checks the Retry-After header is a positive number of seconds
// and returns it.
func wantRetryAfter(t *testing.T, rec *httptest.ResponseRecorder) int {
	t.Helper()
	got := rec.Header().Get("Retry-After")
	n, err := strconv.Atoi(got)
	if err != nil || n <= 0 {
		t.Fatalf("Retry-After = %q, want a positive number of seconds", got)
	}
	return n
}

// withTenantA edits tenant-a's limits.
func withTenantA(f func(*config.Tenant)) func(*acctEnv) {
	return func(e *acctEnv) { f(&e.cfg.Tenants[0]) }
}

// promptBytesOf is the size the gateway reserves for a prompt.
func promptBytesOf(content string) int {
	return len(policy.FeatureText([]provider.Message{{Role: provider.RoleUser, Content: content}}))
}

// costResp mirrors the cost fields of a response.
type costResp struct {
	RouteForge struct {
		CostUSD    *float64 `json:"cost_usd"`
		CostSource *string  `json:"cost_source"`
	} `json:"routeforge"`
}

func decodeCost(t *testing.T, rec *httptest.ResponseRecorder) costResp {
	t.Helper()
	var c costResp
	if err := json.Unmarshal(rec.Body.Bytes(), &c); err != nil {
		t.Fatalf("body is not a chat response: %v; body %s", err, rec.Body)
	}
	return c
}

func TestBudgetExceededReturns429(t *testing.T) {
	// A default request reserves its prompt and 4096 output tokens: about
	// 8.2 million nano-USD, more than the 1 million this tenant has.
	e := newAcctEnv(t, withTenantA(func(ten *config.Tenant) { ten.DailyBudgetUSD = 0.001 }))
	rec := e.post(t, okBody, nil)
	wantError(t, rec, http.StatusTooManyRequests, "budget_exceeded")
	// 22:00 UTC: two hours to the next budget day.
	if got := wantRetryAfter(t, rec); got != 7200 {
		t.Errorf("Retry-After = %d, want 7200", got)
	}
	if n := e.prov.callCount(); n != 0 {
		t.Errorf("provider called %d times, want 0", n)
	}
	if st := e.budget(t, "tenant-a"); st != (accounting.BudgetState{}) {
		t.Errorf("budget state = %+v, want untouched", st)
	}
}

func TestBudgetExceededBeforeAStreamStarts(t *testing.T) {
	e := newAcctEnv(t, withTenantA(func(ten *config.Tenant) { ten.DailyBudgetUSD = 0.001 }))
	rec := e.post(t, streamBody, nil)
	wantError(t, rec, http.StatusTooManyRequests, "budget_exceeded")
	if ct := rec.Header().Get("Content-Type"); ct != "application/json" {
		t.Errorf("Content-Type = %q, want a plain JSON error, not an event stream", ct)
	}
	wantRetryAfter(t, rec)
}

func TestRateLimitedReturns429(t *testing.T) {
	t.Run("requests per minute", func(t *testing.T) {
		e := newAcctEnv(t, withTenantA(func(ten *config.Tenant) { ten.RPM = 1 }))
		if rec := e.post(t, okBody, nil); rec.Code != http.StatusOK {
			t.Fatalf("first request: status = %d; body %s", rec.Code, rec.Body)
		}
		rec := e.post(t, okBody, nil)
		wantError(t, rec, http.StatusTooManyRequests, "rate_limited")
		if got := wantRetryAfter(t, rec); got != 60 {
			t.Errorf("Retry-After = %d, want 60 (one request a minute)", got)
		}
		if n := e.prov.callCount(); n != 1 {
			t.Errorf("provider called %d times, want 1", n)
		}
	})

	t.Run("tokens per minute", func(t *testing.T) {
		// The request needs its prompt tokens plus 4096 output tokens.
		e := newAcctEnv(t, withTenantA(func(ten *config.Tenant) { ten.TPM = 100 }))
		rec := e.post(t, okBody, nil)
		wantError(t, rec, http.StatusTooManyRequests, "rate_limited")
		wantRetryAfter(t, rec)
		if n := e.prov.callCount(); n != 0 {
			t.Errorf("provider called %d times, want 0", n)
		}
	})

	t.Run("a request refused before routing leaves the rate alone", func(t *testing.T) {
		// Admission follows the body check: a malformed request costs no rate.
		e := newAcctEnv(t, withTenantA(func(ten *config.Tenant) { ten.RPM = 1 }))
		if rec := e.post(t, `{`, nil); rec.Code != http.StatusBadRequest {
			t.Fatalf("malformed request: status = %d, want 400; body %s", rec.Code, rec.Body)
		}
		if rec := e.post(t, okBody, nil); rec.Code != http.StatusOK {
			t.Fatalf("the one request of the minute: status = %d, want 200; body %s", rec.Code, rec.Body)
		}
	})

	t.Run("admission precedes routing", func(t *testing.T) {
		// Admit runs before Decide, so a request the router then refuses has
		// used up its slot of the minute (but reserved no budget).
		e := newAcctEnv(t, func(e *acctEnv) { e.cfg.Tenants[1].RPM = 1 })
		const fastBody = `{"model":"m-fast","messages":[{"role":"user","content":"hi"}]}`
		rec := do(t, e.srv.Handler(), http.MethodPost, "/v1/chat/completions", keyB, okBody, nil) // m-strong: not allowed
		wantError(t, rec, http.StatusForbidden, "model_not_allowed")
		rec = do(t, e.srv.Handler(), http.MethodPost, "/v1/chat/completions", keyB, fastBody, nil)
		wantError(t, rec, http.StatusTooManyRequests, "rate_limited")
	})

	t.Run("the explain endpoint is not rate limited", func(t *testing.T) {
		e := newAcctEnv(t, withTenantA(func(ten *config.Tenant) { ten.RPM = 1 }))
		for range 3 {
			rec := do(t, e.srv.Handler(), http.MethodPost, "/v1/routing/explain", keyA, okBody, nil)
			if rec.Code != http.StatusOK {
				t.Fatalf("explain: status = %d; body %s", rec.Code, rec.Body)
			}
		}
		if rec := e.post(t, okBody, nil); rec.Code != http.StatusOK {
			t.Fatalf("chat after explains: status = %d; body %s", rec.Code, rec.Body)
		}
	})
}

func TestAccountingBackendDownFailsClosed(t *testing.T) {
	limited := withTenantA(func(ten *config.Tenant) { ten.DailyBudgetUSD = 1 })

	cases := []struct {
		name  string
		tweak func(*acctEnv)
		body  string
	}{
		{"the budget store at reservation", func(e *acctEnv) { limited(e); e.budgets.down.Store(true) }, okBody},
		{"the budget store at routing time", func(e *acctEnv) {
			limited(e)
			e.budgets.down.Store(true)
			e.useRoute = true
		}, autoBody},
		{"the budget store before a stream", func(e *acctEnv) { limited(e); e.budgets.down.Store(true) }, streamBody},
		{"the rate limiter", func(e *acctEnv) {
			withTenantA(func(ten *config.Tenant) { ten.RPM = 10 })(e)
			e.limiter = brokenLimiter{}
		}, okBody},
		{"a full usage queue", func(e *acctEnv) { e.usageSt = fullUsage{e.usage} }, okBody},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			e := newAcctEnv(t, tc.tweak)
			rec := e.post(t, tc.body, nil)
			wantError(t, rec, http.StatusServiceUnavailable, "accounting_unavailable")
			if got := wantRetryAfter(t, rec); got != 1 {
				t.Errorf("Retry-After = %d, want 1", got)
			}
			if n := e.prov.callCount(); n != 0 {
				t.Errorf("provider called %d times, want 0: no upstream call without accounting", n)
			}
			if got := e.usage.Recent(); len(got) != 0 {
				t.Errorf("%d usage records, want 0", len(got))
			}
			// The error text must not echo the backend's.
			if strings.Contains(rec.Body.String(), errAcctDown.Error()) {
				t.Errorf("body leaks the backend error: %s", rec.Body)
			}
		})
	}
}

func TestAccountingOpenModeProceeds(t *testing.T) {
	limited := withTenantA(func(ten *config.Tenant) { ten.DailyBudgetUSD = 1; ten.RPM = 10 })

	t.Run("budget store down", func(t *testing.T) {
		e := newAcctEnv(t, limited, func(e *acctEnv) {
			e.mode = config.FailModeOpen
			e.budgets.down.Store(true)
			e.prov.complete = func(context.Context, provider.Request) (*provider.Response, error) {
				return &provider.Response{Content: "ok", FinishReason: "stop", Usage: &provider.Usage{PromptTokens: 11, CompletionTokens: 7}}, nil
			}
		})
		rec := e.post(t, okBody, nil)
		if rec.Code != http.StatusOK {
			t.Fatalf("status = %d, want 200; body %s", rec.Code, rec.Body)
		}
		if n := e.prov.callCount(); n != 1 {
			t.Errorf("provider called %d times, want 1", n)
		}
		// The bill is still written, though the budget could not take it.
		got := e.usage.Recent()
		if len(got) != 1 || got[0].CostNanos != 11*acctInNanos+7*acctOutNanos {
			t.Fatalf("usage records = %+v, want one with the cost", got)
		}
	})

	t.Run("budget signal down at routing time", func(t *testing.T) {
		e := newAcctEnv(t, limited, func(e *acctEnv) {
			e.mode = config.FailModeOpen
			e.budgets.down.Store(true)
			e.useRoute = true
		})
		if rec := e.post(t, autoBody, nil); rec.Code != http.StatusOK {
			t.Fatalf("status = %d, want 200; body %s", rec.Code, rec.Body)
		}
	})

	t.Run("rate limiter down", func(t *testing.T) {
		e := newAcctEnv(t, limited, func(e *acctEnv) {
			e.mode = config.FailModeOpen
			e.limiter = brokenLimiter{}
		})
		if rec := e.post(t, okBody, nil); rec.Code != http.StatusOK {
			t.Fatalf("status = %d, want 200; body %s", rec.Code, rec.Body)
		}
	})
}

func TestResponseCarriesCost(t *testing.T) {
	t.Run("reported usage", func(t *testing.T) {
		e := newAcctEnv(t, func(e *acctEnv) {
			e.prov.complete = func(context.Context, provider.Request) (*provider.Response, error) {
				return &provider.Response{Content: "hello", FinishReason: "stop", Usage: &provider.Usage{PromptTokens: 11, CompletionTokens: 7}}, nil
			}
		})
		rec := e.post(t, okBody, nil)
		if rec.Code != http.StatusOK {
			t.Fatalf("status = %d; body %s", rec.Code, rec.Body)
		}
		c := decodeCost(t, rec)
		want := accounting.NanosToUSD(11*acctInNanos + 7*acctOutNanos)
		if c.RouteForge.CostUSD == nil || *c.RouteForge.CostUSD != want {
			t.Errorf("cost_usd = %v, want %v", c.RouteForge.CostUSD, want)
		}
		if c.RouteForge.CostSource == nil || *c.RouteForge.CostSource != "reported" {
			t.Errorf("cost_source = %v, want reported", c.RouteForge.CostSource)
		}
		if !strings.Contains(rec.Body.String(), `"cost_usd":0.000025`) {
			t.Errorf("cost_usd is not a plain JSON number: %s", rec.Body)
		}
	})

	t.Run("estimated when the provider sends no usage", func(t *testing.T) {
		e := newAcctEnv(t, func(e *acctEnv) { e.prov.complete = replyWith("hello") })
		rec := e.post(t, okBody, nil)
		if rec.Code != http.StatusOK {
			t.Fatalf("status = %d; body %s", rec.Code, rec.Body)
		}
		c := decodeCost(t, rec)
		prompt := accounting.EstimateTokens(promptBytesOf("hi"))
		want := accounting.NanosToUSD(int64(prompt)*acctInNanos + int64(accounting.EstimateTokens(len("hello")))*acctOutNanos)
		if c.RouteForge.CostUSD == nil || *c.RouteForge.CostUSD != want {
			t.Errorf("cost_usd = %v, want %v", c.RouteForge.CostUSD, want)
		}
		if c.RouteForge.CostSource == nil || *c.RouteForge.CostSource != "estimated" {
			t.Errorf("cost_source = %v, want estimated", c.RouteForge.CostSource)
		}
	})

	t.Run("a retried request sums its attempts", func(t *testing.T) {
		calls := 0
		e := newAcctEnv(t, func(e *acctEnv) {
			e.prov.complete = func(context.Context, provider.Request) (*provider.Response, error) {
				calls++
				if calls == 1 {
					return nil, &provider.Error{Kind: provider.KindUnavailable, Provider: "p"}
				}
				return &provider.Response{Content: "ok", Usage: &provider.Usage{PromptTokens: 11, CompletionTokens: 7}}, nil
			}
			e.opts = append(e.opts, func(o *Options) { o.Router = &fallbackRouter{inner: o.Router, fallback: "m-fast"} })
			e.opts = append(e.opts, func(o *Options) { o.Executor = retryingExecutor(e.cfg, e.prov) })
		})
		rec := e.post(t, okBody, nil)
		if rec.Code != http.StatusOK {
			t.Fatalf("status = %d; body %s", rec.Code, rec.Body)
		}
		c := decodeCost(t, rec)
		want := accounting.NanosToUSD(11*acctInNanos + 7*acctOutNanos)
		if c.RouteForge.CostUSD == nil || *c.RouteForge.CostUSD != want {
			t.Errorf("cost_usd = %v, want %v: the failed attempt costs nothing", c.RouteForge.CostUSD, want)
		}
		// The failed attempt's zero is an assumption, not a report.
		if c.RouteForge.CostSource == nil || *c.RouteForge.CostSource != "estimated" {
			t.Errorf("cost_source = %v, want estimated", c.RouteForge.CostSource)
		}
		if got := e.usage.Recent(); len(got) != 2 || got[0].Attempt != 1 || got[1].Attempt != 2 {
			t.Errorf("usage records = %+v, want attempts 1 and 2", got)
		}
	})

	t.Run("an unlimited tenant is still billed", func(t *testing.T) {
		e := newAcctEnv(t, func(e *acctEnv) {
			e.prov.complete = func(context.Context, provider.Request) (*provider.Response, error) {
				return &provider.Response{Content: "hello", Usage: &provider.Usage{PromptTokens: 11, CompletionTokens: 7}}, nil
			}
		})
		e.post(t, okBody, nil)
		if e.budgets.reserves.Load() != 0 {
			t.Errorf("Reserve called %d times for a tenant without a budget", e.budgets.reserves.Load())
		}
		if st := e.budget(t, "tenant-a"); st.Spent != 11*acctInNanos+7*acctOutNanos {
			t.Errorf("spent = %d, want the cost of the request", st.Spent)
		}
	})
}

func TestChatHasNoCostWithoutAnAccountant(t *testing.T) {
	srv := newServer(t, testConfig(), map[string]provider.Provider{"p-strong": &stubProvider{complete: replyWith("ok")}})
	rec := do(t, srv.Handler(), http.MethodPost, "/v1/chat/completions", keyA, okBody, nil)
	if rec.Code != http.StatusOK {
		t.Fatalf("status = %d; body %s", rec.Code, rec.Body)
	}
	c := decodeCost(t, rec)
	if c.RouteForge.CostUSD != nil || c.RouteForge.CostSource != nil {
		t.Errorf("cost fields = %v / %v, want none without accounting", c.RouteForge.CostUSD, c.RouteForge.CostSource)
	}
}

// fallbackRouter adds a fallback model to every decision.
type fallbackRouter struct {
	inner    router.Router
	fallback string
}

func (f *fallbackRouter) Decide(ctx context.Context, req *router.Request) (*router.Decision, error) {
	d, err := f.inner.Decide(ctx, req)
	if d != nil {
		d.Fallbacks = append(d.Fallbacks, f.fallback)
	}
	return d, err
}

// retryingExecutor is an executor that makes up to three attempts without
// waiting between them, over every configured model served by prov.
func retryingExecutor(cfg *config.Config, prov provider.Provider) *executor.Executor {
	targets := make(map[string]executor.Target, len(cfg.Models))
	for _, m := range cfg.Models {
		targets[m.Name] = executor.Target{Model: m.Name, Upstream: m.UpstreamModel, Provider: prov}
	}
	return executor.NewWithOptions(executor.Options{
		Targets:     targets,
		MaxAttempts: 3,
		Backoff:     reliability.Backoff{Base: time.Millisecond, Cap: time.Millisecond},
		Sleep:       func(ctx context.Context, _ time.Duration) error { return ctx.Err() },
	})
}

func TestFailoverStopsWhenBudgetRunsOut(t *testing.T) {
	const prompt = "hello world"
	body := autoBody
	b := int64(promptBytesOf(prompt))
	// The cheapest passing model, qwen, is chosen first and deepseek is its
	// first fallback. A reservation is the prompt bytes at the input price plus
	// the 4096-token output cap at the output price. The budget fits qwen's
	// exactly and so is one nano-USD short of deepseek's.
	qwenReserve := b*5000 + 4096*10000
	budget := accounting.NanosToUSD(qwenReserve)

	var e *poolEnv
	var acct *accounting.Accountant
	var usage *accounting.MemoryUsage
	var budgets *spyBudgets
	e = newPoolEnv(t, func(p *poolEnv) {
		p.cfg.Tenants[0].DailyBudgetUSD = budget
		p.prov.complete = func(context.Context, provider.Request) (*provider.Response, error) {
			return nil, &provider.Error{Kind: provider.KindUnavailable, Provider: "p-pool"}
		}
		usage = accounting.NewMemoryUsage(10)
		budgets = &spyBudgets{BudgetStore: accounting.NewMemoryBudgets()}
		acct = accounting.NewAccountant(accounting.AccountantOptions{
			Tenants: p.cfg.Tenants,
			Prices:  poolPrices(t),
			Budgets: budgets,
			Limiter: accounting.NewMemoryLimiter(func() time.Time { return acctNow }),
			Usage:   usage,
			Now:     func() time.Time { return acctNow },
		})
		p.opts = append(p.opts, func(o *Options) {
			o.Accountant = acct
			o.Executor = retryingExecutor(p.cfg, p.prov)
		})
	})

	rec := e.post(t, "/v1/chat/completions", keyA, body, nil)
	// The last upstream error is what the client gets, not the budget error of
	// the attempt that never ran.
	wantError(t, rec, http.StatusBadGateway, "upstream_error")
	if n := e.prov.callCount(); n != 1 {
		t.Fatalf("provider called %d times, want 1: the second reservation fails", n)
	}
	if got := budgets.reserves.Load(); got != 2 {
		t.Errorf("Reserve called %d times, want 2 (qwen, then the refused deepseek)", got)
	}
	records := usage.Recent()
	if len(records) != 1 {
		t.Fatalf("%d usage records, want 1: %+v", len(records), records)
	}
	if r := records[0]; r.Model != "qwen3-235b-a22b-2507" || r.Outcome != router.OutcomeError || r.CostNanos != 0 {
		t.Errorf("record = %+v, want qwen's failed attempt at no cost", r)
	}
	st, err := budgets.BudgetStore.State(context.Background(), accounting.BudgetKey{Tenant: "tenant-a", Day: acctDay})
	if err != nil || st != (accounting.BudgetState{}) {
		t.Errorf("budget = %+v, %v; want nothing held or spent", st, err)
	}
}

func TestNoReservationWhenTheDecisionFails(t *testing.T) {
	for _, tc := range []struct {
		name   string
		key    string
		body   string
		status int
		code   string
	}{
		{"a model the tenant may not use", keyB, okBody, http.StatusForbidden, "model_not_allowed"},
		{"an unknown model", keyA, `{"model":"no-such-model","messages":[{"role":"user","content":"hi"}]}`, http.StatusBadRequest, "invalid_request"},
	} {
		t.Run(tc.name, func(t *testing.T) {
			e := newAcctEnv(t, func(e *acctEnv) {
				for i := range e.cfg.Tenants {
					e.cfg.Tenants[i].DailyBudgetUSD = 1
				}
			})
			rec := do(t, e.srv.Handler(), http.MethodPost, "/v1/chat/completions", tc.key, tc.body, nil)
			wantError(t, rec, tc.status, tc.code)
			if n := e.budgets.reserves.Load(); n != 0 {
				t.Errorf("Reserve called %d times for a request that was never routed", n)
			}
			if n := e.prov.callCount(); n != 0 {
				t.Errorf("provider called %d times", n)
			}
		})
	}
}

func TestExplainDoesNotReserve(t *testing.T) {
	e := newAcctEnv(t, withTenantA(func(ten *config.Tenant) { ten.DailyBudgetUSD = 1 }))
	rec := do(t, e.srv.Handler(), http.MethodPost, "/v1/routing/explain", keyA, okBody, nil)
	if rec.Code != http.StatusOK {
		t.Fatalf("status = %d; body %s", rec.Code, rec.Body)
	}
	if n := e.budgets.reserves.Load(); n != 0 {
		t.Errorf("Reserve called %d times by explain", n)
	}
}

func TestExplainFailsClosedWhenTheBudgetSignalIsDown(t *testing.T) {
	e := newAcctEnv(t, withTenantA(func(ten *config.Tenant) { ten.DailyBudgetUSD = 1 }), func(e *acctEnv) {
		e.budgets.down.Store(true)
		e.useRoute = true
	})
	rec := do(t, e.srv.Handler(), http.MethodPost, "/v1/routing/explain", keyA, autoBody, nil)
	wantError(t, rec, http.StatusServiceUnavailable, "accounting_unavailable")
	wantRetryAfter(t, rec)
}

func TestStreamIsMeteredAndSettled(t *testing.T) {
	limited := withTenantA(func(ten *config.Tenant) { ten.DailyBudgetUSD = 1 })

	t.Run("a finished stream settles on its reported usage", func(t *testing.T) {
		st := &stubStream{events: []provider.StreamEvent{
			{Delta: "Hel"},
			{Delta: "lo"},
			{FinishReason: "stop"},
			{Usage: &provider.Usage{PromptTokens: 5, CompletionTokens: 2}},
		}}
		e := newAcctEnv(t, limited, func(e *acctEnv) { e.prov = streamingProvider(st) })
		rec := e.post(t, streamBody, nil)
		if _, done, errEvent := readSSE(t, rec.Body.String()); !done || errEvent != nil {
			t.Fatalf("done = %v, error event = %v; body %s", done, errEvent, rec.Body)
		}
		got := e.usage.Recent()
		const cost = 5*acctInNanos + 2*acctOutNanos
		if len(got) != 1 || got[0].Partial || got[0].CostNanos != cost || got[0].UsageSource != accounting.SourceReported {
			t.Fatalf("usage records = %+v, want one reported record costing %d", got, cost)
		}
		if state := e.budget(t, "tenant-a"); state != (accounting.BudgetState{Spent: cost}) {
			t.Errorf("budget = %+v, want %d spent and nothing reserved", state, cost)
		}
		if strings.Contains(rec.Body.String(), "cost_usd") {
			t.Errorf("a stream carries no cost: %s", rec.Body)
		}
	})

	t.Run("a stream cut off after output settles as partial", func(t *testing.T) {
		st := &stubStream{
			events: []provider.StreamEvent{{Delta: "0123456789"}},
			err:    &provider.Error{Kind: provider.KindUnavailable, Provider: "p"},
		}
		e := newAcctEnv(t, limited, func(e *acctEnv) { e.prov = streamingProvider(st) })
		rec := e.post(t, streamBody, nil)
		if _, _, errEvent := readSSE(t, rec.Body.String()); errEvent == nil {
			t.Fatalf("want a terminal error event; body %s", rec.Body)
		}
		got := e.usage.Recent()
		// The prompt is estimated from its bytes and 10 delivered bytes are 3 tokens.
		prompt := int64(accounting.EstimateTokens(promptBytesOf("hi")))
		cost := prompt*acctInNanos + 3*acctOutNanos
		if len(got) != 1 || !got[0].Partial || got[0].CompletionTokens != 3 || got[0].CostNanos != cost {
			t.Fatalf("usage records = %+v, want one partial record costing %d", got, cost)
		}
		if state := e.budget(t, "tenant-a"); state != (accounting.BudgetState{Spent: cost}) {
			t.Errorf("budget = %+v, want the partial cost spent and nothing reserved", state)
		}
	})
}

func TestAdminAccountingEndpoint(t *testing.T) {
	withAdmin := func(e *acctEnv) { e.cfg.Admin.APIKeySHA256 = []string{auth.HashKey(adminKey)} }
	limited := withTenantA(func(ten *config.Tenant) { ten.DailyBudgetUSD = 0.0007 })

	get := func(e *acctEnv, path, key string) *httptest.ResponseRecorder {
		return do(t, e.srv.Handler(), http.MethodGet, path, key, "", nil)
	}

	t.Run("reports spend, reservations, overshoot and usage", func(t *testing.T) {
		e := newAcctEnv(t, withAdmin, limited, func(e *acctEnv) {
			e.prov.complete = func(context.Context, provider.Request) (*provider.Response, error) {
				// 900 prompt tokens cost 900,000 nano-USD against a 700,000 budget.
				return &provider.Response{Content: "ok", Usage: &provider.Usage{PromptTokens: 900}}, nil
			}
			e.cfg.Tenants[0].MaxOutputTokens = 10 // keeps the reservation within the budget
		})
		if rec := e.post(t, okBody, nil); rec.Code != http.StatusOK {
			t.Fatalf("chat: status = %d; body %s", rec.Code, rec.Body)
		}

		rec := get(e, "/admin/accounting?tenant=tenant-a&day="+acctDay, adminKey)
		if rec.Code != http.StatusOK {
			t.Fatalf("status = %d; body %s", rec.Code, rec.Body)
		}
		var got map[string]any
		if err := json.Unmarshal(rec.Body.Bytes(), &got); err != nil {
			t.Fatal(err)
		}
		want := map[string]any{
			"tenant":           "tenant-a",
			"day":              acctDay,
			"spent_usd":        accounting.NanosToUSD(900_000),
			"reserved_usd":     0.0,
			"overshoot_usd":    accounting.NanosToUSD(200_000),
			"spent_nanos":      900_000.0,
			"reserved_nanos":   0.0,
			"overshoot_nanos":  200_000.0,
			"usage_records":    1.0,
			"usage_cost_nanos": 900_000.0,
		}
		if len(got) != len(want) {
			t.Errorf("response has %d fields, want %d: %v", len(got), len(want), got)
		}
		for k, w := range want {
			if got[k] != w {
				t.Errorf("%s = %v, want %v", k, got[k], w)
			}
		}
	})

	t.Run("the day defaults to today in UTC", func(t *testing.T) {
		e := newAcctEnv(t, withAdmin, limited)
		rec := get(e, "/admin/accounting?tenant=tenant-a", adminKey)
		if rec.Code != http.StatusOK {
			t.Fatalf("status = %d; body %s", rec.Code, rec.Body)
		}
		var got struct {
			Day string `json:"day"`
		}
		if err := json.Unmarshal(rec.Body.Bytes(), &got); err != nil || got.Day != acctDay {
			t.Errorf("day = %q, %v; want %s", got.Day, err, acctDay)
		}
	})

	t.Run("an unknown tenant has zeros", func(t *testing.T) {
		e := newAcctEnv(t, withAdmin)
		rec := get(e, "/admin/accounting?tenant=nobody", adminKey)
		if rec.Code != http.StatusOK {
			t.Fatalf("status = %d; body %s", rec.Code, rec.Body)
		}
		var got map[string]any
		if err := json.Unmarshal(rec.Body.Bytes(), &got); err != nil {
			t.Fatal(err)
		}
		for _, k := range []string{"spent_usd", "reserved_usd", "overshoot_usd", "spent_nanos", "reserved_nanos", "overshoot_nanos", "usage_records", "usage_cost_nanos"} {
			if got[k] != 0.0 {
				t.Errorf("%s = %v, want 0", k, got[k])
			}
		}
	})

	for _, tc := range []struct {
		name string
		path string
	}{
		{"no tenant", "/admin/accounting"},
		{"an empty tenant", "/admin/accounting?tenant="},
		{"a malformed day", "/admin/accounting?tenant=tenant-a&day=2026-10-27"},
		{"an impossible day", "/admin/accounting?tenant=tenant-a&day=20261342"},
	} {
		t.Run("rejects "+tc.name, func(t *testing.T) {
			e := newAcctEnv(t, withAdmin)
			wantError(t, get(e, tc.path, adminKey), http.StatusBadRequest, "invalid_request")
		})
	}

	t.Run("needs the admin key", func(t *testing.T) {
		e := newAcctEnv(t, withAdmin)
		wantError(t, get(e, "/admin/accounting?tenant=tenant-a", ""), http.StatusUnauthorized, "invalid_api_key")
		wantError(t, get(e, "/admin/accounting?tenant=tenant-a", "wrong-key"), http.StatusUnauthorized, "invalid_api_key")
		// A tenant key is not an admin key.
		wantError(t, get(e, "/admin/accounting?tenant=tenant-a", keyA), http.StatusUnauthorized, "invalid_api_key")
	})

	t.Run("does not exist without admin keys", func(t *testing.T) {
		e := newAcctEnv(t)
		wantError(t, get(e, "/admin/accounting?tenant=tenant-a", adminKey), http.StatusNotFound, "not_found")
	})

	t.Run("does not exist without an accountant", func(t *testing.T) {
		cfg := testConfig()
		cfg.Admin.APIKeySHA256 = []string{auth.HashKey(adminKey)}
		srv := newServer(t, cfg, nil)
		rec := do(t, srv.Handler(), http.MethodGet, "/admin/accounting?tenant=tenant-a", adminKey, "", nil)
		wantError(t, rec, http.StatusNotFound, "not_found")
	})

	t.Run("works without a policy registry", func(t *testing.T) {
		// The policy routes need a registry; this route needs only the accountant.
		e := newAcctEnv(t, withAdmin)
		if e.srv.policies != nil {
			t.Fatal("test setup: the server has a policy registry")
		}
		if rec := get(e, "/admin/accounting?tenant=tenant-a", adminKey); rec.Code != http.StatusOK {
			t.Fatalf("status = %d; body %s", rec.Code, rec.Body)
		}
	})

	t.Run("a budget store that is down is accounting_unavailable", func(t *testing.T) {
		e := newAcctEnv(t, withAdmin, func(e *acctEnv) { e.budgets.down.Store(true) })
		rec := get(e, "/admin/accounting?tenant=tenant-a", adminKey)
		wantError(t, rec, http.StatusServiceUnavailable, "accounting_unavailable")
		wantRetryAfter(t, rec)
	})
}
