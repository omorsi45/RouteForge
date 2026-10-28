package accounting_test

import (
	"context"
	"errors"
	"log/slog"
	"testing"

	"github.com/omorsi45/routeforge/internal/accounting"
	"github.com/omorsi45/routeforge/internal/config"
	"github.com/omorsi45/routeforge/internal/telemetry"
	"github.com/omorsi45/routeforge/internal/telemetry/telemetrytest"
)

type labels = map[string]string

// metered is a rig whose accountant reports to m. It shares the rig's clock and
// stores, so the helpers of the other tests apply.
func metered(t *testing.T, m *telemetry.Metrics, tenants ...config.Tenant) *rig {
	t.Helper()
	r := newRig(t, config.FailModeClosed, tenants...)
	price, err := accounting.NewPrice("m-fast", 1, 2)
	if err != nil {
		t.Fatal(err)
	}
	r.acct = accounting.NewAccountant(accounting.AccountantOptions{
		Tenants:  tenants,
		Prices:   map[string]accounting.Price{"m-fast": price},
		Budgets:  r.budgets,
		Limiter:  r.limiter,
		Usage:    r.usage,
		FailMode: config.FailModeClosed,
		Now:      r.clk.Now,
		Logger:   slog.New(slog.DiscardHandler),
		Metrics:  m,
	})
	return r
}

func TestMetricsRateLimited(t *testing.T) {
	ctx := context.Background()

	t.Run("requests per minute", func(t *testing.T) {
		m := telemetry.NewMetrics()
		ten := config.Tenant{Name: "t", RPM: 1}
		r := metered(t, m, ten)

		if err := r.acct.Admit(ctx, &ten, 40, 20); err != nil {
			t.Fatal(err)
		}
		// An admitted request is not a rate limit.
		telemetrytest.Absent(t, m, "routeforge_rate_limited_total")
		for want := 1.0; want <= 2; want++ {
			var rl *accounting.RateLimitedError
			if err := r.acct.Admit(ctx, &ten, 40, 20); !errors.As(err, &rl) {
				t.Fatalf("err = %v, want a *RateLimitedError", err)
			}
			if v := telemetrytest.Value(t, m, "routeforge_rate_limited_total", labels{"kind": "rpm"}); v != want {
				t.Errorf("rate_limited_total{rpm} = %g, want %g", v, want)
			}
		}
	})

	t.Run("tokens per minute", func(t *testing.T) {
		m := telemetry.NewMetrics()
		ten := config.Tenant{Name: "t", TPM: 50}
		r := metered(t, m, ten)

		if err := r.acct.Admit(ctx, &ten, 40, 20); err != nil { // 30 of 50 tokens
			t.Fatal(err)
		}
		if err := r.acct.Admit(ctx, &ten, 40, 20); err == nil {
			t.Fatal("second request was admitted, want the token limit")
		}
		if v := telemetrytest.Value(t, m, "routeforge_rate_limited_total", labels{"kind": "tpm"}); v != 1 {
			t.Errorf("rate_limited_total{tpm} = %g, want 1", v)
		}
	})

	t.Run("a backend failure is not a rate limit", func(t *testing.T) {
		m := telemetry.NewMetrics()
		ten := config.Tenant{Name: "t", RPM: 1}
		r := metered(t, m, ten)
		r.limiter.setDown(true)

		if err := r.acct.Admit(ctx, &ten, 40, 20); !errors.Is(err, accounting.ErrAccountingUnavailable) {
			t.Fatalf("err = %v, want ErrAccountingUnavailable", err)
		}
		telemetrytest.Absent(t, m, "routeforge_rate_limited_total")
	})
}

func TestMetricsBudgetRejection(t *testing.T) {
	m := telemetry.NewMetrics()
	// 500,000 nano-USD cannot hold the 600,000 reservation.
	ten := config.Tenant{Name: "t", DailyBudgetUSD: 0.0005}
	roomy := config.Tenant{Name: "roomy", DailyBudgetUSD: 1}
	r := metered(t, m, ten, roomy)

	for want := 1.0; want <= 2; want++ {
		_, err := r.meter(&ten).Begin(context.Background(), "m-fast")
		var be *accounting.BudgetExceededError
		if !errors.As(err, &be) {
			t.Fatalf("Begin error = %v, want a *BudgetExceededError", err)
		}
		if v := telemetrytest.Value(t, m, "routeforge_budget_rejections_total", labels{}); v != want {
			t.Errorf("budget_rejections_total = %g, want %g", v, want)
		}
	}

	// A reservation that fits is not a rejection.
	begin(t, r.meter(&roomy)).End(settleOK(1, 1))
	if v := telemetrytest.Value(t, m, "routeforge_budget_rejections_total", labels{}); v != 2 {
		t.Errorf("budget_rejections_total = %g after an admitted reservation, want 2", v)
	}
}

func TestMetricsSettlementRecordsCostAndTokens(t *testing.T) {
	ten := config.Tenant{Name: "t"}

	t.Run("reported usage", func(t *testing.T) {
		m := telemetry.NewMetrics()
		r := metered(t, m, ten)
		// 30 prompt tokens at 1000 and 20 completion tokens at 2000 nano-USD.
		begin(t, r.meter(&ten)).End(settleOK(30, 20))

		cost := labels{"model": "m-fast", "cost_source": "reported"}
		if v, want := telemetrytest.Value(t, m, "routeforge_cost_usd_total", cost), accounting.NanosToUSD(70_000); v != want {
			t.Errorf("cost_usd_total%v = %g, want %g", cost, v, want)
		}
		for direction, want := range map[string]float64{"prompt": 30, "completion": 20} {
			l := labels{"model": "m-fast", "direction": direction, "usage_source": "reported"}
			if v := telemetrytest.Value(t, m, "routeforge_tokens_total", l); v != want {
				t.Errorf("tokens_total%v = %g, want %g", l, v, want)
			}
		}
	})

	t.Run("estimated usage", func(t *testing.T) {
		m := telemetry.NewMetrics()
		r := metered(t, m, ten)
		// 400 prompt bytes are 100 tokens; 40 delivered bytes are 10 tokens.
		begin(t, r.meter(&ten)).End(accounting.Settlement{Outcome: accounting.OutcomeOK, DeliveredBytes: 40})

		cost := labels{"model": "m-fast", "cost_source": "estimated"}
		if v, want := telemetrytest.Value(t, m, "routeforge_cost_usd_total", cost), accounting.NanosToUSD(120_000); v != want {
			t.Errorf("cost_usd_total%v = %g, want %g", cost, v, want)
		}
		for direction, want := range map[string]float64{"prompt": 100, "completion": 10} {
			l := labels{"model": "m-fast", "direction": direction, "usage_source": "estimated"}
			if v := telemetrytest.Value(t, m, "routeforge_tokens_total", l); v != want {
				t.Errorf("tokens_total%v = %g, want %g", l, v, want)
			}
		}
	})

	t.Run("an attempt that failed before any output", func(t *testing.T) {
		m := telemetry.NewMetrics()
		r := metered(t, m, ten)
		begin(t, r.meter(&ten)).End(accounting.Settlement{Outcome: accounting.OutcomeError})

		cost := labels{"model": "m-fast", "cost_source": "estimated"}
		if v := telemetrytest.Value(t, m, "routeforge_cost_usd_total", cost); v != 0 {
			t.Errorf("cost_usd_total%v = %g, want 0", cost, v)
		}
	})

	t.Run("every attempt counts", func(t *testing.T) {
		m := telemetry.NewMetrics()
		r := metered(t, m, ten)
		meter := r.meter(&ten)
		begin(t, meter).End(accounting.Settlement{Outcome: accounting.OutcomeError})
		begin(t, meter).End(settleOK(30, 20))
		begin(t, meter).End(settleOK(30, 20))

		l := labels{"model": "m-fast", "direction": "prompt", "usage_source": "reported"}
		if v := telemetrytest.Value(t, m, "routeforge_tokens_total", l); v != 60 {
			t.Errorf("tokens_total%v = %g, want 60", l, v)
		}
	})
}

func TestMetricsBudgetOvershoot(t *testing.T) {
	t.Run("a settlement over the budget", func(t *testing.T) {
		m := telemetry.NewMetrics()
		// Exactly one reservation (600,000) fits, and the call then costs
		// 400*1000 + 300*2000 = 1,000,000: 400,000 over.
		ten := config.Tenant{Name: "t", DailyBudgetUSD: 0.0006}
		r := metered(t, m, ten)

		begin(t, r.meter(&ten)).End(settleOK(400, 300))

		l := labels{"tenant": "t"}
		if v, want := telemetrytest.Value(t, m, "routeforge_budget_overshoot_usd_total", l), accounting.NanosToUSD(400_000); v != want {
			t.Errorf("budget_overshoot_usd_total%v = %g, want %g", l, v, want)
		}
		// The store agrees with the metric.
		if st, _, err := r.acct.State(context.Background(), "t", testDay); err != nil || accounting.NanosToUSD(st.Overshoot) != 0.0004 {
			t.Errorf("State = %+v, %v; want an overshoot of 0.0004 USD", st, err)
		}
	})

	t.Run("a settlement within the budget adds nothing", func(t *testing.T) {
		m := telemetry.NewMetrics()
		ten := config.Tenant{Name: "t", DailyBudgetUSD: 1}
		r := metered(t, m, ten)

		begin(t, r.meter(&ten)).End(settleOK(30, 20))

		telemetrytest.Absent(t, m, "routeforge_budget_overshoot_usd_total")
	})

	t.Run("a tenant without a budget never overshoots", func(t *testing.T) {
		m := telemetry.NewMetrics()
		ten := config.Tenant{Name: "t"}
		r := metered(t, m, ten)

		begin(t, r.meter(&ten)).End(settleOK(4000, 3000))

		telemetrytest.Absent(t, m, "routeforge_budget_overshoot_usd_total")
	})
}
