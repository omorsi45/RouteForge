package accounting_test

import (
	"math"
	"math/rand/v2"
	"strings"
	"testing"

	"github.com/omorsi45/routeforge/internal/accounting"
)

func mustPrice(t *testing.T, name string, in, out float64) accounting.Price {
	t.Helper()
	p, err := accounting.NewPrice(name, in, out)
	if err != nil {
		t.Fatalf("NewPrice(%s): %v", name, err)
	}
	return p
}

func TestEstimateTokens(t *testing.T) {
	for _, tc := range []struct{ bytes, want int }{
		{0, 0}, {1, 1}, {3, 1}, {4, 1}, {5, 2}, {8, 2}, {9, 3}, {4000, 1000}, {4001, 1001},
	} {
		if got := accounting.EstimateTokens(tc.bytes); got != tc.want {
			t.Errorf("EstimateTokens(%d) = %d, want %d", tc.bytes, got, tc.want)
		}
	}
}

func TestCostNanosExact(t *testing.T) {
	qwen := mustPrice(t, "qwen3-235b-a22b-2507", 0.09, 0.60)
	// 1000 * 90 + 500 * 600 nano-USD.
	if got, want := accounting.CostNanos(qwen, 1000, 500), int64(390_000); got != want {
		t.Errorf("CostNanos = %d, want %d", got, want)
	}
	if got := accounting.CostNanos(qwen, 0, 0); got != 0 {
		t.Errorf("CostNanos(0, 0) = %d, want 0", got)
	}

	// One token at the cheapest rate is a whole number of nano-USD, with no
	// float rounding anywhere.
	gpt := mustPrice(t, "gpt-5", 1.25, 10)
	if got, want := accounting.CostNanos(gpt, 1, 1), int64(1250+10000); got != want {
		t.Errorf("CostNanos(1, 1) = %d, want %d", got, want)
	}
	// A million tokens at 1.25 USD per million is exactly 1.25 USD.
	if got, want := accounting.CostNanos(gpt, 1_000_000, 0), int64(1_250_000_000); got != want {
		t.Errorf("CostNanos(1M, 0) = %d, want %d", got, want)
	}
}

func TestReservationUsesBytes(t *testing.T) {
	qwen := mustPrice(t, "qwen3-235b-a22b-2507", 0.09, 0.60)
	got := accounting.ReservationNanos(qwen, 1000, 200)
	// 1000 prompt bytes at the per-token input price, not 250 tokens.
	if want := int64(1000*90 + 200*600); got != want {
		t.Errorf("ReservationNanos = %d, want %d", got, want)
	}
	if tokens := accounting.CostNanos(qwen, accounting.EstimateTokens(1000), 200); got == tokens {
		t.Errorf("reservation %d equals the token-based cost %d; it must price bytes", got, tokens)
	}
}

func TestNanosToUSD(t *testing.T) {
	for _, tc := range []struct {
		nanos int64
		want  float64
	}{
		{0, 0}, {1, 1e-9}, {1_500_000_000, 1.5}, {390_000, 0.00039},
	} {
		if got := accounting.NanosToUSD(tc.nanos); got != tc.want {
			t.Errorf("NanosToUSD(%d) = %v, want %v", tc.nanos, got, tc.want)
		}
	}
}

func TestUSDToNanos(t *testing.T) {
	ok := []struct {
		usd  float64
		want int64
	}{
		{0, 0},
		{5.0, 5_000_000_000},
		{0.1, 100_000_000},
		{0.000000001, 1},
		{12.345678901, 12_345_678_901},
		{1234.5, 1_234_500_000_000},
	}
	for _, tc := range ok {
		got, err := accounting.USDToNanos(tc.usd)
		if err != nil || got != tc.want {
			t.Errorf("USDToNanos(%v) = %d, %v; want %d", tc.usd, got, err, tc.want)
		}
	}

	bad := []struct {
		name string
		usd  float64
	}{
		{"finer than a nano-USD", 0.0000000001},
		{"between two nano-USD", 0.0000000015},
		{"negative", -1},
		{"NaN", math.NaN()},
		{"infinite", math.Inf(1)},
		{"too large to be exact", 1e12},
	}
	for _, tc := range bad {
		t.Run(tc.name, func(t *testing.T) {
			if got, err := accounting.USDToNanos(tc.usd); err == nil {
				t.Fatalf("USDToNanos(%v) = %d, want an error", tc.usd, got)
			}
		})
	}
}

func TestUSDToNanosErrorNamesTheValue(t *testing.T) {
	_, err := accounting.USDToNanos(0.0000000001)
	if err == nil || !strings.Contains(err.Error(), "1e-10") {
		t.Fatalf("err = %v, want one naming the value", err)
	}
}

// TestUSDToNanosRoundTrips checks the claim behind the exactness test: every
// whole count of nano-USD up to the limit survives a trip through float64.
func TestUSDToNanosRoundTrips(t *testing.T) {
	const limit = 1 << 50
	rng := rand.New(rand.NewPCG(1, 2))
	samples := []int64{0, 1, 2, 999_999_999, 1_000_000_000, 1_000_000_001, limit - 1, limit}
	for range 50000 {
		samples = append(samples, rng.Int64N(limit))
	}
	for _, n := range samples {
		usd := accounting.NanosToUSD(n)
		got, err := accounting.USDToNanos(usd)
		if err != nil || got != n {
			t.Fatalf("USDToNanos(NanosToUSD(%d)) = %d, %v", n, got, err)
		}
	}
	if got, err := accounting.USDToNanos(accounting.NanosToUSD(limit + 1)); err == nil {
		t.Errorf("USDToNanos above the limit = %d, want an error", got)
	}
}
