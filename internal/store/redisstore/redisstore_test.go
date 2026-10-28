package redisstore_test

import (
	"context"
	"math"
	"math/rand/v2"
	"os"
	"slices"
	"strconv"
	"testing"
	"time"

	"github.com/alicebob/miniredis/v2"
	"github.com/redis/go-redis/v9"
	"github.com/redis/go-redis/v9/logging"

	"github.com/omorsi45/routeforge/internal/accounting"
	"github.com/omorsi45/routeforge/internal/accounting/storetest"
	"github.com/omorsi45/routeforge/internal/store/redisstore"
)

var clockStart = time.Date(2026, 10, 27, 12, 0, 0, 0, time.UTC)

func TestMain(m *testing.M) {
	// The client logs every failed dial. TestBackendDownReturnsError causes
	// those on purpose and checks the errors the store returns instead.
	logging.Disable()
	os.Exit(m.Run())
}

// newServer starts an in-process Redis whose clock reads clockStart. The
// limiter reads that clock through the TIME command, so tests move it with
// SetTime; FastForward moves only key expiry.
func newServer(t *testing.T) *miniredis.Miniredis {
	t.Helper()
	mr := miniredis.RunT(t)
	mr.SetTime(clockStart)
	return mr
}

// newClient returns a client for mr without command retries: the scripts are
// not idempotent, so a test must see every failure instead of a replay. A
// single dial attempt keeps a test against a closed server fast.
func newClient(t *testing.T, mr *miniredis.Miniredis) *redis.Client {
	t.Helper()
	c := redis.NewClient(&redis.Options{Addr: mr.Addr(), MaxRetries: -1, DialerRetries: 1})
	t.Cleanup(func() {
		if err := c.Close(); err != nil {
			t.Errorf("close client: %v", err)
		}
	})
	return c
}

func newStore(t *testing.T) (*redisstore.Store, *miniredis.Miniredis) {
	t.Helper()
	mr := newServer(t)
	return redisstore.New(newClient(t, mr)), mr
}

func TestBudgetStoreContract(t *testing.T) {
	storetest.RunBudgetStore(t, func(t *testing.T) accounting.BudgetStore {
		s, _ := newStore(t)
		return s
	})
}

func TestRateLimiterContract(t *testing.T) {
	storetest.RunRateLimiter(t, func(t *testing.T) (accounting.RateLimiter, func(time.Duration)) {
		s, mr := newStore(t)
		now := clockStart
		return s, func(d time.Duration) {
			now = now.Add(d)
			mr.SetTime(now)
		}
	})
}

func TestKeysAndTTL(t *testing.T) {
	s, mr := newStore(t)
	ctx := context.Background()
	at := clockStart.Add(1234 * time.Microsecond)
	mr.SetTime(at)
	bk := accounting.BudgetKey{Tenant: "tenant-a", Day: "20261027"}

	if d, err := s.Allow(ctx, "tenant-a", accounting.KindRPM, 10, 60); err != nil || !d.OK {
		t.Fatalf("Allow rpm = %+v, %v", d, err)
	}
	if d, err := s.Allow(ctx, "tenant-a", accounting.KindTPM, 500, 1000); err != nil || !d.OK {
		t.Fatalf("Allow tpm = %+v, %v", d, err)
	}
	if ok, err := s.Reserve(ctx, bk, 40, 100); err != nil || !ok {
		t.Fatalf("Reserve = %v, %v", ok, err)
	}
	if _, err := s.Settle(ctx, bk, 40, 130, 100); err != nil {
		t.Fatalf("Settle: %v", err)
	}

	keys := mr.Keys()
	slices.Sort(keys)
	wantKeys := []string{"rf:budget:{tenant-a}:20261027", "rf:rl:{tenant-a}:rpm", "rf:rl:{tenant-a}:tpm"}
	if !slices.Equal(keys, wantKeys) {
		t.Fatalf("keys = %q, want %q", keys, wantKeys)
	}
	for _, k := range keys {
		if got := mr.TTL(k); got != 48*time.Hour {
			t.Errorf("TTL(%s) = %v, want 48h", k, got)
		}
	}

	// Buckets hold micro-tokens and the server clock in microseconds; budgets
	// hold integer nano-USD.
	for _, tc := range []struct{ key, field, want string }{
		{"rf:rl:{tenant-a}:rpm", "tokens", "50000000"},
		{"rf:rl:{tenant-a}:rpm", "ts_us", strconv.FormatInt(at.UnixMicro(), 10)},
		{"rf:rl:{tenant-a}:tpm", "tokens", "500000000"},
		{"rf:rl:{tenant-a}:tpm", "ts_us", strconv.FormatInt(at.UnixMicro(), 10)},
		{"rf:budget:{tenant-a}:20261027", "spent", "130"},
		{"rf:budget:{tenant-a}:20261027", "reserved", "0"},
		{"rf:budget:{tenant-a}:20261027", "overshoot", "30"},
	} {
		if got := mr.HGet(tc.key, tc.field); got != tc.want {
			t.Errorf("%s %s = %q, want %q", tc.key, tc.field, got, tc.want)
		}
	}

	t.Run("a read leaves the expiry alone and a write renews it", func(t *testing.T) {
		mr.FastForward(30 * time.Hour)
		if _, err := s.State(ctx, bk); err != nil {
			t.Fatal(err)
		}
		if got := mr.TTL("rf:budget:{tenant-a}:20261027"); got != 18*time.Hour {
			t.Errorf("TTL after a read = %v, want 18h", got)
		}
		if _, err := s.Settle(ctx, bk, 0, 1, 100); err != nil {
			t.Fatal(err)
		}
		if got := mr.TTL("rf:budget:{tenant-a}:20261027"); got != 48*time.Hour {
			t.Errorf("TTL after Settle = %v, want 48h", got)
		}
		if _, err := s.Allow(ctx, "tenant-a", accounting.KindRPM, 1, 60); err != nil {
			t.Fatal(err)
		}
		if got := mr.TTL("rf:rl:{tenant-a}:rpm"); got != 48*time.Hour {
			t.Errorf("TTL after Allow = %v, want 48h", got)
		}
		mr.FastForward(5 * time.Hour)
		ok, err := s.Reserve(ctx, bk, 1, 1000)
		if err != nil || !ok {
			t.Fatalf("Reserve = %v, %v", ok, err)
		}
		if got := mr.TTL("rf:budget:{tenant-a}:20261027"); got != 48*time.Hour {
			t.Errorf("TTL after Reserve = %v, want 48h", got)
		}
		mr.FastForward(time.Hour)
		if ok, err := s.Reserve(ctx, bk, 10_000, 1000); err != nil || ok {
			t.Fatalf("Reserve above the limit = %v, %v; want a rejection", ok, err)
		}
		if got := mr.TTL("rf:budget:{tenant-a}:20261027"); got != 47*time.Hour {
			t.Errorf("TTL after a rejected Reserve = %v, want 47h (nothing was written)", got)
		}
	})

	t.Run("expired keys start from scratch", func(t *testing.T) {
		mr.FastForward(48 * time.Hour)
		if len(mr.Keys()) != 0 {
			t.Fatalf("keys after 48h = %q, want none", mr.Keys())
		}
		st, err := s.State(ctx, bk)
		if err != nil || st != (accounting.BudgetState{}) {
			t.Fatalf("State = %+v, %v; want zeros", st, err)
		}
		if d, err := s.Allow(ctx, "tenant-a", accounting.KindRPM, 60, 60); err != nil || !d.OK {
			t.Fatalf("Allow of a full bucket = %+v, %v", d, err)
		}
	})
}

func TestBackendDownReturnsError(t *testing.T) {
	s, mr := newStore(t)
	mr.Close()
	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()
	k := accounting.BudgetKey{Tenant: "a", Day: "20261027"}

	t.Run("allow", func(t *testing.T) {
		d, err := s.Allow(ctx, "a", accounting.KindRPM, 1, 60)
		if err == nil {
			t.Fatalf("Allow = %+v, want an error", d)
		}
		if d.OK {
			t.Errorf("Allow with the backend down allowed the request: %+v", d)
		}
	})
	t.Run("reserve", func(t *testing.T) {
		ok, err := s.Reserve(ctx, k, 1, 100)
		if err == nil {
			t.Fatalf("Reserve = %v, want an error", ok)
		}
		if ok {
			t.Error("Reserve with the backend down admitted the request")
		}
	})
	t.Run("settle", func(t *testing.T) {
		if o, err := s.Settle(ctx, k, 1, 1, 100); err == nil {
			t.Fatalf("Settle = %d, want an error", o)
		}
	})
	t.Run("state", func(t *testing.T) {
		if st, err := s.State(ctx, k); err == nil {
			t.Fatalf("State = %+v, want an error", st)
		}
	})
	t.Run("ping", func(t *testing.T) {
		if err := s.Ping(ctx); err == nil {
			t.Fatal("Ping succeeded with the backend down")
		}
	})
}

func TestReplicasShareTheServerClock(t *testing.T) {
	mr := newServer(t)
	a := redisstore.New(newClient(t, mr))
	b := redisstore.New(newClient(t, mr))
	ctx := context.Background()

	if d, err := a.Allow(ctx, "t", accounting.KindRPM, 60, 60); err != nil || !d.OK {
		t.Fatalf("a.Allow = %+v, %v", d, err)
	}
	if d, err := b.Allow(ctx, "t", accounting.KindRPM, 1, 60); err != nil || d.OK || d.RetryAfter != time.Second {
		t.Fatalf("b.Allow after a drained the bucket = %+v, %v; want a 1s denial", d, err)
	}
	mr.SetTime(clockStart.Add(30 * time.Second))
	if d, err := b.Allow(ctx, "t", accounting.KindRPM, 30, 60); err != nil || !d.OK {
		t.Fatalf("b.Allow after 30s = %+v, %v", d, err)
	}
	if d, err := a.Allow(ctx, "t", accounting.KindRPM, 1, 60); err != nil || d.OK {
		t.Fatalf("a.Allow after b took the refill = %+v, %v; want a denial", d, err)
	}
}

func TestScriptsRunAfterAScriptFlush(t *testing.T) {
	mr := newServer(t)
	c := newClient(t, mr)
	s := redisstore.New(c)
	ctx := context.Background()
	k := accounting.BudgetKey{Tenant: "a", Day: "20261027"}

	// A server that lost its script cache answers EVALSHA with NOSCRIPT; the
	// store must fall back to sending the script.
	flush := func(t *testing.T) {
		t.Helper()
		if err := c.ScriptFlush(ctx).Err(); err != nil {
			t.Fatalf("SCRIPT FLUSH: %v", err)
		}
	}
	flush(t)
	if d, err := s.Allow(ctx, "a", accounting.KindRPM, 1, 60); err != nil || !d.OK {
		t.Fatalf("Allow after a flush = %+v, %v", d, err)
	}
	flush(t)
	if ok, err := s.Reserve(ctx, k, 5, 100); err != nil || !ok {
		t.Fatalf("Reserve after a flush = %v, %v", ok, err)
	}
	flush(t)
	if _, err := s.Settle(ctx, k, 5, 3, 100); err != nil {
		t.Fatalf("Settle after a flush: %v", err)
	}
	if st, err := s.State(ctx, k); err != nil || st != (accounting.BudgetState{Spent: 3}) {
		t.Fatalf("State = %+v, %v; want spent 3", st, err)
	}
}

func TestBudgetStoreMatchesMemory(t *testing.T) {
	// The memory store is the reference: the same random operations must give
	// the same answers and the same totals.
	ctx := context.Background()
	s, _ := newStore(t)
	mem := accounting.NewMemoryBudgets()
	rng := rand.New(rand.NewPCG(3, 5))
	keys := []accounting.BudgetKey{
		{Tenant: "a", Day: "20261027"},
		{Tenant: "a", Day: "20261028"},
		{Tenant: "b", Day: "20261027"},
	}
	held := make(map[accounting.BudgetKey][]int64)

	for step := range 3000 {
		k := keys[rng.IntN(len(keys))]
		limit := int64(100 + rng.IntN(400))
		if rng.IntN(2) == 0 {
			amount := rng.Int64N(150)
			got, err := s.Reserve(ctx, k, amount, limit)
			if err != nil {
				t.Fatalf("step %d: Reserve: %v", step, err)
			}
			want, _ := mem.Reserve(ctx, k, amount, limit)
			if got != want {
				t.Fatalf("step %d: Reserve(%+v, %d, limit %d) = %v, memory says %v", step, k, amount, limit, got, want)
			}
			if got {
				held[k] = append(held[k], amount)
			}
		} else {
			// Settle an earlier reservation, or a charge that never reserved.
			var reserved int64
			if h := held[k]; len(h) > 0 && rng.IntN(4) != 0 {
				i := rng.IntN(len(h))
				reserved = h[i]
				held[k] = slices.Delete(h, i, i+1)
			}
			actual := rng.Int64N(200)
			got, err := s.Settle(ctx, k, reserved, actual, limit)
			if err != nil {
				t.Fatalf("step %d: Settle: %v", step, err)
			}
			want, _ := mem.Settle(ctx, k, reserved, actual, limit)
			if got != want {
				t.Fatalf("step %d: Settle(%+v, %d, %d, limit %d) overshoot = %d, memory says %d", step, k, reserved, actual, limit, got, want)
			}
		}
		gotState, err := s.State(ctx, k)
		if err != nil {
			t.Fatalf("step %d: State: %v", step, err)
		}
		wantState, _ := mem.State(ctx, k)
		if gotState != wantState {
			t.Fatalf("step %d: State(%+v) = %+v, memory says %+v", step, k, gotState, wantState)
		}
	}
}

func TestBudgetStoreKeepsLargeAmountsExact(t *testing.T) {
	s, _ := newStore(t)
	ctx := context.Background()
	k := accounting.BudgetKey{Tenant: "a", Day: "20261027"}
	const budgetCap = int64(1) << 50 // the largest daily budget, in nano-USD

	if ok, err := s.Reserve(ctx, k, budgetCap, budgetCap); err != nil || !ok {
		t.Fatalf("Reserve of the whole budget = %v, %v", ok, err)
	}
	if ok, err := s.Reserve(ctx, k, 1, budgetCap); err != nil || ok {
		t.Fatalf("Reserve of one more nano = %v, %v; want a rejection", ok, err)
	}
	if o, err := s.Settle(ctx, k, budgetCap, budgetCap-1, budgetCap); err != nil || o != 0 {
		t.Fatalf("Settle = %d, %v; want no overshoot", o, err)
	}
	if st, err := s.State(ctx, k); err != nil || st != (accounting.BudgetState{Spent: budgetCap - 1}) {
		t.Fatalf("State = %+v, %v", st, err)
	}
	// A tenant without a budget settles against the largest limit: its spend is
	// recorded and never counts as overshoot.
	if o, err := s.Settle(ctx, k, 0, budgetCap, math.MaxInt64); err != nil || o != 0 {
		t.Fatalf("Settle with no limit = %d, %v; want no overshoot", o, err)
	}
	if st, err := s.State(ctx, k); err != nil || st != (accounting.BudgetState{Spent: 2*budgetCap - 1}) {
		t.Fatalf("State = %+v, %v; want spent %d", st, err, 2*budgetCap-1)
	}
}

func TestBudgetStoreRejectsAmountsItCannotKeepExact(t *testing.T) {
	s, _ := newStore(t)
	ctx := context.Background()
	k := accounting.BudgetKey{Tenant: "a", Day: "20261027"}
	const tooBig = int64(1)<<50 + 1

	if ok, err := s.Reserve(ctx, k, -1, 100); err == nil {
		t.Errorf("Reserve of -1 = %v, want an error", ok)
	}
	if ok, err := s.Reserve(ctx, k, tooBig, math.MaxInt64); err == nil {
		t.Errorf("Reserve above 2^50 = %v, want an error", ok)
	}
	if o, err := s.Settle(ctx, k, -1, 0, 100); err == nil {
		t.Errorf("Settle with reserved -1 = %d, want an error", o)
	}
	if o, err := s.Settle(ctx, k, 0, -1, 100); err == nil {
		t.Errorf("Settle with actual -1 = %d, want an error", o)
	}
	if o, err := s.Settle(ctx, k, tooBig, 0, 100); err == nil {
		t.Errorf("Settle with reserved above 2^50 = %d, want an error", o)
	}
	if o, err := s.Settle(ctx, k, 0, tooBig, 100); err == nil {
		t.Errorf("Settle with actual above 2^50 = %d, want an error", o)
	}
	if st, err := s.State(ctx, k); err != nil || st != (accounting.BudgetState{}) {
		t.Errorf("State after rejected calls = %+v, %v; want zeros", st, err)
	}
}

func TestStateReportsACorruptField(t *testing.T) {
	s, mr := newStore(t)
	mr.HSet("rf:budget:{a}:20261027", "spent", "not-a-number")
	st, err := s.State(context.Background(), accounting.BudgetKey{Tenant: "a", Day: "20261027"})
	if err == nil {
		t.Fatalf("State = %+v, want an error for a field that is not an integer", st)
	}
}

func TestRateLimiterMatchesMemory(t *testing.T) {
	// The memory limiter is the reference. Rates that are multiples of 60 and
	// whole-second steps keep its float arithmetic exact, so the two must agree
	// on every decision and every wait.
	ctx := context.Background()
	for _, perMinute := range []int64{60, 120, 600, 6000} {
		t.Run(strconv.FormatInt(perMinute, 10), func(t *testing.T) {
			s, mr := newStore(t)
			now := clockStart
			mem := accounting.NewMemoryLimiter(func() time.Time { return now })
			rng := rand.New(rand.NewPCG(uint64(perMinute), 9))
			for step := range 1500 {
				switch rng.IntN(10) {
				case 0:
					now = now.Add(time.Duration(30+rng.IntN(90)) * time.Second)
				default:
					now = now.Add(time.Duration(rng.IntN(4)) * time.Second)
				}
				mr.SetTime(now)
				n := rng.Int64N(perMinute/4 + 1)
				if rng.IntN(25) == 0 {
					n = perMinute + 1 + rng.Int64N(10)
				}
				kind := accounting.KindRPM
				if rng.IntN(2) == 0 {
					kind = accounting.KindTPM
				}
				got, err := s.Allow(ctx, "t", kind, n, perMinute)
				if err != nil {
					t.Fatalf("step %d: Allow: %v", step, err)
				}
				want, _ := mem.Allow(ctx, "t", kind, n, perMinute)
				if got != want {
					t.Fatalf("step %d: Allow(%s, %d of %d) = %+v, memory says %+v", step, kind, n, perMinute, got, want)
				}
			}
		})
	}
}

func TestRateLimiterRateThatDoesNotDivideAMinute(t *testing.T) {
	// 7 a minute is one token every 8.57 s. Refill is kept in whole
	// micro-tokens, so 8 s is 0.933333 of a token.
	s, mr := newStore(t)
	ctx := context.Background()
	allow := func(n int64) accounting.RateDecision {
		t.Helper()
		d, err := s.Allow(ctx, "a", accounting.KindRPM, n, 7)
		if err != nil {
			t.Fatal(err)
		}
		return d
	}
	if d := allow(7); !d.OK {
		t.Fatalf("a full bucket denied 7: %+v", d)
	}
	mr.SetTime(clockStart.Add(8 * time.Second))
	if d := allow(1); d.OK || d.RetryAfter != time.Second {
		t.Fatalf("after 8s = %+v, want a 1s denial", d)
	}
	mr.SetTime(clockStart.Add(9 * time.Second))
	if d := allow(1); !d.OK {
		t.Fatalf("after 9s = %+v, want allowed", d)
	}
}

func TestRateLimiterAtTheLargestRate(t *testing.T) {
	// 100 million a minute is the largest rate: its products come closest to
	// 2^53, and the waits must still be exact.
	const rate = 100_000_000
	s, mr := newStore(t)
	ctx := context.Background()
	allow := func(n int64) accounting.RateDecision {
		t.Helper()
		d, err := s.Allow(ctx, "a", accounting.KindTPM, n, rate)
		if err != nil {
			t.Fatal(err)
		}
		return d
	}
	if d := allow(rate); !d.OK {
		t.Fatalf("a full bucket denied %d: %+v", rate, d)
	}
	if d := allow(1); d.OK || d.RetryAfter != time.Second {
		t.Fatalf("an empty bucket = %+v, want a 1s denial", d)
	}
	mr.SetTime(clockStart.Add(30 * time.Second))
	if d := allow(rate); d.OK || d.RetryAfter != 30*time.Second {
		t.Fatalf("half a bucket = %+v, want a 30s denial", d)
	}
	mr.SetTime(clockStart.Add(24 * time.Hour))
	if d := allow(rate); !d.OK {
		t.Fatalf("after a day = %+v, want allowed", d)
	}
}

func TestRateLimiterKeepsRefillsSmallerThanOneMicroToken(t *testing.T) {
	// At 1 a minute a microsecond refills a sixtieth of a micro-token. Calls
	// ten microseconds apart must still add up to the refill of the whole
	// span instead of each one rounding to nothing.
	s, mr := newStore(t)
	ctx := context.Background()
	if d, err := s.Allow(ctx, "a", accounting.KindRPM, 1, 1); err != nil || !d.OK {
		t.Fatalf("Allow = %+v, %v", d, err)
	}
	for i := 1; i <= 60; i++ {
		mr.SetTime(clockStart.Add(time.Duration(10*i) * time.Microsecond))
		if d, err := s.Allow(ctx, "a", accounting.KindRPM, 1, 1); err != nil || d.OK {
			t.Fatalf("call %d = %+v, %v; want a denial", i, d, err)
		}
	}
	// 600 us at 1/60 micro-token a microsecond.
	if got := mr.HGet("rf:rl:{a}:rpm", "tokens"); got != "10" {
		t.Errorf("tokens after 600us = %q micro-tokens, want 10", got)
	}
}

func TestRateLimiterIgnoresAClockThatGoesBack(t *testing.T) {
	s, mr := newStore(t)
	ctx := context.Background()
	allow := func(n int64) accounting.RateDecision {
		t.Helper()
		d, err := s.Allow(ctx, "a", accounting.KindRPM, n, 60)
		if err != nil {
			t.Fatal(err)
		}
		return d
	}
	if d := allow(60); !d.OK {
		t.Fatalf("Allow = %+v", d)
	}
	mr.SetTime(clockStart.Add(-time.Hour))
	if d := allow(1); d.OK {
		t.Fatalf("after the clock went back Allow = %+v; want a denial", d)
	}
	// Time moving on from the rewound reading must not refill what the earlier
	// reading already counted: one second after the first call is one token.
	mr.SetTime(clockStart.Add(time.Second))
	if d := allow(1); !d.OK {
		t.Fatalf("Allow = %+v; want one token after one second", d)
	}
	if d := allow(1); d.OK {
		t.Fatalf("Allow = %+v; want the bucket empty again", d)
	}
}

func TestRateLimiterAdoptsALowerRate(t *testing.T) {
	// A tenant whose limit is lowered keeps no more than the new capacity.
	s, _ := newStore(t)
	ctx := context.Background()
	for _, tc := range []struct {
		n, perMinute int64
		want         bool
	}{
		{1, 600, true},
		{11, 10, false},
		{10, 10, true},
		{1, 10, false},
	} {
		d, err := s.Allow(ctx, "a", accounting.KindRPM, tc.n, tc.perMinute)
		if err != nil || d.OK != tc.want {
			t.Fatalf("Allow of %d against %d a minute = %+v, %v; want allowed=%v", tc.n, tc.perMinute, d, err, tc.want)
		}
	}
}

func TestRateLimiterRejectsBadArguments(t *testing.T) {
	s, mr := newStore(t)
	ctx := context.Background()
	for _, tc := range []struct {
		name         string
		n, perMinute int64
	}{
		{"zero rate", 1, 0},
		{"negative rate", 1, -5},
		{"negative amount", -1, 60},
		{"a rate whose arithmetic cannot stay exact", 1, 1 << 40},
	} {
		t.Run(tc.name, func(t *testing.T) {
			if d, err := s.Allow(ctx, "a", accounting.KindRPM, tc.n, tc.perMinute); err == nil {
				t.Fatalf("Allow = %+v, want an error", d)
			}
			if len(mr.Keys()) != 0 {
				t.Errorf("a rejected call wrote %q", mr.Keys())
			}
		})
	}
}
