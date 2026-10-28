// Package storetest holds the contract tests every accounting store must pass.
// The memory stores run them in this repository; the Redis stores run the same
// tests, so all backends agree on admission, settlement and refill.
package storetest

import (
	"context"
	"math/rand/v2"
	"sync"
	"testing"
	"time"

	"github.com/omorsi45/routeforge/internal/accounting"
)

// RunBudgetStore runs the BudgetStore contract. newStore returns an empty store
// and is called once per subtest.
func RunBudgetStore(t *testing.T, newStore func(t *testing.T) accounting.BudgetStore) {
	t.Helper()
	ctx := context.Background()
	key := accounting.BudgetKey{Tenant: "tenant-a", Day: "20261027"}

	mustReserve := func(t *testing.T, s accounting.BudgetStore, k accounting.BudgetKey, amount, limit int64, want bool) {
		t.Helper()
		got, err := s.Reserve(ctx, k, amount, limit)
		if err != nil {
			t.Fatalf("Reserve(%d, limit %d): %v", amount, limit, err)
		}
		if got != want {
			t.Fatalf("Reserve(%d, limit %d) = %v, want %v", amount, limit, got, want)
		}
	}
	mustSettle := func(t *testing.T, s accounting.BudgetStore, k accounting.BudgetKey, reserved, actual, limit, wantOvershoot int64) {
		t.Helper()
		got, err := s.Settle(ctx, k, reserved, actual, limit)
		if err != nil {
			t.Fatalf("Settle(%d, %d, limit %d): %v", reserved, actual, limit, err)
		}
		if got != wantOvershoot {
			t.Fatalf("Settle(%d, %d, limit %d) overshoot = %d, want %d", reserved, actual, limit, got, wantOvershoot)
		}
	}
	wantState := func(t *testing.T, s accounting.BudgetStore, k accounting.BudgetKey, want accounting.BudgetState) {
		t.Helper()
		got, err := s.State(ctx, k)
		if err != nil {
			t.Fatalf("State: %v", err)
		}
		if got != want {
			t.Fatalf("State = %+v, want %+v", got, want)
		}
	}

	t.Run("an unknown key has a zero state", func(t *testing.T) {
		wantState(t, newStore(t), key, accounting.BudgetState{})
	})

	t.Run("reserve admits up to the limit exactly and rejects one nano over", func(t *testing.T) {
		s := newStore(t)
		mustReserve(t, s, key, 60, 100, true)
		mustReserve(t, s, key, 40, 100, true)
		wantState(t, s, key, accounting.BudgetState{Reserved: 100})
		mustReserve(t, s, key, 1, 100, false)
		// A rejection changes nothing.
		wantState(t, s, key, accounting.BudgetState{Reserved: 100})
	})

	t.Run("reserve rejects an amount above the limit on an empty budget", func(t *testing.T) {
		s := newStore(t)
		mustReserve(t, s, key, 101, 100, false)
		wantState(t, s, key, accounting.BudgetState{})
	})

	t.Run("reserve counts what is spent as well as what is reserved", func(t *testing.T) {
		s := newStore(t)
		mustReserve(t, s, key, 50, 100, true)
		mustSettle(t, s, key, 50, 30, 100, 0)
		wantState(t, s, key, accounting.BudgetState{Spent: 30})
		mustReserve(t, s, key, 71, 100, false)
		mustReserve(t, s, key, 70, 100, true)
		wantState(t, s, key, accounting.BudgetState{Spent: 30, Reserved: 70})
	})

	t.Run("settle returns the reservation and charges the actual cost", func(t *testing.T) {
		s := newStore(t)
		mustReserve(t, s, key, 80, 100, true)
		mustSettle(t, s, key, 80, 50, 100, 0)
		wantState(t, s, key, accounting.BudgetState{Spent: 50})
	})

	t.Run("settle of an attempt that cost nothing frees the whole reservation", func(t *testing.T) {
		s := newStore(t)
		mustReserve(t, s, key, 80, 100, true)
		mustSettle(t, s, key, 80, 0, 100, 0)
		wantState(t, s, key, accounting.BudgetState{})
		mustReserve(t, s, key, 100, 100, true)
	})

	t.Run("settle without a reservation still charges", func(t *testing.T) {
		s := newStore(t)
		mustSettle(t, s, key, 0, 40, 100, 0)
		wantState(t, s, key, accounting.BudgetState{Spent: 40})
	})

	t.Run("overshoot is zero while spend stays within the limit", func(t *testing.T) {
		s := newStore(t)
		mustReserve(t, s, key, 100, 100, true)
		mustSettle(t, s, key, 100, 100, 100, 0)
		wantState(t, s, key, accounting.BudgetState{Spent: 100})
	})

	t.Run("overshoot counts only the part beyond the limit when crossing it", func(t *testing.T) {
		s := newStore(t)
		mustSettle(t, s, key, 0, 60, 100, 0)
		mustReserve(t, s, key, 40, 100, true)
		mustSettle(t, s, key, 40, 70, 100, 30)
		wantState(t, s, key, accounting.BudgetState{Spent: 130, Overshoot: 30})
	})

	t.Run("overshoot counts the whole charge when already over the limit", func(t *testing.T) {
		s := newStore(t)
		mustSettle(t, s, key, 0, 130, 100, 30)
		mustSettle(t, s, key, 0, 20, 100, 20)
		wantState(t, s, key, accounting.BudgetState{Spent: 150, Overshoot: 50})
		// Over the limit nothing more is admitted.
		mustReserve(t, s, key, 1, 100, false)
	})

	t.Run("overshoot always equals spend beyond the limit", func(t *testing.T) {
		s := newStore(t)
		rng := rand.New(rand.NewPCG(7, 11))
		const limit = 1000
		var spent int64
		for range 200 {
			actual := rng.Int64N(120)
			spent += actual
			if _, err := s.Settle(ctx, key, 0, actual, limit); err != nil {
				t.Fatal(err)
			}
			st, err := s.State(ctx, key)
			if err != nil {
				t.Fatal(err)
			}
			if want := max(0, spent-limit); st.Spent != spent || st.Overshoot != want {
				t.Fatalf("after spending %d: state = %+v, want spent %d overshoot %d", spent, st, spent, want)
			}
		}
	})

	t.Run("tenants and days have separate budgets", func(t *testing.T) {
		s := newStore(t)
		other := accounting.BudgetKey{Tenant: "tenant-b", Day: key.Day}
		tomorrow := accounting.BudgetKey{Tenant: key.Tenant, Day: "20261028"}
		mustReserve(t, s, key, 100, 100, true)
		mustReserve(t, s, other, 100, 100, true)
		mustReserve(t, s, tomorrow, 100, 100, true)
		mustSettle(t, s, key, 100, 100, 100, 0)
		wantState(t, s, other, accounting.BudgetState{Reserved: 100})
		wantState(t, s, tomorrow, accounting.BudgetState{Reserved: 100})
	})

	t.Run("concurrent reserves never exceed the limit", func(t *testing.T) {
		s := newStore(t)
		const (
			workers = 100
			amount  = 7
			limit   = 100
		)
		var (
			wg       sync.WaitGroup
			start    = make(chan struct{})
			mu       sync.Mutex
			admitted int
		)
		for range workers {
			wg.Add(1)
			go func() {
				defer wg.Done()
				<-start
				ok, err := s.Reserve(ctx, key, amount, limit)
				if err != nil {
					t.Errorf("Reserve: %v", err)
					return
				}
				if ok {
					mu.Lock()
					admitted++
					mu.Unlock()
				}
			}()
		}
		close(start)
		wg.Wait()
		if want := limit / amount; admitted != want {
			t.Errorf("admitted %d reservations of %d against %d, want %d", admitted, amount, limit, want)
		}
		wantState(t, s, key, accounting.BudgetState{Reserved: int64(admitted) * amount})
	})

	t.Run("ping succeeds on a working store", func(t *testing.T) {
		if err := newStore(t).Ping(ctx); err != nil {
			t.Fatalf("Ping: %v", err)
		}
	})
}

// RunRateLimiter runs the RateLimiter contract. newLimiter returns an empty
// limiter and a function that moves the limiter's clock forward; it is called
// once per subtest.
func RunRateLimiter(t *testing.T, newLimiter func(t *testing.T) (accounting.RateLimiter, func(time.Duration))) {
	t.Helper()
	ctx := context.Background()

	allow := func(t *testing.T, l accounting.RateLimiter, tenant, kind string, n, perMinute int64) accounting.RateDecision {
		t.Helper()
		d, err := l.Allow(ctx, tenant, kind, n, perMinute)
		if err != nil {
			t.Fatalf("Allow(%s, %s, %d, %d): %v", tenant, kind, n, perMinute, err)
		}
		return d
	}
	wantAllowed := func(t *testing.T, l accounting.RateLimiter, tenant, kind string, n, perMinute int64) {
		t.Helper()
		if d := allow(t, l, tenant, kind, n, perMinute); !d.OK {
			t.Fatalf("Allow(%s, %s, %d, %d) denied (retry after %v), want allowed", tenant, kind, n, perMinute, d.RetryAfter)
		}
	}
	wantDenied := func(t *testing.T, l accounting.RateLimiter, tenant, kind string, n, perMinute int64, retryAfter time.Duration) {
		t.Helper()
		d := allow(t, l, tenant, kind, n, perMinute)
		if d.OK {
			t.Fatalf("Allow(%s, %s, %d, %d) allowed, want denied", tenant, kind, n, perMinute)
		}
		if d.RetryAfter != retryAfter {
			t.Fatalf("Allow(%s, %s, %d, %d) retry after %v, want %v", tenant, kind, n, perMinute, d.RetryAfter, retryAfter)
		}
	}

	t.Run("a bucket starts full", func(t *testing.T) {
		l, _ := newLimiter(t)
		wantAllowed(t, l, "a", "rpm", 60, 60)
		wantDenied(t, l, "a", "rpm", 1, 60, time.Second)
	})

	t.Run("single requests drain the bucket one token at a time", func(t *testing.T) {
		l, _ := newLimiter(t)
		for i := range 5 {
			if d := allow(t, l, "a", "rpm", 1, 5); !d.OK {
				t.Fatalf("request %d denied, want allowed", i+1)
			}
		}
		wantDenied(t, l, "a", "rpm", 1, 5, 12*time.Second)
	})

	t.Run("a denial reports the wait for the missing tokens", func(t *testing.T) {
		l, _ := newLimiter(t)
		wantAllowed(t, l, "a", "tpm", 60, 60)
		wantDenied(t, l, "a", "tpm", 30, 60, 30*time.Second)
		wantDenied(t, l, "a", "tpm", 60, 60, 60*time.Second)
	})

	t.Run("the wait is rounded up to a whole second", func(t *testing.T) {
		l, advance := newLimiter(t)
		wantAllowed(t, l, "a", "tpm", 60, 60)
		advance(500 * time.Millisecond)
		// Half a token is back; one more needs another half second.
		wantDenied(t, l, "a", "tpm", 1, 60, time.Second)
		// Two need one and a half seconds, reported as two.
		wantDenied(t, l, "a", "tpm", 2, 60, 2*time.Second)
	})

	t.Run("the wait is at least one second", func(t *testing.T) {
		l, _ := newLimiter(t)
		wantAllowed(t, l, "a", "rpm", 6000, 6000)
		// 100 tokens a second: one token is 10 ms away, reported as 1 s.
		wantDenied(t, l, "a", "rpm", 1, 6000, time.Second)
	})

	t.Run("a denial leaves the bucket unchanged", func(t *testing.T) {
		l, advance := newLimiter(t)
		wantAllowed(t, l, "a", "tpm", 60, 60)
		wantDenied(t, l, "a", "tpm", 30, 60, 30*time.Second)
		advance(time.Second)
		// Had the denial taken tokens, one second would not be enough.
		wantAllowed(t, l, "a", "tpm", 1, 60)
	})

	t.Run("tokens refill with time", func(t *testing.T) {
		l, advance := newLimiter(t)
		wantAllowed(t, l, "a", "rpm", 60, 60)
		wantDenied(t, l, "a", "rpm", 1, 60, time.Second)
		advance(30 * time.Second)
		wantAllowed(t, l, "a", "rpm", 30, 60)
		wantDenied(t, l, "a", "rpm", 1, 60, time.Second)
	})

	t.Run("partial refills add up", func(t *testing.T) {
		l, advance := newLimiter(t)
		wantAllowed(t, l, "a", "rpm", 60, 60)
		advance(500 * time.Millisecond)
		wantDenied(t, l, "a", "rpm", 1, 60, time.Second)
		advance(500 * time.Millisecond)
		wantAllowed(t, l, "a", "rpm", 1, 60)
	})

	t.Run("a refill never goes above the capacity", func(t *testing.T) {
		l, advance := newLimiter(t)
		wantAllowed(t, l, "a", "rpm", 60, 60)
		advance(time.Hour)
		wantAllowed(t, l, "a", "rpm", 60, 60)
		wantDenied(t, l, "a", "rpm", 1, 60, time.Second)
	})

	t.Run("a request above the capacity is denied for a minute", func(t *testing.T) {
		l, _ := newLimiter(t)
		wantDenied(t, l, "a", "tpm", 61, 60, time.Minute)
		// The denial took nothing: the full bucket is still there.
		wantAllowed(t, l, "a", "tpm", 60, 60)
	})

	t.Run("tenants and kinds have separate buckets", func(t *testing.T) {
		l, _ := newLimiter(t)
		wantAllowed(t, l, "a", "rpm", 10, 10)
		wantDenied(t, l, "a", "rpm", 1, 10, 6*time.Second)
		wantAllowed(t, l, "b", "rpm", 10, 10)
		wantAllowed(t, l, "a", "tpm", 10, 10)
	})

	t.Run("ping succeeds on a working limiter", func(t *testing.T) {
		l, _ := newLimiter(t)
		if err := l.Ping(ctx); err != nil {
			t.Fatalf("Ping: %v", err)
		}
	})
}
