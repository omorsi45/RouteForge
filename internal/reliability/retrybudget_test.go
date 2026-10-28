package reliability

import (
	"sync"
	"sync/atomic"
	"testing"
	"time"
)

// specBudget is the default retry budget of spec 10 with the floor of Ruling R5.
func specBudget() (*RetryBudget, *fakeClock) {
	clk := newFakeClock()
	return NewRetryBudget(0.2, 10*time.Second, 3, clk.Now), clk
}

// drain takes retries until one is refused and returns how many were granted.
func drain(r *RetryBudget) int {
	n := 0
	for r.TryRetry() {
		n++
		if n > 10000 {
			panic("retry budget never refuses")
		}
	}
	return n
}

func recordRequests(r *RetryBudget, n int) {
	for range n {
		r.RecordRequest()
	}
}

func TestRetryBudgetFloorAndRatio(t *testing.T) {
	t.Run("no requests still allow the floor", func(t *testing.T) {
		r, _ := specBudget()
		if got := drain(r); got != 3 {
			t.Fatalf("retries with 0 requests = %d, want 3", got)
		}
	})

	t.Run("one hundred requests allow twenty three", func(t *testing.T) {
		r, _ := specBudget()
		recordRequests(r, 100)
		if got := drain(r); got != 23 {
			t.Fatalf("retries with 100 requests = %d, want floor(0.2*100)+3 = 23", got)
		}
	})

	t.Run("the ratio is floored", func(t *testing.T) {
		r, _ := specBudget()
		recordRequests(r, 14) // floor(2.8) + 3
		if got := drain(r); got != 5 {
			t.Fatalf("retries with 14 requests = %d, want 5", got)
		}
	})

	t.Run("later requests raise the allowance", func(t *testing.T) {
		r, _ := specBudget()
		if got := drain(r); got != 3 {
			t.Fatalf("first drain = %d, want 3", got)
		}
		recordRequests(r, 10) // floor(2) + 3 = 5 in all, 3 already used
		if got := drain(r); got != 2 {
			t.Fatalf("second drain = %d, want 2", got)
		}
	})

	t.Run("a refused retry is not counted", func(t *testing.T) {
		r, _ := specBudget()
		drain(r)
		for range 5 {
			if r.TryRetry() {
				t.Fatal("TryRetry granted a retry past the budget")
			}
		}
		recordRequests(r, 5) // floor(1) + 3 = 4 in all, 3 used: exactly one more
		if got := drain(r); got != 1 {
			t.Fatalf("drain after refusals = %d, want 1", got)
		}
	})
}

func TestRetryBudgetWindowSlides(t *testing.T) {
	t.Run("a full window frees the whole budget", func(t *testing.T) {
		r, clk := specBudget()
		recordRequests(r, 100)
		if got := drain(r); got != 23 {
			t.Fatalf("drain = %d, want 23", got)
		}
		clk.Advance(10 * time.Second)
		if got := drain(r); got != 3 {
			t.Fatalf("drain after the window slid = %d, want the floor of 3", got)
		}
	})

	t.Run("nothing is freed inside the window", func(t *testing.T) {
		r, clk := specBudget()
		drain(r)
		clk.Advance(9 * time.Second)
		if r.TryRetry() {
			t.Fatal("TryRetry granted a retry 9 s after the budget was spent")
		}
	})

	t.Run("requests leave while their retries stay", func(t *testing.T) {
		r, clk := specBudget()
		recordRequests(r, 50) // allowance floor(10) + 3 = 13
		clk.Advance(6 * time.Second)
		recordRequests(r, 50) // 100 requests in the window: 23
		if got := drain(r); got != 23 {
			t.Fatalf("drain = %d, want 23", got)
		}
		clk.Advance(4 * time.Second) // the first 50 requests leave, the 23 retries stay
		if r.TryRetry() {
			t.Fatal("TryRetry granted a retry although 23 retries exceed the 13 allowed by 50 requests")
		}
	})

	t.Run("retries leave while their requests stay", func(t *testing.T) {
		r, clk := specBudget()
		if got := drain(r); got != 3 {
			t.Fatalf("first drain = %d, want 3", got)
		}
		clk.Advance(6 * time.Second)
		recordRequests(r, 100)
		clk.Advance(4 * time.Second) // the 3 retries leave, the 100 requests stay
		if got := drain(r); got != 23 {
			t.Fatalf("drain = %d, want 23 (it would be 20 if the early retries still counted)", got)
		}
	})
}

func TestRetryBudgetConcurrent(t *testing.T) {
	r, _ := specBudget()
	var (
		granted atomic.Int32
		wg      sync.WaitGroup
	)
	for range 16 {
		wg.Add(1)
		go func() {
			defer wg.Done()
			for range 50 {
				if r.TryRetry() {
					granted.Add(1)
				}
			}
		}()
	}
	wg.Wait()
	if got := granted.Load(); got != 3 {
		t.Fatalf("%d retries granted to concurrent callers, want exactly the floor of 3", got)
	}
}

func TestNewRetryBudgetDefaultsClock(t *testing.T) {
	r := NewRetryBudget(0.2, 10*time.Second, 3, nil)
	r.RecordRequest()
	if !r.TryRetry() {
		t.Fatal("TryRetry refused with a fresh budget")
	}
}
