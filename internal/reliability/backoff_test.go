package reliability

import (
	"testing"
	"time"
)

func TestBackoffFullJitter(t *testing.T) {
	half := func() float64 { return 0.5 }
	b := Backoff{Base: 100 * time.Millisecond, Cap: 2 * time.Second, Rand: half}

	cases := []struct {
		retry int
		want  time.Duration
	}{
		{1, 50 * time.Millisecond},  // 0.5 * 100 ms
		{2, 100 * time.Millisecond}, // 0.5 * 200 ms
		{3, 200 * time.Millisecond}, // 0.5 * 400 ms
		{5, 800 * time.Millisecond}, // 0.5 * 1.6 s
		{6, time.Second},            // 0.5 * min(2 s, 3.2 s)
		{7, time.Second},            // still capped
		{200, time.Second},          // no overflow for a huge retry count
		{0, 0},                      // there is no zeroth retry
		{-3, 0},                     // nor a negative one
	}
	for _, tc := range cases {
		if got := b.Delay(tc.retry); got != tc.want {
			t.Errorf("Delay(%d) = %v, want %v", tc.retry, got, tc.want)
		}
	}
}

func TestBackoffSpansZeroToCeiling(t *testing.T) {
	at := func(r float64) Backoff {
		return Backoff{Base: 100 * time.Millisecond, Cap: 2 * time.Second, Rand: func() float64 { return r }}
	}
	if got := at(0).Delay(3); got != 0 {
		t.Errorf("Delay with Rand 0 = %v, want 0", got)
	}
	if got := at(0.999999).Delay(3); got <= 399*time.Millisecond || got >= 400*time.Millisecond {
		t.Errorf("Delay with Rand just below 1 = %v, want just under 400ms", got)
	}
	if got := at(0.999999).Delay(10); got <= 1999*time.Millisecond || got >= 2*time.Second {
		t.Errorf("capped Delay with Rand just below 1 = %v, want just under 2s", got)
	}
}

func TestBackoffCapBelowBase(t *testing.T) {
	b := Backoff{Base: time.Second, Cap: 100 * time.Millisecond, Rand: func() float64 { return 1 }}
	if got := b.Delay(1); got != 100*time.Millisecond {
		t.Errorf("Delay(1) = %v, want the cap", got)
	}
}

func TestBackoffDefaultRand(t *testing.T) {
	b := Backoff{Base: 100 * time.Millisecond, Cap: 2 * time.Second}
	for range 100 {
		if got := b.Delay(2); got < 0 || got >= 200*time.Millisecond {
			t.Fatalf("Delay(2) = %v, want a value in [0, 200ms)", got)
		}
	}
}
