package accounting_test

import (
	"context"
	"testing"
	"time"

	"github.com/omorsi45/routeforge/internal/accounting"
	"github.com/omorsi45/routeforge/internal/accounting/storetest"
)

var clockStart = time.Date(2026, 10, 27, 12, 0, 0, 0, time.UTC)

func TestMemoryLimiterContract(t *testing.T) {
	storetest.RunRateLimiter(t, func(*testing.T) (accounting.RateLimiter, func(time.Duration)) {
		now := clockStart
		l := accounting.NewMemoryLimiter(func() time.Time { return now })
		return l, func(d time.Duration) { now = now.Add(d) }
	})
}

func TestMemoryLimiterRejectsBadArguments(t *testing.T) {
	l := accounting.NewMemoryLimiter(func() time.Time { return clockStart })
	ctx := context.Background()
	for _, tc := range []struct {
		name         string
		n, perMinute int64
	}{
		{"zero rate", 1, 0},
		{"negative rate", 1, -5},
		{"negative amount", -1, 60},
	} {
		t.Run(tc.name, func(t *testing.T) {
			if d, err := l.Allow(ctx, "a", "rpm", tc.n, tc.perMinute); err == nil {
				t.Fatalf("Allow = %+v, want an error", d)
			}
		})
	}
}

func TestMemoryLimiterIgnoresAClockThatGoesBack(t *testing.T) {
	now := clockStart
	l := accounting.NewMemoryLimiter(func() time.Time { return now })
	ctx := context.Background()
	if d, err := l.Allow(ctx, "a", "rpm", 60, 60); err != nil || !d.OK {
		t.Fatalf("Allow = %+v, %v", d, err)
	}
	now = now.Add(-time.Hour)
	if d, err := l.Allow(ctx, "a", "rpm", 1, 60); err != nil || d.OK {
		t.Fatalf("after the clock went back Allow = %+v, %v; want a denial", d, err)
	}
	// Time moving on from the rewound reading must not refill what the earlier
	// reading already counted: one second after the first call is one token.
	now = clockStart.Add(time.Second)
	if d, err := l.Allow(ctx, "a", "rpm", 1, 60); err != nil || !d.OK {
		t.Fatalf("Allow = %+v, %v; want one token after one second", d, err)
	}
	if d, err := l.Allow(ctx, "a", "rpm", 1, 60); err != nil || d.OK {
		t.Fatalf("Allow = %+v, %v; want the bucket empty again", d, err)
	}
}

func TestMemoryLimiterAdoptsALowerRate(t *testing.T) {
	// A tenant whose limit is lowered keeps no more than the new capacity.
	l := accounting.NewMemoryLimiter(func() time.Time { return clockStart })
	ctx := context.Background()
	if d, err := l.Allow(ctx, "a", "rpm", 1, 600); err != nil || !d.OK {
		t.Fatalf("Allow = %+v, %v", d, err)
	}
	if d, err := l.Allow(ctx, "a", "rpm", 11, 10); err != nil || d.OK {
		t.Fatalf("Allow of 11 against 10 a minute = %+v, %v; want a denial", d, err)
	}
	if d, err := l.Allow(ctx, "a", "rpm", 10, 10); err != nil || !d.OK {
		t.Fatalf("Allow of 10 against 10 a minute = %+v, %v; want allowed", d, err)
	}
	if d, err := l.Allow(ctx, "a", "rpm", 1, 10); err != nil || d.OK {
		t.Fatalf("Allow after the lowered capacity is spent = %+v, %v; want a denial", d, err)
	}
}
