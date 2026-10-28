package accounting_test

import (
	"context"
	"testing"
	"time"

	"github.com/omorsi45/routeforge/internal/accounting"
	"github.com/omorsi45/routeforge/internal/accounting/storetest"
)

func TestMemoryBudgetsContract(t *testing.T) {
	storetest.RunBudgetStore(t, func(*testing.T) accounting.BudgetStore {
		return accounting.NewMemoryBudgets()
	})
}

func TestDayKeyIsUTC(t *testing.T) {
	west := time.FixedZone("west", -5*3600)
	east := time.FixedZone("east", 9*3600)
	for _, tc := range []struct {
		name string
		at   time.Time
		want string
	}{
		{"utc", time.Date(2026, 10, 27, 12, 0, 0, 0, time.UTC), "20261027"},
		{"last nanosecond of the day", time.Date(2026, 10, 27, 23, 59, 59, 999_999_999, time.UTC), "20261027"},
		{"first nanosecond of the day", time.Date(2026, 10, 28, 0, 0, 0, 0, time.UTC), "20261028"},
		{"evening in a western zone is already tomorrow in UTC", time.Date(2026, 10, 27, 20, 30, 0, 0, west), "20261028"},
		{"morning in an eastern zone is still yesterday in UTC", time.Date(2026, 10, 28, 7, 0, 0, 0, east), "20261027"},
		{"year end", time.Date(2026, 12, 31, 23, 0, 0, 0, time.UTC), "20261231"},
	} {
		t.Run(tc.name, func(t *testing.T) {
			if got := accounting.DayKey(tc.at); got != tc.want {
				t.Errorf("DayKey(%v) = %q, want %q", tc.at, got, tc.want)
			}
		})
	}
}

func TestMemoryBudgetsHonourASettledContext(t *testing.T) {
	// The memory store does no I/O, so a canceled context must not stop the
	// settlement of money that was already spent.
	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	s := accounting.NewMemoryBudgets()
	k := accounting.BudgetKey{Tenant: "a", Day: "20261027"}
	if _, err := s.Settle(ctx, k, 0, 10, 100); err != nil {
		t.Fatalf("Settle with a canceled context: %v", err)
	}
	st, err := s.State(ctx, k)
	if err != nil || st.Spent != 10 {
		t.Fatalf("State = %+v, %v; want spent 10", st, err)
	}
}
