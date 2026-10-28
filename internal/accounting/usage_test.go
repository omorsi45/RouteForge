package accounting_test

import (
	"context"
	"fmt"
	"sync"
	"testing"
	"time"

	"github.com/omorsi45/routeforge/internal/accounting"
)

func usageRecord(id, tenant, day string, cost int64) accounting.UsageRecord {
	return accounting.UsageRecord{
		ID:          id,
		RequestID:   "req_" + id,
		DecisionID:  "dec_" + id,
		Tenant:      tenant,
		Day:         day,
		Model:       "m",
		Attempt:     1,
		Outcome:     "ok",
		UsageSource: "reported",
		CostNanos:   cost,
		CostSource:  "reported",
		CreatedAt:   time.Date(2026, 10, 27, 12, 0, 0, 0, time.UTC),
	}
}

func TestMemoryUsageTotalsPerTenantDay(t *testing.T) {
	ctx := context.Background()
	u := accounting.NewMemoryUsage(100)
	for _, r := range []accounting.UsageRecord{
		usageRecord("1", "a", "20261027", 100),
		usageRecord("2", "a", "20261027", 250),
		usageRecord("3", "a", "20261028", 7),
		usageRecord("4", "b", "20261027", 9),
	} {
		if err := u.Write(ctx, r); err != nil {
			t.Fatalf("Write(%s): %v", r.ID, err)
		}
	}
	for _, tc := range []struct {
		tenant, day string
		want        accounting.UsageTotals
	}{
		{"a", "20261027", accounting.UsageTotals{Records: 2, CostNanos: 350}},
		{"a", "20261028", accounting.UsageTotals{Records: 1, CostNanos: 7}},
		{"b", "20261027", accounting.UsageTotals{Records: 1, CostNanos: 9}},
		{"b", "20261028", accounting.UsageTotals{}},
		{"nobody", "20261027", accounting.UsageTotals{}},
	} {
		got, err := u.Totals(ctx, tc.tenant, tc.day)
		if err != nil || got != tc.want {
			t.Errorf("Totals(%s, %s) = %+v, %v; want %+v", tc.tenant, tc.day, got, err, tc.want)
		}
	}
}

func TestMemoryUsageKeepsTheMostRecentRecords(t *testing.T) {
	ctx := context.Background()
	u := accounting.NewMemoryUsage(3)
	for i := 1; i <= 5; i++ {
		if err := u.Write(ctx, usageRecord(fmt.Sprint(i), "a", "20261027", int64(i))); err != nil {
			t.Fatal(err)
		}
	}
	var ids []string
	for _, r := range u.Recent() {
		ids = append(ids, r.ID)
	}
	if want := []string{"3", "4", "5"}; fmt.Sprint(ids) != fmt.Sprint(want) {
		t.Errorf("Recent ids = %v, want %v (oldest first)", ids, want)
	}
	// Old records leave the window but never the totals: no write is dropped.
	got, err := u.Totals(ctx, "a", "20261027")
	if want := (accounting.UsageTotals{Records: 5, CostNanos: 15}); err != nil || got != want {
		t.Errorf("Totals = %+v, %v; want %+v", got, err, want)
	}
	if u.Full() {
		t.Error("Full = true, want false: the memory store never refuses a write")
	}
}

func TestMemoryUsageRecentBeforeTheWindowFills(t *testing.T) {
	ctx := context.Background()
	u := accounting.NewMemoryUsage(10)
	if got := u.Recent(); len(got) != 0 {
		t.Fatalf("Recent of an empty store = %v", got)
	}
	for i := 1; i <= 2; i++ {
		if err := u.Write(ctx, usageRecord(fmt.Sprint(i), "a", "20261027", 1)); err != nil {
			t.Fatal(err)
		}
	}
	got := u.Recent()
	if len(got) != 2 || got[0].ID != "1" || got[1].ID != "2" {
		t.Errorf("Recent = %+v, want records 1 and 2", got)
	}
	// The result is a copy.
	got[0].ID = "changed"
	if again := u.Recent(); again[0].ID != "1" {
		t.Errorf("Recent shares its storage: first id is now %q", again[0].ID)
	}
}

func TestMemoryUsageWithoutRetentionStillTotals(t *testing.T) {
	ctx := context.Background()
	u := accounting.NewMemoryUsage(0)
	if err := u.Write(ctx, usageRecord("1", "a", "20261027", 5)); err != nil {
		t.Fatal(err)
	}
	if got := u.Recent(); len(got) != 0 {
		t.Errorf("Recent = %v, want none", got)
	}
	if got, _ := u.Totals(ctx, "a", "20261027"); got.Records != 1 || got.CostNanos != 5 {
		t.Errorf("Totals = %+v, want 1 record costing 5", got)
	}
}

func TestMemoryUsageConcurrentWrites(t *testing.T) {
	ctx := context.Background()
	u := accounting.NewMemoryUsage(10)
	var wg sync.WaitGroup
	for i := range 50 {
		wg.Add(1)
		go func() {
			defer wg.Done()
			if err := u.Write(ctx, usageRecord(fmt.Sprint(i), "a", "20261027", 2)); err != nil {
				t.Errorf("Write: %v", err)
			}
		}()
	}
	wg.Wait()
	got, err := u.Totals(ctx, "a", "20261027")
	if want := (accounting.UsageTotals{Records: 50, CostNanos: 100}); err != nil || got != want {
		t.Errorf("Totals = %+v, %v; want %+v", got, err, want)
	}
	if n := len(u.Recent()); n != 10 {
		t.Errorf("Recent holds %d records, want 10", n)
	}
}

func TestMemoryUsagePing(t *testing.T) {
	if err := accounting.NewMemoryUsage(1).Ping(context.Background()); err != nil {
		t.Fatalf("Ping: %v", err)
	}
}
