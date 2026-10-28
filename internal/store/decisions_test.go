package store_test

import (
	"context"
	"fmt"
	"reflect"
	"sync"
	"testing"
	"time"

	"github.com/omorsi45/routeforge/internal/router"
	"github.com/omorsi45/routeforge/internal/store"
)

// fullDecision returns a decision with every pointer and slice field set, so
// a copy that shares any of them is caught.
func fullDecision(id, tenant string) *router.Decision {
	f := func(v float64) *float64 { return &v }
	n := func(v int) *int { return &v }
	return &router.Decision{
		ID:               id,
		Tenant:           tenant,
		CreatedAt:        time.Date(2026, 10, 27, 12, 0, 0, 0, time.UTC),
		PolicyVersion:    "pv-1",
		Strategy:         router.StrategyQuality,
		Profile:          "balanced",
		Chosen:           "m-fast",
		Reason:           router.ReasonCheapestPassing,
		Lambda:           f(0.6),
		GuaranteeApplies: true,
		DecidedInUS:      42,
		Candidates: []router.Candidate{{
			Model:             "m-fast",
			Eligible:          false,
			IneligibleReasons: []string{router.IneligibleBudget},
			Raw:               f(0.5),
			P:                 f(0.9),
			LCB:               f(0.8),
			BlockN:            n(100),
			BlockK:            n(90),
			ExpectedCostUSD:   f(0.001),
			PredictedP95MS:    f(250),
		}},
		Fallbacks: []string{"m-strong"},
		Attempts:  []router.Attempt{{Model: "m-fast", Outcome: router.OutcomeOK, DurationMS: 1.5}},
	}
}

func mustGet(t *testing.T, s store.DecisionStore, tenant, id string) *router.Decision {
	t.Helper()
	d, ok, err := s.Get(context.Background(), tenant, id)
	if err != nil {
		t.Fatalf("Get(%q, %q) error: %v", tenant, id, err)
	}
	if !ok {
		t.Fatalf("Get(%q, %q) found nothing", tenant, id)
	}
	return d
}

func wantMissing(t *testing.T, s store.DecisionStore, tenant, id string) {
	t.Helper()
	d, ok, err := s.Get(context.Background(), tenant, id)
	if err != nil {
		t.Fatalf("Get(%q, %q) error: %v", tenant, id, err)
	}
	if ok || d != nil {
		t.Fatalf("Get(%q, %q) = %+v, %v; want nothing", tenant, id, d, ok)
	}
}

func put(t *testing.T, s store.DecisionStore, d *router.Decision) {
	t.Helper()
	if err := s.Put(d); err != nil {
		t.Fatalf("Put(%s) error: %v", d.ID, err)
	}
}

func TestRingEvictsOldest(t *testing.T) {
	r := store.NewRing(3)
	for i := 1; i <= 4; i++ {
		put(t, r, fullDecision(fmt.Sprintf("dec_%d", i), "tenant-a"))
	}
	wantMissing(t, r, "tenant-a", "dec_1")
	for i := 2; i <= 4; i++ {
		id := fmt.Sprintf("dec_%d", i)
		if got := mustGet(t, r, "tenant-a", id); got.ID != id {
			t.Errorf("Get(%s).ID = %q", id, got.ID)
		}
	}

	// The ring keeps evicting in insertion order as it wraps.
	put(t, r, fullDecision("dec_5", "tenant-a"))
	put(t, r, fullDecision("dec_6", "tenant-a"))
	wantMissing(t, r, "tenant-a", "dec_2")
	wantMissing(t, r, "tenant-a", "dec_3")
	for _, id := range []string{"dec_4", "dec_5", "dec_6"} {
		mustGet(t, r, "tenant-a", id)
	}
}

func TestRingCapacityOne(t *testing.T) {
	r := store.NewRing(1)
	put(t, r, fullDecision("dec_1", "tenant-a"))
	put(t, r, fullDecision("dec_2", "tenant-a"))
	wantMissing(t, r, "tenant-a", "dec_1")
	mustGet(t, r, "tenant-a", "dec_2")
}

func TestRingTenantScoped(t *testing.T) {
	r := store.NewRing(3)
	put(t, r, fullDecision("dec_1", "tenant-a"))

	mustGet(t, r, "tenant-a", "dec_1")
	wantMissing(t, r, "tenant-b", "dec_1")
	wantMissing(t, r, "", "dec_1")
	wantMissing(t, r, "tenant-a", "dec_unknown")
}

func TestRingReturnsCopy(t *testing.T) {
	r := store.NewRing(3)
	put(t, r, fullDecision("dec_1", "tenant-a"))

	got := mustGet(t, r, "tenant-a", "dec_1")
	if !reflect.DeepEqual(got, fullDecision("dec_1", "tenant-a")) {
		t.Fatalf("Get returned %+v, want what was stored", got)
	}

	// Change every pointer and slice the copy holds.
	got.Chosen = "changed"
	*got.Lambda = 9
	got.Fallbacks[0] = "changed"
	got.Attempts[0].Outcome = "changed"
	got.Attempts = append(got.Attempts, router.Attempt{Model: "extra"})
	c := &got.Candidates[0]
	c.IneligibleReasons[0] = "changed"
	*c.Raw, *c.P, *c.LCB, *c.BlockN, *c.BlockK, *c.ExpectedCostUSD, *c.PredictedP95MS = 9, 9, 9, 9, 9, 9, 9

	if again := mustGet(t, r, "tenant-a", "dec_1"); !reflect.DeepEqual(again, fullDecision("dec_1", "tenant-a")) {
		t.Errorf("stored decision changed through a returned copy: %+v", again)
	}
}

func TestRingPutCopies(t *testing.T) {
	r := store.NewRing(3)
	d := fullDecision("dec_1", "tenant-a")
	put(t, r, d)

	// The caller keeps using its decision after Put.
	d.Chosen = "changed"
	*d.Lambda = 9
	d.Attempts[0].Outcome = "changed"
	d.Candidates[0].IneligibleReasons[0] = "changed"
	*d.Candidates[0].P = 9

	if got := mustGet(t, r, "tenant-a", "dec_1"); !reflect.DeepEqual(got, fullDecision("dec_1", "tenant-a")) {
		t.Errorf("stored decision changed through the original: %+v", got)
	}
}

func TestRingNilFieldsSurviveCopy(t *testing.T) {
	r := store.NewRing(3)
	put(t, r, &router.Decision{ID: "dec_1", Tenant: "tenant-a", Candidates: []router.Candidate{{Model: "m"}}})

	got := mustGet(t, r, "tenant-a", "dec_1")
	if got.Lambda != nil || got.Fallbacks != nil || got.Attempts != nil {
		t.Errorf("nil fields became non-nil: lambda %v, fallbacks %#v, attempts %#v", got.Lambda, got.Fallbacks, got.Attempts)
	}
	c := got.Candidates[0]
	if c.Raw != nil || c.P != nil || c.LCB != nil || c.BlockN != nil || c.BlockK != nil || c.ExpectedCostUSD != nil || c.PredictedP95MS != nil || c.IneligibleReasons != nil {
		t.Errorf("nil candidate fields became non-nil: %+v", c)
	}
}

func TestRingPutSameIDReplaces(t *testing.T) {
	r := store.NewRing(2)
	first := fullDecision("dec_1", "tenant-a")
	put(t, r, first)
	put(t, r, fullDecision("dec_2", "tenant-a"))

	second := fullDecision("dec_1", "tenant-a")
	second.Chosen = "updated"
	put(t, r, second)

	if got := mustGet(t, r, "tenant-a", "dec_1"); got.Chosen != "updated" {
		t.Errorf("Chosen = %q, want the replacement", got.Chosen)
	}
	// Replacing does not use a slot: both entries are still there.
	mustGet(t, r, "tenant-a", "dec_2")
}

func TestRingPutNil(t *testing.T) {
	if err := store.NewRing(1).Put(nil); err == nil {
		t.Fatal("Put(nil) error = nil, want an error")
	}
}

func TestNewRingRejectsNonPositiveCapacity(t *testing.T) {
	for _, capacity := range []int{0, -1} {
		func() {
			defer func() {
				if recover() == nil {
					t.Errorf("NewRing(%d) did not panic", capacity)
				}
			}()
			store.NewRing(capacity)
		}()
	}
}

func TestRingConcurrent(t *testing.T) {
	r := store.NewRing(16)
	var wg sync.WaitGroup
	for g := range 8 {
		wg.Add(1)
		go func() {
			defer wg.Done()
			for i := range 200 {
				id := fmt.Sprintf("dec_%d_%d", g, i)
				if err := r.Put(fullDecision(id, "tenant-a")); err != nil {
					t.Errorf("Put(%s): %v", id, err)
					return
				}
				if _, _, err := r.Get(context.Background(), "tenant-a", id); err != nil {
					t.Errorf("Get(%s): %v", id, err)
					return
				}
			}
		}()
	}
	wg.Wait()
}
