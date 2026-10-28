package policy

import (
	"errors"
	"slices"
	"strings"
	"sync"
	"sync/atomic"
	"testing"
	"time"
)

// testClock returns a clock that advances one second per call, starting at a
// fixed instant.
func testClock() func() time.Time {
	base := time.Date(2026, 10, 27, 12, 0, 0, 0, time.UTC)
	var calls atomic.Int64
	return func() time.Time {
		return base.Add(time.Duration(calls.Add(1)) * time.Second)
	}
}

// buildPolicy loads the small committed policy after mutate has edited its
// generic tree. The version is recomputed, so the result is a valid policy.
func buildPolicy(t *testing.T, mutate func(m map[string]any)) *Policy {
	t.Helper()
	m := smallPolicyMap(t)
	if mutate != nil {
		mutate(m)
	}
	p, err := Load(render(t, m, false))
	if err != nil {
		t.Fatalf("Load() error: %v", err)
	}
	return p
}

// policyPair returns two valid policies with different versions: B differs
// from A by one gbm initial value.
func policyPair(t *testing.T) (a, b *Policy) {
	t.Helper()
	a = buildPolicy(t, nil)
	b = buildPolicy(t, func(m map[string]any) {
		setAt(t, m, jn("0.123"), "models", 0, "predictor", "init")
	})
	if a.Version == b.Version {
		t.Fatalf("test policies share version %s", a.Version)
	}
	return a, b
}

func statuses(r *Registry) map[string]string {
	out := make(map[string]string)
	for _, e := range r.List() {
		out[e.Policy.Version] = e.Status
	}
	return out
}

func TestRegistryActivate(t *testing.T) {
	a, b := policyPair(t)
	clock := testClock()
	r := NewRegistry(poolOrder, clock)
	if r.Active() != nil {
		t.Fatal("Active() before any activation is not nil")
	}
	for _, p := range []*Policy{a, b} {
		if err := r.Add(p); err != nil {
			t.Fatalf("Add(%s) error: %v", p.Version, err)
		}
	}
	if r.Active() != nil {
		t.Fatal("Add activated a policy")
	}
	if got := statuses(r); got[a.Version] != StatusInactive || got[b.Version] != StatusInactive {
		t.Errorf("statuses after Add = %v, want both %q", got, StatusInactive)
	}

	act, err := r.Activate(a.Version, "alice")
	if err != nil {
		t.Fatalf("Activate(A) error: %v", err)
	}
	if act.Version != a.Version || act.Previous != "" || act.Actor != "alice" || act.At.IsZero() {
		t.Errorf("Activation = %+v, want version %s, no previous, actor alice, a time", act, a.Version)
	}
	if got := r.Active(); got != a {
		t.Errorf("Active() = %v, want the policy added as %s", got, a.Version)
	}
	if got := statuses(r); got[a.Version] != StatusActive || got[b.Version] != StatusInactive {
		t.Errorf("statuses after Activate(A) = %v, want A active and B inactive", got)
	}
}

func TestRegistryActivateRejectsUnmappedModel(t *testing.T) {
	a, _ := policyPair(t)
	// Without qwen3-235b-a22b-2507 (index 5 in pool order) the policy has a
	// model nothing is configured for.
	r := NewRegistry(poolOrder[:5], testClock())
	err := r.Add(a)
	ve := requireValidationError(t, err)
	if ve.Path != "models[5].name" {
		t.Errorf("error path = %q, want models[5].name", ve.Path)
	}
	if want := "qwen3-235b-a22b-2507"; !errorMentions(err, want) {
		t.Errorf("error %q does not name the model %q", err, want)
	}
	if len(r.List()) != 0 {
		t.Errorf("rejected policy was stored: %v", r.List())
	}
	if _, err := r.Activate(a.Version, "alice"); !errors.Is(err, ErrUnknownVersion) {
		t.Errorf("Activate(rejected) error = %v, want ErrUnknownVersion", err)
	}

	// The first unmapped model, in pool order, is the one reported.
	r = NewRegistry(poolOrder[1:4], testClock())
	if ve := requireValidationError(t, r.Add(a)); ve.Path != "models[0].name" {
		t.Errorf("error path = %q, want models[0].name", ve.Path)
	}
}

func errorMentions(err error, text string) bool {
	return err != nil && strings.Contains(err.Error(), text)
}

func TestRegistryRollback(t *testing.T) {
	a, b := policyPair(t)
	r := NewRegistry(poolOrder, testClock())
	for _, p := range []*Policy{a, b} {
		if err := r.Add(p); err != nil {
			t.Fatalf("Add(%s) error: %v", p.Version, err)
		}
	}
	if _, err := r.Activate(a.Version, "alice"); err != nil {
		t.Fatalf("Activate(A) error: %v", err)
	}
	if _, err := r.Activate(b.Version, "bob"); err != nil {
		t.Fatalf("Activate(B) error: %v", err)
	}
	if got := r.Active(); got != b {
		t.Fatalf("Active() = %v, want B", got)
	}
	if got := statuses(r); got[a.Version] != StatusRetired || got[b.Version] != StatusActive {
		t.Errorf("statuses after Activate(B) = %v, want A retired and B active", got)
	}

	act, err := r.Rollback("carol")
	if err != nil {
		t.Fatalf("Rollback() error: %v", err)
	}
	if act.Version != a.Version || act.Previous != b.Version || act.Actor != "carol" {
		t.Errorf("Rollback activation = %+v, want version A, previous B, actor carol", act)
	}
	if got := r.Active(); got != a {
		t.Errorf("Active() after Rollback = %v, want A", got)
	}
	if got := statuses(r); got[a.Version] != StatusActive || got[b.Version] != StatusInactive {
		t.Errorf("statuses after Rollback = %v, want A active and B inactive", got)
	}

	log := r.Activations()
	if len(log) != 3 {
		t.Fatalf("Activations() has %d entries, want 3: %+v", len(log), log)
	}
	want := []struct{ version, previous, actor string }{
		{a.Version, "", "alice"},
		{b.Version, a.Version, "bob"},
		{a.Version, b.Version, "carol"},
	}
	for i, w := range want {
		if log[i].Version != w.version || log[i].Previous != w.previous || log[i].Actor != w.actor {
			t.Errorf("Activations()[%d] = %+v, want %+v", i, log[i], w)
		}
		if i > 0 && !log[i].At.After(log[i-1].At) {
			t.Errorf("Activations()[%d].At = %v, not after %v", i, log[i].At, log[i-1].At)
		}
	}
}

func TestRegistryRollbackTwiceTogglesBack(t *testing.T) {
	a, b := policyPair(t)
	r := NewRegistry(poolOrder, testClock())
	mustAdd(t, r, a, b)
	mustActivate(t, r, a, "x")
	mustActivate(t, r, b, "x")
	for i, want := range []*Policy{a, b, a} {
		if _, err := r.Rollback("x"); err != nil {
			t.Fatalf("Rollback #%d error: %v", i+1, err)
		}
		if got := r.Active(); got != want {
			t.Errorf("Active() after Rollback #%d = %s, want %s", i+1, got.Version, want.Version)
		}
	}
}

func mustAdd(t *testing.T, r *Registry, ps ...*Policy) {
	t.Helper()
	for _, p := range ps {
		if err := r.Add(p); err != nil {
			t.Fatalf("Add(%s) error: %v", p.Version, err)
		}
	}
}

func mustActivate(t *testing.T, r *Registry, p *Policy, actor string) Activation {
	t.Helper()
	act, err := r.Activate(p.Version, actor)
	if err != nil {
		t.Fatalf("Activate(%s) error: %v", p.Version, err)
	}
	return act
}

func TestRegistryRollbackWithoutPrevious(t *testing.T) {
	a, _ := policyPair(t)
	r := NewRegistry(poolOrder, testClock())
	if _, err := r.Rollback("alice"); !errors.Is(err, ErrNoPrevious) {
		t.Errorf("Rollback() on an empty registry error = %v, want ErrNoPrevious", err)
	}
	mustAdd(t, r, a)
	mustActivate(t, r, a, "alice")
	if _, err := r.Rollback("alice"); !errors.Is(err, ErrNoPrevious) {
		t.Errorf("Rollback() after one activation error = %v, want ErrNoPrevious", err)
	}
	if got := r.Active(); got != a {
		t.Errorf("failed Rollback changed the active policy to %v", got)
	}
	if n := len(r.Activations()); n != 1 {
		t.Errorf("failed Rollback left %d activation records, want 1", n)
	}
}

func TestRegistryActivateUnknown(t *testing.T) {
	a, _ := policyPair(t)
	r := NewRegistry(poolOrder, testClock())
	mustAdd(t, r, a)
	mustActivate(t, r, a, "alice")
	_, err := r.Activate("000000000000", "bob")
	if !errors.Is(err, ErrUnknownVersion) {
		t.Fatalf("Activate(unknown) error = %v, want ErrUnknownVersion", err)
	}
	if !errorMentions(err, "000000000000") {
		t.Errorf("error %q does not name the version", err)
	}
	if got := r.Active(); got != a {
		t.Errorf("failed Activate changed the active policy to %v", got)
	}
	if n := len(r.Activations()); n != 1 {
		t.Errorf("failed Activate left %d activation records, want 1", n)
	}
}

func TestRegistryActivateActiveVersionIsNoop(t *testing.T) {
	a, b := policyPair(t)
	r := NewRegistry(poolOrder, testClock())
	mustAdd(t, r, a, b)
	mustActivate(t, r, a, "alice")
	mustActivate(t, r, b, "alice")

	act := mustActivate(t, r, b, "bob")
	if act.Version != b.Version || act.Previous != b.Version || act.Actor != "bob" {
		t.Errorf("no-op Activation = %+v, want version and previous B, actor bob", act)
	}
	if n := len(r.Activations()); n != 2 {
		t.Errorf("no-op activation was recorded: %d records, want 2", n)
	}
	// The previous version is still A, not B.
	if _, err := r.Rollback("carol"); err != nil {
		t.Fatalf("Rollback() error: %v", err)
	}
	if got := r.Active(); got != a {
		t.Errorf("Active() after Rollback = %s, want A", got.Version)
	}
}

func TestRegistryAddIsIdempotent(t *testing.T) {
	a, _ := policyPair(t)
	r := NewRegistry(poolOrder, testClock())
	mustAdd(t, r, a)
	mustActivate(t, r, a, "alice")
	before := r.List()

	// The same version again, even as a distinct value, changes nothing.
	again := buildPolicy(t, nil)
	if again == a || again.Version != a.Version {
		t.Fatal("test setup: want an equal policy that is a different value")
	}
	if err := r.Add(again); err != nil {
		t.Fatalf("second Add() error: %v", err)
	}
	after := r.List()
	if len(after) != 1 {
		t.Fatalf("List() has %d entries after a repeated Add, want 1", len(after))
	}
	if after[0].Policy != a || after[0].Status != StatusActive || !after[0].AddedAt.Equal(before[0].AddedAt) {
		t.Errorf("repeated Add changed the entry: before %+v, after %+v", before[0], after[0])
	}
}

func TestRegistryAddNil(t *testing.T) {
	r := NewRegistry(poolOrder, testClock())
	if err := r.Add(nil); err == nil {
		t.Error("Add(nil) error = nil, want an error")
	}
}

func TestRegistryListSortedByAddedAt(t *testing.T) {
	a, b := policyPair(t)
	c := buildPolicy(t, func(m map[string]any) {
		setAt(t, m, jn("0.456"), "models", 0, "predictor", "init")
	})
	// A clock that runs backwards: List must order by AddedAt, not by insertion.
	base := time.Date(2026, 10, 27, 12, 0, 0, 0, time.UTC)
	var calls int
	clock := func() time.Time {
		calls++
		return base.Add(-time.Duration(calls) * time.Hour)
	}
	r := NewRegistry(poolOrder, clock)
	mustAdd(t, r, a, b, c)
	var got []string
	for _, e := range r.List() {
		got = append(got, e.Policy.Version)
	}
	if want := []string{c.Version, b.Version, a.Version}; !slices.Equal(got, want) {
		t.Errorf("List() versions = %v, want %v", got, want)
	}
}

func TestRegistryReturnsCopies(t *testing.T) {
	a, b := policyPair(t)
	r := NewRegistry(poolOrder, testClock())
	mustAdd(t, r, a, b)
	mustActivate(t, r, a, "alice")

	list := r.List()
	list[0].Status = "tampered"
	list[0], list[1] = list[1], list[0]
	log := r.Activations()
	log[0].Actor = "tampered"

	if got := statuses(r); got[a.Version] != StatusActive || got[b.Version] != StatusInactive {
		t.Errorf("List() result aliases registry state: statuses %v", got)
	}
	if got := r.Activations()[0].Actor; got != "alice" {
		t.Errorf("Activations() result aliases registry state: actor %q", got)
	}
}

func TestRegistryDefaultClock(t *testing.T) {
	a, _ := policyPair(t)
	r := NewRegistry(poolOrder, nil)
	mustAdd(t, r, a)
	act := mustActivate(t, r, a, "alice")
	if d := time.Since(act.At); d < 0 || d > time.Minute {
		t.Errorf("Activation.At = %v, want about now", act.At)
	}
}

func TestRegistryConcurrentActiveReads(t *testing.T) {
	a, b := policyPair(t)
	r := NewRegistry(poolOrder, testClock())
	mustAdd(t, r, a, b)
	mustActivate(t, r, a, "init")

	var (
		wg      sync.WaitGroup
		done    atomic.Bool
		started sync.WaitGroup
		bad     atomic.Int64
		reads   atomic.Int64
	)
	for range 8 {
		wg.Add(1)
		started.Add(1)
		go func() {
			defer wg.Done()
			started.Done()
			for !done.Load() {
				if p := r.Active(); p != a && p != b {
					bad.Add(1)
				}
				reads.Add(1)
			}
		}()
	}
	started.Wait()
	for i := range 1000 {
		next := b
		if i%2 == 1 {
			next = a
		}
		if _, err := r.Activate(next.Version, "writer"); err != nil {
			t.Fatalf("Activate() error: %v", err)
		}
	}
	done.Store(true)
	wg.Wait()
	if n := bad.Load(); n != 0 {
		t.Errorf("%d reads returned a policy that is neither A nor B", n)
	}
	if reads.Load() == 0 {
		t.Error("readers never ran")
	}
	if got := r.Active(); got != a {
		t.Errorf("final Active() = %s, want A after an even number of flips", got.Version)
	}
}
