package policy

import (
	"context"
	"errors"
	"fmt"
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

// fakePersister is a Persister that records what it is asked, in order, and
// can be made to fail.
type fakePersister struct {
	mu        sync.Mutex
	calls     []string
	raw       map[string][]byte
	active    string
	hasActive bool

	saveErr     error
	activateErr error
	activeErr   error
	loadErr     error
	// atActivate and atLoad run inside the corresponding call, before it
	// answers, with no lock of the fake held.
	atActivate func()
	atLoad     func()
}

func newFakePersister() *fakePersister {
	return &fakePersister{raw: make(map[string][]byte)}
}

func (f *fakePersister) SavePolicy(_ context.Context, p *Policy) error {
	f.mu.Lock()
	defer f.mu.Unlock()
	f.calls = append(f.calls, "save "+p.Version)
	if f.saveErr != nil {
		return f.saveErr
	}
	f.raw[p.Version] = p.Raw
	return nil
}

func (f *fakePersister) Activate(_ context.Context, version, previous, previousStatus, actor string) error {
	f.mu.Lock()
	hook := f.atActivate
	f.mu.Unlock()
	if hook != nil {
		hook()
	}
	f.mu.Lock()
	defer f.mu.Unlock()
	f.calls = append(f.calls, fmt.Sprintf("activate %s previous=%q status=%s actor=%s", version, previous, previousStatus, actor))
	if f.activateErr != nil {
		return f.activateErr
	}
	f.active, f.hasActive = version, true
	return nil
}

func (f *fakePersister) ActiveVersion(context.Context) (string, bool, error) {
	f.mu.Lock()
	defer f.mu.Unlock()
	if f.activeErr != nil {
		return "", false, f.activeErr
	}
	return f.active, f.hasActive, nil
}

func (f *fakePersister) LoadRaw(_ context.Context, version string) ([]byte, error) {
	f.mu.Lock()
	hook := f.atLoad
	f.mu.Unlock()
	if hook != nil {
		hook()
	}
	f.mu.Lock()
	defer f.mu.Unlock()
	f.calls = append(f.calls, "load "+version)
	if f.loadErr != nil {
		return nil, f.loadErr
	}
	raw, ok := f.raw[version]
	if !ok {
		return nil, fmt.Errorf("%w: %s", ErrUnknownVersion, version)
	}
	return raw, nil
}

// setActive plays another replica activating version.
func (f *fakePersister) setActive(version string) {
	f.mu.Lock()
	defer f.mu.Unlock()
	f.active, f.hasActive = version, true
}

func (f *fakePersister) callLog() []string {
	f.mu.Lock()
	defer f.mu.Unlock()
	return slices.Clone(f.calls)
}

func versionOrEmpty(p *Policy) string {
	if p == nil {
		return ""
	}
	return p.Version
}

func requireCalls(t *testing.T, step string, f *fakePersister, want ...string) {
	t.Helper()
	if got := f.callLog(); !slices.Equal(got, want) {
		t.Fatalf("after %s: persister calls = %q, want %q", step, got, want)
	}
}

func TestRegistryWithPersisterWritesThrough(t *testing.T) {
	a, b := policyPair(t)
	fp := newFakePersister()
	r := NewRegistry(poolOrder, testClock())
	r.SetPersister(fp)

	// What the registry serves at the moment the persister is asked to write an
	// activation: the pointer must not move before the write has succeeded.
	var servedAtActivate []string
	fp.atActivate = func() { servedAtActivate = append(servedAtActivate, versionOrEmpty(r.Active())) }

	mustAdd(t, r, a)
	requireCalls(t, "Add(a)", fp, "save "+a.Version)
	mustAdd(t, r, a) // already stored: nothing to save again
	requireCalls(t, "Add(a) again", fp, "save "+a.Version)
	mustAdd(t, r, b)
	requireCalls(t, "Add(b)", fp, "save "+a.Version, "save "+b.Version)

	mustActivate(t, r, a, "alice")
	mustActivate(t, r, b, "bob")
	if _, err := r.Rollback("carol"); err != nil {
		t.Fatalf("Rollback: %v", err)
	}
	requireCalls(t, "activations", fp,
		"save "+a.Version,
		"save "+b.Version,
		fmt.Sprintf("activate %s previous=%q status=retired actor=alice", a.Version, ""),
		fmt.Sprintf("activate %s previous=%q status=retired actor=bob", b.Version, a.Version),
		fmt.Sprintf("activate %s previous=%q status=inactive actor=carol", a.Version, b.Version),
	)
	if want := []string{"", a.Version, b.Version}; !slices.Equal(servedAtActivate, want) {
		t.Errorf("active policy when each activation was written = %q, want %q", servedAtActivate, want)
	}
	if got := r.Active(); got != a {
		t.Errorf("Active() after the rollback = %s, want %s", versionOrEmpty(got), a.Version)
	}
	if got := statuses(r); got[a.Version] != StatusActive || got[b.Version] != StatusInactive {
		t.Errorf("statuses = %v, want a active and b inactive", got)
	}

	// Activating the active version is a no-op, so nothing is written.
	mustActivate(t, r, a, "dave")
	if got := len(fp.callLog()); got != 5 {
		t.Errorf("a repeated activation wrote to the persister: %q", fp.callLog())
	}
}

func TestRegistryActivateFailsWhenPersistFails(t *testing.T) {
	a, b := policyPair(t)
	fp := newFakePersister()
	r := NewRegistry(poolOrder, testClock())
	r.SetPersister(fp)
	mustAdd(t, r, a, b)
	mustActivate(t, r, a, "alice")

	var announced []Activation
	r.OnActivate(func(act Activation) { announced = append(announced, act) })
	logBefore := r.Activations()

	boom := errors.New("connection reset by peer")
	fp.activateErr = boom
	_, err := r.Activate(b.Version, "bob")
	if !errors.Is(err, ErrPersist) || !errors.Is(err, boom) {
		t.Fatalf("Activate with a failing persister = %v, want ErrPersist wrapping the cause", err)
	}
	if got := r.Active(); got != a {
		t.Errorf("Active() = %s after a failed activation, want %s", versionOrEmpty(got), a.Version)
	}
	if got := statuses(r); got[a.Version] != StatusActive || got[b.Version] != StatusInactive {
		t.Errorf("statuses = %v, want them unchanged", got)
	}
	if !slices.Equal(r.Activations(), logBefore) || len(announced) != 0 {
		t.Errorf("a failed activation was logged or announced: log %v, announced %v", r.Activations(), announced)
	}

	// The failed attempt left no state behind: once the persister recovers the
	// same activation works, and so does a rollback after it.
	fp.activateErr = nil
	mustActivate(t, r, b, "bob")
	fp.activateErr = boom
	if _, err := r.Rollback("carol"); !errors.Is(err, ErrPersist) {
		t.Fatalf("Rollback with a failing persister = %v, want ErrPersist", err)
	}
	if got := r.Active(); got != b {
		t.Errorf("Active() = %s after a failed rollback, want %s", versionOrEmpty(got), b.Version)
	}
	if got := statuses(r); got[a.Version] != StatusRetired || got[b.Version] != StatusActive {
		t.Errorf("statuses after the failed rollback = %v, want a retired and b active", got)
	}
	fp.activateErr = nil
	if _, err := r.Rollback("carol"); err != nil {
		t.Fatalf("Rollback after the persister recovered: %v", err)
	}
	if got := r.Active(); got != a {
		t.Errorf("Active() = %s after the rollback, want %s", versionOrEmpty(got), a.Version)
	}
}

func TestRegistryAddFailsWhenSaveFails(t *testing.T) {
	a, _ := policyPair(t)
	fp := newFakePersister()
	r := NewRegistry(poolOrder, testClock())
	r.SetPersister(fp)

	boom := errors.New("disk full")
	fp.saveErr = boom
	err := r.Add(a)
	if !errors.Is(err, ErrPersist) || !errors.Is(err, boom) {
		t.Fatalf("Add with a failing persister = %v, want ErrPersist wrapping the cause", err)
	}
	if len(r.List()) != 0 {
		t.Errorf("a policy that could not be saved was stored: %v", r.List())
	}
	if _, err := r.Activate(a.Version, "alice"); !errors.Is(err, ErrUnknownVersion) {
		t.Errorf("Activate of the unsaved policy = %v, want ErrUnknownVersion", err)
	}

	// A validation problem the persister finds stays a validation problem.
	problem := &ValidationError{Path: "created_at", Problem: "must be an RFC 3339 timestamp"}
	fp.saveErr = problem
	err = r.Add(a)
	if ve := requireValidationError(t, err); ve.Path != "created_at" || !errors.Is(err, ErrPersist) {
		t.Errorf("Add = %v, want the persister's validation error, also an ErrPersist", err)
	}

	fp.saveErr = nil
	mustAdd(t, r, a)
	if len(r.List()) != 1 {
		t.Errorf("Add after the persister recovered stored %d policies, want 1", len(r.List()))
	}
}

func TestRegistryRejectsAnUnmappedModelBeforeSaving(t *testing.T) {
	a, _ := policyPair(t)
	fp := newFakePersister()
	r := NewRegistry(poolOrder[:5], testClock())
	r.SetPersister(fp)
	requireValidationError(t, r.Add(a))
	requireCalls(t, "rejected Add", fp)
}

func TestRegistrySetPersisterNilRemovesIt(t *testing.T) {
	a, _ := policyPair(t)
	fp := newFakePersister()
	r := NewRegistry(poolOrder, testClock())
	r.SetPersister(fp)
	r.SetPersister(nil)
	mustAdd(t, r, a)
	mustActivate(t, r, a, "alice")
	requireCalls(t, "work without a persister", fp)
}
