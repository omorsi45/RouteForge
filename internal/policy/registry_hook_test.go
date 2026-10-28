package policy

import (
	"slices"
	"testing"
)

func TestRegistryActivationHook(t *testing.T) {
	a, b := policyPair(t)
	r := NewRegistry(poolOrder, testClock())
	for _, p := range []*Policy{a, b} {
		if err := r.Add(p); err != nil {
			t.Fatalf("Add(%s) error: %v", p.Version, err)
		}
	}

	var got []Activation
	r.OnActivate(func(act Activation) { got = append(got, act) })
	check := func(step string, want ...[2]string) {
		t.Helper()
		var have [][2]string
		for _, act := range got {
			have = append(have, [2]string{act.Version, act.Previous})
		}
		if !slices.Equal(have, want) {
			t.Errorf("after %s: hook saw %v, want %v", step, have, want)
		}
	}

	check("registering")
	if _, err := r.Activate(a.Version, "alice"); err != nil {
		t.Fatal(err)
	}
	check("the first activation", [2]string{a.Version, ""})
	if got[0].Actor != "alice" || got[0].At.IsZero() {
		t.Errorf("hook activation = %+v, want actor alice and a time", got[0])
	}

	// Activating the active version changes nothing, so it is not announced.
	if _, err := r.Activate(a.Version, "alice"); err != nil {
		t.Fatal(err)
	}
	check("a repeated activation", [2]string{a.Version, ""})

	// A failed activation is not announced either.
	if _, err := r.Activate("no-such-version", "alice"); err == nil {
		t.Fatal("Activate of an unknown version succeeded")
	}
	check("a failed activation", [2]string{a.Version, ""})

	if _, err := r.Activate(b.Version, "bob"); err != nil {
		t.Fatal(err)
	}
	check("the second activation", [2]string{a.Version, ""}, [2]string{b.Version, a.Version})

	if _, err := r.Rollback("carol"); err != nil {
		t.Fatal(err)
	}
	check("a rollback", [2]string{a.Version, ""}, [2]string{b.Version, a.Version}, [2]string{a.Version, b.Version})

	// A nil hook removes it.
	r.OnActivate(nil)
	if _, err := r.Rollback("carol"); err != nil {
		t.Fatal(err)
	}
	check("removing the hook", [2]string{a.Version, ""}, [2]string{b.Version, a.Version}, [2]string{a.Version, b.Version})
}

func TestRegistryHookRunsInActivationOrder(t *testing.T) {
	a, b := policyPair(t)
	r := NewRegistry(poolOrder, testClock())
	for _, p := range []*Policy{a, b} {
		if err := r.Add(p); err != nil {
			t.Fatal(err)
		}
	}
	// The hook runs while the registry is locked, so announcements follow the
	// order of the activations even when callers race.
	var last Activation
	r.OnActivate(func(act Activation) { last = act })
	done := make(chan struct{})
	for range 20 {
		go func() {
			defer func() { done <- struct{}{} }()
			_, _ = r.Activate(a.Version, "x")
			_, _ = r.Activate(b.Version, "x")
		}()
	}
	for range 20 {
		<-done
	}
	if active := r.Active(); active == nil || last.Version != active.Version {
		t.Errorf("last announced version = %q, but the active one is %v", last.Version, active)
	}
}
