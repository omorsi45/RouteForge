package router_test

import (
	"context"
	"errors"
	"strings"
	"testing"
	"time"

	"github.com/omorsi45/routeforge/internal/config"
	"github.com/omorsi45/routeforge/internal/router"
)

func models() []config.Model {
	return []config.Model{
		{Name: "m-fast"},
		{Name: "m-strong", Strongest: true},
		{Name: "m-other", Strongest: true},
	}
}

func tenant(allowed ...string) *config.Tenant {
	return &config.Tenant{Name: "tenant-a", AllowedModels: allowed}
}

func decide(t *testing.T, req *router.Request) (*router.Decision, error) {
	t.Helper()
	return router.NewBasic(models(), router.BasicOptions{}).Decide(context.Background(), req)
}

func TestPinned(t *testing.T) {
	d, err := decide(t, &router.Request{Tenant: tenant(), Model: "m-fast", Profile: "strict"})
	if err != nil {
		t.Fatal(err)
	}
	if d.Chosen != "m-fast" || d.Strategy != router.StrategyPinned || d.Reason != router.ReasonPinned {
		t.Fatalf("decision = %+v", d)
	}
	if d.Tenant != "tenant-a" || d.Profile != "strict" {
		t.Fatalf("tenant/profile = %q/%q", d.Tenant, d.Profile)
	}
	if len(d.Candidates) != 1 || d.Candidates[0].Model != "m-fast" || !d.Candidates[0].Eligible {
		t.Fatalf("candidates = %+v", d.Candidates)
	}
}

func TestPinnedUnknown(t *testing.T) {
	_, err := decide(t, &router.Request{Tenant: tenant(), Model: "nope"})
	if !errors.Is(err, router.ErrUnknownModel) {
		t.Fatalf("err = %v, want ErrUnknownModel", err)
	}
}

func TestPinnedNotAllowed(t *testing.T) {
	_, err := decide(t, &router.Request{Tenant: tenant("m-strong"), Model: "m-fast"})
	if !errors.Is(err, router.ErrModelNotAllowed) {
		t.Fatalf("err = %v, want ErrModelNotAllowed", err)
	}
}

func TestAutoStrongestFirstMarked(t *testing.T) {
	d, err := decide(t, &router.Request{Tenant: tenant(), Model: config.AutoModel})
	if err != nil {
		t.Fatal(err)
	}
	if d.Chosen != "m-strong" || d.Strategy != router.StrategyStrongest || d.Reason != router.ReasonStrongest {
		t.Fatalf("decision = %+v", d)
	}
	if len(d.Candidates) != 3 {
		t.Fatalf("candidates = %d, want 3", len(d.Candidates))
	}
	for _, c := range d.Candidates {
		if !c.Eligible || c.Raw != nil || c.P != nil || c.LCB != nil {
			t.Fatalf("candidate = %+v", c)
		}
	}
}

func TestAutoRespectsAllowed(t *testing.T) {
	d, err := decide(t, &router.Request{Tenant: tenant("m-fast", "m-other"), Model: config.AutoModel})
	if err != nil {
		t.Fatal(err)
	}
	if d.Chosen != "m-other" || len(d.Candidates) != 2 {
		t.Fatalf("chosen = %q, candidates = %d", d.Chosen, len(d.Candidates))
	}
}

func TestAutoNoStrongest(t *testing.T) {
	_, err := decide(t, &router.Request{Tenant: tenant("m-fast"), Model: config.AutoModel})
	if !errors.Is(err, router.ErrNoEligibleModel) {
		t.Fatalf("err = %v, want ErrNoEligibleModel", err)
	}
}

func TestAutoExplicitStrongest(t *testing.T) {
	d, err := decide(t, &router.Request{Tenant: tenant(), Model: config.AutoModel, Strategy: "strongest"})
	if err != nil {
		t.Fatal(err)
	}
	if d.Chosen != "m-strong" {
		t.Fatalf("chosen = %q", d.Chosen)
	}
}

func TestAutoStrategyUnavailable(t *testing.T) {
	_, err := decide(t, &router.Request{Tenant: tenant(), Model: config.AutoModel, Strategy: "quality"})
	if !errors.Is(err, router.ErrStrategyUnavailable) {
		t.Fatalf("err = %v, want ErrStrategyUnavailable", err)
	}
	if !strings.Contains(err.Error(), `"quality"`) {
		t.Fatalf("message %q lacks the quoted strategy", err.Error())
	}
}

func TestDecisionIDsInjected(t *testing.T) {
	fixed := time.Date(2026, 10, 26, 12, 0, 0, 0, time.UTC)
	b := router.NewBasic(models(), router.BasicOptions{
		NewID: func() string { return "dec_x" },
		Now:   func() time.Time { return fixed },
	})
	d, err := b.Decide(context.Background(), &router.Request{Tenant: tenant(), Model: "m-fast"})
	if err != nil {
		t.Fatal(err)
	}
	if d.ID != "dec_x" || !d.CreatedAt.Equal(fixed) {
		t.Fatalf("id = %q, created_at = %v", d.ID, d.CreatedAt)
	}
}

func TestTenantNotMutated(t *testing.T) {
	ten := tenant("m-fast", "m-strong")
	if _, err := decide(t, &router.Request{Tenant: ten, Model: config.AutoModel}); err != nil {
		t.Fatal(err)
	}
	if ten.Name != "tenant-a" || len(ten.AllowedModels) != 2 || ten.AllowedModels[0] != "m-fast" {
		t.Fatalf("tenant mutated: %+v", ten)
	}
}
