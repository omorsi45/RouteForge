package router_test

import (
	"encoding/json"
	"maps"
	"regexp"
	"slices"
	"testing"

	"github.com/omorsi45/routeforge/internal/router"
)

func TestNewDecisionID(t *testing.T) {
	re := regexp.MustCompile(`^dec_[0-9a-f]{32}$`)
	a, b := router.NewDecisionID(), router.NewDecisionID()
	if !re.MatchString(a) || !re.MatchString(b) {
		t.Fatalf("ids = %q, %q", a, b)
	}
	if a == b {
		t.Fatalf("ids repeat: %q", a)
	}
}

func sortedKeys(m map[string]any) []string {
	return slices.Sorted(maps.Keys(m))
}

func TestDecisionJSONShape(t *testing.T) {
	d, err := decide(t, &router.Request{Tenant: tenant(), Model: "m-fast", Profile: "balanced"})
	if err != nil {
		t.Fatal(err)
	}
	raw, err := json.Marshal(d)
	if err != nil {
		t.Fatal(err)
	}
	var got map[string]any
	if err := json.Unmarshal(raw, &got); err != nil {
		t.Fatal(err)
	}
	wantKeys := []string{
		"attempts", "candidates", "chosen", "created_at", "decided_in_us", "fallbacks",
		"guarantee_applies", "id", "lambda", "policy_version", "profile", "reason",
		"strategy", "tenant",
	}
	if k := sortedKeys(got); !slices.Equal(k, wantKeys) {
		t.Fatalf("keys = %v, want %v", k, wantKeys)
	}
	if got["lambda"] != nil {
		t.Fatalf("lambda = %v, want null", got["lambda"])
	}
	for _, k := range []string{"fallbacks", "attempts"} {
		arr, ok := got[k].([]any)
		if !ok || len(arr) != 0 {
			t.Fatalf("%s = %#v, want empty array", k, got[k])
		}
	}
	cands, ok := got["candidates"].([]any)
	if !ok || len(cands) != 1 {
		t.Fatalf("candidates = %#v", got["candidates"])
	}
	cand, ok := cands[0].(map[string]any)
	if !ok {
		t.Fatalf("candidate = %#v", cands[0])
	}
	wantCand := []string{
		"block_k", "block_n", "eligible", "expected_cost_usd", "ineligible_reasons",
		"lcb", "model", "p", "predicted_p95_ms", "raw",
	}
	if k := sortedKeys(cand); !slices.Equal(k, wantCand) {
		t.Fatalf("candidate keys = %v, want %v", k, wantCand)
	}
	if reasons, ok := cand["ineligible_reasons"].([]any); !ok || len(reasons) != 0 {
		t.Fatalf("ineligible_reasons = %#v, want empty array", cand["ineligible_reasons"])
	}
}

func TestValidStrategy(t *testing.T) {
	for _, s := range []string{"quality", "strongest", "cheapest", "static", "latency"} {
		if !router.ValidStrategy(s) {
			t.Errorf("ValidStrategy(%q) = false", s)
		}
	}
	for _, s := range []string{"pinned", "", "Quality"} {
		if router.ValidStrategy(s) {
			t.Errorf("ValidStrategy(%q) = true", s)
		}
	}
}
