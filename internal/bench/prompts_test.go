package bench

import (
	"slices"
	"strings"
	"testing"
)

func TestSyntheticPromptsDeterministic(t *testing.T) {
	a := SyntheticPrompts(400, 7)
	b := SyntheticPrompts(400, 7)
	if !slices.Equal(a, b) {
		t.Fatal("the same seed produced different prompts")
	}
	if len(a) != 400 {
		t.Fatalf("got %d prompts, want 400", len(a))
	}
	if slices.Equal(a, SyntheticPrompts(400, 8)) {
		t.Error("different seeds produced the same prompts")
	}
	// A longer list starts with the shorter one: the first prompts do not
	// depend on how many are asked for.
	if !slices.Equal(a, SyntheticPrompts(1000, 7)[:400]) {
		t.Error("the first 400 of 1000 prompts differ from 400 prompts")
	}
	if got := SyntheticPrompts(0, 7); len(got) != 0 {
		t.Errorf("SyntheticPrompts(0) = %v, want none", got)
	}
}

func TestSyntheticPromptsUseFourTemplates(t *testing.T) {
	markers := []string{"average speed", "function", "Which", "short story"}
	seen := map[string]int{}
	shortest, longest := 1<<30, 0
	distinct := map[string]bool{}
	for _, p := range SyntheticPrompts(2000, 1) {
		if strings.TrimSpace(p) == "" {
			t.Fatal("empty prompt")
		}
		for _, m := range markers {
			if strings.Contains(p, m) {
				seen[m]++
			}
		}
		shortest, longest = min(shortest, len(p)), max(longest, len(p))
		distinct[p] = true
	}
	for _, m := range markers {
		if seen[m] != 500 {
			t.Errorf("%d prompts use the template with %q, want 500 (the templates alternate)", seen[m], m)
		}
	}
	if longest <= shortest {
		t.Errorf("every prompt is %d bytes long, want a spread of lengths", shortest)
	}
	if len(distinct) < 1900 {
		t.Errorf("only %d of 2000 prompts are distinct", len(distinct))
	}
}
