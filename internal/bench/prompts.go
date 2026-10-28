package bench

import (
	"fmt"
	"math/rand/v2"
	"strings"
)

// The word lists the synthetic prompts are built from. They are fixed, so the
// same seed gives the same prompts on every machine.
var (
	nouns      = []string{"river", "engine", "garden", "bridge", "market", "signal", "harbor", "ladder", "window", "forest", "planet", "violin", "castle", "pencil", "desert", "island"}
	adjectives = []string{"quiet", "rapid", "ancient", "bright", "narrow", "golden", "hollow", "steady", "gentle", "rugged", "silver", "distant", "curious", "patient", "vivid", "humble"}
	verbs      = []string{"sorts", "filters", "merges", "counts", "reverses", "groups", "splits", "ranks"}
	places     = []string{"Lisbon", "Nairobi", "Osaka", "Quebec", "Lima", "Oslo", "Cairo", "Perth"}
)

// SyntheticPrompts returns n deterministic prompts for load and fault runs.
// They come from four templates, in turn: arithmetic, code, trivia and
// writing, filled with words from fixed lists and followed by a few filler
// sentences, so that they differ in content and in length. No dataset text is
// used. Prompt i depends only on seed and i, so a longer list starts with a
// shorter one.
func SyntheticPrompts(n int, seed uint64) []string {
	out := make([]string, max(n, 0))
	for i := range out {
		out[i] = syntheticPrompt(i, seed)
	}
	return out
}

func syntheticPrompt(i int, seed uint64) string {
	rng := rand.New(rand.NewPCG(seed, uint64(i)))
	pick := func(list []string) string { return list[rng.IntN(len(list))] }
	num := func() int { return 1 + rng.IntN(999) }

	var sb strings.Builder
	switch i % 4 {
	case 0:
		fmt.Fprintf(&sb, "A %s %s travels %d kilometers in %d hours, then %d more kilometers in %d hours. What is the average speed for the whole trip? Show your steps.",
			pick(adjectives), pick(nouns), num(), num(), num(), num())
	case 1:
		fmt.Fprintf(&sb, "Write a function that %s a list of %s records by their %s field and returns the %s one. Include tests for an empty list and for ties.",
			pick(verbs), pick(nouns), pick(nouns), pick(adjectives))
	case 2:
		fmt.Fprintf(&sb, "Which %s is known for the %s %s of %s, and in what year was it first described?",
			pick(nouns), pick(adjectives), pick(nouns), pick(places))
	default:
		fmt.Fprintf(&sb, "Write a short story of about %d words about a %s %s who visits %s.",
			50+rng.IntN(450), pick(adjectives), pick(nouns), pick(places))
	}
	for extra := rng.IntN(7); extra > 0; extra-- {
		fmt.Fprintf(&sb, " Consider the %s %s as well.", pick(adjectives), pick(nouns))
	}
	return sb.String()
}
