package health

import (
	"math"
	"sync"
	"testing"
	"time"
)

const tolerance = 1e-9

func near(a, b float64) bool { return math.Abs(a-b) <= tolerance }

func ms(v float64) time.Duration { return time.Duration(v * float64(time.Millisecond)) }

func TestPriorPredictionMatchesTrainerFormula(t *testing.T) {
	// The trainer's predicted_p95_ms: (ttft + 1000 * tokens / decode) scaled by
	// 1 + 1.645 * jitter / sqrt(3).
	r := NewRegistry(map[string]Prior{"m": {TTFTMS: 200, DecodeTokensPerS: 120, Jitter: 0.1}}, DefaultAlpha)
	got := r.PredictedP95MS("m", 100)
	want := (200 + 1000*100.0/120) * (1 + 1.645*0.1/math.Sqrt(3))
	if !near(got, want) {
		t.Fatalf("PredictedP95MS = %v, want %v", got, want)
	}
}

func TestPriorPredictionAcrossProfiles(t *testing.T) {
	priors := map[string]Prior{
		"qwen":   {TTFTMS: 200, DecodeTokensPerS: 120, Jitter: 0.1},
		"slow":   {TTFTMS: 700, DecodeTokensPerS: 45, Jitter: 0.1},
		"steady": {TTFTMS: 500, DecodeTokensPerS: 75, Jitter: 0},
		"jumpy":  {TTFTMS: 300, DecodeTokensPerS: 105, Jitter: 0.5},
	}
	r := NewRegistry(priors, DefaultAlpha)
	for name, p := range priors {
		for _, tokens := range []float64{0, 1, 313, 2150} {
			want := (p.TTFTMS + 1000*tokens/p.DecodeTokensPerS) * (1 + 1.645*p.Jitter/math.Sqrt(3))
			if got := r.PredictedP95MS(name, tokens); !near(got, want) {
				t.Errorf("%s, %v tokens: got %v, want %v", name, tokens, got, want)
			}
		}
	}
}

func TestObserveMovesMean(t *testing.T) {
	// 100 tokens/s is 10 ms per token. A streamed call with TTFT 400 ms and 10
	// further tokens in 100 ms matches the per-token prior, so only the TTFT
	// estimate moves: mean 200 + 0.1 * (400 - 200) = 220.
	r := NewRegistry(map[string]Prior{"m": {TTFTMS: 200, DecodeTokensPerS: 100, Jitter: 0.1}}, 0.1)
	r.Observe("m", Observation{TTFT: ms(400), Total: ms(500), Tokens: 11, Streamed: true})

	e := r.models["m"]
	if !near(e.ttft.mean, 220) {
		t.Errorf("ttft mean = %v, want 220", e.ttft.mean)
	}
	// var = (1 - alpha) * (var + alpha * d^2) with var = (200 * 0.1 / sqrt 3)^2.
	if want := 0.9 * (400.0/3 + 0.1*200*200); !near(e.ttft.variance, want) {
		t.Errorf("ttft variance = %v, want %v", e.ttft.variance, want)
	}
	if !near(e.perToken.mean, 10) {
		t.Errorf("per-token mean = %v, want it unchanged at 10", e.perToken.mean)
	}
	if want := 0.9 * (1.0 / 3); !near(e.perToken.variance, want) {
		t.Errorf("per-token variance = %v, want %v", e.perToken.variance, want)
	}

	want := (220 + 1.645*math.Sqrt(0.9*(400.0/3+0.1*200*200))) +
		10*(10+1.645*math.Sqrt(0.9*(1.0/3)))
	if got := r.PredictedP95MS("m", 10); !near(got, want) {
		t.Errorf("PredictedP95MS = %v, want %v", got, want)
	}
}

func TestObserveStreamedPerToken(t *testing.T) {
	r := NewRegistry(map[string]Prior{"m": {TTFTMS: 200, DecodeTokensPerS: 100, Jitter: 0.1}}, 0.1)
	// 300 ms of decode over 5 further tokens is 60 ms per token:
	// 10 + 0.1 * (60 - 10) = 15. TTFT 200 equals the prior, so it stays 200.
	r.Observe("m", Observation{TTFT: ms(200), Total: ms(500), Tokens: 6, Streamed: true})
	e := r.models["m"]
	if !near(e.perToken.mean, 15) || !near(e.ttft.mean, 200) {
		t.Fatalf("per-token mean %v, ttft mean %v; want 15 and 200", e.perToken.mean, e.ttft.mean)
	}

	// One token, or none, divides by 1: the decode time is the per-token time.
	for _, tokens := range []int{1, 0, -3} {
		r := NewRegistry(map[string]Prior{"m": {TTFTMS: 200, DecodeTokensPerS: 100, Jitter: 0.1}}, 0.1)
		r.Observe("m", Observation{TTFT: ms(200), Total: ms(250), Tokens: tokens, Streamed: true})
		if want := 10 + 0.1*(50-10); !near(r.models["m"].perToken.mean, want) {
			t.Errorf("%d tokens: per-token mean = %v, want %v", tokens, r.models["m"].perToken.mean, want)
		}
	}
}

func TestObserveNonStreamed(t *testing.T) {
	prior := map[string]Prior{"m": {TTFTMS: 200, DecodeTokensPerS: 100, Jitter: 0.1}}

	// Only the per-token estimate moves, using the current TTFT mean: 500 ms
	// beyond 200 ms over 10 tokens is 50 ms per token, so 10 + 0.1 * 40 = 14.
	r := NewRegistry(prior, 0.1)
	r.Observe("m", Observation{TTFT: ms(999), Total: ms(700), Tokens: 10})
	e := r.models["m"]
	if !near(e.perToken.mean, 14) {
		t.Errorf("per-token mean = %v, want 14", e.perToken.mean)
	}
	if !near(e.ttft.mean, 200) || !near(e.ttft.variance, 400.0/3) {
		t.Errorf("ttft = %+v, want it untouched", e.ttft)
	}

	// A call faster than the TTFT estimate counts as zero decode time.
	r = NewRegistry(prior, 0.1)
	r.Observe("m", Observation{Total: ms(150), Tokens: 10})
	if want := 10 + 0.1*(0-10); !near(r.models["m"].perToken.mean, want) {
		t.Errorf("per-token mean = %v, want %v", r.models["m"].perToken.mean, want)
	}

	// Zero tokens divide by 1.
	r = NewRegistry(prior, 0.1)
	r.Observe("m", Observation{Total: ms(260), Tokens: 0})
	if want := 10 + 0.1*(60-10); !near(r.models["m"].perToken.mean, want) {
		t.Errorf("per-token mean = %v, want %v", r.models["m"].perToken.mean, want)
	}
}

func TestObservationsConvergeAndNarrow(t *testing.T) {
	r := NewRegistry(map[string]Prior{"m": {TTFTMS: 200, DecodeTokensPerS: 100, Jitter: 0.1}}, 0.1)
	for range 500 {
		r.Observe("m", Observation{TTFT: ms(400), Total: ms(500), Tokens: 11, Streamed: true})
	}
	// Steady observations: the mean reaches 400 ms + 10 tokens * 10 ms and the
	// spread vanishes, so the p95 is the mean.
	if got, want := r.PredictedP95MS("m", 10), 400.0+10*10; math.Abs(got-want) > 1e-6 {
		t.Fatalf("PredictedP95MS = %v, want about %v", got, want)
	}
}

func TestUnknownModelIsInf(t *testing.T) {
	r := NewRegistry(map[string]Prior{"m": {TTFTMS: 200, DecodeTokensPerS: 100, Jitter: 0.1}}, DefaultAlpha)
	if got := r.PredictedP95MS("other", 10); !math.IsInf(got, 1) {
		t.Fatalf("PredictedP95MS(unknown) = %v, want +Inf", got)
	}
	// Observing a model with no prior is ignored.
	r.Observe("other", Observation{TTFT: ms(1), Total: ms(2), Tokens: 1, Streamed: true})
	if got := r.PredictedP95MS("other", 10); !math.IsInf(got, 1) {
		t.Fatalf("PredictedP95MS after Observe = %v, want +Inf", got)
	}
}

func TestPriorsAreCopied(t *testing.T) {
	priors := map[string]Prior{"m": {TTFTMS: 200, DecodeTokensPerS: 100, Jitter: 0}}
	r := NewRegistry(priors, DefaultAlpha)
	priors["m"] = Prior{TTFTMS: 1, DecodeTokensPerS: 1}
	delete(priors, "m")
	if got, want := r.PredictedP95MS("m", 10), 200.0+10*10; !near(got, want) {
		t.Fatalf("PredictedP95MS = %v, want %v", got, want)
	}
}

func TestConcurrentObserve(t *testing.T) {
	r := NewRegistry(map[string]Prior{
		"a": {TTFTMS: 200, DecodeTokensPerS: 100, Jitter: 0.1},
		"b": {TTFTMS: 300, DecodeTokensPerS: 50, Jitter: 0.1},
	}, DefaultAlpha)
	var wg sync.WaitGroup
	for i := range 8 {
		model := []string{"a", "b"}[i%2]
		wg.Add(2)
		go func() {
			defer wg.Done()
			for j := range 200 {
				r.Observe(model, Observation{TTFT: ms(float64(100 + j)), Total: ms(float64(300 + j)), Tokens: 11, Streamed: j%2 == 0})
			}
		}()
		go func() {
			defer wg.Done()
			for range 200 {
				_ = r.PredictedP95MS(model, 100)
			}
		}()
	}
	wg.Wait()
	for _, model := range []string{"a", "b"} {
		if got := r.PredictedP95MS(model, 100); math.IsNaN(got) || math.IsInf(got, 0) {
			t.Errorf("%s: PredictedP95MS = %v after concurrent use", model, got)
		}
	}
}
