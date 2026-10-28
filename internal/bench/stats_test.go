package bench

import (
	"math"
	"slices"
	"sort"
	"testing"
)

func TestQuantileMatchesNumpyLinear(t *testing.T) {
	// numpy.quantile([1, 2, 3, 4], q) with its default linear interpolation.
	sorted := []float64{1, 2, 3, 4}
	for _, c := range []struct{ q, want float64 }{
		{0, 1},
		{0.25, 1.75},
		{0.5, 2.5},
		{0.95, 3.85},
		{0.99, 3.97},
		{1, 4},
	} {
		if got := Quantile(sorted, c.q); math.Abs(got-c.want) > 1e-12 {
			t.Errorf("Quantile(%v, %v) = %v, want %v", sorted, c.q, got, c.want)
		}
	}
}

func TestQuantileEdges(t *testing.T) {
	if got := Quantile([]float64{7}, 0.95); got != 7 {
		t.Errorf("one value: %v, want 7", got)
	}
	if got := Quantile(nil, 0.5); !math.IsNaN(got) {
		t.Errorf("no values: %v, want NaN", got)
	}
	// A q outside [0, 1] is clamped to the ends instead of reading out of range.
	if got := Quantile([]float64{1, 2, 3}, -1); got != 1 {
		t.Errorf("q below 0: %v, want the minimum", got)
	}
	if got := Quantile([]float64{1, 2, 3}, 2); got != 3 {
		t.Errorf("q above 1: %v, want the maximum", got)
	}
}

func TestSummaryKeys(t *testing.T) {
	got := Summary([]float64{4, 1, 3, 2})
	keys := make([]string, 0, len(got))
	for k := range got {
		keys = append(keys, k)
	}
	sort.Strings(keys)
	if want := []string{"max", "mean", "n", "p50", "p95", "p99"}; !slices.Equal(keys, want) {
		t.Fatalf("keys = %v, want %v", keys, want)
	}
	for k, want := range map[string]float64{"n": 4, "mean": 2.5, "p50": 2.5, "p95": 3.85, "p99": 3.97, "max": 4} {
		if math.Abs(got[k]-want) > 1e-12 {
			t.Errorf("%s = %v, want %v", k, got[k], want)
		}
	}
}

func TestSummaryDoesNotReorderItsInput(t *testing.T) {
	xs := []float64{3, 1, 2}
	Summary(xs)
	if !slices.Equal(xs, []float64{3, 1, 2}) {
		t.Errorf("input is now %v", xs)
	}
}

func TestSummaryOfNothingHasOnlyACount(t *testing.T) {
	// There is no mean or percentile of no values, and none is made up.
	got := Summary(nil)
	if len(got) != 1 || got["n"] != 0 {
		t.Errorf("Summary(nil) = %v, want only n = 0", got)
	}
}
