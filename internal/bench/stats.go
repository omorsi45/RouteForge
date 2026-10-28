package bench

import (
	"math"
	"slices"
)

// Quantile returns the q-quantile of sorted, which must be in ascending order,
// by linear interpolation between the closest ranks: the default of
// numpy.quantile. A q outside [0, 1] gives the minimum or the maximum. It
// returns NaN for no values.
func Quantile(sorted []float64, q float64) float64 {
	n := len(sorted)
	switch {
	case n == 0:
		return math.NaN()
	case q <= 0:
		return sorted[0]
	case q >= 1:
		return sorted[n-1]
	}
	pos := q * float64(n-1)
	lo := int(math.Floor(pos))
	hi := min(lo+1, n-1)
	return sorted[lo] + float64((pos-float64(lo))*(sorted[hi]-sorted[lo]))
}

// Summary describes xs by its count n, mean, median (p50), p95, p99 and max.
// xs is not changed. For no values it holds only n = 0: a mean or percentile of
// nothing would be made up.
func Summary(xs []float64) map[string]float64 {
	out := map[string]float64{"n": float64(len(xs))}
	if len(xs) == 0 {
		return out
	}
	sorted := slices.Clone(xs)
	slices.Sort(sorted)
	var sum float64
	for _, x := range sorted {
		sum += x
	}
	out["mean"] = sum / float64(len(sorted))
	out["p50"] = Quantile(sorted, 0.50)
	out["p95"] = Quantile(sorted, 0.95)
	out["p99"] = Quantile(sorted, 0.99)
	out["max"] = sorted[len(sorted)-1]
	return out
}
