// Package telemetrytest reads back what a telemetry.Metrics has recorded, so
// the tests of every package that records a metric can check it without
// parsing the text exposition format.
package telemetrytest

import (
	"maps"
	"math"
	"slices"
	"strings"
	"testing"

	"github.com/omorsi45/routeforge/internal/telemetry"
)

// Series is one labeled series of a metric family.
type Series struct {
	Labels map[string]string
	Value  float64   // counters and gauges
	Count  uint64    // histograms: the number of observations
	Sum    float64   // histograms: the sum of the observations
	Bounds []float64 // histograms: the bucket upper bounds, +Inf left out
}

// Family is a gathered metric family.
type Family struct {
	Kind   string // "counter", "gauge" or "histogram"
	Series []Series
}

// LabelNames returns the sorted label names of the family's series. A family
// whose series differ in their label names is a bug in the metrics and fails
// the test.
func (f Family) LabelNames(t testing.TB) []string {
	t.Helper()
	var names []string
	for i, s := range f.Series {
		got := slices.Sorted(maps.Keys(s.Labels))
		if i > 0 && !slices.Equal(got, names) {
			t.Fatalf("series disagree on label names: %v and %v", names, got)
		}
		names = got
	}
	return names
}

// Find returns the series whose labels equal labels exactly.
func (f Family) Find(labels map[string]string) (Series, bool) {
	for _, s := range f.Series {
		if maps.Equal(s.Labels, labels) {
			return s, true
		}
	}
	return Series{}, false
}

// Gather returns every family m exposes, by name.
func Gather(t testing.TB, m *telemetry.Metrics) map[string]Family {
	t.Helper()
	fams, err := m.Gatherer().Gather()
	if err != nil {
		t.Fatalf("gather metrics: %v", err)
	}
	out := make(map[string]Family, len(fams))
	for _, mf := range fams {
		f := Family{Kind: strings.ToLower(mf.GetType().String())}
		for _, metric := range mf.GetMetric() {
			s := Series{Labels: map[string]string{}}
			for _, l := range metric.GetLabel() {
				s.Labels[l.GetName()] = l.GetValue()
			}
			switch {
			case metric.GetHistogram() != nil:
				h := metric.GetHistogram()
				s.Count, s.Sum = h.GetSampleCount(), h.GetSampleSum()
				for _, b := range h.GetBucket() {
					if !math.IsInf(b.GetUpperBound(), 1) {
						s.Bounds = append(s.Bounds, b.GetUpperBound())
					}
				}
			case metric.GetCounter() != nil:
				s.Value = metric.GetCounter().GetValue()
			case metric.GetGauge() != nil:
				s.Value = metric.GetGauge().GetValue()
			}
			f.Series = append(f.Series, s)
		}
		out[mf.GetName()] = f
	}
	return out
}

// Find returns the series of the named family whose labels equal labels
// exactly, and whether there is one.
func Find(t testing.TB, m *telemetry.Metrics, name string, labels map[string]string) (Series, bool) {
	t.Helper()
	f, ok := Gather(t, m)[name]
	if !ok {
		return Series{}, false
	}
	return f.Find(labels)
}

// Value returns the value of a counter or gauge series and fails the test when
// the series does not exist.
func Value(t testing.TB, m *telemetry.Metrics, name string, labels map[string]string) float64 {
	t.Helper()
	return mustFind(t, m, name, labels).Value
}

// Hist returns a histogram series and fails the test when it does not exist.
func Hist(t testing.TB, m *telemetry.Metrics, name string, labels map[string]string) Series {
	t.Helper()
	return mustFind(t, m, name, labels)
}

// Absent fails the test when the named family has any series.
func Absent(t testing.TB, m *telemetry.Metrics, name string) {
	t.Helper()
	if f, ok := Gather(t, m)[name]; ok && len(f.Series) > 0 {
		t.Errorf("%s has %d series, want none: %+v", name, len(f.Series), f.Series)
	}
}

func mustFind(t testing.TB, m *telemetry.Metrics, name string, labels map[string]string) Series {
	t.Helper()
	s, ok := Find(t, m, name, labels)
	if !ok {
		t.Fatalf("no series %s%v; have %+v", name, labels, Gather(t, m)[name].Series)
	}
	return s
}
