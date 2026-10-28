// Package health keeps a running estimate of each model's latency. The router
// reads it to predict a p95 latency before a call, and the executor feeds it
// the timings of calls that succeed.
//
// Each model has two estimates: time to first token (TTFT) and time per output
// token, both in milliseconds. An estimate is an exponentially weighted mean
// and variance, seeded from the model's configured latency prior so that
// before any call the prediction equals the one the trainer computes
// offline (trainer/routeforge_train/latency.py predicted_p95_ms).
package health

import (
	"math"
	"sync"
	"time"
)

// P95Z is the normal quantile of the 95th percentile.
const P95Z = 1.645

// DefaultAlpha is the EWMA weight of a new observation.
const DefaultAlpha = 0.1

// Prior is the latency assumed for a model before any is observed. Jitter is
// the half-width of the uniform noise as a fraction of the mean, so the
// standard deviation is mean * Jitter / sqrt(3).
type Prior struct {
	TTFTMS           float64
	DecodeTokensPerS float64
	Jitter           float64
}

// Observation is the timing of one successful attempt.
type Observation struct {
	// TTFT is the time from the start of the attempt to its first output.
	// Only streamed attempts have one.
	TTFT time.Duration
	// Total is the time from the start of the attempt to its last output.
	Total time.Duration
	// Tokens is the number of completion tokens delivered.
	Tokens int
	// Streamed reports whether TTFT was measured.
	Streamed bool
}

// moments is an exponentially weighted mean and variance.
type moments struct {
	mean     float64
	variance float64
}

// update folds in one observation x with weight alpha.
func (m *moments) update(x, alpha float64) {
	d := x - m.mean
	m.mean += alpha * d
	m.variance = (1 - alpha) * (m.variance + alpha*d*d)
}

// p95 is the mean plus P95Z standard deviations.
func (m *moments) p95() float64 {
	return m.mean + P95Z*math.Sqrt(m.variance)
}

// seed returns the moments of a uniform noise of half-width jitter*mean.
func seed(mean, jitter float64) moments {
	sd := mean * jitter / math.Sqrt(3)
	return moments{mean: mean, variance: sd * sd}
}

// estimator is the latency state of one model.
type estimator struct {
	mu       sync.Mutex
	ttft     moments // ms
	perToken moments // ms per output token
}

// Registry holds the estimators of the configured models. It is safe for
// concurrent use.
type Registry struct {
	alpha  float64
	models map[string]*estimator // fixed at construction; each guards itself
}

// NewRegistry returns a Registry with one estimator per prior, keyed by
// configured model name and seeded from its prior. alpha, the weight of a new
// observation, must be in (0, 1]; DefaultAlpha is the usual choice.
func NewRegistry(priors map[string]Prior, alpha float64) *Registry {
	r := &Registry{alpha: alpha, models: make(map[string]*estimator, len(priors))}
	for name, p := range priors {
		r.models[name] = &estimator{
			ttft:     seed(p.TTFTMS, p.Jitter),
			perToken: seed(1000/p.DecodeTokensPerS, p.Jitter),
		}
	}
	return r
}

// milliseconds converts d to fractional milliseconds.
func milliseconds(d time.Duration) float64 {
	return float64(d) / float64(time.Millisecond)
}

// Observe folds the timing of a successful attempt into the estimates of
// model. The executor calls it for successful attempts only: a failure's
// timing says nothing about how fast the model answers. A model without a
// prior is ignored.
//
// A streamed attempt updates both estimates: its TTFT, and its decode time
// (Total - TTFT) per token after the first. A non-streamed attempt has no
// separate TTFT, so it updates only the per-token estimate, from the time left
// after the current TTFT estimate.
func (r *Registry) Observe(model string, o Observation) {
	e, ok := r.models[model]
	if !ok {
		return
	}
	e.mu.Lock()
	defer e.mu.Unlock()
	if o.Streamed {
		e.ttft.update(milliseconds(o.TTFT), r.alpha)
		e.perToken.update(milliseconds(o.Total-o.TTFT)/float64(max(o.Tokens-1, 1)), r.alpha)
		return
	}
	e.perToken.update(math.Max(0, milliseconds(o.Total)-e.ttft.mean)/float64(max(o.Tokens, 1)), r.alpha)
}

// PredictedP95MS predicts the p95 latency in milliseconds of a response of
// outputTokens tokens: the p95 TTFT plus outputTokens times the p95 time per
// token. A model without a prior has no prediction: +Inf.
func (r *Registry) PredictedP95MS(model string, outputTokens float64) float64 {
	e, ok := r.models[model]
	if !ok {
		return math.Inf(1)
	}
	e.mu.Lock()
	defer e.mu.Unlock()
	return e.ttft.p95() + outputTokens*e.perToken.p95()
}
