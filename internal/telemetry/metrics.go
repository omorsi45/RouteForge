// Package telemetry holds what the gateway tells the outside world about
// itself: Prometheus metrics (this file) and OpenTelemetry spans.
//
// No metric label and no span attribute ever carries prompt or completion text.
package telemetry

import (
	"net/http"
	"strconv"
	"sync/atomic"
	"time"

	"github.com/prometheus/client_golang/prometheus"
	"github.com/prometheus/client_golang/prometheus/collectors"
	"github.com/prometheus/client_golang/prometheus/promauto"
	"github.com/prometheus/client_golang/prometheus/promhttp"
)

// noPolicy is the policy_version label of a request served with no policy.
const noPolicy = "none"

// StateSources supplies the values of the gauges that mirror the gateway's
// state. They are read at every scrape, so they are always current and need no
// bookkeeping where the state changes. Each function may be nil, and then its
// gauge has no series (routeforge_inflight reads zero).
type StateSources struct {
	// Models are the configured model names; breaker_state and queue_depth have
	// one series for each.
	Models []string
	// BreakerState returns a model's circuit breaker state: 0 closed, 1
	// half-open, 2 open.
	BreakerState func(model string) int
	// QueueDepth returns how many requests wait for a slot of the model.
	QueueDepth func(model string) int
	// InFlight returns how many requests the gateway is serving.
	InFlight func() int
}

// Metrics is the gateway's set of Prometheus series (spec 14 and 11), on a
// registry of its own. Every method is safe to call on a nil *Metrics, where it
// does nothing, so a package takes an optional *Metrics without checking it.
// Methods are safe for concurrent use.
type Metrics struct {
	reg *prometheus.Registry

	requests     *prometheus.CounterVec
	decisions    *prometheus.CounterVec
	decisionTime prometheus.Histogram
	overhead     *prometheus.HistogramVec
	ttft         *prometheus.HistogramVec
	upstream     *prometheus.HistogramVec
	retries      *prometheus.CounterVec
	failovers    prometheus.Counter
	rateLimited  *prometheus.CounterVec
	budgetReject prometheus.Counter
	cost         *prometheus.CounterVec
	tokens       *prometheus.CounterVec
	overshoot    *prometheus.CounterVec
	predicted    *prometheus.HistogramVec
	policyInfo   *prometheus.GaugeVec
	state        *stateCollector
}

// NewMetrics returns a Metrics on a new registry that also exposes the Go
// runtime and process collectors.
func NewMetrics() *Metrics {
	reg := prometheus.NewRegistry()
	reg.MustRegister(
		collectors.NewGoCollector(),
		collectors.NewProcessCollector(collectors.ProcessCollectorOpts{}),
	)
	f := promauto.With(reg)
	m := &Metrics{
		reg: reg,
		requests: f.NewCounterVec(prometheus.CounterOpts{
			Name: "routeforge_requests_total",
			Help: "Chat requests by chosen model, strategy, profile, policy version and HTTP status; a client that left is 499.",
		}, []string{"model", "strategy", "profile", "policy_version", "status"}),
		decisions: f.NewCounterVec(prometheus.CounterOpts{
			Name: "routeforge_route_decisions_total",
			Help: "Routing decisions by chosen model, reason, strategy and policy version.",
		}, []string{"model", "reason", "strategy", "policy_version"}),
		decisionTime: f.NewHistogram(prometheus.HistogramOpts{
			Name:    "routeforge_route_decision_seconds",
			Help:    "Time to make a routing decision.",
			Buckets: prometheus.ExponentialBuckets(0.00001, 2, 16),
		}),
		overhead: f.NewHistogramVec(prometheus.HistogramOpts{
			Name:    "routeforge_gateway_overhead_seconds",
			Help:    "Time the gateway adds to a request: its duration minus the upstream attempts; for a stream, the time to the first byte minus the wait for the first upstream event.",
			Buckets: prometheus.ExponentialBuckets(0.0001, 2, 16),
		}, []string{"stream"}),
		ttft: f.NewHistogramVec(prometheus.HistogramOpts{
			Name:    "routeforge_upstream_ttft_seconds",
			Help:    "Time from the start of a streaming upstream attempt to its first event.",
			Buckets: prometheus.DefBuckets,
		}, []string{"model"}),
		upstream: f.NewHistogramVec(prometheus.HistogramOpts{
			Name:    "routeforge_upstream_duration_seconds",
			Help:    "Duration of upstream attempts by model and outcome.",
			Buckets: prometheus.DefBuckets,
		}, []string{"model", "outcome"}),
		retries: f.NewCounterVec(prometheus.CounterOpts{
			Name: "routeforge_retries_total",
			Help: "Retries made, by the model whose attempt failed and the kind of the failure.",
		}, []string{"model", "kind"}),
		failovers: f.NewCounter(prometheus.CounterOpts{
			Name: "routeforge_failovers_total",
			Help: "Attempts made on another model than the one before them.",
		}),
		rateLimited: f.NewCounterVec(prometheus.CounterOpts{
			Name: "routeforge_rate_limited_total",
			Help: "Requests refused by a tenant rate limit, by limit (rpm or tpm).",
		}, []string{"kind"}),
		budgetReject: f.NewCounter(prometheus.CounterOpts{
			Name: "routeforge_budget_rejections_total",
			Help: "Attempts refused because the tenant's daily budget could not hold the reservation.",
		}),
		cost: f.NewCounterVec(prometheus.CounterOpts{
			Name: "routeforge_cost_usd_total",
			Help: "Settled cost in USD by model and cost source (reported or estimated).",
		}, []string{"model", "cost_source"}),
		tokens: f.NewCounterVec(prometheus.CounterOpts{
			Name: "routeforge_tokens_total",
			Help: "Settled tokens by model, direction (prompt or completion) and usage source.",
		}, []string{"model", "direction", "usage_source"}),
		overshoot: f.NewCounterVec(prometheus.CounterOpts{
			Name: "routeforge_budget_overshoot_usd_total",
			Help: "USD by which settled spend went over a tenant's daily budget.",
		}, []string{"tenant"}),
		predicted: f.NewHistogramVec(prometheus.HistogramOpts{
			Name:    "routeforge_predicted_success",
			Help:    "Calibrated success probability of the model chosen for a request.",
			Buckets: prometheus.LinearBuckets(0.05, 0.05, 20),
		}, []string{"model"}),
		policyInfo: f.NewGaugeVec(prometheus.GaugeOpts{
			Name: "routeforge_policy_info",
			Help: "1 for the version of the active routing policy.",
		}, []string{"version"}),
		state: newStateCollector(),
	}
	reg.MustRegister(m.state)
	return m
}

// Gatherer returns the registry's gatherer, for tests and for embedding the
// metrics in another exposition. It is nil on a nil *Metrics.
func (m *Metrics) Gatherer() prometheus.Gatherer {
	if m == nil {
		return nil
	}
	return m.reg
}

// Handler returns the handler of GET /metrics, which serves the text
// exposition format. It does no authentication; restrict it at the proxy. On a
// nil *Metrics it answers 404.
func (m *Metrics) Handler() http.Handler {
	if m == nil {
		return http.NotFoundHandler()
	}
	return promhttp.HandlerFor(m.reg, promhttp.HandlerOpts{})
}

// PolicyLabel is how a policy version is written in a metric label and a span
// attribute: the version itself, or "none" when the request was served with no
// policy. It matches the X-RouteForge-Policy-Version header.
func PolicyLabel(version string) string {
	if version == "" {
		return noPolicy
	}
	return version
}

// Request counts one finished chat request. The model, strategy, profile and
// policy version are those of its routing decision, empty when the request
// failed before one was made. status is the HTTP status, 499 for a client that
// left.
func (m *Metrics) Request(model, strategy, profile, policyVersion string, status int) {
	if m == nil {
		return
	}
	m.requests.WithLabelValues(model, strategy, profile, PolicyLabel(policyVersion), strconv.Itoa(status)).Inc()
}

// Decision counts one routing decision that chose a model.
func (m *Metrics) Decision(model, reason, strategy, policyVersion string) {
	if m == nil {
		return
	}
	m.decisions.WithLabelValues(model, reason, strategy, PolicyLabel(policyVersion)).Inc()
}

// DecisionTime records how long a routing decision took.
func (m *Metrics) DecisionTime(d time.Duration) {
	if m == nil {
		return
	}
	m.decisionTime.Observe(d.Seconds())
}

// PredictedSuccess records the calibrated success probability p of the model a
// decision chose.
func (m *Metrics) PredictedSuccess(model string, p float64) {
	if m == nil {
		return
	}
	m.predicted.WithLabelValues(model).Observe(p)
}

// GatewayOverhead records the time the gateway added to one request.
func (m *Metrics) GatewayOverhead(stream bool, d time.Duration) {
	if m == nil {
		return
	}
	m.overhead.WithLabelValues(strconv.FormatBool(stream)).Observe(d.Seconds())
}

// UpstreamTTFT records the time from the start of a streaming attempt to its
// first event.
func (m *Metrics) UpstreamTTFT(model string, d time.Duration) {
	if m == nil {
		return
	}
	m.ttft.WithLabelValues(model).Observe(d.Seconds())
}

// UpstreamDuration records how long an upstream attempt took and how it ended
// (ok, error, canceled or partial).
func (m *Metrics) UpstreamDuration(model, outcome string, d time.Duration) {
	if m == nil {
		return
	}
	m.upstream.WithLabelValues(model, outcome).Observe(d.Seconds())
}

// Retry counts one retry after an attempt on model failed with the given kind
// of error.
func (m *Metrics) Retry(model, kind string) {
	if m == nil {
		return
	}
	m.retries.WithLabelValues(model, kind).Inc()
}

// Failover counts one attempt made on another model than the one before it.
func (m *Metrics) Failover() {
	if m == nil {
		return
	}
	m.failovers.Inc()
}

// RateLimited counts one request refused by a tenant rate limit of the given
// kind (rpm or tpm).
func (m *Metrics) RateLimited(kind string) {
	if m == nil {
		return
	}
	m.rateLimited.WithLabelValues(kind).Inc()
}

// BudgetRejected counts one attempt refused for lack of budget.
func (m *Metrics) BudgetRejected() {
	if m == nil {
		return
	}
	m.budgetReject.Inc()
}

// Cost adds the settled cost of an attempt, in USD.
func (m *Metrics) Cost(model, costSource string, usd float64) {
	if m == nil {
		return
	}
	m.cost.WithLabelValues(model, costSource).Add(usd)
}

// Tokens adds the settled tokens of an attempt.
func (m *Metrics) Tokens(model, usageSource string, prompt, completion int) {
	if m == nil {
		return
	}
	m.tokens.WithLabelValues(model, "prompt", usageSource).Add(float64(prompt))
	m.tokens.WithLabelValues(model, "completion", usageSource).Add(float64(completion))
}

// BudgetOvershoot adds the USD by which a settlement took a tenant over its
// daily budget.
func (m *Metrics) BudgetOvershoot(tenant string, usd float64) {
	if m == nil {
		return
	}
	m.overshoot.WithLabelValues(tenant).Add(usd)
}

// PolicyActivated makes version the one policy_info reports and drops the
// series of the version it replaced. previous may be empty.
func (m *Metrics) PolicyActivated(version, previous string) {
	if m == nil {
		return
	}
	if previous != "" && previous != version {
		m.policyInfo.DeleteLabelValues(previous)
	}
	m.policyInfo.WithLabelValues(version).Set(1)
}

// SetStateSources sets where the breaker_state, queue_depth and inflight gauges
// read their values. The model list is copied.
func (m *Metrics) SetStateSources(s StateSources) {
	if m == nil {
		return
	}
	s.Models = append([]string(nil), s.Models...)
	m.state.src.Store(&s)
}

// stateCollector computes the state gauges at scrape time from the sources.
type stateCollector struct {
	src      atomic.Pointer[StateSources]
	breaker  *prometheus.Desc
	queue    *prometheus.Desc
	inflight *prometheus.Desc
}

func newStateCollector() *stateCollector {
	return &stateCollector{
		breaker: prometheus.NewDesc("routeforge_breaker_state",
			"Circuit breaker state of a model: 0 closed, 1 half-open, 2 open.", []string{"model"}, nil),
		queue: prometheus.NewDesc("routeforge_queue_depth",
			"Requests waiting for a slot of a model.", []string{"model"}, nil),
		inflight: prometheus.NewDesc("routeforge_inflight",
			"Requests the gateway is serving.", nil, nil),
	}
}

// Describe implements prometheus.Collector.
func (c *stateCollector) Describe(ch chan<- *prometheus.Desc) {
	ch <- c.breaker
	ch <- c.queue
	ch <- c.inflight
}

// Collect implements prometheus.Collector.
func (c *stateCollector) Collect(ch chan<- prometheus.Metric) {
	src := c.src.Load()
	var inflight int
	if src != nil && src.InFlight != nil {
		inflight = src.InFlight()
	}
	ch <- prometheus.MustNewConstMetric(c.inflight, prometheus.GaugeValue, float64(inflight))
	if src == nil {
		return
	}
	for _, model := range src.Models {
		if src.BreakerState != nil {
			ch <- prometheus.MustNewConstMetric(c.breaker, prometheus.GaugeValue, float64(src.BreakerState(model)), model)
		}
		if src.QueueDepth != nil {
			ch <- prometheus.MustNewConstMetric(c.queue, prometheus.GaugeValue, float64(src.QueueDepth(model)), model)
		}
	}
}
