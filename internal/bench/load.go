package bench

import (
	"context"
	"errors"
	"fmt"
	"math"
	"math/rand/v2"
	"sync"
	"time"
)

// The kinds of load run, which name the result files.
const (
	// KindLoadZero runs against providers that answer at once, so that the
	// time a request takes is the gateway's.
	KindLoadZero = "load-zero"
	// KindLoadProfiled runs against providers that take the time of the
	// simulated latency profiles.
	KindLoadProfiled = "load-profiled"
)

// The ways a scenario offers load.
const (
	// ModeOpen sends requests at a fixed average rate whatever the gateway
	// does, with Poisson arrivals.
	ModeOpen = "open"
	// ModeClosed keeps a fixed number of requests in flight, each new one
	// sent when another returns.
	ModeClosed = "closed"
)

const (
	// OverheadTargetMS is the hypothesis the profiled run tests: the p95 of
	// the time the gateway adds to a request is below it.
	OverheadTargetMS = 20.0
	// OverheadScenario is the scenario the overhead target is measured on.
	OverheadScenario      = "open_loop_200rps"
	overheadScenarioLabel = "profiled open loop 200 rps"
)

// loadHeaders select the routing every load request asks for.
var loadHeaders = map[string]string{
	"X-RouteForge-Strategy": "quality",
	"X-RouteForge-Profile":  "balanced",
}

// Scenario is one load run against a gateway started for it.
type Scenario struct {
	Name string
	Mode string // ModeOpen or ModeClosed

	// RPS and Duration set an open loop: Poisson arrivals at RPS per second
	// for Duration.
	RPS      float64
	Duration time.Duration
	// Concurrency and Requests set a closed loop: Requests requests with
	// Concurrency of them in flight, half of them streaming.
	Concurrency int
	Requests    int
}

// DefaultScenarios are the six scenarios of the load benchmark.
func DefaultScenarios() []Scenario {
	return []Scenario{
		{Name: "open_loop_50rps", Mode: ModeOpen, RPS: 50, Duration: 15 * time.Second},
		{Name: "open_loop_200rps", Mode: ModeOpen, RPS: 200, Duration: 15 * time.Second},
		{Name: "open_loop_500rps", Mode: ModeOpen, RPS: 500, Duration: 15 * time.Second},
		{Name: "closed_loop_c1", Mode: ModeClosed, Concurrency: 1, Requests: 3000},
		{Name: "closed_loop_c16", Mode: ModeClosed, Concurrency: 16, Requests: 3000},
		{Name: "closed_loop_c64", Mode: ModeClosed, Concurrency: 64, Requests: 3000},
	}
}

func (s Scenario) validate() error {
	if s.Name == "" {
		return errors.New("a scenario has no name")
	}
	switch s.Mode {
	case ModeOpen:
		if !(s.RPS > 0) || s.Duration <= 0 {
			return fmt.Errorf("scenario %s: an open loop needs a positive rps and duration", s.Name)
		}
	case ModeClosed:
		if s.Concurrency < 1 || s.Requests < 1 {
			return fmt.Errorf("scenario %s: a closed loop needs a positive concurrency and request count", s.Name)
		}
	default:
		return fmt.Errorf("scenario %s: unknown mode %q", s.Name, s.Mode)
	}
	return nil
}

// GatewayStarter starts a fresh gateway for the named scenario. A fresh gateway
// per scenario means that its decision store holds the decisions of that
// scenario alone, which the overhead measurement relies on (it reads every
// request's decision back), and that no scenario starts with the latency
// estimates, queues or breakers another left behind.
type GatewayStarter func(ctx context.Context, scenario string) (Gateway, error)

// SampleFunc reads the resident size and CPU use of a process.
type SampleFunc func(pid int) (ResourceSample, error)

// LoadOptions configures RunLoad.
type LoadOptions struct {
	// Kind is KindLoadProfiled for the run that tests the overhead target.
	Kind  string
	Start GatewayStarter
	// Key is the API key of the tenant the requests are sent as.
	Key string
	// Scenarios to run in order; DefaultScenarios when empty.
	Scenarios []Scenario
	// Seed fixes the prompts and the arrival times.
	Seed uint64
	// SampleInterval is how often the gateway process is sampled (default 1 s)
	// and Sample how (default ps).
	SampleInterval time.Duration
	Sample         SampleFunc
	// RequestTimeout bounds one request (default 60 s).
	RequestTimeout time.Duration
}

// ScenarioResult is what one scenario measured.
type ScenarioResult struct {
	Mode        string  `json:"mode"`
	RPS         float64 `json:"rps,omitempty"`
	DurationS   float64 `json:"duration_s,omitempty"`
	Concurrency int     `json:"concurrency,omitempty"`

	Requests       int `json:"requests"`        // requests sent
	StreamRequests int `json:"stream_requests"` // of them, streaming
	// Completed counts the requests that returned 200 and, for a stream, ended
	// with [DONE]. ThroughputRPS is Completed per second of ElapsedS, the time
	// from the first request to the last response.
	Completed     int                `json:"completed"`
	ElapsedS      float64            `json:"elapsed_s"`
	ThroughputRPS float64            `json:"throughput_rps"`
	ErrorRate     float64            `json:"error_rate"`
	Errors        map[string]int     `json:"errors"` // failed requests by error code
	LatencyMS     map[string]float64 `json:"latency_ms"`
	// OverheadMS is the time a request took end to end less the time of its
	// attempts, as the stored decision records them: the gateway's share,
	// queueing included. DecisionUS is the time routing took.
	OverheadMS       map[string]float64 `json:"overhead_ms"`
	DecisionUS       map[string]float64 `json:"decision_us"`
	DecisionsFetched int                `json:"decisions_fetched"`
	DecisionsMissing int                `json:"decisions_missing"` // not stored when asked for

	// Samples counts the resource samples of the gateway process taken. When
	// none could be, RSSBytes and CPUPercent are absent and SamplingError says
	// why. CPUPercent is what ps reports: CPU time over the process's life.
	Samples       int                `json:"samples"`
	SamplingError string             `json:"sampling_error,omitempty"`
	RSSBytes      map[string]float64 `json:"rss_bytes,omitempty"`
	CPUPercent    map[string]float64 `json:"cpu_percent,omitempty"`
}

// OverheadTarget is the verdict on the hypothesis that the gateway adds under
// 20 ms at the 95th percentile.
type OverheadTarget struct {
	P95MSBelow float64 `json:"p95_ms_below"`
	Scenario   string  `json:"scenario"`
	P95MS      float64 `json:"p95_ms"`
	Pass       bool    `json:"pass"`
}

func newOverheadTarget(p95 float64) OverheadTarget {
	return OverheadTarget{P95MSBelow: OverheadTargetMS, Scenario: overheadScenarioLabel, P95MS: p95, Pass: p95 < OverheadTargetMS}
}

// LoadResults is what RunLoad measured.
type LoadResults struct {
	Scenarios map[string]ScenarioResult `json:"scenarios"`
	// OverheadTarget is set for the profiled run only.
	OverheadTarget *OverheadTarget `json:"overhead_target,omitempty"`
}

// RunLoad runs the scenarios in order, each against a gateway started for it and
// stopped after it, and returns what they measured. Requests that fail are
// results, counted by error code; an error ends the run: a gateway that cannot
// be started or stopped cleanly, a decision that cannot be read, or a context
// that ends.
func RunLoad(ctx context.Context, o LoadOptions) (LoadResults, error) {
	if o.Start == nil {
		return LoadResults{}, errors.New("load options need a gateway starter")
	}
	if o.Key == "" {
		return LoadResults{}, errors.New("load options need an API key")
	}
	scenarios := o.Scenarios
	if len(scenarios) == 0 {
		scenarios = DefaultScenarios()
	}
	names := map[string]bool{}
	for _, sc := range scenarios {
		if err := sc.validate(); err != nil {
			return LoadResults{}, err
		}
		if names[sc.Name] {
			return LoadResults{}, fmt.Errorf("scenario %s is listed twice", sc.Name)
		}
		names[sc.Name] = true
	}
	if o.SampleInterval <= 0 {
		o.SampleInterval = time.Second
	}
	if o.Sample == nil {
		o.Sample = psSample
	}
	if o.RequestTimeout <= 0 {
		o.RequestTimeout = 60 * time.Second
	}

	res := LoadResults{Scenarios: make(map[string]ScenarioResult, len(scenarios))}
	for i, sc := range scenarios {
		r, err := runScenario(ctx, o, sc, i)
		if err != nil {
			return LoadResults{}, err
		}
		res.Scenarios[sc.Name] = r
	}

	if o.Kind == KindLoadProfiled {
		sc, ok := res.Scenarios[OverheadScenario]
		if !ok {
			return LoadResults{}, fmt.Errorf("the profiled run measures its overhead target on scenario %s, which was not run", OverheadScenario)
		}
		p95, ok := sc.OverheadMS["p95"]
		if !ok {
			return LoadResults{}, fmt.Errorf("scenario %s has no overhead samples, so the overhead target cannot be judged", OverheadScenario)
		}
		target := newOverheadTarget(p95)
		res.OverheadTarget = &target
	}
	return res, nil
}

// runScenario runs sc against a fresh gateway and stops the gateway whatever
// happens.
func runScenario(ctx context.Context, o LoadOptions, sc Scenario, index int) (ScenarioResult, error) {
	gw, err := o.Start(ctx, sc.Name)
	if err != nil {
		return ScenarioResult{}, fmt.Errorf("scenario %s: start gateway: %w", sc.Name, err)
	}
	res, runErr := measure(ctx, o, sc, index, gw)
	if stopErr := gw.Stop(); stopErr != nil {
		stopErr = fmt.Errorf("scenario %s: stop gateway: %w", sc.Name, stopErr)
		if runErr != nil {
			return ScenarioResult{}, errors.Join(runErr, stopErr)
		}
		return ScenarioResult{}, stopErr
	}
	if runErr != nil {
		return ScenarioResult{}, runErr
	}
	return res, nil
}

// loadRecord is what one load request came to.
type loadRecord struct {
	stream     bool
	ok         bool
	errCode    string  // why it failed
	e2eMS      float64 // end to end, for a request that completed
	decisionID string

	haveDecision bool // the stored decision was read
	missing      bool // it was asked for and the gateway did not have it
	decidedUS    int64
	attempts     []AttemptRecord
}

// sender sends the requests of a scenario.
type sender struct {
	client  *Client
	prompts []string
	timeout time.Duration
}

// send sends request i, streaming or not, and records how it ended.
func (s *sender) send(ctx context.Context, i int, stream bool) loadRecord {
	rctx, cancel := context.WithTimeout(ctx, s.timeout)
	defer cancel()
	out, err := s.client.Do(rctx, ChatRequest{Prompt: s.prompts[i], Stream: stream, Headers: loadHeaders})
	rec := loadRecord{stream: stream, decisionID: out.DecisionID}
	switch {
	case errors.Is(err, context.DeadlineExceeded) && ctx.Err() == nil:
		rec.errCode = "client_timeout"
	case err != nil:
		rec.errCode = "transport_error"
	case out.Status != 200 || out.ErrorCode != "":
		rec.errCode = out.ErrorCode
	default:
		rec.ok = true
		rec.e2eMS = float64(out.Duration) / float64(time.Millisecond)
	}
	return rec
}

// measure offers the load of sc to gw and summarizes what happened.
func measure(ctx context.Context, o LoadOptions, sc Scenario, index int, gw Gateway) (ScenarioResult, error) {
	seed := o.Seed + uint64(index)
	var offsets []time.Duration
	n := sc.Requests
	if sc.Mode == ModeOpen {
		offsets = arrivalOffsets(sc.RPS, sc.Duration, seed)
		n = len(offsets)
	}
	snd := &sender{client: NewClient(gw.BaseURL(), o.Key), prompts: SyntheticPrompts(n, seed), timeout: o.RequestTimeout}
	defer snd.client.Close() // the gateway is stopped next, and waits for idle connections to go

	smp := startSampler(gw.PID(), o.SampleInterval, o.Sample)
	start := time.Now()
	var (
		records []loadRecord
		err     error
	)
	if sc.Mode == ModeOpen {
		records, err = openLoop(ctx, snd, offsets)
	} else {
		records, err = closedLoop(ctx, snd, sc.Concurrency, sc.Requests)
	}
	elapsed := time.Since(start)
	samples, sampleErr := smp.stop()
	if err != nil {
		return ScenarioResult{}, fmt.Errorf("scenario %s: %w", sc.Name, err)
	}

	if err := fetchDecisions(ctx, snd.client, records); err != nil {
		return ScenarioResult{}, fmt.Errorf("scenario %s: %w", sc.Name, err)
	}
	return buildScenarioResult(sc, records, elapsed, samples, sampleErr), nil
}

// arrivalOffsets returns the times, from the start of a run, at which the
// requests of an open loop are sent: a Poisson process of rate rps over d, so
// the gaps are exponential with mean 1/rps. The same seed gives the same times.
func arrivalOffsets(rps float64, d time.Duration, seed uint64) []time.Duration {
	rng := rand.New(rand.NewPCG(seed, 0x6c6f6164))
	limit := d.Seconds()
	var (
		offsets []time.Duration
		t       float64 // seconds
	)
	for {
		// 1-u is in (0, 1], so the logarithm is finite.
		t += -math.Log(1-rng.Float64()) / rps
		if t >= limit {
			return offsets
		}
		off := time.Duration(t * float64(time.Second))
		if n := len(offsets); n > 0 && off <= offsets[n-1] {
			off = offsets[n-1] + 1 // keep the arrivals in strict order at nanosecond resolution
		}
		offsets = append(offsets, off)
	}
}

// openLoop sends request i at offsets[i] after the start whatever the gateway
// has done with the earlier ones, and waits for all of them. A request that is
// late is sent at once, so a stall does not slow the rest down.
func openLoop(ctx context.Context, s *sender, offsets []time.Duration) ([]loadRecord, error) {
	records := make([]loadRecord, len(offsets))
	var wg sync.WaitGroup
	start := time.Now()
	for i, off := range offsets {
		if wait := time.Until(start.Add(off)); wait > 0 {
			timer := time.NewTimer(wait)
			select {
			case <-timer.C:
			case <-ctx.Done():
				timer.Stop()
				wg.Wait()
				return nil, ctx.Err()
			}
		}
		wg.Add(1)
		go func() {
			defer wg.Done()
			records[i] = s.send(ctx, i, false)
		}()
	}
	wg.Wait()
	return records, ctx.Err()
}

// closedLoop sends n requests with concurrency of them in flight; every second
// one streams.
func closedLoop(ctx context.Context, s *sender, concurrency, n int) ([]loadRecord, error) {
	records := make([]loadRecord, n)
	err := forEach(ctx, n, concurrency, func(ctx context.Context, i int) error {
		records[i] = s.send(ctx, i, i%2 == 1)
		return nil
	})
	if err != nil {
		return nil, err
	}
	return records, nil
}

// fetchDecisions reads back the stored decision of every request that
// completed and has one, and notes in its record the routing time and the time
// of its attempts. A decision the gateway does not have is noted as missing;
// any other failure to read one is an error.
func fetchDecisions(ctx context.Context, c *Client, records []loadRecord) error {
	var ids []int
	for i, r := range records {
		if r.ok && r.decisionID != "" {
			ids = append(ids, i)
		}
	}
	return forEach(ctx, len(ids), 16, func(ctx context.Context, k int) error {
		r := &records[ids[k]]
		d, err := c.Decision(ctx, r.decisionID)
		if errors.Is(err, ErrDecisionNotFound) {
			r.missing = true
			return nil
		}
		if err != nil {
			return err
		}
		r.haveDecision, r.decidedUS, r.attempts = true, d.DecidedInUS, d.Attempts
		return nil
	})
}

// overheadMS is the time of a request that was not spent in its attempts.
func overheadMS(e2eMS float64, attempts []AttemptRecord) float64 {
	for _, a := range attempts {
		e2eMS -= a.DurationMS
	}
	return e2eMS
}

// buildScenarioResult summarizes the records of a scenario.
func buildScenarioResult(sc Scenario, records []loadRecord, elapsed time.Duration, samples []ResourceSample, sampleErr error) ScenarioResult {
	res := ScenarioResult{
		Mode:        sc.Mode,
		RPS:         sc.RPS,
		DurationS:   sc.Duration.Seconds(),
		Concurrency: sc.Concurrency,
		Requests:    len(records),
		ElapsedS:    elapsed.Seconds(),
		Errors:      map[string]int{},
		Samples:     len(samples),
	}
	var latency, overhead, decision []float64
	for _, r := range records {
		if r.stream {
			res.StreamRequests++
		}
		if !r.ok {
			res.Errors[r.errCode]++
			continue
		}
		res.Completed++
		latency = append(latency, r.e2eMS)
		if r.haveDecision {
			res.DecisionsFetched++
			overhead = append(overhead, overheadMS(r.e2eMS, r.attempts))
			decision = append(decision, float64(r.decidedUS))
		}
		if r.missing {
			res.DecisionsMissing++
		}
	}
	if res.Requests > 0 {
		res.ErrorRate = float64(res.Requests-res.Completed) / float64(res.Requests)
	}
	if res.ElapsedS > 0 {
		res.ThroughputRPS = float64(res.Completed) / res.ElapsedS
	}
	res.LatencyMS, res.OverheadMS, res.DecisionUS = Summary(latency), Summary(overhead), Summary(decision)

	if sampleErr != nil {
		res.SamplingError = sampleErr.Error()
	}
	if len(samples) > 0 {
		var rssMax, rssSum, cpuSum float64
		for _, s := range samples {
			rss := float64(s.RSSBytes)
			rssMax = math.Max(rssMax, rss)
			rssSum += rss
			cpuSum += s.CPUPercent
		}
		n := float64(len(samples))
		res.RSSBytes = map[string]float64{"max": rssMax, "mean": rssSum / n}
		res.CPUPercent = map[string]float64{"mean": cpuSum / n}
	}
	return res
}

// sampler reads the resources of a process at a fixed interval.
type sampler struct {
	stopCh  chan struct{}
	done    chan struct{}
	samples []ResourceSample
	err     error // the first sample that failed
}

// startSampler samples process pid at once, then every interval, then once more
// when stopped.
func startSampler(pid int, interval time.Duration, sample SampleFunc) *sampler {
	s := &sampler{stopCh: make(chan struct{}), done: make(chan struct{})}
	take := func() {
		smp, err := sample(pid)
		if err != nil {
			if s.err == nil {
				s.err = err
			}
			return
		}
		s.samples = append(s.samples, smp)
	}
	go func() {
		defer close(s.done)
		tick := time.NewTicker(interval)
		defer tick.Stop()
		take()
		for {
			select {
			case <-tick.C:
				take()
			case <-s.stopCh:
				take()
				return
			}
		}
	}()
	return s
}

// stop ends the sampling and returns the samples taken and the first error.
func (s *sampler) stop() ([]ResourceSample, error) {
	close(s.stopCh)
	<-s.done
	return s.samples, s.err
}
