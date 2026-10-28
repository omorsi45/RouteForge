package bench

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"math"
	"net/http"
	"net/http/httptest"
	"os"
	"slices"
	"strings"
	"sync"
	"sync/atomic"
	"testing"
	"time"
)

func TestOpenLoopSchedule(t *testing.T) {
	const rps, seconds = 1000.0, 10
	offsets := arrivalOffsets(rps, seconds*time.Second, 42)

	// About rps * seconds arrivals: 10,000 draws. Poisson counts vary by about
	// 1%, so 5% leaves no room for a wrong rate and none for flakiness: the
	// seed fixes the draws.
	if n := float64(len(offsets)); math.Abs(n-rps*seconds) > 0.05*rps*seconds {
		t.Fatalf("%d arrivals in %d s at %v rps, want within 5%% of %v", len(offsets), seconds, rps, rps*seconds)
	}
	var gapSum float64
	prev := time.Duration(0)
	for i, off := range offsets {
		if off <= prev && i > 0 {
			t.Fatalf("arrival %d at %s does not follow %s", i, off, prev)
		}
		if off < 0 || off >= seconds*time.Second {
			t.Fatalf("arrival %d at %s is outside [0, %ds)", i, off, seconds)
		}
		gapSum += (off - prev).Seconds()
		prev = off
	}
	if meanGap := gapSum / float64(len(offsets)); math.Abs(meanGap-1/rps) > 0.05/rps {
		t.Errorf("mean gap %v s, want 1/%v within 5%%", meanGap, rps)
	}
	if !slices.Equal(offsets, arrivalOffsets(rps, seconds*time.Second, 42)) {
		t.Error("the same seed gave different arrivals")
	}
	if slices.Equal(offsets, arrivalOffsets(rps, seconds*time.Second, 43)) {
		t.Error("different seeds gave the same arrivals")
	}
}

func TestOpenLoopScheduleIsPoisson(t *testing.T) {
	// For Poisson arrivals the gaps are exponential: their standard deviation
	// equals their mean. A fixed-rate schedule would have a deviation of 0.
	offsets := arrivalOffsets(2000, 10*time.Second, 5)
	gaps := make([]float64, len(offsets))
	prev := time.Duration(0)
	for i, off := range offsets {
		gaps[i], prev = (off - prev).Seconds(), off
	}
	var mean, sq float64
	for _, g := range gaps {
		mean += g
	}
	mean /= float64(len(gaps))
	for _, g := range gaps {
		sq += (g - mean) * (g - mean)
	}
	if sd := math.Sqrt(sq / float64(len(gaps))); math.Abs(sd-mean) > 0.1*mean {
		t.Errorf("gap deviation %v vs mean %v, want them equal for exponential gaps", sd, mean)
	}
}

func TestOverheadSubtractsAttemptTime(t *testing.T) {
	attempts := []AttemptRecord{{DurationMS: 10.25}, {DurationMS: 5}}
	if got, want := overheadMS(25.5, attempts), 10.25; math.Abs(got-want) > 1e-12 {
		t.Errorf("overhead = %v, want %v", got, want)
	}
	if got := overheadMS(25.5, nil); got != 25.5 {
		t.Errorf("overhead without attempts = %v, want the whole time", got)
	}
}

// loadStub is a gateway for the load tests. Every chat request takes delay and
// gets a decision id; GET of a decision answers with attempts that took
// attemptMS in all and a routing time of decidedUS.
type loadStub struct {
	*httptest.Server
	mu        sync.Mutex
	chats     int
	streams   int
	headers   []http.Header
	delay     time.Duration
	attemptMS float64
	decidedUS int64
	failEvery int      // every nth chat request is refused with 503, 0 for none
	goneEvery int      // every nth decision is unknown, 0 for none
	fetched   []string // decision ids asked for
	nextID    atomic.Int64
}

func newLoadStub(t *testing.T) *loadStub {
	t.Helper()
	s := &loadStub{attemptMS: 20, decidedUS: 300}
	mux := http.NewServeMux()
	mux.HandleFunc("POST /v1/chat/completions", s.chat)
	mux.HandleFunc("GET /v1/routing/decisions/{id}", s.decision)
	s.Server = httptest.NewServer(mux)
	t.Cleanup(s.Close)
	return s
}

func (s *loadStub) chat(w http.ResponseWriter, r *http.Request) {
	var req struct {
		Stream bool `json:"stream"`
	}
	body, _ := io.ReadAll(r.Body)
	_ = json.Unmarshal(body, &req)
	n := s.nextID.Add(1)

	s.mu.Lock()
	s.chats++
	if req.Stream {
		s.streams++
	}
	s.headers = append(s.headers, r.Header.Clone())
	s.mu.Unlock()
	time.Sleep(s.delay)

	if s.failEvery > 0 && n%int64(s.failEvery) == 0 {
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(http.StatusServiceUnavailable)
		_, _ = io.WriteString(w, `{"error":{"message":"m","type":"server_error","code":"overloaded"}}`)
		return
	}
	w.Header().Set("X-RouteForge-Decision-Id", fmt.Sprintf("dec_%d", n))
	w.Header().Set("X-RouteForge-Model", "m1")
	if req.Stream {
		w.Header().Set("Content-Type", "text/event-stream")
		_, _ = io.WriteString(w, "data: "+contentFrame+"\n\ndata: [DONE]\n\n")
		return
	}
	_, _ = io.WriteString(w, fmt.Sprintf(`{"routeforge":{"decision_id":"dec_%d"}}`, n))
}

func (s *loadStub) decision(w http.ResponseWriter, r *http.Request) {
	id := r.PathValue("id")
	s.mu.Lock()
	s.fetched = append(s.fetched, id)
	s.mu.Unlock()
	var n int64
	_, _ = fmt.Sscanf(id, "dec_%d", &n)
	if s.goneEvery > 0 && n%int64(s.goneEvery) == 0 {
		w.WriteHeader(http.StatusNotFound)
		_, _ = io.WriteString(w, `{"error":{"message":"decision not found","type":"invalid_request_error","code":"not_found"}}`)
		return
	}
	half := s.attemptMS / 2
	_, _ = fmt.Fprintf(w, `{"id":%q,"chosen":"m1","decided_in_us":%d,"attempts":[{"model":"m1","outcome":"error","duration_ms":%v},{"model":"m2","outcome":"ok","duration_ms":%v}]}`,
		id, s.decidedUS, half, half)
}

// fakeGatewayRecord lists, in order, the gateways a run started and stopped.
type fakeGatewayRecord struct {
	mu     sync.Mutex
	events []string
}

func (r *fakeGatewayRecord) add(e string) {
	r.mu.Lock()
	defer r.mu.Unlock()
	r.events = append(r.events, e)
}

type loadGateway struct {
	url     string
	name    string
	rec     *fakeGatewayRecord
	stopErr error
}

func (g *loadGateway) BaseURL() string { return g.url }
func (g *loadGateway) PID() int        { return os.Getpid() }
func (g *loadGateway) Stop() error {
	g.rec.add("stop " + g.name)
	return g.stopErr
}

// loadOptions runs the scenarios against s, recording starts and stops in rec.
func loadOptions(s *loadStub, rec *fakeGatewayRecord, scenarios ...Scenario) LoadOptions {
	return LoadOptions{
		Kind: KindLoadZero,
		Start: func(_ context.Context, name string) (Gateway, error) {
			rec.add("start " + name)
			return &loadGateway{url: s.URL, name: name, rec: rec}, nil
		},
		Key:            testKey,
		Scenarios:      scenarios,
		Seed:           1,
		SampleInterval: 5 * time.Millisecond,
		Sample:         func(int) (ResourceSample, error) { return ResourceSample{RSSBytes: 1 << 20, CPUPercent: 2}, nil },
	}
}

func closedScenario(name string, concurrency, requests int) Scenario {
	return Scenario{Name: name, Mode: ModeClosed, Concurrency: concurrency, Requests: requests}
}

func TestOverheadSubtractsAttempts(t *testing.T) {
	s := newLoadStub(t)
	s.delay = 30 * time.Millisecond // every request takes at least this long end to end
	s.attemptMS = 20                // and 20 ms of that was upstream, as its decision says

	res, err := RunLoad(context.Background(), loadOptions(s, &fakeGatewayRecord{}, closedScenario("closed_loop_c4", 4, 24)))
	if err != nil {
		t.Fatal(err)
	}
	sc := res.Scenarios["closed_loop_c4"]
	if sc.Completed != 24 || sc.DecisionsFetched != 24 || sc.DecisionsMissing != 0 {
		t.Fatalf("completed = %d, fetched = %d, missing = %d, want 24, 24, 0", sc.Completed, sc.DecisionsFetched, sc.DecisionsMissing)
	}
	// Overhead is each request's end-to-end time less its attempts' time, so
	// the means differ by exactly the attempt time.
	if d := sc.LatencyMS["mean"] - sc.OverheadMS["mean"]; math.Abs(d-20) > 1e-9 {
		t.Errorf("latency mean - overhead mean = %v, want 20", d)
	}
	if sc.OverheadMS["n"] != 24 || sc.OverheadMS["p50"] < 10 {
		t.Errorf("overhead = %v, want 24 samples of at least 10 ms (30 ms - 20 ms)", sc.OverheadMS)
	}
	if sc.DecisionUS["n"] != 24 || sc.DecisionUS["p50"] != 300 || sc.DecisionUS["max"] != 300 {
		t.Errorf("decision_us = %v, want 24 samples of 300", sc.DecisionUS)
	}
	if got := len(s.fetched); got != 24 {
		t.Errorf("fetched %d decisions, want one per request", got)
	}
}

func TestRunLoadClosedLoopHalfStreams(t *testing.T) {
	s := newLoadStub(t)
	res, err := RunLoad(context.Background(), loadOptions(s, &fakeGatewayRecord{}, closedScenario("closed_loop_c8", 8, 100)))
	if err != nil {
		t.Fatal(err)
	}
	sc := res.Scenarios["closed_loop_c8"]
	if sc.Mode != ModeClosed || sc.Concurrency != 8 || sc.Requests != 100 || sc.Completed != 100 {
		t.Errorf("scenario = %+v", sc)
	}
	if sc.StreamRequests != 50 || s.streams != 50 || s.chats != 100 {
		t.Errorf("stream requests = %d (gateway saw %d of %d), want 50 of 100", sc.StreamRequests, s.streams, s.chats)
	}
	if sc.ErrorRate != 0 || len(sc.Errors) != 0 {
		t.Errorf("error rate = %v, errors = %v, want none", sc.ErrorRate, sc.Errors)
	}
	if sc.ThroughputRPS <= 0 || sc.ElapsedS <= 0 {
		t.Errorf("throughput = %v, elapsed = %v", sc.ThroughputRPS, sc.ElapsedS)
	}
	if want := float64(sc.Completed) / sc.ElapsedS; math.Abs(sc.ThroughputRPS-want) > 1e-9 {
		t.Errorf("throughput = %v, want completed / elapsed = %v", sc.ThroughputRPS, want)
	}
}

func TestRunLoadSendsQualityBalancedWithTheKey(t *testing.T) {
	s := newLoadStub(t)
	if _, err := RunLoad(context.Background(), loadOptions(s, &fakeGatewayRecord{}, closedScenario("closed_loop_c2", 2, 6))); err != nil {
		t.Fatal(err)
	}
	for i, h := range s.headers {
		if h.Get("X-RouteForge-Strategy") != "quality" || h.Get("X-RouteForge-Profile") != "balanced" || h.Get("Authorization") != "Bearer "+testKey {
			t.Errorf("request %d headers = %v", i, h)
		}
	}
}

func TestRunLoadOpenLoop(t *testing.T) {
	s := newLoadStub(t)
	sc := Scenario{Name: "open_loop_100rps", Mode: ModeOpen, RPS: 100, Duration: 500 * time.Millisecond}
	res, err := RunLoad(context.Background(), loadOptions(s, &fakeGatewayRecord{}, sc))
	if err != nil {
		t.Fatal(err)
	}
	got := res.Scenarios["open_loop_100rps"]
	// About 50 Poisson arrivals, none streaming.
	if got.Requests < 20 || got.Requests > 100 || got.Completed != got.Requests {
		t.Errorf("requests = %d, completed = %d, want about 50 and all completed", got.Requests, got.Completed)
	}
	if got.StreamRequests != 0 || s.streams != 0 {
		t.Errorf("open loop requests streamed: %d", s.streams)
	}
	if got.RPS != 100 || got.DurationS != 0.5 {
		t.Errorf("rps = %v, duration = %v, want the scenario's", got.RPS, got.DurationS)
	}
}

func TestRunLoadFreshGatewayPerScenario(t *testing.T) {
	s := newLoadStub(t)
	rec := &fakeGatewayRecord{}
	_, err := RunLoad(context.Background(), loadOptions(s, rec, closedScenario("a", 1, 2), closedScenario("b", 1, 2)))
	if err != nil {
		t.Fatal(err)
	}
	// Each scenario starts a gateway and stops it before the next one starts,
	// so the decision ring holds every request of the scenario and nothing else.
	want := []string{"start a", "stop a", "start b", "stop b"}
	if !slices.Equal(rec.events, want) {
		t.Errorf("events = %v, want %v", rec.events, want)
	}
}

func TestRunLoadCountsErrorsAndMissingDecisions(t *testing.T) {
	s := newLoadStub(t)
	s.failEvery = 4 // 5 of 20 requests are refused
	s.goneEvery = 5 // and of the 15 decisions asked for, those numbered 5, 10 and 15 are gone
	res, err := RunLoad(context.Background(), loadOptions(s, &fakeGatewayRecord{}, closedScenario("closed_loop_c1", 1, 20)))
	if err != nil {
		t.Fatal(err)
	}
	sc := res.Scenarios["closed_loop_c1"]
	if sc.Completed != 15 || sc.Errors["overloaded"] != 5 || math.Abs(sc.ErrorRate-0.25) > 1e-12 {
		t.Errorf("completed = %d, errors = %v, error rate = %v, want 15, 5 overloaded, 0.25", sc.Completed, sc.Errors, sc.ErrorRate)
	}
	if sc.DecisionsMissing != 3 || sc.DecisionsFetched != 12 {
		t.Errorf("fetched = %d, missing = %d, want 12 and 3", sc.DecisionsFetched, sc.DecisionsMissing)
	}
	// Throughput counts the requests that completed, not the ones that were sent.
	if want := float64(sc.Completed) / sc.ElapsedS; math.Abs(sc.ThroughputRPS-want) > 1e-9 {
		t.Errorf("throughput = %v, want completed / elapsed = %v", sc.ThroughputRPS, want)
	}
	// Latency covers the 15 requests that completed, overhead the 12 whose
	// decision could be read.
	if sc.LatencyMS["n"] != 15 || sc.OverheadMS["n"] != 12 {
		t.Errorf("latency n = %v, overhead n = %v, want 15 and 12", sc.LatencyMS["n"], sc.OverheadMS["n"])
	}
}

func TestRunLoadRecordsTransportFailures(t *testing.T) {
	s := newLoadStub(t)
	opts := loadOptions(s, &fakeGatewayRecord{}, closedScenario("closed_loop_c1", 1, 5))
	s.Close() // every request fails to connect
	res, err := RunLoad(context.Background(), opts)
	if err != nil {
		t.Fatalf("a gateway that cannot be reached is a result, not an error: %v", err)
	}
	sc := res.Scenarios["closed_loop_c1"]
	if sc.Completed != 0 || sc.Errors["transport_error"] != 5 || sc.ErrorRate != 1 {
		t.Errorf("completed = %d, errors = %v, error rate = %v", sc.Completed, sc.Errors, sc.ErrorRate)
	}
	if sc.LatencyMS["n"] != 0 || len(sc.LatencyMS) != 1 {
		t.Errorf("latency = %v, want only n = 0", sc.LatencyMS)
	}
}

func TestRunLoadSamplesTheGatewayProcess(t *testing.T) {
	s := newLoadStub(t)
	s.delay = 5 * time.Millisecond
	var (
		mu     sync.Mutex
		seen   []ResourceSample
		pids   = map[int]bool{}
		counts int
	)
	opts := loadOptions(s, &fakeGatewayRecord{}, closedScenario("closed_loop_c2", 2, 40))
	opts.Sample = func(pid int) (ResourceSample, error) {
		mu.Lock()
		defer mu.Unlock()
		counts++
		pids[pid] = true
		smp := ResourceSample{RSSBytes: int64(counts) * 4096, CPUPercent: float64(counts), CPUMethod: "test_method"}
		seen = append(seen, smp)
		return smp, nil
	}
	res, err := RunLoad(context.Background(), opts)
	if err != nil {
		t.Fatal(err)
	}
	sc := res.Scenarios["closed_loop_c2"]

	mu.Lock()
	defer mu.Unlock()
	if len(seen) < 2 || sc.Samples != len(seen) {
		t.Fatalf("result counts %d samples, the sampler was called %d times, want the same and at least 2", sc.Samples, len(seen))
	}
	if !pids[os.Getpid()] || len(pids) != 1 {
		t.Errorf("sampled pids %v, want only the gateway's (%d)", pids, os.Getpid())
	}
	var rssSum, cpuSum float64
	for _, smp := range seen {
		rssSum += float64(smp.RSSBytes)
		cpuSum += smp.CPUPercent
	}
	last := seen[len(seen)-1]
	if sc.RSSBytes["max"] != float64(last.RSSBytes) || math.Abs(sc.RSSBytes["mean"]-rssSum/float64(len(seen))) > 1e-6 {
		t.Errorf("rss = %v, want max %d and mean %v", sc.RSSBytes, last.RSSBytes, rssSum/float64(len(seen)))
	}
	if math.Abs(sc.CPUPercent["mean"]-cpuSum/float64(len(seen))) > 1e-9 {
		t.Errorf("cpu = %v, want mean %v", sc.CPUPercent, cpuSum/float64(len(seen)))
	}
	if sc.CPUMethod != "test_method" {
		t.Errorf("cpu method = %q, want the one the samples carry", sc.CPUMethod)
	}
	if sc.SamplingError != "" {
		t.Errorf("sampling error %q with every sample read", sc.SamplingError)
	}
}

func TestRunLoadMeansCPUOverIntervalsOnly(t *testing.T) {
	s := newLoadStub(t)
	s.delay = 5 * time.Millisecond
	var calls atomic.Int32
	opts := loadOptions(s, &fakeGatewayRecord{}, closedScenario("closed_loop_c2", 2, 30))
	opts.Sample = func(int) (ResourceSample, error) {
		smp := ResourceSample{RSSBytes: 1 << 20, CPUPercent: 40, CPUMethod: "proc_stat_interval"}
		if calls.Add(1) == 1 {
			// The first reading of a process is only a baseline: it carries no
			// CPU figure, and a 0 in its place must not pull the mean down.
			smp.NoCPU, smp.CPUPercent = true, 0
		}
		return smp, nil
	}
	res, err := RunLoad(context.Background(), opts)
	if err != nil {
		t.Fatal(err)
	}
	sc := res.Scenarios["closed_loop_c2"]
	if sc.Samples < 2 {
		t.Fatalf("samples = %d, want the baseline and at least one more", sc.Samples)
	}
	if sc.CPUPercent["mean"] != 40 {
		t.Errorf("cpu mean = %v over %d samples, want 40 (the baseline is not an interval)", sc.CPUPercent, sc.Samples)
	}
	if sc.CPUMethod != "proc_stat_interval" {
		t.Errorf("cpu method = %q", sc.CPUMethod)
	}
	// The resident size of the baseline counts: memory needs no interval.
	if sc.RSSBytes["mean"] != 1<<20 {
		t.Errorf("rss = %v, want every sample counted", sc.RSSBytes)
	}
}

func TestRunLoadWithOnlyABaselineHasNoCPU(t *testing.T) {
	s := newLoadStub(t)
	opts := loadOptions(s, &fakeGatewayRecord{}, closedScenario("closed_loop_c1", 1, 3))
	opts.Sample = func(int) (ResourceSample, error) {
		return ResourceSample{RSSBytes: 1 << 20, NoCPU: true, CPUMethod: "proc_stat_interval"}, nil
	}
	res, err := RunLoad(context.Background(), opts)
	if err != nil {
		t.Fatal(err)
	}
	sc := res.Scenarios["closed_loop_c1"]
	if sc.CPUPercent != nil || sc.CPUMethod != "" {
		t.Errorf("cpu = %v, method = %q, want neither when no interval was measured", sc.CPUPercent, sc.CPUMethod)
	}
	if sc.RSSBytes == nil {
		t.Error("the resident size was read and is missing")
	}
}

func TestRunLoadSamplesPerIntervalByDefault(t *testing.T) {
	s := newLoadStub(t)
	s.delay = 5 * time.Millisecond
	opts := loadOptions(s, &fakeGatewayRecord{}, closedScenario("closed_loop_c2", 2, 40))
	opts.Sample = nil // the default sampler reads the gateway process, which is this test
	res, err := RunLoad(context.Background(), opts)
	if err != nil {
		t.Fatal(err)
	}
	sc := res.Scenarios["closed_loop_c2"]
	want := CPUMethodPSLifetime
	if _, err := os.Stat(fmt.Sprintf("/proc/%d/stat", os.Getpid())); err == nil {
		want = CPUMethodProcStat
	}
	if sc.Samples < 2 || sc.SamplingError != "" || sc.RSSBytes["max"] <= 0 {
		t.Fatalf("samples = %d, error %q, rss = %v, want the process read", sc.Samples, sc.SamplingError, sc.RSSBytes)
	}
	if sc.CPUMethod != want || sc.CPUPercent["mean"] < 0 {
		t.Errorf("cpu method = %q, cpu = %v, want method %s and a mean of at least 0", sc.CPUMethod, sc.CPUPercent, want)
	}
}

func TestRunLoadSamplingFailureIsRecordedNotInvented(t *testing.T) {
	s := newLoadStub(t)
	s.delay = 5 * time.Millisecond
	opts := loadOptions(s, &fakeGatewayRecord{}, closedScenario("closed_loop_c1", 1, 10))
	opts.Sample = func(int) (ResourceSample, error) { return ResourceSample{}, errors.New("ps: not found") }
	res, err := RunLoad(context.Background(), opts)
	if err != nil {
		t.Fatal(err)
	}
	sc := res.Scenarios["closed_loop_c1"]
	if sc.Samples != 0 || sc.RSSBytes != nil || sc.CPUPercent != nil {
		t.Errorf("samples = %d, rss = %v, cpu = %v, want none", sc.Samples, sc.RSSBytes, sc.CPUPercent)
	}
	if !strings.Contains(sc.SamplingError, "ps: not found") {
		t.Errorf("sampling error = %q, want the reason", sc.SamplingError)
	}
	data, err := json.Marshal(sc)
	if err != nil {
		t.Fatal(err)
	}
	if strings.Contains(string(data), "rss_bytes") {
		t.Errorf("result has rss_bytes without samples: %s", data)
	}
}

func TestRunLoadOverheadTarget(t *testing.T) {
	s := newLoadStub(t)
	scenario := Scenario{Name: OverheadScenario, Mode: ModeOpen, RPS: 200, Duration: 300 * time.Millisecond}

	t.Run("profiled run", func(t *testing.T) {
		opts := loadOptions(s, &fakeGatewayRecord{}, scenario)
		opts.Kind = KindLoadProfiled
		res, err := RunLoad(context.Background(), opts)
		if err != nil {
			t.Fatal(err)
		}
		target := res.OverheadTarget
		if target == nil {
			t.Fatal("a profiled run reports the overhead target")
		}
		if target.P95MSBelow != 20 || target.Scenario != "profiled open loop 200 rps" {
			t.Errorf("target = %+v", target)
		}
		if want := res.Scenarios[OverheadScenario].OverheadMS["p95"]; target.P95MS != want {
			t.Errorf("p95 = %v, want the scenario's overhead p95 %v", target.P95MS, want)
		}
		data, err := json.Marshal(target)
		if err != nil {
			t.Fatal(err)
		}
		for _, key := range []string{`"p95_ms_below":20`, `"scenario":"profiled open loop 200 rps"`, `"p95_ms":`, `"pass":`} {
			if !strings.Contains(string(data), key) {
				t.Errorf("target JSON %s lacks %s", data, key)
			}
		}
	})

	t.Run("zero latency run has no target", func(t *testing.T) {
		res, err := RunLoad(context.Background(), loadOptions(s, &fakeGatewayRecord{}, scenario))
		if err != nil {
			t.Fatal(err)
		}
		if res.OverheadTarget != nil {
			t.Errorf("target = %+v, want none outside the profiled run", res.OverheadTarget)
		}
	})

	t.Run("profiled run without the scenario", func(t *testing.T) {
		opts := loadOptions(s, &fakeGatewayRecord{}, closedScenario("closed_loop_c1", 1, 3))
		opts.Kind = KindLoadProfiled
		if _, err := RunLoad(context.Background(), opts); err == nil || !strings.Contains(err.Error(), OverheadScenario) {
			t.Fatalf("err = %v, want one that names %s", err, OverheadScenario)
		}
	})
}

func TestOverheadTargetVerdict(t *testing.T) {
	for _, c := range []struct {
		p95  float64
		pass bool
	}{{0.5, true}, {19.999, true}, {20, false}, {35, false}} {
		got := newOverheadTarget(c.p95)
		if got.Pass != c.pass || got.P95MS != c.p95 || got.P95MSBelow != 20 {
			t.Errorf("newOverheadTarget(%v) = %+v, want pass = %v (below 20 ms)", c.p95, got, c.pass)
		}
	}
}

func TestRunLoadStopsTheGatewayOnEveryPath(t *testing.T) {
	s := newLoadStub(t)
	t.Run("start fails", func(t *testing.T) {
		opts := loadOptions(s, &fakeGatewayRecord{}, closedScenario("a", 1, 2))
		opts.Start = func(context.Context, string) (Gateway, error) { return nil, errors.New("no port") }
		if _, err := RunLoad(context.Background(), opts); err == nil || !strings.Contains(err.Error(), "no port") {
			t.Fatalf("err = %v, want the start failure", err)
		}
	})
	t.Run("context ends mid run", func(t *testing.T) {
		rec := &fakeGatewayRecord{}
		slow := newLoadStub(t)
		slow.delay = 200 * time.Millisecond
		ctx, cancel := context.WithTimeout(context.Background(), 100*time.Millisecond)
		defer cancel()
		_, err := RunLoad(ctx, loadOptions(slow, rec, closedScenario("a", 1, 50)))
		if err == nil {
			t.Fatal("want an error when the context ends mid run")
		}
		if want := []string{"start a", "stop a"}; !slices.Equal(rec.events, want) {
			t.Errorf("events = %v, want %v", rec.events, want)
		}
	})
	t.Run("stop fails", func(t *testing.T) {
		rec := &fakeGatewayRecord{}
		opts := loadOptions(s, rec, closedScenario("a", 1, 2))
		opts.Start = func(_ context.Context, name string) (Gateway, error) {
			return &loadGateway{url: s.URL, name: name, rec: rec, stopErr: errors.New("had to kill it")}, nil
		}
		if _, err := RunLoad(context.Background(), opts); err == nil || !strings.Contains(err.Error(), "had to kill it") {
			t.Fatalf("err = %v, want the stop failure reported", err)
		}
	})
}

func TestRunLoadRejectsBadOptions(t *testing.T) {
	s := newLoadStub(t)
	good := func() LoadOptions {
		return loadOptions(s, &fakeGatewayRecord{}, closedScenario("a", 1, 2))
	}
	for _, c := range []struct {
		name   string
		mutate func(o *LoadOptions)
	}{
		{"no starter", func(o *LoadOptions) { o.Start = nil }},
		{"no scenario name", func(o *LoadOptions) { o.Scenarios[0].Name = "" }},
		{"duplicate scenario name", func(o *LoadOptions) { o.Scenarios = append(o.Scenarios, o.Scenarios[0]) }},
		{"closed loop without concurrency", func(o *LoadOptions) { o.Scenarios[0].Concurrency = 0 }},
		{"closed loop without requests", func(o *LoadOptions) { o.Scenarios[0].Requests = 0 }},
		{"open loop without a rate", func(o *LoadOptions) { o.Scenarios[0] = Scenario{Name: "a", Mode: ModeOpen, Duration: time.Second} }},
		{"open loop without a duration", func(o *LoadOptions) { o.Scenarios[0] = Scenario{Name: "a", Mode: ModeOpen, RPS: 10} }},
		{"unknown mode", func(o *LoadOptions) { o.Scenarios[0].Mode = "spiral" }},
	} {
		t.Run(c.name, func(t *testing.T) {
			o := good()
			c.mutate(&o)
			if _, err := RunLoad(context.Background(), o); err == nil {
				t.Fatal("want an error")
			}
		})
	}
}

func TestDefaultScenarios(t *testing.T) {
	got := DefaultScenarios()
	type shape struct {
		name   string
		mode   string
		rps    float64
		conc   int
		reqs   int
		length time.Duration
	}
	var have []shape
	for _, sc := range got {
		have = append(have, shape{sc.Name, sc.Mode, sc.RPS, sc.Concurrency, sc.Requests, sc.Duration})
	}
	want := []shape{
		{"open_loop_50rps", ModeOpen, 50, 0, 0, 15 * time.Second},
		{"open_loop_200rps", ModeOpen, 200, 0, 0, 15 * time.Second},
		{"open_loop_500rps", ModeOpen, 500, 0, 0, 15 * time.Second},
		{"closed_loop_c1", ModeClosed, 0, 1, 3000, 0},
		{"closed_loop_c16", ModeClosed, 0, 16, 3000, 0},
		{"closed_loop_c64", ModeClosed, 0, 64, 3000, 0},
	}
	if !slices.Equal(have, want) {
		t.Errorf("scenarios = %+v, want %+v", have, want)
	}
	if OverheadScenario != "open_loop_200rps" {
		t.Errorf("overhead scenario = %q", OverheadScenario)
	}
}

func TestParsePS(t *testing.T) {
	for _, c := range []struct {
		name    string
		in      string
		want    ResourceSample
		wantErr bool
	}{
		{"normal", "  51234  1.5\n", ResourceSample{RSSBytes: 51234 * 1024, CPUPercent: 1.5}, false},
		{"one line no padding", "100 0.0", ResourceSample{RSSBytes: 100 * 1024}, false},
		{"empty", "", ResourceSample{}, true},
		{"one field", "51234\n", ResourceSample{}, true},
		{"three fields", "1 2 3\n", ResourceSample{}, true},
		{"rss not a number", "lots 1.0\n", ResourceSample{}, true},
		{"cpu not a number", "100 high\n", ResourceSample{}, true},
		{"negative rss", "-5 1.0\n", ResourceSample{}, true},
	} {
		t.Run(c.name, func(t *testing.T) {
			got, err := parsePS([]byte(c.in))
			if (err != nil) != c.wantErr {
				t.Fatalf("err = %v, want error = %v", err, c.wantErr)
			}
			if !c.wantErr && got != c.want {
				t.Errorf("got %+v, want %+v", got, c.want)
			}
		})
	}
}
