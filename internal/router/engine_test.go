package router_test

import (
	"context"
	"errors"
	"math"
	"reflect"
	"slices"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/omorsi45/routeforge/internal/accounting"
	"github.com/omorsi45/routeforge/internal/config"
	"github.com/omorsi45/routeforge/internal/policy"
	"github.com/omorsi45/routeforge/internal/provider"
	"github.com/omorsi45/routeforge/internal/router"
)

// The six pool models, in policy order. That order is the tie-break order.
const (
	gpt5     = "gpt-5"
	sonnet   = "claude-sonnet-4"
	pro      = "gemini-2.5-pro"
	flash    = "gemini-2.5-flash"
	deepseek = "deepseek-v3.1-terminus"
	qwen     = "qwen3-235b-a22b-2507"
)

var poolNames = []string{gpt5, sonnet, pro, flash, deepseek, qwen}

// fixedPredictor scores every prompt the same. The calibration of each test
// model has a single block, so p and lcb come straight from the spec.
type fixedPredictor float64

func (f fixedPredictor) Raw(*policy.Features, *[8]float64) float64 { return float64(f) }

// modelSpec describes one synthetic pool model.
type modelSpec struct {
	p, lcb  float64 // calibrated success rate and its lower bound
	eot     float64 // expected output tokens
	in, out float64 // USD per 1M tokens
}

// world is a synthetic policy plus the gateway state around it. With 400
// prompt bytes (100 tokens) the expected costs ascend as qwen, deepseek,
// flash, sonnet, pro, gpt-5, which differs from the pool order on purpose.
type world struct {
	specs     map[string]modelSpec
	lambda    map[string]float64 // by profile name
	strongest string
	models    []config.Model
	signals   router.Signals
	now       time.Time
}

func newWorld() *world {
	w := &world{
		specs: map[string]modelSpec{
			gpt5:     {p: 0.97, lcb: 0.95, eot: 100, in: 5.0, out: 10.0},
			sonnet:   {p: 0.75, lcb: 0.70, eot: 100, in: 3.0, out: 6.0},
			pro:      {p: 0.92, lcb: 0.90, eot: 100, in: 4.0, out: 8.0},
			flash:    {p: 0.65, lcb: 0.60, eot: 100, in: 0.5, out: 1.0},
			deepseek: {p: 0.55, lcb: 0.50, eot: 100, in: 0.2, out: 0.4},
			qwen:     {p: 0.40, lcb: 0.30, eot: 100, in: 0.1, out: 0.2},
		},
		lambda:    map[string]float64{"strict": 0.8, "balanced": 0.6, "economy": 0.4},
		strongest: pro,
		now:       time.Date(2026, 10, 26, 12, 0, 0, 0, time.UTC),
	}
	for _, name := range poolNames {
		w.models = append(w.models, config.Model{Name: name, PolicyName: name, Strongest: name == pro})
	}
	return w
}

// model returns the configured model with the given name for editing.
func (w *world) model(name string) *config.Model {
	for i := range w.models {
		if w.models[i].Name == name {
			return &w.models[i]
		}
	}
	panic("no configured model " + name)
}

func (w *world) policy(version string) *policy.Policy {
	p := &policy.Policy{
		Version:       version,
		PredictorType: "stub",
		Strongest:     w.strongest,
		Cheapest:      qwen,
		Profiles:      map[string]policy.Profile{},
		Rules: policy.StaticRules{
			FamilyKeywords: map[string][]string{"qa": {"who"}, "reasoning": {"prove"}, "code": {"python"}},
			FamilyModel:    map[string]string{"qa": qwen, "reasoning": gpt5, "code": sonnet},
			FamilyOrder:    []string{"qa", "reasoning", "code"},
			DefaultFamily:  "qa",
		},
	}
	for i := range p.DenseStd {
		p.DenseStd[i] = 1
	}
	for name, l := range w.lambda {
		p.Profiles[name] = policy.Profile{Name: name, Lambda: l}
	}
	for _, name := range poolNames {
		s := w.specs[name]
		p.Models = append(p.Models, policy.Model{
			Name:      name,
			Predictor: fixedPredictor(0.5),
			Calibration: policy.Calibration{Blocks: []policy.Block{
				{Lo: 0, Hi: 1, N: 100, K: int(math.Round(s.p * 100)), P: s.p, LCB: s.lcb},
			}},
			ExpectedOutputTokens: s.eot,
		})
	}
	return p
}

func (w *world) prices(t *testing.T) map[string]accounting.Price {
	t.Helper()
	prices := make(map[string]accounting.Price, len(w.specs))
	for name, s := range w.specs {
		p, err := accounting.NewPrice(name, s.in, s.out)
		if err != nil {
			t.Fatalf("NewPrice(%s): %v", name, err)
		}
		prices[name] = p
	}
	return prices
}

// registry returns a registry with the world's policy added and activated.
func (w *world) registry(t *testing.T, version string) *policy.Registry {
	t.Helper()
	reg := policy.NewRegistry(poolNames, nil)
	pol := w.policy(version)
	if err := reg.Add(pol); err != nil {
		t.Fatalf("Add: %v", err)
	}
	if _, err := reg.Activate(pol.Version, "test"); err != nil {
		t.Fatalf("Activate: %v", err)
	}
	return reg
}

func (w *world) engineOver(t *testing.T, reg *policy.Registry) *router.Engine {
	t.Helper()
	return router.NewEngine(router.EngineOptions{
		Models:   w.models,
		Prices:   w.prices(t),
		Policies: reg,
		Signals:  w.signals,
		NewID:    func() string { return "dec_test" },
		Now:      func() time.Time { return w.now },
	})
}

func (w *world) engine(t *testing.T) *router.Engine {
	t.Helper()
	return w.engineOver(t, w.registry(t, "pv-1"))
}

// engineNoPolicy returns an engine whose registry has no active policy.
func (w *world) engineNoPolicy(t *testing.T) *router.Engine {
	t.Helper()
	return w.engineOver(t, policy.NewRegistry(poolNames, nil))
}

// autoReq is an auto-routed request of 400 prompt bytes (100 tokens) that
// reserves 100 output tokens. The text is short on purpose: the stub
// predictors ignore it and the engine trusts PromptBytes for sizes.
func autoReq(strategy, profile string) *router.Request {
	return &router.Request{
		Tenant:      &config.Tenant{Name: "tenant-a"},
		Model:       config.AutoModel,
		Strategy:    strategy,
		Profile:     profile,
		Text:        "hello world",
		PromptBytes: 400,
		MaxTokens:   100,
	}
}

func decideWith(e *router.Engine, req *router.Request) (*router.Decision, error) {
	return e.Decide(context.Background(), req)
}

func mustDecide(t *testing.T, e *router.Engine, req *router.Request) *router.Decision {
	t.Helper()
	d, err := decideWith(e, req)
	if err != nil {
		t.Fatalf("Decide() error: %v", err)
	}
	return d
}

func candidate(t *testing.T, d *router.Decision, model string) router.Candidate {
	t.Helper()
	for _, c := range d.Candidates {
		if c.Model == model {
			return c
		}
	}
	t.Fatalf("decision has no candidate %q: %+v", model, d.Candidates)
	return router.Candidate{}
}

func candidateNames(d *router.Decision) []string {
	names := make([]string, len(d.Candidates))
	for i, c := range d.Candidates {
		names[i] = c.Model
	}
	return names
}

func wantPick(t *testing.T, d *router.Decision, chosen, reason string) {
	t.Helper()
	if d.Chosen != chosen || d.Reason != reason {
		t.Fatalf("chosen = %q (%s), want %q (%s)", d.Chosen, d.Reason, chosen, reason)
	}
}

func wantFallbacks(t *testing.T, d *router.Decision, want ...string) {
	t.Helper()
	if want == nil {
		want = []string{}
	}
	if d.Fallbacks == nil || !slices.Equal(d.Fallbacks, want) {
		t.Fatalf("fallbacks = %#v, want %#v", d.Fallbacks, want)
	}
}

func TestEngineQualityCheapestPassing(t *testing.T) {
	w := newWorld()
	w.lambda["economy"] = 0
	d := mustDecide(t, w.engine(t), autoReq("quality", "economy"))

	wantPick(t, d, qwen, router.ReasonCheapestPassing)
	if d.ID != "dec_test" || d.Tenant != "tenant-a" || !d.CreatedAt.Equal(w.now) {
		t.Fatalf("id/tenant/created_at = %q/%q/%v", d.ID, d.Tenant, d.CreatedAt)
	}
	if d.Strategy != router.StrategyQuality || d.Profile != "economy" || d.PolicyVersion != "pv-1" {
		t.Fatalf("strategy/profile/version = %q/%q/%q", d.Strategy, d.Profile, d.PolicyVersion)
	}
	if d.Lambda == nil || *d.Lambda != 0 {
		t.Fatalf("lambda = %v, want 0", d.Lambda)
	}
	if !d.GuaranteeApplies {
		t.Fatal("GuaranteeApplies = false, want true")
	}
	if got := candidateNames(d); !slices.Equal(got, poolNames) {
		t.Fatalf("candidates = %v, want pool order %v", got, poolNames)
	}
	for _, c := range d.Candidates {
		s := w.specs[c.Model]
		if !c.Eligible || c.IneligibleReasons == nil || len(c.IneligibleReasons) != 0 {
			t.Errorf("%s: eligible = %v, reasons = %#v", c.Model, c.Eligible, c.IneligibleReasons)
		}
		if c.Raw == nil || *c.Raw != 0.5 || c.P == nil || *c.P != s.p || c.LCB == nil || *c.LCB != s.lcb {
			t.Errorf("%s: raw/p/lcb = %v/%v/%v", c.Model, c.Raw, c.P, c.LCB)
		}
		if c.BlockN == nil || *c.BlockN != 100 || c.BlockK == nil || *c.BlockK != int(math.Round(s.p*100)) {
			t.Errorf("%s: block n/k = %v/%v", c.Model, c.BlockN, c.BlockK)
		}
		if c.PredictedP95MS != nil {
			t.Errorf("%s: predicted_p95_ms = %v, want null without a latency signal", c.Model, *c.PredictedP95MS)
		}
	}
	// 400 bytes are 100 tokens: (100 * 0.1 + 100 * 0.2) / 1e6.
	if got := candidate(t, d, qwen).ExpectedCostUSD; got == nil || math.Abs(*got-3e-5) > 1e-15 {
		t.Fatalf("qwen expected cost = %v, want 3e-5", got)
	}
}

func TestEngineQualityProfiles(t *testing.T) {
	// Lambdas are strict 0.8, balanced 0.6, economy 0.4. A model passes when
	// lcb >= lambda, so flash (lcb 0.60) passes balanced exactly at the edge.
	cases := []struct {
		profile string
		chosen  string
	}{
		{"strict", pro},       // sonnet 0.70 fails; pro 0.90 is cheaper than gpt-5
		{"balanced", flash},   // qwen 0.30 and deepseek 0.50 fail; flash 0.60 == lambda
		{"economy", deepseek}, // qwen 0.30 fails; deepseek 0.50 passes
	}
	for _, tc := range cases {
		t.Run(tc.profile, func(t *testing.T) {
			d := mustDecide(t, newWorld().engine(t), autoReq("quality", tc.profile))
			wantPick(t, d, tc.chosen, router.ReasonCheapestPassing)
			want := newWorld().lambda[tc.profile]
			if d.Lambda == nil || *d.Lambda != want {
				t.Fatalf("lambda = %v, want %v", d.Lambda, want)
			}
		})
	}
}

func TestEngineSentinelLambdaEscalates(t *testing.T) {
	w := newWorld()
	w.lambda["strict"] = 1.01
	d := mustDecide(t, w.engine(t), autoReq("quality", "strict"))

	wantPick(t, d, pro, router.ReasonEscalated)
	if d.Lambda == nil || *d.Lambda != 1.01 {
		t.Fatalf("lambda = %v, want 1.01", d.Lambda)
	}
	if !d.GuaranteeApplies {
		t.Fatal("GuaranteeApplies = false, want true when every model is eligible")
	}
	// Nothing passed, so the fallbacks are every other model by descending p.
	wantFallbacks(t, d, gpt5, sonnet, flash, deepseek, qwen)
}

func TestEngineEscalationUsesHighestPWhenStrongestIneligible(t *testing.T) {
	w := newWorld()
	w.lambda["strict"] = 1.01
	w.signals.BreakerOpen = func(m string) bool { return m == pro }
	d := mustDecide(t, w.engine(t), autoReq("quality", "strict"))

	wantPick(t, d, gpt5, router.ReasonEscalated)
	if d.GuaranteeApplies {
		t.Fatal("GuaranteeApplies = true with the strongest model ineligible")
	}
	wantFallbacks(t, d, sonnet, flash, deepseek, qwen)
}

func TestEngineCostTiesBreakByPoolOrder(t *testing.T) {
	cheap := func(w *world, names ...string) {
		for _, n := range names {
			w.specs[n] = modelSpec{p: w.specs[n].p, lcb: w.specs[n].lcb, eot: 50, in: 0.01, out: 0.02}
		}
	}

	t.Run("gpt-5 and gemini-2.5-pro tie", func(t *testing.T) {
		w := newWorld()
		w.lambda["economy"] = 0
		cheap(w, gpt5, pro)
		d := mustDecide(t, w.engine(t), autoReq("quality", "economy"))

		a, b := candidate(t, d, gpt5), candidate(t, d, pro)
		if a.ExpectedCostUSD == nil || b.ExpectedCostUSD == nil || *a.ExpectedCostUSD != *b.ExpectedCostUSD {
			t.Fatalf("costs do not tie: %v vs %v", a.ExpectedCostUSD, b.ExpectedCostUSD)
		}
		wantPick(t, d, gpt5, router.ReasonCheapestPassing)
		if len(d.Fallbacks) == 0 || d.Fallbacks[0] != pro {
			t.Fatalf("fallbacks = %v, want %s first", d.Fallbacks, pro)
		}
	})

	t.Run("six way tie keeps pool order", func(t *testing.T) {
		w := newWorld()
		w.lambda["economy"] = 0
		cheap(w, poolNames...)
		d := mustDecide(t, w.engine(t), autoReq("quality", "economy"))

		wantPick(t, d, gpt5, router.ReasonCheapestPassing)
		wantFallbacks(t, d, sonnet, pro, flash, deepseek, qwen)
	})

	t.Run("cheapest strategy tie", func(t *testing.T) {
		w := newWorld()
		cheap(w, pro, flash)
		d := mustDecide(t, w.engine(t), autoReq("cheapest", "balanced"))
		wantPick(t, d, pro, router.ReasonCheapest)
	})
}

func TestEngineIneligibleReasonsOrdered(t *testing.T) {
	t.Run("breaker queue latency", func(t *testing.T) {
		w := newWorld()
		w.signals.BreakerOpen = func(m string) bool { return m == flash }
		w.signals.QueueFull = func(m string) bool { return m == flash }
		w.signals.PredictedP95MS = func(m string, _ float64) float64 {
			if m == flash {
				return 900
			}
			return 100
		}
		req := autoReq("quality", "balanced")
		req.Deadline = w.now.Add(500 * time.Millisecond)
		d := mustDecide(t, w.engine(t), req)

		c := candidate(t, d, flash)
		want := []string{router.IneligibleBreakerOpen, router.IneligibleQueueFull, router.IneligibleLatencySLO}
		if c.Eligible || !slices.Equal(c.IneligibleReasons, want) {
			t.Fatalf("flash eligible = %v, reasons = %v, want %v", c.Eligible, c.IneligibleReasons, want)
		}
		if d.Chosen == flash {
			t.Fatal("an ineligible model was chosen")
		}
	})

	t.Run("all five", func(t *testing.T) {
		w := newWorld()
		w.model(flash).ContextWindow = 10
		w.signals.BreakerOpen = func(m string) bool { return m == flash }
		w.signals.QueueFull = func(m string) bool { return m == flash }
		w.signals.PredictedP95MS = func(m string, _ float64) float64 {
			if m == flash {
				return 900
			}
			return 100
		}
		w.signals.BudgetRemainingNanos = func(context.Context, string) (int64, bool, error) { return 1, true, nil }
		req := autoReq("quality", "balanced")
		req.Deadline = w.now.Add(500 * time.Millisecond)
		d, err := decideWith(w.engine(t), req)
		if !errors.Is(err, router.ErrNoEligibleModel) {
			t.Fatalf("err = %v, want ErrNoEligibleModel (a 1 nano budget fits nothing)", err)
		}
		want := []string{
			router.IneligibleBreakerOpen, router.IneligibleQueueFull, router.IneligibleContextWindow,
			router.IneligibleLatencySLO, router.IneligibleBudget,
		}
		if got := candidate(t, d, flash).IneligibleReasons; !slices.Equal(got, want) {
			t.Fatalf("reasons = %v, want %v", got, want)
		}
	})

	t.Run("constants", func(t *testing.T) {
		got := []string{
			router.IneligibleBreakerOpen, router.IneligibleQueueFull, router.IneligibleContextWindow,
			router.IneligibleLatencySLO, router.IneligibleBudget,
		}
		want := []string{"breaker_open", "queue_full", "context_window", "latency_slo", "budget"}
		if !slices.Equal(got, want) {
			t.Fatalf("ineligibility reasons = %v, want %v", got, want)
		}
		reasons := []string{
			router.ReasonCheapestPassing, router.ReasonEscalated, router.ReasonStrongest, router.ReasonCheapest,
			router.ReasonStaticFamily, router.ReasonLatency, router.ReasonPickIneligible, router.ReasonPinned,
		}
		wantReasons := []string{
			"cheapest_passing", "escalated_no_model_passed", "strongest", "cheapest",
			"static_family", "lowest_latency", "strategy_pick_ineligible", "pinned",
		}
		if !slices.Equal(reasons, wantReasons) {
			t.Fatalf("decision reasons = %v, want %v", reasons, wantReasons)
		}
	})
}

func TestEngineGuaranteeFalseWhenModelIneligible(t *testing.T) {
	w := newWorld()
	w.signals.BreakerOpen = func(m string) bool { return m == qwen }
	d := mustDecide(t, w.engine(t), autoReq("quality", "balanced"))

	wantPick(t, d, flash, router.ReasonCheapestPassing)
	if d.GuaranteeApplies {
		t.Fatal("GuaranteeApplies = true with one pool model ineligible")
	}
}

func TestEngineGuaranteeFalseWhenTenantRestricts(t *testing.T) {
	w := newWorld()
	req := autoReq("quality", "balanced")
	req.Tenant = &config.Tenant{Name: "tenant-a", AllowedModels: []string{sonnet, pro, flash, deepseek, qwen}}
	d := mustDecide(t, w.engine(t), req)

	if got, want := candidateNames(d), []string{sonnet, pro, flash, deepseek, qwen}; !slices.Equal(got, want) {
		t.Fatalf("candidates = %v, want %v", got, want)
	}
	if d.GuaranteeApplies {
		t.Fatal("GuaranteeApplies = true although the tenant excludes a pool model")
	}
}

func TestEngineGuaranteeFalseWhenModelNotConfigured(t *testing.T) {
	w := newWorld()
	w.models = slices.DeleteFunc(w.models, func(m config.Model) bool { return m.Name == deepseek })
	d := mustDecide(t, w.engine(t), autoReq("quality", "balanced"))

	if len(d.Candidates) != 5 {
		t.Fatalf("candidates = %v, want 5", candidateNames(d))
	}
	if d.GuaranteeApplies {
		t.Fatal("GuaranteeApplies = true with a pool model that is not configured")
	}
}

func TestEngineNoEligibleReturnsDecisionAndError(t *testing.T) {
	w := newWorld()
	w.signals.BreakerOpen = func(string) bool { return true }
	d, err := decideWith(w.engine(t), autoReq("quality", "balanced"))

	if !errors.Is(err, router.ErrNoEligibleModel) || errors.Is(err, router.ErrOverloaded) {
		t.Fatalf("err = %v, want ErrNoEligibleModel only", err)
	}
	if d == nil {
		t.Fatal("decision is nil, want the candidates with their reasons")
	}
	if d.Chosen != "" || d.GuaranteeApplies || d.Strategy != router.StrategyQuality || d.PolicyVersion != "pv-1" {
		t.Fatalf("decision = %+v", d)
	}
	if len(d.Candidates) != 6 || d.Fallbacks == nil || len(d.Fallbacks) != 0 {
		t.Fatalf("candidates = %d, fallbacks = %#v", len(d.Candidates), d.Fallbacks)
	}
	for _, c := range d.Candidates {
		if c.Eligible || !slices.Equal(c.IneligibleReasons, []string{router.IneligibleBreakerOpen}) {
			t.Errorf("%s: eligible = %v, reasons = %v", c.Model, c.Eligible, c.IneligibleReasons)
		}
	}
}

func TestEngineAllQueuesFullIsOverloaded(t *testing.T) {
	t.Run("every model queue full", func(t *testing.T) {
		w := newWorld()
		w.signals.QueueFull = func(string) bool { return true }
		d, err := decideWith(w.engine(t), autoReq("quality", "balanced"))
		if !errors.Is(err, router.ErrOverloaded) {
			t.Fatalf("err = %v, want ErrOverloaded", err)
		}
		if errors.Is(err, router.ErrNoEligibleModel) {
			t.Fatalf("err = %v also matches ErrNoEligibleModel", err)
		}
		if d == nil || len(d.Candidates) != 6 {
			t.Fatalf("decision = %+v", d)
		}
	})

	t.Run("queue full next to another reason", func(t *testing.T) {
		w := newWorld()
		w.signals.QueueFull = func(string) bool { return true }
		w.signals.BreakerOpen = func(m string) bool { return m == gpt5 }
		_, err := decideWith(w.engine(t), autoReq("quality", "balanced"))
		if !errors.Is(err, router.ErrOverloaded) {
			t.Fatalf("err = %v, want ErrOverloaded when every list contains queue_full", err)
		}
	})

	t.Run("one model without queue full", func(t *testing.T) {
		w := newWorld()
		w.signals.BreakerOpen = func(string) bool { return true }
		w.signals.QueueFull = func(m string) bool { return m != qwen }
		_, err := decideWith(w.engine(t), autoReq("quality", "balanced"))
		if !errors.Is(err, router.ErrNoEligibleModel) || errors.Is(err, router.ErrOverloaded) {
			t.Fatalf("err = %v, want ErrNoEligibleModel", err)
		}
	})

	t.Run("no candidates is not overloaded", func(t *testing.T) {
		w := newWorld()
		req := autoReq("quality", "balanced")
		req.Tenant = &config.Tenant{Name: "tenant-a", AllowedModels: []string{"not-a-pool-model"}}
		d, err := decideWith(w.engine(t), req)
		if !errors.Is(err, router.ErrNoEligibleModel) || errors.Is(err, router.ErrOverloaded) {
			t.Fatalf("err = %v, want ErrNoEligibleModel", err)
		}
		if d == nil || len(d.Candidates) != 0 {
			t.Fatalf("decision = %+v", d)
		}
	})
}

func TestEngineFallbackOrder(t *testing.T) {
	t.Run("passing by cost then rest by descending p", func(t *testing.T) {
		w := newWorld()
		// balanced lambda 0.6: gpt-5, sonnet, pro and flash pass; deepseek and
		// qwen do not. Passing by cost: flash, sonnet, pro, gpt-5.
		d := mustDecide(t, w.engine(t), autoReq("quality", "balanced"))
		wantPick(t, d, flash, router.ReasonCheapestPassing)
		wantFallbacks(t, d, sonnet, pro, gpt5, deepseek, qwen)
	})

	t.Run("ineligible models are left out", func(t *testing.T) {
		w := newWorld()
		w.signals.QueueFull = func(m string) bool { return m == sonnet || m == deepseek }
		d := mustDecide(t, w.engine(t), autoReq("quality", "balanced"))
		wantPick(t, d, flash, router.ReasonCheapestPassing)
		wantFallbacks(t, d, pro, gpt5, qwen)
	})

	t.Run("equal p keeps pool order", func(t *testing.T) {
		w := newWorld()
		w.lambda["strict"] = 1.01
		s := w.specs[flash]
		s.p = w.specs[sonnet].p
		w.specs[flash] = s
		d := mustDecide(t, w.engine(t), autoReq("quality", "strict"))
		wantPick(t, d, pro, router.ReasonEscalated)
		wantFallbacks(t, d, gpt5, sonnet, flash, deepseek, qwen)
	})
}

func TestEngineStrategies(t *testing.T) {
	w := newWorld()
	w.signals.PredictedP95MS = func(m string, _ float64) float64 {
		return map[string]float64{gpt5: 900, sonnet: 700, pro: 800, flash: 150, deepseek: 300, qwen: 200}[m]
	}
	e := w.engine(t)

	cases := []struct {
		strategy  string
		text      string
		chosen    string
		reason    string
		fallbacks []string // every other model by descending p
	}{
		{router.StrategyStrongest, "hello", pro, router.ReasonStrongest, []string{gpt5, sonnet, flash, deepseek, qwen}},
		{router.StrategyCheapest, "hello", qwen, router.ReasonCheapest, []string{gpt5, pro, sonnet, flash, deepseek}},
		{router.StrategyStatic, "write python code", sonnet, router.ReasonStaticFamily, []string{gpt5, pro, flash, deepseek, qwen}},
		{router.StrategyStatic, "prove it", gpt5, router.ReasonStaticFamily, []string{pro, sonnet, flash, deepseek, qwen}},
		{router.StrategyStatic, "no keyword here", qwen, router.ReasonStaticFamily, []string{gpt5, pro, sonnet, flash, deepseek}},
		{router.StrategyLatency, "hello", flash, router.ReasonLatency, []string{gpt5, pro, sonnet, deepseek, qwen}},
	}
	for _, tc := range cases {
		t.Run(tc.strategy+"/"+tc.text, func(t *testing.T) {
			req := autoReq(tc.strategy, "balanced")
			req.Text = tc.text
			d := mustDecide(t, e, req)
			wantPick(t, d, tc.chosen, tc.reason)
			wantFallbacks(t, d, tc.fallbacks...)
			if d.Strategy != tc.strategy || d.PolicyVersion != "pv-1" {
				t.Fatalf("strategy/version = %q/%q", d.Strategy, d.PolicyVersion)
			}
			if d.Lambda != nil || d.GuaranteeApplies {
				t.Fatalf("lambda = %v, guarantee = %v; only quality sets them", d.Lambda, d.GuaranteeApplies)
			}
			if len(d.Candidates) != 6 {
				t.Fatalf("candidates = %v", candidateNames(d))
			}
		})
	}
}

func TestEngineLatencyStrategy(t *testing.T) {
	t.Run("lowest predicted p95 with pool order ties", func(t *testing.T) {
		w := newWorld()
		w.signals.PredictedP95MS = func(m string, _ float64) float64 {
			if m == deepseek || m == qwen {
				return 100
			}
			return 500
		}
		d := mustDecide(t, w.engine(t), autoReq("latency", "balanced"))
		wantPick(t, d, deepseek, router.ReasonLatency)
	})

	t.Run("unavailable without a latency signal", func(t *testing.T) {
		d, err := decideWith(newWorld().engine(t), autoReq("latency", "balanced"))
		if !errors.Is(err, router.ErrStrategyUnavailable) {
			t.Fatalf("err = %v, want ErrStrategyUnavailable", err)
		}
		if d != nil {
			t.Fatalf("decision = %+v, want nil", d)
		}
		if !strings.Contains(err.Error(), `"latency"`) {
			t.Fatalf("message %q lacks the quoted strategy", err.Error())
		}
	})
}

func TestEnginePickIneligibleFallsBack(t *testing.T) {
	t.Run("strongest", func(t *testing.T) {
		w := newWorld()
		s := w.specs[sonnet]
		s.p = w.specs[gpt5].p // gpt-5 and sonnet tie for the highest p
		w.specs[sonnet] = s
		w.signals.BreakerOpen = func(m string) bool { return m == pro }
		d := mustDecide(t, w.engine(t), autoReq("strongest", "balanced"))

		wantPick(t, d, gpt5, router.ReasonPickIneligible)
		if d.Strategy != router.StrategyStrongest {
			t.Fatalf("strategy = %q, want it kept", d.Strategy)
		}
		wantFallbacks(t, d, sonnet, flash, deepseek, qwen)
	})

	t.Run("cheapest", func(t *testing.T) {
		w := newWorld()
		w.signals.QueueFull = func(m string) bool { return m == qwen }
		d := mustDecide(t, w.engine(t), autoReq("cheapest", "balanced"))
		wantPick(t, d, gpt5, router.ReasonPickIneligible)
	})

	t.Run("static model not allowed for the tenant", func(t *testing.T) {
		w := newWorld()
		req := autoReq("static", "balanced")
		req.Text = "write python code" // static maps this to sonnet
		req.Tenant = &config.Tenant{Name: "tenant-a", AllowedModels: []string{gpt5, pro}}
		d := mustDecide(t, w.engine(t), req)
		wantPick(t, d, gpt5, router.ReasonPickIneligible)
	})

	t.Run("latency", func(t *testing.T) {
		w := newWorld()
		w.signals.PredictedP95MS = func(m string, _ float64) float64 {
			if m == flash {
				return 10
			}
			return 100
		}
		w.signals.BreakerOpen = func(m string) bool { return m == flash }
		d := mustDecide(t, w.engine(t), autoReq("latency", "balanced"))
		wantPick(t, d, gpt5, router.ReasonPickIneligible)
	})

	t.Run("nothing eligible", func(t *testing.T) {
		w := newWorld()
		w.signals.BreakerOpen = func(string) bool { return true }
		d, err := decideWith(w.engine(t), autoReq("strongest", "balanced"))
		if !errors.Is(err, router.ErrNoEligibleModel) || d == nil {
			t.Fatalf("decision = %v, err = %v", d, err)
		}
	})
}

func TestEnginePinned(t *testing.T) {
	w := newWorld()
	e := w.engine(t)

	t.Run("works with a policy active", func(t *testing.T) {
		req := autoReq("", "strict")
		req.Model = flash
		d := mustDecide(t, e, req)
		wantPick(t, d, flash, router.ReasonPinned)
		if d.Strategy != router.StrategyPinned || len(d.Candidates) != 1 || !d.Candidates[0].Eligible {
			t.Fatalf("decision = %+v", d)
		}
		if d.Fallbacks == nil || len(d.Fallbacks) != 0 || d.Attempts == nil {
			t.Fatalf("fallbacks/attempts = %#v/%#v", d.Fallbacks, d.Attempts)
		}
	})

	t.Run("unknown", func(t *testing.T) {
		req := autoReq("", "strict")
		req.Model = "nope"
		if _, err := decideWith(e, req); !errors.Is(err, router.ErrUnknownModel) {
			t.Fatalf("err = %v, want ErrUnknownModel", err)
		}
	})

	t.Run("not allowed", func(t *testing.T) {
		req := autoReq("", "strict")
		req.Model = flash
		req.Tenant = &config.Tenant{Name: "tenant-a", AllowedModels: []string{pro}}
		if _, err := decideWith(e, req); !errors.Is(err, router.ErrModelNotAllowed) {
			t.Fatalf("err = %v, want ErrModelNotAllowed", err)
		}
	})
}

func TestEnginePinnedIneligible(t *testing.T) {
	w := newWorld()
	w.signals.BreakerOpen = func(m string) bool { return m == flash }
	req := autoReq("", "strict")
	req.Model = flash
	d, err := decideWith(w.engine(t), req)

	if !errors.Is(err, router.ErrNoEligibleModel) {
		t.Fatalf("err = %v, want ErrNoEligibleModel", err)
	}
	if !strings.Contains(err.Error(), router.IneligibleBreakerOpen) {
		t.Fatalf("message %q lacks the reason", err.Error())
	}
	if d == nil || len(d.Candidates) != 1 || d.Chosen != "" {
		t.Fatalf("decision = %+v", d)
	}
	c := d.Candidates[0]
	if c.Model != flash || c.Eligible || !slices.Equal(c.IneligibleReasons, []string{router.IneligibleBreakerOpen}) {
		t.Fatalf("candidate = %+v", c)
	}
	if d.Strategy != router.StrategyPinned {
		t.Fatalf("strategy = %q", d.Strategy)
	}
}

// A pinned request has one candidate, so a full queue for it is the case spec
// 10 calls overloaded: the model is fine, the gateway is busy. It is not "no
// eligible model", which tells the client to pick another model.
func TestEnginePinnedQueueFullIsOverloaded(t *testing.T) {
	pinned := func() *router.Request {
		req := autoReq("", "strict")
		req.Model = flash
		return req
	}

	t.Run("queue full is the only reason", func(t *testing.T) {
		w := newWorld()
		w.signals.QueueFull = func(m string) bool { return m == flash }
		d, err := decideWith(w.engine(t), pinned())

		if !errors.Is(err, router.ErrOverloaded) {
			t.Fatalf("err = %v, want ErrOverloaded", err)
		}
		if errors.Is(err, router.ErrNoEligibleModel) {
			t.Fatalf("err = %v also matches ErrNoEligibleModel", err)
		}
		for _, want := range []string{flash, router.IneligibleQueueFull} {
			if !strings.Contains(err.Error(), want) {
				t.Errorf("message %q lacks %q", err.Error(), want)
			}
		}
		if d == nil || d.Chosen != "" || d.Strategy != router.StrategyPinned || len(d.Candidates) != 1 {
			t.Fatalf("decision = %+v, want the pinned candidate with its reason and no choice", d)
		}
		c := d.Candidates[0]
		if c.Model != flash || c.Eligible || !slices.Equal(c.IneligibleReasons, []string{router.IneligibleQueueFull}) {
			t.Fatalf("candidate = %+v", c)
		}
	})

	t.Run("without an active policy", func(t *testing.T) {
		w := newWorld()
		w.signals.QueueFull = func(m string) bool { return m == flash }
		_, err := decideWith(w.engineNoPolicy(t), pinned())
		if !errors.Is(err, router.ErrOverloaded) {
			t.Fatalf("err = %v, want ErrOverloaded", err)
		}
	})

	t.Run("another reason as well stays no eligible model", func(t *testing.T) {
		w := newWorld()
		w.signals.QueueFull = func(m string) bool { return m == flash }
		w.signals.BreakerOpen = func(m string) bool { return m == flash }
		_, err := decideWith(w.engine(t), pinned())
		if !errors.Is(err, router.ErrNoEligibleModel) || errors.Is(err, router.ErrOverloaded) {
			t.Fatalf("err = %v, want ErrNoEligibleModel only", err)
		}
	})

	t.Run("another model's full queue does not matter", func(t *testing.T) {
		w := newWorld()
		w.signals.QueueFull = func(m string) bool { return m == qwen }
		d := mustDecide(t, w.engine(t), pinned())
		wantPick(t, d, flash, router.ReasonPinned)
	})
}

func TestEnginePinnedChecksEveryReason(t *testing.T) {
	w := newWorld()
	w.model(flash).ContextWindow = 10
	w.signals.QueueFull = func(m string) bool { return m == flash }
	w.signals.PredictedP95MS = func(string, float64) float64 { return 900 }
	w.signals.BudgetRemainingNanos = func(context.Context, string) (int64, bool, error) { return 1, true, nil }
	req := autoReq("", "strict")
	req.Model = flash
	req.Deadline = w.now.Add(500 * time.Millisecond)
	d, err := decideWith(w.engine(t), req)

	if !errors.Is(err, router.ErrNoEligibleModel) || d == nil {
		t.Fatalf("decision = %v, err = %v", d, err)
	}
	want := []string{
		router.IneligibleQueueFull, router.IneligibleContextWindow, router.IneligibleLatencySLO,
		router.IneligibleBudget,
	}
	if got := d.Candidates[0].IneligibleReasons; !slices.Equal(got, want) {
		t.Fatalf("reasons = %v, want %v", got, want)
	}
}

func TestEngineWithoutPolicyMatchesBasic(t *testing.T) {
	models := []config.Model{
		{Name: "m-fast"},
		{Name: "m-strong", Strongest: true},
		{Name: "m-other", Strongest: true},
	}
	engine := router.NewEngine(router.EngineOptions{
		Models:   models,
		Policies: policy.NewRegistry(nil, nil),
	})
	basic := router.NewBasic(models, router.BasicOptions{})

	tenants := map[string]*config.Tenant{
		"all":          {Name: "tenant-a"},
		"fast+other":   {Name: "tenant-a", AllowedModels: []string{"m-fast", "m-other"}},
		"fast only":    {Name: "tenant-a", AllowedModels: []string{"m-fast"}},
		"strong first": {Name: "tenant-a", AllowedModels: []string{"m-other", "m-strong"}},
	}
	modelNames := []string{config.AutoModel, "m-fast", "m-strong", "nope"}
	strategies := []string{"", router.StrategyStrongest, router.StrategyQuality, router.StrategyCheapest, router.StrategyStatic, router.StrategyLatency}

	for tname, ten := range tenants {
		for _, model := range modelNames {
			for _, strategy := range strategies {
				name := tname + "/" + model + "/" + strategy
				req := &router.Request{Tenant: ten, Model: model, Strategy: strategy, Profile: "balanced", MaxTokens: 10}
				want, wantErr := basic.Decide(context.Background(), req)
				got, gotErr := engine.Decide(context.Background(), req)

				for _, sentinel := range []error{
					router.ErrUnknownModel, router.ErrModelNotAllowed, router.ErrStrategyUnavailable, router.ErrNoEligibleModel,
				} {
					if errors.Is(wantErr, sentinel) != errors.Is(gotErr, sentinel) {
						t.Errorf("%s: err = %v, Basic returned %v", name, gotErr, wantErr)
					}
				}
				if (wantErr == nil) != (gotErr == nil) {
					t.Errorf("%s: err = %v, Basic returned %v", name, gotErr, wantErr)
					continue
				}
				if wantErr != nil {
					continue
				}
				if got.Chosen != want.Chosen || got.Strategy != want.Strategy || got.Reason != want.Reason {
					t.Errorf("%s: got %q/%s/%s, Basic %q/%s/%s", name,
						got.Chosen, got.Strategy, got.Reason, want.Chosen, want.Strategy, want.Reason)
				}
				if !reflect.DeepEqual(candidateNames(got), candidateNames(want)) {
					t.Errorf("%s: candidates = %v, Basic %v", name, candidateNames(got), candidateNames(want))
				}
				if got.PolicyVersion != "" || got.Lambda != nil || got.GuaranteeApplies {
					t.Errorf("%s: policy fields set without a policy: %+v", name, got)
				}
			}
		}
	}
}

func TestEngineWithoutPolicyEligibility(t *testing.T) {
	w := newWorld()
	w.model(pro).Strongest = false
	w.model(sonnet).Strongest = true
	w.model(gpt5).Strongest = true
	w.signals.BreakerOpen = func(m string) bool { return m == gpt5 }
	e := w.engineNoPolicy(t)

	t.Run("skips an ineligible strongest model", func(t *testing.T) {
		d := mustDecide(t, e, autoReq("", "balanced"))
		wantPick(t, d, sonnet, router.ReasonStrongest)
		if d.Strategy != router.StrategyStrongest || len(d.Candidates) != 6 {
			t.Fatalf("strategy = %q, candidates = %v", d.Strategy, candidateNames(d))
		}
		if c := candidate(t, d, gpt5); c.Eligible || !slices.Equal(c.IneligibleReasons, []string{router.IneligibleBreakerOpen}) {
			t.Fatalf("gpt-5 candidate = %+v", c)
		}
	})

	t.Run("no strongest eligible", func(t *testing.T) {
		w.signals.BreakerOpen = func(m string) bool { return m == gpt5 || m == sonnet }
		d, err := decideWith(w.engineNoPolicy(t), autoReq("", "balanced"))
		if !errors.Is(err, router.ErrNoEligibleModel) || d == nil {
			t.Fatalf("decision = %v, err = %v", d, err)
		}
	})

	t.Run("quality is unavailable", func(t *testing.T) {
		_, err := decideWith(e, autoReq("quality", "balanced"))
		if !errors.Is(err, router.ErrStrategyUnavailable) || !strings.Contains(err.Error(), `"quality"`) {
			t.Fatalf("err = %v", err)
		}
	})
}

func TestEngineNilRegistryBehavesLikeNoPolicy(t *testing.T) {
	e := router.NewEngine(router.EngineOptions{Models: []config.Model{{Name: "m", Strongest: true}}})
	req := &router.Request{Tenant: &config.Tenant{Name: "t"}, Model: config.AutoModel, Profile: "balanced"}
	d := mustDecide(t, e, req)
	wantPick(t, d, "m", router.ReasonStrongest)
}

func TestEngineDefaultStrategy(t *testing.T) {
	d := mustDecide(t, newWorld().engine(t), autoReq("", "balanced"))
	if d.Strategy != router.StrategyQuality || d.Lambda == nil {
		t.Fatalf("strategy = %q, lambda = %v, want quality", d.Strategy, d.Lambda)
	}
}

func TestEngineUnknownStrategy(t *testing.T) {
	for _, strategy := range []string{"bogus", router.StrategyPinned} {
		_, err := decideWith(newWorld().engine(t), autoReq(strategy, "balanced"))
		if !errors.Is(err, router.ErrStrategyUnavailable) {
			t.Errorf("strategy %q: err = %v, want ErrStrategyUnavailable", strategy, err)
		}
	}
}

func TestEngineUnknownProfile(t *testing.T) {
	d, err := decideWith(newWorld().engine(t), autoReq("quality", "nope"))
	if err == nil || d != nil {
		t.Fatalf("decision = %v, err = %v; want an error and no decision", d, err)
	}
	if !strings.Contains(err.Error(), `"nope"`) {
		t.Fatalf("message %q lacks the profile", err.Error())
	}
}

func TestEngineContextWindow(t *testing.T) {
	t.Run("too small", func(t *testing.T) {
		w := newWorld()
		w.model(flash).ContextWindow = 10
		d := mustDecide(t, w.engine(t), autoReq("quality", "balanced"))
		c := candidate(t, d, flash)
		if c.Eligible || !slices.Equal(c.IneligibleReasons, []string{router.IneligibleContextWindow}) {
			t.Fatalf("flash candidate = %+v", c)
		}
		wantPick(t, d, sonnet, router.ReasonCheapestPassing)
	})

	t.Run("exact fit is eligible", func(t *testing.T) {
		// ceil(400/4) + 100 output tokens = 200.
		w := newWorld()
		w.model(flash).ContextWindow = 200
		w.model(deepseek).ContextWindow = 199
		d := mustDecide(t, w.engine(t), autoReq("quality", "balanced"))
		if c := candidate(t, d, flash); !c.Eligible {
			t.Fatalf("flash with window 200 is ineligible: %+v", c)
		}
		if c := candidate(t, d, deepseek); c.Eligible {
			t.Fatalf("deepseek with window 199 is eligible: %+v", c)
		}
	})

	t.Run("rounds prompt tokens up", func(t *testing.T) {
		// 401 bytes are 101 tokens: 101 + 100 = 201.
		w := newWorld()
		w.model(flash).ContextWindow = 200
		req := autoReq("quality", "balanced")
		req.PromptBytes = 401
		d := mustDecide(t, w.engine(t), req)
		if c := candidate(t, d, flash); c.Eligible {
			t.Fatalf("flash eligible with 101 + 100 > 200: %+v", c)
		}
	})

	t.Run("unknown window is unchecked", func(t *testing.T) {
		req := autoReq("quality", "balanced")
		req.PromptBytes = 1 << 30
		d := mustDecide(t, newWorld().engine(t), req)
		if !candidate(t, d, flash).Eligible {
			t.Fatal("a model with ContextWindow 0 was checked")
		}
	})
}

func TestEngineLatencySLO(t *testing.T) {
	signal := func(w *world, seen *[]float64) {
		w.signals.PredictedP95MS = func(m string, eot float64) float64 {
			if seen != nil {
				*seen = append(*seen, eot)
			}
			return map[string]float64{gpt5: 900, sonnet: 700, pro: 800, flash: 150, deepseek: 300, qwen: 200}[m]
		}
	}

	t.Run("models slower than the remaining deadline", func(t *testing.T) {
		w := newWorld()
		var seen []float64
		signal(w, &seen)
		req := autoReq("quality", "balanced")
		req.Deadline = w.now.Add(500 * time.Millisecond)
		d := mustDecide(t, w.engine(t), req)

		for _, m := range []string{gpt5, sonnet, pro} {
			if c := candidate(t, d, m); c.Eligible || !slices.Equal(c.IneligibleReasons, []string{router.IneligibleLatencySLO}) {
				t.Errorf("%s candidate = %+v", m, c)
			}
		}
		for _, m := range []string{flash, deepseek, qwen} {
			if !candidate(t, d, m).Eligible {
				t.Errorf("%s is ineligible", m)
			}
		}
		if c := candidate(t, d, gpt5); c.PredictedP95MS == nil || *c.PredictedP95MS != 900 {
			t.Errorf("gpt-5 predicted_p95_ms = %v, want 900", c.PredictedP95MS)
		}
		for _, eot := range seen {
			if eot != 100 {
				t.Fatalf("signal got expected output tokens %v, want 100", eot)
			}
		}
		wantPick(t, d, flash, router.ReasonCheapestPassing)
	})

	t.Run("equal to the remaining deadline passes", func(t *testing.T) {
		w := newWorld()
		signal(w, nil)
		req := autoReq("quality", "balanced")
		req.Deadline = w.now.Add(150 * time.Millisecond)
		if !candidate(t, mustDecide(t, w.engine(t), req), flash).Eligible {
			t.Fatal("flash (150 ms) was ineligible with exactly 150 ms left")
		}
		req.Deadline = w.now.Add(150*time.Millisecond - time.Microsecond)
		d, err := decideWith(w.engine(t), req)
		if !errors.Is(err, router.ErrNoEligibleModel) || d == nil {
			t.Fatalf("decision = %v, err = %v; every model is slower than 149.999 ms", d, err)
		}
		if candidate(t, d, flash).Eligible {
			t.Fatal("flash (150 ms) was eligible with 149.999 ms left")
		}
	})

	t.Run("fractional milliseconds", func(t *testing.T) {
		w := newWorld()
		w.signals.PredictedP95MS = func(string, float64) float64 { return 1.5 }
		req := autoReq("quality", "balanced")
		req.Deadline = w.now.Add(1500 * time.Microsecond)
		if !candidate(t, mustDecide(t, w.engine(t), req), flash).Eligible {
			t.Fatal("1.5 ms predicted was ineligible with 1.5 ms left")
		}
	})

	t.Run("zero deadline skips the check", func(t *testing.T) {
		w := newWorld()
		signal(w, nil)
		d := mustDecide(t, w.engine(t), autoReq("quality", "balanced"))
		for _, c := range d.Candidates {
			if !c.Eligible || c.PredictedP95MS == nil {
				t.Errorf("%s: eligible = %v, p95 = %v", c.Model, c.Eligible, c.PredictedP95MS)
			}
		}
	})

	t.Run("no signal leaves the prediction null", func(t *testing.T) {
		w := newWorld()
		req := autoReq("quality", "balanced")
		req.Deadline = w.now.Add(time.Millisecond)
		d := mustDecide(t, w.engine(t), req)
		for _, c := range d.Candidates {
			if !c.Eligible || c.PredictedP95MS != nil {
				t.Errorf("%s: eligible = %v, p95 = %v", c.Model, c.Eligible, c.PredictedP95MS)
			}
		}
	})
}

func TestEngineBudget(t *testing.T) {
	// Reservations at 400 prompt bytes and 100 output tokens, in nano-USD:
	// qwen 60,000; deepseek 120,000; flash 300,000; sonnet 1,800,000;
	// pro 2,400,000; gpt-5 3,000,000.
	budget := func(w *world, remaining int64, limited bool, calls *int, tenants *[]string) {
		w.signals.BudgetRemainingNanos = func(_ context.Context, tenant string) (int64, bool, error) {
			if calls != nil {
				*calls++
			}
			if tenants != nil {
				*tenants = append(*tenants, tenant)
			}
			return remaining, limited, nil
		}
	}

	t.Run("reservation above the remaining budget", func(t *testing.T) {
		w := newWorld()
		var calls int
		var tenants []string
		budget(w, 1_000_000, true, &calls, &tenants)
		d := mustDecide(t, w.engine(t), autoReq("quality", "strict"))

		for _, m := range []string{gpt5, sonnet, pro} {
			if c := candidate(t, d, m); c.Eligible || !slices.Equal(c.IneligibleReasons, []string{router.IneligibleBudget}) {
				t.Errorf("%s candidate = %+v", m, c)
			}
		}
		if calls != 1 || !slices.Equal(tenants, []string{"tenant-a"}) {
			t.Fatalf("budget signal calls = %d for %v, want one call for tenant-a", calls, tenants)
		}
		// strict needs lcb 0.8: only gpt-5 and pro pass and both are out of budget.
		wantPick(t, d, flash, router.ReasonEscalated)
	})

	t.Run("unlimited tenant", func(t *testing.T) {
		w := newWorld()
		budget(w, 0, false, nil, nil)
		d := mustDecide(t, w.engine(t), autoReq("quality", "balanced"))
		for _, c := range d.Candidates {
			if !c.Eligible {
				t.Errorf("%s ineligible for an unlimited tenant", c.Model)
			}
		}
	})

	t.Run("free model always fits", func(t *testing.T) {
		w := newWorld()
		w.specs[qwen] = modelSpec{p: 0.4, lcb: 0.3, eot: 100, in: 0, out: 0}
		budget(w, 0, true, nil, nil)
		d := mustDecide(t, w.engine(t), autoReq("cheapest", "balanced"))
		wantPick(t, d, qwen, router.ReasonCheapest)
	})

	t.Run("signal error marks no reason", func(t *testing.T) {
		w := newWorld()
		w.signals.BudgetRemainingNanos = func(context.Context, string) (int64, bool, error) {
			return 0, true, errors.New("budget backend down")
		}
		d := mustDecide(t, w.engine(t), autoReq("quality", "balanced"))
		for _, c := range d.Candidates {
			if !c.Eligible {
				t.Errorf("%s ineligible after a signal error: %v", c.Model, c.IneligibleReasons)
			}
		}
	})
}

func TestEngineBudgetExactFit(t *testing.T) {
	w := newWorld()
	w.signals.BudgetRemainingNanos = func(context.Context, string) (int64, bool, error) { return 60_000, true, nil }
	d, err := decideWith(w.engine(t), autoReq("cheapest", "balanced"))
	if err != nil {
		t.Fatal(err)
	}
	wantPick(t, d, qwen, router.ReasonCheapest)
	if c := candidate(t, d, deepseek); c.Eligible {
		t.Fatalf("deepseek needs 120,000 nano-USD but was eligible with 60,000: %+v", c)
	}
}

func TestEngineTextAndBytes(t *testing.T) {
	t.Run("expected cost follows PromptBytes", func(t *testing.T) {
		// qwen: in 0.1, out 0.2, 100 expected output tokens.
		cases := []struct {
			bytes int
			want  float64
		}{
			{1, 20.1e-6},   // 1 token
			{4, 20.1e-6},   // 1 token
			{5, 20.2e-6},   // 2 tokens
			{400, 3e-5},    // 100 tokens
			{4000, 1.2e-4}, // 1000 tokens
		}
		for _, tc := range cases {
			req := autoReq("quality", "balanced")
			req.PromptBytes = tc.bytes
			req.Text = "x"
			d := mustDecide(t, newWorld().engine(t), req)
			got := candidate(t, d, qwen).ExpectedCostUSD
			if got == nil || math.Abs(*got-tc.want) > 1e-15 {
				t.Errorf("bytes %d: cost = %v, want %v", tc.bytes, got, tc.want)
			}
		}
	})

	t.Run("bytes default to the text length", func(t *testing.T) {
		req := autoReq("quality", "balanced")
		req.PromptBytes = 0
		req.Text = strings.Repeat("a", 400)
		d := mustDecide(t, newWorld().engine(t), req)
		if got := candidate(t, d, qwen).ExpectedCostUSD; got == nil || math.Abs(*got-3e-5) > 1e-15 {
			t.Fatalf("cost = %v, want 3e-5", got)
		}
	})

	t.Run("text and bytes come from the messages", func(t *testing.T) {
		req := autoReq("quality", "balanced")
		req.PromptBytes = 0
		req.Text = ""
		req.Messages = []provider.Message{
			{Role: provider.RoleSystem, Content: strings.Repeat("a", 199)},
			{Role: provider.RoleUser, Content: strings.Repeat("b", 200)},
		} // 199 + "\n" + 200 = 400 bytes
		d := mustDecide(t, newWorld().engine(t), req)
		if got := candidate(t, d, qwen).ExpectedCostUSD; got == nil || math.Abs(*got-3e-5) > 1e-15 {
			t.Fatalf("cost = %v, want 3e-5", got)
		}
	})
}

func TestEngineDoesNotMutateTenant(t *testing.T) {
	w := newWorld()
	ten := &config.Tenant{Name: "tenant-a", AllowedModels: []string{gpt5, pro, flash}}
	before := *ten
	before.AllowedModels = slices.Clone(ten.AllowedModels)

	e := w.engine(t)
	for _, strategy := range []string{"quality", "strongest", "cheapest", "static"} {
		req := autoReq(strategy, "balanced")
		req.Tenant = ten
		mustDecide(t, e, req)
	}
	req := autoReq("", "balanced")
	req.Tenant = ten
	req.Model = flash
	mustDecide(t, e, req)

	if !reflect.DeepEqual(*ten, before) {
		t.Fatalf("tenant mutated: %+v, was %+v", *ten, before)
	}
}

func TestEngineDecidedInUS(t *testing.T) {
	d := mustDecide(t, newWorld().engine(t), autoReq("quality", "balanced"))
	if d.DecidedInUS < 0 {
		t.Fatalf("decided_in_us = %d", d.DecidedInUS)
	}
}

func TestEngineDefaultIDAndClock(t *testing.T) {
	w := newWorld()
	e := router.NewEngine(router.EngineOptions{
		Models: w.models, Prices: w.prices(t), Policies: w.registry(t, "pv-1"),
	})
	before := time.Now()
	d := mustDecide(t, e, autoReq("quality", "balanced"))
	if !strings.HasPrefix(d.ID, "dec_") || len(d.ID) != len("dec_")+32 {
		t.Fatalf("id = %q, want a generated decision id", d.ID)
	}
	if d.CreatedAt.Before(before) || d.CreatedAt.After(time.Now()) {
		t.Fatalf("created_at = %v, want the wall clock", d.CreatedAt)
	}
}

// Decisions share nothing mutable: parallel calls under the race detector,
// while another goroutine swaps the active policy.
func TestEngineConcurrentDecide(t *testing.T) {
	w := newWorld()
	reg := policy.NewRegistry(poolNames, nil)
	a, b := w.policy("pv-a"), w.policy("pv-b")
	for _, p := range []*policy.Policy{a, b} {
		if err := reg.Add(p); err != nil {
			t.Fatal(err)
		}
	}
	if _, err := reg.Activate("pv-a", "test"); err != nil {
		t.Fatal(err)
	}
	e := w.engineOver(t, reg)

	var wg sync.WaitGroup
	for i := 0; i < 8; i++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			for j := 0; j < 100; j++ {
				d, err := decideWith(e, autoReq("quality", "balanced"))
				if err != nil {
					t.Errorf("Decide() error: %v", err)
					return
				}
				if d.PolicyVersion != "pv-a" && d.PolicyVersion != "pv-b" {
					t.Errorf("policy version = %q", d.PolicyVersion)
					return
				}
				if d.Chosen != flash {
					t.Errorf("chosen = %q, want %q", d.Chosen, flash)
					return
				}
			}
		}()
	}
	wg.Add(1)
	go func() {
		defer wg.Done()
		for j := 0; j < 100; j++ {
			version := "pv-b"
			if j%2 == 1 {
				version = "pv-a"
			}
			if _, err := reg.Activate(version, "test"); err != nil {
				t.Errorf("Activate: %v", err)
				return
			}
		}
	}()
	wg.Wait()
}
