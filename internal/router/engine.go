package router

import (
	"cmp"
	"context"
	"errors"
	"fmt"
	"maps"
	"slices"
	"strings"
	"time"

	"github.com/omorsi45/routeforge/internal/accounting"
	"github.com/omorsi45/routeforge/internal/config"
	"github.com/omorsi45/routeforge/internal/policy"
)

// Signals are the live inputs to eligibility. A nil function means the model
// is never ineligible for that reason. All functions must be safe for
// concurrent use.
type Signals struct {
	// BreakerOpen reports whether the model's circuit breaker is open.
	BreakerOpen func(model string) bool
	// QueueFull reports whether the model's admission queue is full.
	QueueFull func(model string) bool
	// PredictedP95MS predicts the model's p95 latency in milliseconds for a
	// response of outputTokens tokens. Nil disables the latency check, leaves
	// predicted_p95_ms null and makes the latency strategy unavailable.
	PredictedP95MS func(model string, outputTokens float64) float64
	// BudgetRemainingNanos returns what the tenant may still reserve, in
	// nano-USD. limited is false when the tenant has no budget. An error is
	// not turned into an ineligibility reason; when accounting defines its
	// error this is where it will surface.
	BudgetRemainingNanos func(ctx context.Context, tenant string) (remaining int64, limited bool, err error)
}

// EngineOptions configures an Engine. Nil NewID and Now use NewDecisionID and
// time.Now. Nil Policies behaves like a registry with no active policy.
type EngineOptions struct {
	Models   []config.Model
	Prices   map[string]accounting.Price // by configured model name
	Policies *policy.Registry
	Signals  Signals
	NewID    func() string
	Now      func() time.Time
}

// Engine is the router that serves the trained policy. It implements Router.
// An Engine is immutable after NewEngine and safe for concurrent use.
//
// A model missing from Prices is treated as free, so a deployment must price
// every configured model (cmd/routeforge checks this at startup).
type Engine struct {
	models   []config.Model
	byPolicy map[string]*config.Model // by policy name
	prices   map[string]accounting.Price
	policies *policy.Registry
	signals  Signals
	newID    func() string
	now      func() time.Time
}

// NewEngine returns an Engine over the configured models.
func NewEngine(o EngineOptions) *Engine {
	e := &Engine{
		models:   slices.Clone(o.Models),
		prices:   maps.Clone(o.Prices),
		policies: o.Policies,
		signals:  o.Signals,
		newID:    o.NewID,
		now:      o.Now,
	}
	if e.newID == nil {
		e.newID = NewDecisionID
	}
	if e.now == nil {
		e.now = time.Now
	}
	e.byPolicy = make(map[string]*config.Model, len(e.models))
	for i := range e.models {
		m := &e.models[i]
		if m.PolicyName == "" {
			continue
		}
		if _, taken := e.byPolicy[m.PolicyName]; !taken {
			e.byPolicy[m.PolicyName] = m
		}
	}
	return e
}

// option is a candidate model under evaluation.
type option struct {
	model   *config.Model
	scored  bool // score and cost are set
	score   policy.Score
	cost    float64 // expected cost in USD
	p95     float64 // predicted p95 latency in ms
	hasP95  bool
	reasons []string // why the model is ineligible; empty means eligible
}

func (o *option) eligible() bool { return len(o.reasons) == 0 }

func (o *option) candidate() Candidate {
	c := Candidate{Model: o.model.Name, Eligible: o.eligible(), IneligibleReasons: o.reasons}
	if o.scored {
		c.Raw = ptr(o.score.Raw)
		c.P = ptr(o.score.P)
		c.LCB = ptr(o.score.LCB)
		c.BlockN = ptr(o.score.BlockN)
		c.BlockK = ptr(o.score.BlockK)
		c.ExpectedCostUSD = ptr(o.cost)
	}
	if o.hasP95 {
		c.PredictedP95MS = ptr(o.p95)
	}
	return c
}

func ptr[T any](v T) *T { return &v }

// evaluation holds what eligibility needs to know about one request.
type evaluation struct {
	req          *Request
	text         string  // feature text
	bytes        int     // prompt bytes
	promptTokens int     // ceil(bytes / 4)
	hasDeadline  bool    // the request has a deadline
	remainingMS  float64 // milliseconds left until the deadline
	limited      bool    // the tenant has a budget
	remaining    int64   // nano-USD the tenant may still reserve
}

func (e *Engine) newEvaluation(ctx context.Context, req *Request) *evaluation {
	text := req.Text
	if text == "" {
		text = policy.FeatureText(req.Messages)
	}
	bytes := req.PromptBytes
	if bytes == 0 {
		bytes = len(text)
	}
	ev := &evaluation{req: req, text: text, bytes: bytes, promptTokens: (bytes + 3) / 4}
	if !req.Deadline.IsZero() {
		ev.hasDeadline = true
		ev.remainingMS = float64(req.Deadline.Sub(e.now())) / float64(time.Millisecond)
	}
	if e.signals.BudgetRemainingNanos != nil {
		remaining, limited, err := e.signals.BudgetRemainingNanos(ctx, req.Tenant.Name)
		if err == nil {
			ev.limited, ev.remaining = limited, remaining
		}
	}
	return ev
}

// ineligibility lists every reason o cannot serve the request, in the fixed
// order of the Ineligible constants.
func (e *Engine) ineligibility(ev *evaluation, o *option) []string {
	reasons := []string{}
	name := o.model.Name
	if e.signals.BreakerOpen != nil && e.signals.BreakerOpen(name) {
		reasons = append(reasons, IneligibleBreakerOpen)
	}
	if e.signals.QueueFull != nil && e.signals.QueueFull(name) {
		reasons = append(reasons, IneligibleQueueFull)
	}
	if w := o.model.ContextWindow; w > 0 && ev.promptTokens+ev.req.MaxTokens > w {
		reasons = append(reasons, IneligibleContextWindow)
	}
	if o.hasP95 && ev.hasDeadline && o.p95 > ev.remainingMS {
		reasons = append(reasons, IneligibleLatencySLO)
	}
	if ev.limited && e.reservationNanos(ev, name) > ev.remaining {
		reasons = append(reasons, IneligibleBudget)
	}
	return reasons
}

// reservationNanos is what the request would reserve on the named model:
// prompt bytes at the input price plus the output cap at the output price.
func (e *Engine) reservationNanos(ev *evaluation, model string) int64 {
	price := e.prices[model]
	return int64(ev.bytes)*price.InNanos + int64(ev.req.MaxTokens)*price.OutNanos
}

// expectedCostUSD is the trainer's expected cost: the estimated prompt tokens
// times the input price plus the expected output tokens times the output
// price, per million tokens. Every product is converted to float64 before it
// is added so no compiler can fuse it into a multiply-add; Python never does.
func expectedCostUSD(bytes int, price accounting.Price, expectedOutputTokens float64) float64 {
	tokens := float64((bytes + 3) / 4)
	return (float64(tokens*price.InputPerMTok) + float64(expectedOutputTokens*price.OutputPerMTok)) / 1e6
}

// predictLatency fills o's predicted p95 when a latency signal exists.
func (e *Engine) predictLatency(o *option, expectedOutputTokens float64) {
	if e.signals.PredictedP95MS != nil {
		o.p95 = e.signals.PredictedP95MS(o.model.Name, expectedOutputTokens)
		o.hasP95 = true
	}
}

func (e *Engine) model(name string) *config.Model {
	for i := range e.models {
		if e.models[i].Name == name {
			return &e.models[i]
		}
	}
	return nil
}

func (e *Engine) activePolicy() *policy.Policy {
	if e.policies == nil {
		return nil
	}
	return e.policies.Active()
}

// Decide implements Router. It reads the active policy once, so a request
// scores and routes against a single policy version.
func (e *Engine) Decide(ctx context.Context, req *Request) (*Decision, error) {
	start := time.Now()
	d := &Decision{
		ID:         e.newID(),
		Tenant:     req.Tenant.Name,
		CreatedAt:  e.now(),
		Profile:    req.Profile,
		Fallbacks:  []string{},
		Attempts:   []Attempt{},
		Candidates: []Candidate{},
	}
	var err error
	if req.Model == config.AutoModel {
		err = e.routeAuto(ctx, req, d)
	} else {
		err = e.routePinned(ctx, req, d)
	}
	// Only these two errors come with a decision (Ruling C8).
	if err != nil && !errors.Is(err, ErrNoEligibleModel) && !errors.Is(err, ErrOverloaded) {
		return nil, err
	}
	d.DecidedInUS = time.Since(start).Microseconds()
	return d, err
}

// routePinned serves a request for one named model: the model must exist, be
// allowed and be eligible.
func (e *Engine) routePinned(ctx context.Context, req *Request, d *Decision) error {
	m := e.model(req.Model)
	if m == nil {
		return fmt.Errorf("%w: %q", ErrUnknownModel, req.Model)
	}
	if !req.Tenant.Allows(m.Name) {
		return fmt.Errorf("%w: %q", ErrModelNotAllowed, req.Model)
	}
	d.Strategy = StrategyPinned

	ev := e.newEvaluation(ctx, req)
	o := &option{model: m}
	if pol := e.activePolicy(); pol != nil {
		if i, ok := pol.ModelIndex(m.PolicyName); ok {
			e.predictLatency(o, pol.Models[i].ExpectedOutputTokens)
		}
	}
	o.reasons = e.ineligibility(ev, o)
	d.Candidates = append(d.Candidates, o.candidate())
	if !o.eligible() {
		// A pinned request has one candidate, so a full queue as its only
		// reason is the overload of spec 10: the model is fine and the gateway
		// is busy. Any other reason, or a second one, leaves the request
		// without an eligible model.
		sentinel := ErrNoEligibleModel
		if slices.Equal(o.reasons, []string{IneligibleQueueFull}) {
			sentinel = ErrOverloaded
		}
		return fmt.Errorf("%w: %q: %s", sentinel, m.Name, strings.Join(o.reasons, ", "))
	}
	d.Chosen, d.Reason = m.Name, ReasonPinned
	return nil
}

// routeAuto serves routeforge-auto with the active policy, or like Basic
// when there is none.
func (e *Engine) routeAuto(ctx context.Context, req *Request, d *Decision) error {
	pol := e.activePolicy()
	if pol == nil {
		return e.routeWithoutPolicy(ctx, req, d)
	}
	strategy := cmp.Or(req.Strategy, StrategyQuality)
	if !ValidStrategy(strategy) || (strategy == StrategyLatency && e.signals.PredictedP95MS == nil) {
		return fmt.Errorf("%w: %q", ErrStrategyUnavailable, strategy)
	}
	var lambda float64
	if strategy == StrategyQuality {
		profile, ok := pol.Profiles[req.Profile]
		if !ok {
			return fmt.Errorf("policy %s has no profile %q", pol.Version, req.Profile)
		}
		lambda = profile.Lambda
	}

	ev := e.newEvaluation(ctx, req)
	features := policy.Featurize(ev.text)
	opts := e.options(ev, pol, pol.Score(&features))

	d.Strategy, d.PolicyVersion = strategy, pol.Version
	for _, o := range opts {
		d.Candidates = append(d.Candidates, o.candidate())
	}
	eligible := slices.DeleteFunc(slices.Clone(opts), func(o *option) bool { return !o.eligible() })
	if len(eligible) == 0 {
		return noEligible(opts)
	}

	var (
		chosen    *option
		reason    string
		fallbacks []*option
	)
	if strategy == StrategyQuality {
		d.Lambda = ptr(lambda)
		chosen, reason, fallbacks = chooseQuality(pol, eligible, lambda)
		d.GuaranteeApplies = len(opts) == len(pol.Models) && len(eligible) == len(opts)
	} else {
		chosen, reason, fallbacks = chooseByStrategy(strategy, pol, ev.text, opts, eligible)
	}
	d.Chosen, d.Reason = chosen.model.Name, reason
	for _, f := range fallbacks {
		d.Fallbacks = append(d.Fallbacks, f.model.Name)
	}
	return nil
}

// options returns the candidates of a policy decision in pool order: the
// configured models that map to a policy model and that the tenant allows,
// each scored and checked for eligibility.
func (e *Engine) options(ev *evaluation, pol *policy.Policy, scores []policy.Score) []*option {
	opts := make([]*option, 0, len(pol.Models))
	for i := range pol.Models {
		pm := &pol.Models[i]
		m := e.byPolicy[pm.Name]
		if m == nil || !ev.req.Tenant.Allows(m.Name) {
			continue
		}
		o := &option{
			model:  m,
			scored: true,
			score:  scores[i],
			cost:   expectedCostUSD(ev.bytes, e.prices[m.Name], pm.ExpectedOutputTokens),
		}
		e.predictLatency(o, pm.ExpectedOutputTokens)
		o.reasons = e.ineligibility(ev, o)
		opts = append(opts, o)
	}
	return opts
}

// chooseQuality applies the quality rule (spec 8.4) to the eligible
// candidates, which are in pool order. Candidates are stably sorted by
// expected cost, so equal costs keep pool order, like the trainer's
// route_quality. The first with lcb >= lambda wins; with none, the policy's
// strongest model wins when eligible, else the eligible candidate with the
// highest p. Fallbacks are the other passing candidates by cost, then the
// rest by descending p.
func chooseQuality(pol *policy.Policy, eligible []*option, lambda float64) (chosen *option, reason string, fallbacks []*option) {
	byCost := slices.Clone(eligible)
	slices.SortStableFunc(byCost, func(a, b *option) int { return cmp.Compare(a.cost, b.cost) })
	passes := func(o *option) bool { return o.score.LCB >= lambda }
	passing := slices.DeleteFunc(slices.Clone(byCost), func(o *option) bool { return !passes(o) })

	var rest []*option
	if len(passing) > 0 {
		chosen, reason = passing[0], ReasonCheapestPassing
		fallbacks = passing[1:]
		rest = slices.DeleteFunc(slices.Clone(eligible), passes)
	} else {
		chosen, reason = byPolicyName(eligible, pol.Strongest), ReasonEscalated
		if chosen == nil {
			chosen = highestP(eligible)
		}
		rest = slices.DeleteFunc(slices.Clone(eligible), func(o *option) bool { return o == chosen })
	}
	sortByDescendingP(rest)
	return chosen, reason, append(slices.Clone(fallbacks), rest...)
}

// chooseByStrategy serves the reference strategies. The strategy picks among
// all candidates; when its pick is ineligible the eligible candidate with the
// highest p is chosen instead (Ruling R4). Fallbacks are the other eligible
// candidates by descending p.
func chooseByStrategy(strategy string, pol *policy.Policy, text string, opts, eligible []*option) (chosen *option, reason string, fallbacks []*option) {
	var pick *option
	switch strategy {
	case StrategyStrongest:
		pick, reason = byPolicyName(opts, pol.Strongest), ReasonStrongest
	case StrategyCheapest:
		pick, reason = lowest(opts, func(o *option) float64 { return o.cost }), ReasonCheapest
	case StrategyStatic:
		family := pol.Rules.Classify(text)
		pick, reason = byPolicyName(opts, pol.Rules.FamilyModel[family]), ReasonStaticFamily
	case StrategyLatency:
		pick, reason = lowest(opts, func(o *option) float64 { return o.p95 }), ReasonLatency
	}
	if pick == nil || !pick.eligible() {
		pick, reason = highestP(eligible), ReasonPickIneligible
	}
	fallbacks = slices.DeleteFunc(slices.Clone(eligible), func(o *option) bool { return o == pick })
	sortByDescendingP(fallbacks)
	return pick, reason, fallbacks
}

// routeWithoutPolicy reproduces Basic: only the strongest strategy exists and
// it picks the first allowed model marked strongest. A marked model that is
// ineligible is skipped.
func (e *Engine) routeWithoutPolicy(ctx context.Context, req *Request, d *Decision) error {
	strategy := cmp.Or(req.Strategy, StrategyStrongest)
	if strategy != StrategyStrongest {
		return fmt.Errorf("%w: %q", ErrStrategyUnavailable, strategy)
	}
	d.Strategy = strategy

	ev := e.newEvaluation(ctx, req)
	var opts []*option
	for i := range e.models {
		m := &e.models[i]
		if !req.Tenant.Allows(m.Name) {
			continue
		}
		o := &option{model: m}
		o.reasons = e.ineligibility(ev, o)
		opts = append(opts, o)
		d.Candidates = append(d.Candidates, o.candidate())
	}

	marked := false
	for _, o := range opts {
		if !o.model.Strongest {
			continue
		}
		marked = true
		if o.eligible() {
			d.Chosen, d.Reason = o.model.Name, ReasonStrongest
			return nil
		}
	}
	if !marked {
		return fmt.Errorf("%w: no allowed model is marked strongest", ErrNoEligibleModel)
	}
	return noEligible(opts)
}

// noEligible builds the error for a decision with no eligible candidate:
// ErrOverloaded when every candidate's queue is full, else ErrNoEligibleModel.
// The message lists each ineligible model with its reasons.
func noEligible(opts []*option) error {
	sentinel := ErrNoEligibleModel
	if len(opts) > 0 && !slices.ContainsFunc(opts, func(o *option) bool {
		return !slices.Contains(o.reasons, IneligibleQueueFull)
	}) {
		sentinel = ErrOverloaded
	}
	var parts []string
	for _, o := range opts {
		if !o.eligible() {
			parts = append(parts, o.model.Name+" ("+strings.Join(o.reasons, ", ")+")")
		}
	}
	if len(parts) == 0 {
		return fmt.Errorf("%w: no candidate model", sentinel)
	}
	return fmt.Errorf("%w: %s", sentinel, strings.Join(parts, "; "))
}

// byPolicyName returns the option for the named policy model, or nil.
func byPolicyName(opts []*option, name string) *option {
	for _, o := range opts {
		if o.model.PolicyName == name {
			return o
		}
	}
	return nil
}

// lowest returns the option with the smallest key; the first wins a tie, so
// pool order breaks ties. It returns nil for no options.
func lowest(opts []*option, key func(*option) float64) *option {
	var best *option
	for _, o := range opts {
		if best == nil || key(o) < key(best) {
			best = o
		}
	}
	return best
}

// highestP returns the option with the largest calibrated p; the first wins a
// tie, so pool order breaks ties. It returns nil for no options.
func highestP(opts []*option) *option {
	var best *option
	for _, o := range opts {
		if best == nil || o.score.P > best.score.P {
			best = o
		}
	}
	return best
}

// sortByDescendingP sorts by p, largest first; equal p keeps the input order.
func sortByDescendingP(opts []*option) {
	slices.SortStableFunc(opts, func(a, b *option) int { return cmp.Compare(b.score.P, a.score.P) })
}
