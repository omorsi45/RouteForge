package router

import (
	"cmp"
	"context"
	"fmt"
	"slices"
	"time"

	"github.com/omorsi45/routeforge/internal/config"
)

// BasicOptions injects the id generator and clock; nil fields use the
// defaults.
type BasicOptions struct {
	NewID func() string
	Now   func() time.Time
}

// Basic pins named models and routes routeforge-auto to the strongest
// allowed model. It does no scoring.
type Basic struct {
	models []config.Model
	newID  func() string
	now    func() time.Time
}

// NewBasic returns a Basic router over the configured models.
func NewBasic(models []config.Model, o BasicOptions) *Basic {
	b := &Basic{models: slices.Clone(models), newID: o.NewID, now: o.Now}
	if b.newID == nil {
		b.newID = NewDecisionID
	}
	if b.now == nil {
		b.now = time.Now
	}
	return b
}

// Decide implements Router.
func (b *Basic) Decide(_ context.Context, req *Request) (*Decision, error) {
	start := time.Now()
	d := &Decision{
		ID:         b.newID(),
		Tenant:     req.Tenant.Name,
		CreatedAt:  b.now(),
		Profile:    req.Profile,
		Fallbacks:  []string{},
		Attempts:   []Attempt{},
		Candidates: []Candidate{},
	}

	if req.Model != config.AutoModel {
		if !slices.ContainsFunc(b.models, func(m config.Model) bool { return m.Name == req.Model }) {
			return nil, fmt.Errorf("%w: %q", ErrUnknownModel, req.Model)
		}
		if !req.Tenant.Allows(req.Model) {
			return nil, fmt.Errorf("%w: %q", ErrModelNotAllowed, req.Model)
		}
		d.Strategy = StrategyPinned
		d.Reason = ReasonPinned
		d.Chosen = req.Model
		d.Candidates = append(d.Candidates, newCandidate(req.Model))
		d.DecidedInUS = time.Since(start).Microseconds()
		return d, nil
	}

	strategy := cmp.Or(req.Strategy, StrategyStrongest)
	if strategy != StrategyStrongest {
		return nil, fmt.Errorf("%w: %q", ErrStrategyUnavailable, strategy)
	}
	for _, m := range b.models {
		if !req.Tenant.Allows(m.Name) {
			continue
		}
		d.Candidates = append(d.Candidates, newCandidate(m.Name))
		if d.Chosen == "" && m.Strongest {
			d.Chosen = m.Name
		}
	}
	if d.Chosen == "" {
		return nil, fmt.Errorf("%w: no allowed model is marked strongest", ErrNoEligibleModel)
	}
	d.Strategy = strategy
	d.Reason = ReasonStrongest
	d.DecidedInUS = time.Since(start).Microseconds()
	return d, nil
}

func newCandidate(model string) Candidate {
	return Candidate{Model: model, Eligible: true, IneligibleReasons: []string{}}
}
