// Package store holds the gateway's records of past work. The decision store
// keeps routing decisions so clients can read them back.
package store

import (
	"context"
	"errors"
	"slices"
	"sync"

	"github.com/omorsi45/routeforge/internal/router"
)

// DecisionStore keeps routing decisions by tenant and id.
//
// Put stores a copy of d, so the caller may keep using its decision. Get
// returns a copy that the caller owns, and found is false for an unknown id
// and for an id that belongs to another tenant: the two cannot be told apart.
type DecisionStore interface {
	Put(d *router.Decision) error
	Get(ctx context.Context, tenant, id string) (d *router.Decision, found bool, err error)
}

// Ring is an in-memory DecisionStore that keeps the newest decisions and
// evicts the oldest when it is full. It is safe for concurrent use.
type Ring struct {
	mu    sync.Mutex
	slots []*router.Decision // fixed length; nil slots are free
	byID  map[string]int     // decision id to slot
	next  int                // slot the next new decision takes
}

// NewRing returns a ring that holds up to capacity decisions. It panics when
// capacity is not positive, which is a programming error.
func NewRing(capacity int) *Ring {
	if capacity < 1 {
		panic("store: ring capacity must be positive")
	}
	return &Ring{
		slots: make([]*router.Decision, capacity),
		byID:  make(map[string]int, capacity),
	}
}

// Put stores a copy of d. A decision with an id already in the ring replaces
// that entry and keeps its place in the eviction order.
func (r *Ring) Put(d *router.Decision) error {
	if d == nil {
		return errors.New("store decision: decision is nil")
	}
	cp := cloneDecision(d)

	r.mu.Lock()
	defer r.mu.Unlock()
	if slot, ok := r.byID[cp.ID]; ok {
		r.slots[slot] = cp
		return nil
	}
	if old := r.slots[r.next]; old != nil {
		delete(r.byID, old.ID)
	}
	r.slots[r.next] = cp
	r.byID[cp.ID] = r.next
	r.next = (r.next + 1) % len(r.slots)
	return nil
}

// Get returns a copy of the decision with the given id when it belongs to
// tenant.
func (r *Ring) Get(_ context.Context, tenant, id string) (*router.Decision, bool, error) {
	r.mu.Lock()
	defer r.mu.Unlock()
	slot, ok := r.byID[id]
	if !ok || r.slots[slot].Tenant != tenant {
		return nil, false, nil
	}
	return cloneDecision(r.slots[slot]), true, nil
}

// cloneDecision deep-copies d. A nil slice or pointer stays nil, so the copy
// marshals to the same JSON as the original.
func cloneDecision(d *router.Decision) *router.Decision {
	cp := *d
	cp.Lambda = clonePtr(d.Lambda)
	cp.Fallbacks = slices.Clone(d.Fallbacks)
	cp.Attempts = slices.Clone(d.Attempts)
	if d.Candidates != nil {
		cp.Candidates = make([]router.Candidate, len(d.Candidates))
		for i, c := range d.Candidates {
			c.IneligibleReasons = slices.Clone(c.IneligibleReasons)
			c.Raw = clonePtr(c.Raw)
			c.P = clonePtr(c.P)
			c.LCB = clonePtr(c.LCB)
			c.BlockN = clonePtr(c.BlockN)
			c.BlockK = clonePtr(c.BlockK)
			c.ExpectedCostUSD = clonePtr(c.ExpectedCostUSD)
			c.PredictedP95MS = clonePtr(c.PredictedP95MS)
			cp.Candidates[i] = c
		}
	}
	return &cp
}

func clonePtr[T any](p *T) *T {
	if p == nil {
		return nil
	}
	v := *p
	return &v
}
