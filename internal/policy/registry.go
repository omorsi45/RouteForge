package policy

import (
	"context"
	"errors"
	"fmt"
	"slices"
	"sync"
	"sync/atomic"
	"time"
)

// Entry statuses. At most one entry is active. An entry that an activation
// replaced is retired; one that a rollback replaced, and one never activated,
// is inactive.
const (
	StatusInactive = "inactive"
	StatusActive   = "active"
	StatusRetired  = "retired"
)

var (
	// ErrUnknownVersion is returned when a version was never added.
	ErrUnknownVersion = errors.New("unknown policy version")
	// ErrNoPrevious is returned by Rollback when no policy was active before
	// the current one.
	ErrNoPrevious = errors.New("no previous policy to roll back to")
	// ErrPersist is wrapped by the error of an Add, Activate or Rollback whose
	// write to the Persister failed. The registry is then unchanged.
	ErrPersist = errors.New("policy persistence failed")
)

// persistTimeout bounds one call to the Persister made by Add, Activate or
// Rollback.
const persistTimeout = 10 * time.Second

// Persister keeps policies outside the process, so that replicas share them
// and they survive a restart. A Registry with a Persister writes through to it:
// a change is made in memory only after the Persister has taken it.
type Persister interface {
	// SavePolicy stores p as inactive. Saving a version that is already stored
	// changes nothing, its status included.
	SavePolicy(ctx context.Context, p *Policy) error
	// Activate makes version the active policy. previous is the version the
	// caller was running, and previousStatus the status (StatusRetired or
	// StatusInactive) of whichever policy it replaces. It wraps ErrUnknownVersion
	// for a version that was never saved.
	Activate(ctx context.Context, version, previous, previousStatus, actor string) error
	// ActiveVersion returns the active version, and false when there is none.
	ActiveVersion(ctx context.Context) (string, bool, error)
	// LoadRaw returns the exact bytes version was saved with. It wraps
	// ErrUnknownVersion for a version that was never saved.
	LoadRaw(ctx context.Context, version string) ([]byte, error)
}

// Entry is a policy known to a Registry.
type Entry struct {
	Policy  *Policy
	AddedAt time.Time
	Status  string
}

// Activation records one change of the active policy.
type Activation struct {
	Version  string // the policy that became active
	Previous string // the policy that was active before; empty for the first
	Actor    string
	At       time.Time
}

// Registry holds the policies a gateway can serve and which one is active.
// It is safe for concurrent use. Active is lock-free, so a request that
// already holds a policy keeps scoring with it while an activation swaps the
// pointer for the next request.
type Registry struct {
	configured map[string]bool
	now        func() time.Time
	active     atomic.Pointer[Policy]

	mu         sync.Mutex
	entries    map[string]*Entry
	versions   []string // in the order added
	current    string   // active version, empty before the first activation
	previous   string   // version active before current, empty if none
	log        []Activation
	onActivate func(Activation)
	persister  Persister
	epoch      uint64 // counts the activations, so the poller can tell one raced it
}

// NewRegistry returns an empty registry. configured lists the policy_name of
// every configured model; a policy may only use those. A nil now defaults to
// time.Now.
func NewRegistry(configured []string, now func() time.Time) *Registry {
	if now == nil {
		now = time.Now
	}
	names := make(map[string]bool, len(configured))
	for _, name := range configured {
		names[name] = true
	}
	return &Registry{configured: names, now: now, entries: make(map[string]*Entry)}
}

// Add stores p as inactive. A policy with a model that has no configured
// model is rejected with a *ValidationError at models[i].name. Adding a
// version that is already stored changes nothing.
//
// With a Persister the policy is saved first; if that fails the error wraps
// ErrPersist and nothing is stored.
func (r *Registry) Add(p *Policy) error {
	if err := r.check(p); err != nil {
		return err
	}
	r.mu.Lock()
	defer r.mu.Unlock()
	if _, ok := r.entries[p.Version]; ok {
		return nil
	}
	if r.persister != nil {
		ctx, cancel := context.WithTimeout(context.Background(), persistTimeout)
		defer cancel()
		if err := r.persister.SavePolicy(ctx, p); err != nil {
			return fmt.Errorf("%w: save policy %s: %w", ErrPersist, p.Version, err)
		}
	}
	r.insert(p)
	return nil
}

// addStored stores a policy the Persister already holds, without saving it
// again. The caller got it from the Persister.
func (r *Registry) addStored(p *Policy) error {
	if err := r.check(p); err != nil {
		return err
	}
	r.mu.Lock()
	defer r.mu.Unlock()
	if _, ok := r.entries[p.Version]; !ok {
		r.insert(p)
	}
	return nil
}

// check rejects a policy that is nil or has a model nothing is configured for.
func (r *Registry) check(p *Policy) error {
	if p == nil {
		return errors.New("add policy: policy is nil")
	}
	for i := range p.Models {
		if name := p.Models[i].Name; !r.configured[name] {
			return fail(fmt.Sprintf("models[%d].name", i), "policy model %q has no configured model", name)
		}
	}
	return nil
}

// insert stores p as inactive. The caller holds r.mu and has checked that the
// version is new.
func (r *Registry) insert(p *Policy) {
	r.entries[p.Version] = &Entry{Policy: p, AddedAt: r.now(), Status: StatusInactive}
	r.versions = append(r.versions, p.Version)
}

// Activate makes version the active policy and retires the one it replaces.
// Activating the version that is already active changes nothing and adds no
// record; the returned Activation then has Previous equal to Version.
//
// With a Persister the activation is written first and the active policy
// switches only once that has succeeded; if it fails the error wraps
// ErrPersist and the registry is unchanged.
func (r *Registry) Activate(version, actor string) (Activation, error) {
	r.mu.Lock()
	defer r.mu.Unlock()
	target, ok := r.entries[version]
	if !ok {
		return Activation{}, fmt.Errorf("%w: %s", ErrUnknownVersion, version)
	}
	if version == r.current {
		return Activation{Version: version, Previous: version, Actor: actor, At: r.now()}, nil
	}
	return r.activate(target, StatusRetired, actor)
}

// Rollback re-activates the version that was active before the current one;
// the version it replaces becomes inactive. A second Rollback switches back.
// It returns ErrNoPrevious when there is no such version. Like Activate it
// writes through to a Persister first.
func (r *Registry) Rollback(actor string) (Activation, error) {
	r.mu.Lock()
	defer r.mu.Unlock()
	if r.previous == "" {
		return Activation{}, ErrNoPrevious
	}
	return r.activate(r.entries[r.previous], StatusInactive, actor)
}

// activate writes the activation of target to the Persister, if there is one,
// and then switches to it. The caller holds r.mu.
func (r *Registry) activate(target *Entry, replaced, actor string) (Activation, error) {
	if r.persister != nil {
		ctx, cancel := context.WithTimeout(context.Background(), persistTimeout)
		defer cancel()
		if err := r.persister.Activate(ctx, target.Policy.Version, r.current, replaced, actor); err != nil {
			return Activation{}, fmt.Errorf("%w: activate policy %s: %w", ErrPersist, target.Policy.Version, err)
		}
	}
	return r.switchTo(target, replaced, actor), nil
}

// SetPersister makes the registry write every Add, Activate and Rollback
// through to p before it changes; nil removes it. It does not save or activate
// what the registry already holds.
func (r *Registry) SetPersister(p Persister) {
	r.mu.Lock()
	defer r.mu.Unlock()
	r.persister = p
}

// OnActivate registers fn to be told of every change of the active policy,
// replacing any earlier hook; nil removes it. fn runs once for each Activate or
// Rollback that changes the active policy, in the order the changes happen,
// while the registry is locked: it must be quick and must not call the
// registry. It is not called for the policy active when it is registered.
func (r *Registry) OnActivate(fn func(Activation)) {
	r.mu.Lock()
	defer r.mu.Unlock()
	r.onActivate = fn
}

// switchTo activates target, gives the policy it replaces the status
// replaced, and records the change. The caller holds r.mu.
func (r *Registry) switchTo(target *Entry, replaced, actor string) Activation {
	prev := r.current
	if old, ok := r.entries[prev]; ok {
		old.Status = replaced
	}
	target.Status = StatusActive
	r.previous, r.current = prev, target.Policy.Version
	r.epoch++
	act := Activation{Version: r.current, Previous: prev, Actor: actor, At: r.now()}
	r.log = append(r.log, act)
	r.active.Store(target.Policy)
	if r.onActivate != nil {
		r.onActivate(act)
	}
	return act
}

// Active returns the active policy, or nil before the first activation. It
// never blocks.
func (r *Registry) Active() *Policy {
	return r.active.Load()
}

// List returns a snapshot of every entry, sorted by AddedAt.
func (r *Registry) List() []Entry {
	r.mu.Lock()
	defer r.mu.Unlock()
	out := make([]Entry, len(r.versions))
	for i, v := range r.versions {
		out[i] = *r.entries[v]
	}
	slices.SortStableFunc(out, func(a, b Entry) int { return a.AddedAt.Compare(b.AddedAt) })
	return out
}

// Activations returns a copy of the activation log, oldest first.
func (r *Registry) Activations() []Activation {
	r.mu.Lock()
	defer r.mu.Unlock()
	return slices.Clone(r.log)
}
