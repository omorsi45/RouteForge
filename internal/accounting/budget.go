package accounting

import (
	"context"
	"sync"
	"time"
)

// BudgetKey names one tenant-day budget. Day is a DayKey.
type BudgetKey struct {
	Tenant, Day string
}

// DayKey is the budget day of t: its UTC date as yyyymmdd. Budgets reset at
// midnight UTC.
func DayKey(t time.Time) string { return t.UTC().Format("20060102") }

// BudgetState is what a budget has used, in nano-USD.
type BudgetState struct {
	Spent    int64 // settled charges
	Reserved int64 // held back for calls still running
	// Overshoot is how far settled spend went beyond the limit: actual charges
	// are never clipped, so a call that costs more than was reserved can push
	// Spent over it. BudgetStore.Settle says how each charge adds to it.
	Overshoot int64
}

// BudgetStore keeps the spend of tenant-days. Implementations must make
// Reserve atomic: concurrent reservations never admit more than the limit.
type BudgetStore interface {
	// Reserve holds amount back and reports true when spent + reserved +
	// amount stays within limit, else holds nothing and reports false.
	Reserve(ctx context.Context, k BudgetKey, amount, limit int64) (bool, error)
	// Settle returns a reservation of reserved, charges actual and returns the
	// overshoot this charge added: max(0, spent_after - max(limit,
	// spent_before)). Over a day the overshoots add up to max(0, spend -
	// limit). A call that never reserved passes reserved 0.
	Settle(ctx context.Context, k BudgetKey, reserved, actual, limit int64) (overshoot int64, err error)
	// State returns the budget's totals; an unknown key is all zeros.
	State(ctx context.Context, k BudgetKey) (BudgetState, error)
	// Ping checks that the store is reachable.
	Ping(ctx context.Context) error
}

// MemoryBudgets is a BudgetStore in process memory. It is safe for concurrent
// use.
type MemoryBudgets struct {
	mu     sync.Mutex
	states map[BudgetKey]*BudgetState
}

// NewMemoryBudgets returns an empty MemoryBudgets.
func NewMemoryBudgets() *MemoryBudgets {
	return &MemoryBudgets{states: make(map[BudgetKey]*BudgetState)}
}

// state returns the state of k, creating it. The caller holds mu.
func (m *MemoryBudgets) state(k BudgetKey) *BudgetState {
	st, ok := m.states[k]
	if !ok {
		st = &BudgetState{}
		m.states[k] = st
	}
	return st
}

// Reserve implements BudgetStore.
func (m *MemoryBudgets) Reserve(_ context.Context, k BudgetKey, amount, limit int64) (bool, error) {
	m.mu.Lock()
	defer m.mu.Unlock()
	st := m.state(k)
	if st.Spent+st.Reserved+amount > limit {
		return false, nil
	}
	st.Reserved += amount
	return true, nil
}

// Settle implements BudgetStore.
func (m *MemoryBudgets) Settle(_ context.Context, k BudgetKey, reserved, actual, limit int64) (int64, error) {
	m.mu.Lock()
	defer m.mu.Unlock()
	st := m.state(k)
	before := st.Spent
	st.Reserved -= reserved
	st.Spent += actual
	overshoot := max(0, st.Spent-max(limit, before))
	st.Overshoot += overshoot
	return overshoot, nil
}

// State implements BudgetStore.
func (m *MemoryBudgets) State(_ context.Context, k BudgetKey) (BudgetState, error) {
	m.mu.Lock()
	defer m.mu.Unlock()
	if st, ok := m.states[k]; ok {
		return *st, nil
	}
	return BudgetState{}, nil
}

// Ping implements BudgetStore.
func (m *MemoryBudgets) Ping(context.Context) error { return nil }
