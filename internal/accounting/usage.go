package accounting

import (
	"context"
	"sync"
	"time"
)

// Sources of a token count and of the cost computed from it.
const (
	SourceReported  = "reported"  // the provider sent usage
	SourceEstimated = "estimated" // counted from bytes, one token per four
)

// UsageRecord is the bill of one upstream attempt. It never holds prompt or
// completion text.
type UsageRecord struct {
	ID         string
	RequestID  string
	DecisionID string
	Tenant     string
	Day        string // the DayKey the attempt was charged to
	Model      string // configured model name
	Attempt    int    // 1 for the first attempt of the request
	Outcome    string // a router.Outcome* value
	// Partial is set when the attempt delivered some output and then failed.
	Partial          bool
	PromptTokens     int
	CompletionTokens int
	UsageSource      string // SourceReported or SourceEstimated
	CostNanos        int64
	CostSource       string // SourceReported or SourceEstimated
	CreatedAt        time.Time
}

// UsageTotals is what a tenant-day's usage records add up to.
type UsageTotals struct {
	Records   int
	CostNanos int64
}

// UsageStore keeps usage records. A write must not be dropped silently: a
// store that cannot take more reports Full so that admission can refuse work
// before it is done rather than lose its bill.
type UsageStore interface {
	Write(ctx context.Context, r UsageRecord) error
	// Full reports that the store cannot take another record right now.
	Full() bool
	Totals(ctx context.Context, tenant, day string) (UsageTotals, error)
	// Ping checks that the store is reachable.
	Ping(ctx context.Context) error
}

type usageKey struct{ tenant, day string }

// MemoryUsage is a UsageStore in process memory. It keeps the totals of every
// tenant-day and the most recent records up to its capacity. It is safe for
// concurrent use.
type MemoryUsage struct {
	mu       sync.Mutex
	capacity int
	ring     []UsageRecord // at most capacity records
	next     int           // where the next record goes once the ring is full
	totals   map[usageKey]UsageTotals
}

// NewMemoryUsage returns an empty MemoryUsage that keeps the last capacity
// records. A capacity of zero or less keeps none; totals are always kept.
func NewMemoryUsage(capacity int) *MemoryUsage {
	capacity = max(capacity, 0)
	return &MemoryUsage{
		capacity: capacity,
		ring:     make([]UsageRecord, 0, capacity),
		totals:   make(map[usageKey]UsageTotals),
	}
}

// Write implements UsageStore. It never fails and never drops a write: an old
// record may leave the window of recent ones, but it stays in the totals.
func (u *MemoryUsage) Write(_ context.Context, r UsageRecord) error {
	u.mu.Lock()
	defer u.mu.Unlock()
	k := usageKey{r.Tenant, r.Day}
	t := u.totals[k]
	t.Records++
	t.CostNanos += r.CostNanos
	u.totals[k] = t

	switch {
	case u.capacity == 0:
	case len(u.ring) < u.capacity:
		u.ring = append(u.ring, r)
	default:
		u.ring[u.next] = r
		u.next = (u.next + 1) % u.capacity
	}
	return nil
}

// Full implements UsageStore. Memory is never full.
func (u *MemoryUsage) Full() bool { return false }

// Totals implements UsageStore.
func (u *MemoryUsage) Totals(_ context.Context, tenant, day string) (UsageTotals, error) {
	u.mu.Lock()
	defer u.mu.Unlock()
	return u.totals[usageKey{tenant, day}], nil
}

// Ping implements UsageStore.
func (u *MemoryUsage) Ping(context.Context) error { return nil }

// Recent returns a copy of the records still held, oldest first.
func (u *MemoryUsage) Recent() []UsageRecord {
	u.mu.Lock()
	defer u.mu.Unlock()
	out := make([]UsageRecord, 0, len(u.ring))
	out = append(out, u.ring[u.next:]...)
	out = append(out, u.ring[:u.next]...)
	return out
}
