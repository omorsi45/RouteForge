package reliability

import (
	"sync"
	"sync/atomic"
)

// InFlight caps how many requests the gateway serves at once. Requests over
// the cap are refused immediately, never queued: the gateway sheds load rather
// than build a backlog. It is safe for concurrent use.
type InFlight struct {
	limit int64
	n     atomic.Int64
}

// NewInFlight returns a limiter that admits at most limit requests at once; it
// must be at least 1.
func NewInFlight(limit int) *InFlight { return &InFlight{limit: int64(limit)} }

// TryAcquire takes one slot. When all are taken it returns false and a nil
// release. Otherwise the caller must call release when the request is done;
// calling it again does nothing.
func (f *InFlight) TryAcquire() (release func(), ok bool) {
	for {
		n := f.n.Load()
		if n >= f.limit {
			return nil, false
		}
		if f.n.CompareAndSwap(n, n+1) {
			break
		}
	}
	var once sync.Once
	return func() { once.Do(func() { f.n.Add(-1) }) }, true
}

// Count returns how many requests hold a slot.
func (f *InFlight) Count() int { return int(f.n.Load()) }
