package reliability

import (
	"math/rand/v2"
	"time"
)

// Backoff computes full-jitter exponential delays: the delay before retry n is
// drawn uniformly from [0, min(Cap, Base * 2^(n-1))).
type Backoff struct {
	Base, Cap time.Duration
	// Rand returns a number in [0, 1). Nil uses math/rand/v2, which needs no
	// seeding and is safe for concurrent use.
	Rand func() float64
}

// Delay returns how long to wait before retry number retry, counted from 1.
// There is no delay before the first attempt, so a retry below 1 gets none.
func (b Backoff) Delay(retry int) time.Duration {
	if retry < 1 {
		return 0
	}
	// Double up to the cap rather than shifting by retry-1, which would
	// overflow for large counts.
	ceiling := b.Base
	for i := 1; i < retry && ceiling < b.Cap; i++ {
		ceiling *= 2
	}
	ceiling = min(ceiling, b.Cap)

	r := b.Rand
	if r == nil {
		r = rand.Float64
	}
	return time.Duration(r() * float64(ceiling))
}
