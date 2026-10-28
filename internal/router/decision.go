// Package router defines the routing seam: the Router interface, the
// decision record and the Basic router used before the quality router exists.
package router

import (
	"crypto/rand"
	"encoding/hex"
	"slices"
	"time"
)

// Strategies. The first group is selectable by clients through
// X-RouteForge-Strategy; StrategyPinned only labels decisions for explicitly
// named models.
const (
	StrategyPinned    = "pinned"
	StrategyQuality   = "quality"
	StrategyStrongest = "strongest"
	StrategyCheapest  = "cheapest"
	StrategyStatic    = "static"
	StrategyLatency   = "latency"
)

// Decision reasons.
const (
	ReasonPinned    = "pinned"
	ReasonStrongest = "strongest"
)

// Attempt outcomes.
const (
	OutcomeOK       = "ok"
	OutcomeError    = "error"
	OutcomeCanceled = "canceled"
	OutcomePartial  = "partial"
)

var clientStrategies = []string{
	StrategyQuality, StrategyStrongest, StrategyCheapest, StrategyStatic, StrategyLatency,
}

// ValidStrategy reports whether name is a value clients may send in
// X-RouteForge-Strategy.
func ValidStrategy(name string) bool {
	return slices.Contains(clientStrategies, name)
}

// Decision is the routing record for one request. It never holds prompt text.
type Decision struct {
	ID               string      `json:"id"`
	Tenant           string      `json:"tenant"`
	CreatedAt        time.Time   `json:"created_at"`
	PolicyVersion    string      `json:"policy_version"` // empty until a policy is loaded
	Strategy         string      `json:"strategy"`
	Profile          string      `json:"profile"`
	Chosen           string      `json:"chosen"`
	Reason           string      `json:"reason"`
	Lambda           *float64    `json:"lambda"`
	GuaranteeApplies bool        `json:"guarantee_applies"`
	DecidedInUS      int64       `json:"decided_in_us"`
	Candidates       []Candidate `json:"candidates"`
	Fallbacks        []string    `json:"fallbacks"`
	Attempts         []Attempt   `json:"attempts"`
}

// Candidate is one model considered by a decision.
type Candidate struct {
	Model             string   `json:"model"`
	Eligible          bool     `json:"eligible"`
	IneligibleReasons []string `json:"ineligible_reasons"`
	Raw               *float64 `json:"raw"`
	P                 *float64 `json:"p"`
	LCB               *float64 `json:"lcb"`
	BlockN            *int     `json:"block_n"`
	BlockK            *int     `json:"block_k"`
	ExpectedCostUSD   *float64 `json:"expected_cost_usd"`
	PredictedP95MS    *float64 `json:"predicted_p95_ms"`
}

// Attempt is one upstream call made on behalf of a decision.
type Attempt struct {
	Model      string  `json:"model"`
	Outcome    string  `json:"outcome"`              // OutcomeOK, OutcomeError, OutcomeCanceled or OutcomePartial
	ErrorKind  string  `json:"error_kind,omitempty"` // a provider.ErrorKind value
	DurationMS float64 `json:"duration_ms"`          // milliseconds, microsecond resolution
}

// NewDecisionID returns "dec_" followed by 32 lowercase hex characters from
// crypto/rand.
func NewDecisionID() string {
	var b [16]byte
	if _, err := rand.Read(b[:]); err != nil {
		panic("router: crypto/rand failed: " + err.Error())
	}
	return "dec_" + hex.EncodeToString(b[:])
}
