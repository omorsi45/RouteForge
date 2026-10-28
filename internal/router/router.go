package router

import (
	"context"
	"errors"
	"time"

	"github.com/omorsi45/routeforge/internal/config"
	"github.com/omorsi45/routeforge/internal/provider"
)

// Request is what a router needs to choose a model. Tenant is shared and
// read-only.
type Request struct {
	Tenant    *config.Tenant
	Model     string // config.AutoModel or a configured model name
	Strategy  string // "" means the router default
	Profile   string // strict | balanced | economy
	Messages  []provider.Message
	MaxTokens int       // reserved output tokens
	Deadline  time.Time // request deadline; the zero value means none

	// PromptBytes is the length in bytes of the feature text. Zero means
	// "derive it from the text".
	PromptBytes int
	// Text is the feature text (policy.FeatureText of Messages), computed
	// once by the caller. Empty means "derive it from Messages".
	Text string
}

// Router chooses a model for a request.
//
// Decide may return a non-nil Decision together with ErrNoEligibleModel or
// ErrOverloaded; the decision then lists every candidate's ineligibility
// reasons. For every other error the decision is nil.
type Router interface {
	Decide(ctx context.Context, req *Request) (*Decision, error)
}

// Errors returned by Decide. Callers match them with errors.Is.
var (
	ErrUnknownModel        = errors.New("unknown model")
	ErrModelNotAllowed     = errors.New("model not allowed")
	ErrStrategyUnavailable = errors.New("strategy not available")
	ErrNoEligibleModel     = errors.New("no eligible model")
	// ErrOverloaded means no candidate is eligible and every candidate's
	// queue is full.
	ErrOverloaded = errors.New("overloaded")
)
