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
	Deadline  time.Time // request deadline
}

// Router chooses a model for a request.
type Router interface {
	Decide(ctx context.Context, req *Request) (*Decision, error)
}

// Errors returned by Decide. Callers match them with errors.Is.
var (
	ErrUnknownModel        = errors.New("unknown model")
	ErrModelNotAllowed     = errors.New("model not allowed")
	ErrStrategyUnavailable = errors.New("strategy not available")
	ErrNoEligibleModel     = errors.New("no eligible model")
)
