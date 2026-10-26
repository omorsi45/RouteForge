package main

import (
	"fmt"
	"net/http"

	"github.com/omorsi45/routeforge/internal/config"
	"github.com/omorsi45/routeforge/internal/executor"
	"github.com/omorsi45/routeforge/internal/provider"
	"github.com/omorsi45/routeforge/internal/provider/anthropic"
	"github.com/omorsi45/routeforge/internal/provider/fake"
	"github.com/omorsi45/routeforge/internal/provider/openai"
)

// buildProviders constructs every configured provider, keyed by provider
// name. A provider whose api_key_env names an unset or empty variable fails
// the build; the variable name is reported and its value never is.
func buildProviders(cfg *config.Config, getenv func(string) string, client *http.Client) (map[string]provider.Provider, error) {
	out := make(map[string]provider.Provider, len(cfg.Providers))
	for _, p := range cfg.Providers {
		var key string
		if p.APIKeyEnv != "" {
			key = getenv(p.APIKeyEnv)
			if key == "" {
				return nil, fmt.Errorf("provider %q: environment variable %s is not set", p.Name, p.APIKeyEnv)
			}
		}
		switch p.Type {
		case "openai":
			out[p.Name] = openai.New(openai.Options{
				Name:           p.Name,
				BaseURL:        p.BaseURL,
				APIKey:         key,
				MaxTokensParam: p.MaxTokensParam,
				Client:         client,
			})
		case "anthropic":
			out[p.Name] = anthropic.New(anthropic.Options{
				Name:    p.Name,
				BaseURL: p.BaseURL,
				APIKey:  key,
				Client:  client,
			})
		case "fake":
			out[p.Name] = fake.New(p.Name, p.Fake)
		default:
			return nil, fmt.Errorf("provider %q: unsupported type %q", p.Name, p.Type)
		}
	}
	return out, nil
}

// buildTargets binds each configured model to its provider's instance.
func buildTargets(cfg *config.Config, providers map[string]provider.Provider) map[string]executor.Target {
	targets := make(map[string]executor.Target, len(cfg.Models))
	for _, m := range cfg.Models {
		targets[m.Name] = executor.Target{Model: m.Name, Upstream: m.UpstreamModel, Provider: providers[m.Provider]}
	}
	return targets
}
