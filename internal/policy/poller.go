package policy

import (
	"context"
	"fmt"
	"log/slog"
	"time"
)

// pollerActor is recorded as the actor of an activation a replica followed.
// The actor of the original activation is in the Persister.
const pollerActor = "poller"

// StartPoller follows the active policy of p in the background: every `every`
// it reads the active version, and when it differs from the one r serves it
// loads that policy from its stored bytes, adds it to r without saving it again
// and switches r to it. A replica thereby follows an activation made on
// another. It stops when ctx ends.
//
// A failing poll is logged once, when it starts, and again when polling works
// again, not on every tick. A nil logger logs nothing.
func StartPoller(ctx context.Context, r *Registry, p Persister, every time.Duration, logger *slog.Logger) {
	if logger == nil {
		logger = slog.New(slog.DiscardHandler)
	}
	go func() {
		ticker := time.NewTicker(every)
		defer ticker.Stop()
		var failing string // the text of the error being reported, or empty
		for {
			select {
			case <-ctx.Done():
				return
			case <-ticker.C:
			}
			if ctx.Err() != nil {
				return
			}
			pollCtx, cancel := context.WithTimeout(ctx, every)
			followed, err := r.follow(pollCtx, p)
			cancel()
			switch {
			case err != nil && err.Error() != failing:
				failing = err.Error()
				logger.WarnContext(ctx, "policy poll failed", "error", err)
			case err == nil && failing != "":
				failing = ""
				logger.InfoContext(ctx, "policy poll recovered")
			}
			if followed.Version != "" {
				logger.InfoContext(ctx, "policy followed another replica", "version", followed.Version, "previous", followed.Previous)
			}
		}
	}()
}

// follow makes r serve the policy p holds as active, if that differs from the
// one r serves. It returns the activation it made, or the zero Activation when
// it made none. It writes nothing to p.
func (r *Registry) follow(ctx context.Context, p Persister) (Activation, error) {
	r.mu.Lock()
	epoch := r.epoch
	r.mu.Unlock()

	version, ok, err := p.ActiveVersion(ctx)
	if err != nil {
		return Activation{}, fmt.Errorf("read the active policy version: %w", err)
	}
	if !ok {
		return Activation{}, nil
	}

	r.mu.Lock()
	current, known := r.current, r.entries[version] != nil
	r.mu.Unlock()
	if version == current {
		return Activation{}, nil
	}
	if !known {
		raw, err := p.LoadRaw(ctx, version)
		if err != nil {
			return Activation{}, fmt.Errorf("load policy %s: %w", version, err)
		}
		pol, err := Load(raw)
		if err != nil {
			return Activation{}, fmt.Errorf("stored policy %s: %w", version, err)
		}
		if pol.Version != version {
			return Activation{}, fmt.Errorf("stored policy %s holds version %s", version, pol.Version)
		}
		if err := r.addStored(pol); err != nil {
			return Activation{}, fmt.Errorf("stored policy %s: %w", version, err)
		}
	}

	r.mu.Lock()
	defer r.mu.Unlock()
	if r.epoch != epoch {
		// An activation was made here while this poll was reading the store, so
		// the version read may already be out of date. The next poll looks again.
		return Activation{}, nil
	}
	return r.switchTo(r.entries[version], StatusRetired, pollerActor), nil
}
