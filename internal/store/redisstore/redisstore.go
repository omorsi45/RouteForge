// Package redisstore keeps rate limits and daily budgets in Redis, so every
// gateway replica shares one view of them. Store implements both
// accounting.RateLimiter and accounting.BudgetStore, and follows the semantics
// of the memory stores exactly; the same contract tests run against both.
//
// Keys (the braces are a Redis Cluster hash tag around the tenant, so one
// tenant's keys live in one slot):
//
//	rf:rl:{<tenant>}:<kind>         hash tokens, ts_us   token bucket
//	rf:budget:{<tenant>}:<yyyymmdd> hash spent, reserved, overshoot (nano-USD)
//
// Every key expires 48 hours after its last write, so a call that runs across
// midnight can still settle on the day it started and idle tenants leave
// nothing behind.
//
// Redis 5.0 or newer is required: the limiter reads the server clock with TIME
// and then writes, which older servers refuse in a script.
//
// The scripts are not idempotent. A client that retries a command after a
// connection error could apply a reservation or a charge twice, so build the
// client with retries disabled (go-redis MaxRetries of -1) and let the
// accountant's fail mode decide what a failure means.
package redisstore

import (
	"context"
	"fmt"
	"strconv"
	"time"

	"github.com/redis/go-redis/v9"

	"github.com/omorsi45/routeforge/internal/accounting"
)

const (
	// keyTTL is how long a key lives after its last write.
	keyTTL = 48 * time.Hour

	// maxAmount bounds one reservation or charge, in nano-USD. It is the
	// largest daily budget the configuration accepts (about 1.1 million USD),
	// so no real request comes near it, and it leaves room for a day's totals
	// to stay below 2^53, where the scripts' arithmetic is exact.
	maxAmount = int64(1) << 50

	// maxPerMinute bounds a rate limit. It keeps every product in the limiter
	// script below 2^53: capacity is perMinute x 1e6 micro-tokens and a wait
	// is computed from capacity x 60.
	maxPerMinute = int64(100_000_000)
)

// Store is a RateLimiter and a BudgetStore on Redis. It is safe for concurrent
// use.
type Store struct {
	client redis.UniversalClient
}

var (
	_ accounting.RateLimiter = (*Store)(nil)
	_ accounting.BudgetStore = (*Store)(nil)
)

// New returns a Store on client. The caller owns the client and closes it.
func New(client redis.UniversalClient) *Store {
	return &Store{client: client}
}

func rateKey(tenant, kind string) string {
	return "rf:rl:{" + tenant + "}:" + kind
}

func budgetKey(k accounting.BudgetKey) string {
	return "rf:budget:{" + k.Tenant + "}:" + k.Day
}

// ttlSeconds is keyTTL as the script argument.
const ttlSeconds = int64(keyTTL / time.Second)

// Allow implements accounting.RateLimiter. The bucket is refilled from the
// Redis server's clock, never from this process's.
func (s *Store) Allow(ctx context.Context, tenant, kind string, n, perMinute int64) (accounting.RateDecision, error) {
	if perMinute <= 0 || n < 0 {
		return accounting.RateDecision{}, fmt.Errorf("rate limit %s for %q: need a positive rate and a non-negative amount, got n=%d perMinute=%d", kind, tenant, n, perMinute)
	}
	if perMinute > maxPerMinute {
		return accounting.RateDecision{}, fmt.Errorf("rate limit %s for %q: %d a minute is above the supported %d", kind, tenant, perMinute, maxPerMinute)
	}
	res, err := allowScript.Run(ctx, s.client, []string{rateKey(tenant, kind)}, n, perMinute, ttlSeconds).Int64Slice()
	if err != nil {
		return accounting.RateDecision{}, fmt.Errorf("redis rate limit %s for %q: %w", kind, tenant, err)
	}
	if len(res) != 2 {
		return accounting.RateDecision{}, fmt.Errorf("redis rate limit %s for %q: script returned %d values, want 2", kind, tenant, len(res))
	}
	if res[0] == 1 {
		return accounting.RateDecision{OK: true}, nil
	}
	return accounting.RateDecision{RetryAfter: time.Duration(res[1]) * time.Second}, nil
}

// checkAmount rejects a value the scripts could not keep exact.
func checkAmount(what string, v int64) error {
	if v < 0 || v > maxAmount {
		return fmt.Errorf("%s %d is outside 0 to %d nano-USD", what, v, maxAmount)
	}
	return nil
}

// Reserve implements accounting.BudgetStore.
func (s *Store) Reserve(ctx context.Context, k accounting.BudgetKey, amount, limit int64) (bool, error) {
	if err := checkAmount("reservation", amount); err != nil {
		return false, fmt.Errorf("redis budget reserve for %q on %s: %w", k.Tenant, k.Day, err)
	}
	got, err := reserveScript.Run(ctx, s.client, []string{budgetKey(k)}, amount, limit, ttlSeconds).Int64()
	if err != nil {
		return false, fmt.Errorf("redis budget reserve for %q on %s: %w", k.Tenant, k.Day, err)
	}
	return got == 1, nil
}

// Settle implements accounting.BudgetStore.
func (s *Store) Settle(ctx context.Context, k accounting.BudgetKey, reserved, actual, limit int64) (int64, error) {
	if err := checkAmount("reserved amount", reserved); err != nil {
		return 0, fmt.Errorf("redis budget settle for %q on %s: %w", k.Tenant, k.Day, err)
	}
	if err := checkAmount("actual charge", actual); err != nil {
		return 0, fmt.Errorf("redis budget settle for %q on %s: %w", k.Tenant, k.Day, err)
	}
	overshoot, err := settleScript.Run(ctx, s.client, []string{budgetKey(k)}, reserved, actual, limit, ttlSeconds).Int64()
	if err != nil {
		return 0, fmt.Errorf("redis budget settle for %q on %s: %w", k.Tenant, k.Day, err)
	}
	return overshoot, nil
}

// State implements accounting.BudgetStore. It is one HMGET, which Redis
// answers atomically; it does not renew the expiry.
func (s *Store) State(ctx context.Context, k accounting.BudgetKey) (accounting.BudgetState, error) {
	vals, err := s.client.HMGet(ctx, budgetKey(k), "spent", "reserved", "overshoot").Result()
	if err != nil {
		return accounting.BudgetState{}, fmt.Errorf("redis budget state for %q on %s: %w", k.Tenant, k.Day, err)
	}
	if len(vals) != 3 {
		return accounting.BudgetState{}, fmt.Errorf("redis budget state for %q on %s: got %d fields, want 3", k.Tenant, k.Day, len(vals))
	}
	var st accounting.BudgetState
	for i, field := range []struct {
		name string
		dst  *int64
	}{{"spent", &st.Spent}, {"reserved", &st.Reserved}, {"overshoot", &st.Overshoot}} {
		v, err := parseField(vals[i])
		if err != nil {
			return accounting.BudgetState{}, fmt.Errorf("redis budget state for %q on %s: field %s: %w", k.Tenant, k.Day, field.name, err)
		}
		*field.dst = v
	}
	return st, nil
}

// parseField reads an integer hash field; a missing field is zero.
func parseField(v any) (int64, error) {
	if v == nil {
		return 0, nil
	}
	str, ok := v.(string)
	if !ok {
		return 0, fmt.Errorf("unexpected %T value", v)
	}
	return strconv.ParseInt(str, 10, 64)
}

// Ping implements both accounting.RateLimiter and accounting.BudgetStore.
func (s *Store) Ping(ctx context.Context) error {
	if err := s.client.Ping(ctx).Err(); err != nil {
		return fmt.Errorf("redis ping: %w", err)
	}
	return nil
}
