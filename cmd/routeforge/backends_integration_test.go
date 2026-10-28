//go:build integration

package main

import (
	"bytes"
	"context"
	"encoding/json"
	"log/slog"
	"net/http"
	"testing"
	"time"

	"github.com/alicebob/miniredis/v2"
	"github.com/jackc/pgx/v5/pgxpool"

	"github.com/omorsi45/routeforge/internal/accounting"
	"github.com/omorsi45/routeforge/internal/auth"
	"github.com/omorsi45/routeforge/internal/config"
	"github.com/omorsi45/routeforge/internal/store/postgres/pgtest"
)

const itAdminKey = "test-admin-key"

// itConfig is the demo configuration on the Postgres and Redis of the test, with
// no simulated latency. Every gateway of a test gets a configuration of its own.
func itConfig(t *testing.T) *config.Config {
	t.Helper()
	cfg, err := config.Load("../../deploy/configs/policy-fake.yaml")
	if err != nil {
		t.Fatal(err)
	}
	for i := range cfg.Providers {
		cfg.Providers[i].Fake.TTFT = 0
		cfg.Providers[i].Fake.PerToken = 0
	}
	cfg.Admin.APIKeySHA256 = []string{auth.HashKey(itAdminKey)}
	cfg.Tenants[0].RPM = 1000
	cfg.Tenants[0].DailyBudgetUSD = 100
	cfg.Storage = config.Storage{Type: config.StoragePostgres, DSNEnv: testPGDSNEnv}
	cfg.Limiter = config.Limiter{Type: config.LimiterRedis, URLEnv: testRedisURLEnv}
	return cfg
}

type itPolicyList struct {
	Active   *string `json:"active"`
	Policies []struct {
		Version string `json:"version"`
		Status  string `json:"status"`
	} `json:"policies"`
	Activations []struct {
		Version string `json:"version"`
		Actor   string `json:"actor"`
	} `json:"activations"`
}

func itPolicies(t *testing.T, base string) itPolicyList {
	t.Helper()
	resp, raw := apiRequest(t, http.MethodGet, base+"/admin/policies", itAdminKey, "")
	if resp.StatusCode != http.StatusOK {
		t.Fatalf("GET /admin/policies: status %d, body %s", resp.StatusCode, raw)
	}
	var list itPolicyList
	if err := json.Unmarshal(raw, &list); err != nil {
		t.Fatalf("decode %s: %v", raw, err)
	}
	return list
}

func (l itPolicyList) status(version string) string {
	for _, p := range l.Policies {
		if p.Version == version {
			return p.Status
		}
	}
	return "absent"
}

// waitFor polls cond until it holds, for at most d. The wait is for another
// gateway's poller, which has no signal to listen for.
func waitFor(t *testing.T, d time.Duration, what string, cond func() bool) {
	t.Helper()
	deadline := time.Now().Add(d)
	for !cond() {
		if time.Now().After(deadline) {
			t.Fatalf("%s did not happen within %v", what, d)
		}
		time.Sleep(50 * time.Millisecond)
	}
}

// policyOf explains a request on the gateway at base and returns the version of
// the policy that routed it.
func policyOf(t *testing.T, base string) string {
	t.Helper()
	resp, raw := apiRequest(t, http.MethodPost, base+"/v1/routing/explain", demoKey, itChatBody)
	if resp.StatusCode != http.StatusOK {
		t.Fatalf("explain: status %d, body %s", resp.StatusCode, raw)
	}
	var d struct {
		PolicyVersion string `json:"policy_version"`
	}
	if err := json.Unmarshal(raw, &d); err != nil {
		t.Fatalf("decode %s: %v", raw, err)
	}
	return d.PolicyVersion
}

const itChatBody = `{"model":"routeforge-auto","messages":[{"role":"user","content":"hello"}]}`

func TestServeWithPostgresAndRedis(t *testing.T) {
	dsn := pgtest.Start(t)
	mr := miniredis.RunT(t)
	redisURL := "redis://" + mr.Addr() + "/0"
	getenv := func(name string) string {
		switch name {
		case testPGDSNEnv:
			return dsn
		case testRedisURLEnv:
			return redisURL
		}
		return ""
	}
	// The pollers look every 250 ms instead of every 5 s. The gateways below are
	// stopped before this is put back, because their cleanups run first.
	old := policyPollInterval
	policyPollInterval = 250 * time.Millisecond
	t.Cleanup(func() { policyPollInterval = old })

	pool, err := pgxpool.New(context.Background(), dsn)
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(pool.Close)
	queryRow := func(query string, args ...any) func(dest ...any) {
		return func(dest ...any) {
			t.Helper()
			if err := pool.QueryRow(context.Background(), query, args...).Scan(dest...); err != nil {
				t.Fatalf("%s: %v", query, err)
			}
		}
	}

	logger := slog.New(slog.DiscardHandler)
	baseA, _ := startGatewayEnv(t, itConfig(t), logger, getenv)

	// A chat request, metered through Redis and recorded in Postgres.
	resp, raw := apiRequest(t, http.MethodPost, baseA+"/v1/chat/completions", demoKey, itChatBody)
	if resp.StatusCode != http.StatusOK {
		t.Fatalf("chat: status %d, body %s", resp.StatusCode, raw)
	}
	decisionID := resp.Header.Get("X-RouteForge-Decision-Id")
	v1 := resp.Header.Get("X-RouteForge-Policy-Version")
	served := resp.Header.Get("X-RouteForge-Model")
	if decisionID == "" || !policyVersionRE.MatchString(v1) {
		t.Fatalf("decision id %q and policy version %q: want an id and a 12 character version", decisionID, v1)
	}

	// /admin/accounting reads the totals from Postgres, which first writes the
	// records still queued, so the answer needs no waiting.
	resp, raw = apiRequest(t, http.MethodGet, baseA+"/admin/accounting?tenant=demo", itAdminKey, "")
	if resp.StatusCode != http.StatusOK {
		t.Fatalf("GET /admin/accounting: status %d, body %s", resp.StatusCode, raw)
	}
	var acct struct {
		SpentNanos     int64 `json:"spent_nanos"`
		ReservedNanos  int64 `json:"reserved_nanos"`
		UsageRecords   int   `json:"usage_records"`
		UsageCostNanos int64 `json:"usage_cost_nanos"`
	}
	if err := json.Unmarshal(raw, &acct); err != nil {
		t.Fatalf("decode %s: %v", raw, err)
	}
	if acct.UsageRecords != 1 || acct.UsageCostNanos <= 0 {
		t.Fatalf("accounting = %+v, want one usage record with a cost", acct)
	}
	if acct.SpentNanos != acct.UsageCostNanos || acct.ReservedNanos != 0 {
		t.Errorf("Redis spent %d and reserved %d against %d in the usage records, want equal spend and no reservation",
			acct.SpentNanos, acct.ReservedNanos, acct.UsageCostNanos)
	}

	var rows int
	var cost int64
	queryRow(`select count(*), coalesce(sum(cost_nanos), 0)::bigint from usage_records where tenant = 'demo'`)(&rows, &cost)
	if rows != 1 || cost != acct.UsageCostNanos {
		t.Errorf("usage_records has %d rows costing %d, want 1 costing %d", rows, cost, acct.UsageCostNanos)
	}
	var tenant, policyVersion, chosen string
	queryRow(`select tenant, policy_version, chosen from routing_decisions where id = $1`, decisionID)(&tenant, &policyVersion, &chosen)
	if tenant != "demo" || policyVersion != v1 || chosen != served {
		t.Errorf("routing_decisions row = %q %q %q, want demo's decision under policy %s choosing %s", tenant, policyVersion, chosen, v1, served)
	}
	day := accounting.DayKey(time.Now())
	if got := mr.HGet("rf:budget:{demo}:"+day, "spent"); got == "" || got == "0" {
		t.Errorf("Redis holds no spend for demo on %s", day)
	}
	if !mr.Exists("rf:rl:{demo}:rpm") {
		t.Errorf("Redis holds no rate limit bucket; keys %v", mr.Keys())
	}

	// A second gateway on the same database reads the first one's decision from
	// the table and starts on the policy the database holds.
	baseB, _ := startGatewayEnv(t, itConfig(t), logger, getenv)
	resp, raw = apiRequest(t, http.MethodGet, baseB+"/v1/routing/decisions/"+decisionID, demoKey, "")
	if resp.StatusCode != http.StatusOK {
		t.Fatalf("second gateway: GET decision: status %d, body %s", resp.StatusCode, raw)
	}
	if got := policyOf(t, baseB); got != v1 {
		t.Fatalf("second gateway routes with %s, want %s", got, v1)
	}

	// An upload and an activation on the first gateway reach the second one
	// through the poller. The uploaded bytes are stored as they were sent.
	p2 := variantPolicy(t, setFirstInit("0.123"))
	resp, raw = apiRequest(t, http.MethodPost, baseA+"/admin/policies", itAdminKey, string(p2.Raw))
	if resp.StatusCode != http.StatusCreated {
		t.Fatalf("upload: status %d, body %s", resp.StatusCode, raw)
	}
	var stored []byte
	var status string
	queryRow(`select artifact_raw, status from policies where version = $1`, p2.Version)(&stored, &status)
	if !bytes.Equal(stored, p2.Raw) || status != "inactive" {
		t.Errorf("stored policy: %d bytes (sent %d), status %q; want the exact bytes, inactive", len(stored), len(p2.Raw), status)
	}
	if got := itPolicies(t, baseB).status(p2.Version); got != "absent" {
		t.Errorf("the second gateway knows an inactive upload: status %q", got)
	}

	resp, raw = apiRequest(t, http.MethodPost, baseA+"/admin/policies/"+p2.Version+"/activate", itAdminKey, "")
	if resp.StatusCode != http.StatusOK {
		t.Fatalf("activate: status %d, body %s", resp.StatusCode, raw)
	}
	waitFor(t, 15*time.Second, "the second gateway following the activation", func() bool {
		l := itPolicies(t, baseB)
		return l.Active != nil && *l.Active == p2.Version
	})
	if got := policyOf(t, baseB); got != p2.Version {
		t.Errorf("second gateway routes with %s after the activation, want %s", got, p2.Version)
	}
	followed := itPolicies(t, baseB)
	if followed.status(v1) != "retired" || followed.status(p2.Version) != "active" {
		t.Errorf("second gateway statuses: %s %s and %s %s", v1, followed.status(v1), p2.Version, followed.status(p2.Version))
	}
	if last := followed.Activations[len(followed.Activations)-1]; last.Version != p2.Version || last.Actor != "poller" {
		t.Errorf("last activation on the second gateway = %+v, want %s by the poller", last, p2.Version)
	}

	// A gateway that starts now takes the database's policy, not its own file's,
	// and keeps the file's policy for later.
	baseC, _ := startGatewayEnv(t, itConfig(t), logger, getenv)
	if got := policyOf(t, baseC); got != p2.Version {
		t.Errorf("third gateway routes with %s, want the database's %s", got, p2.Version)
	}
	if got := itPolicies(t, baseC).status(v1); got != "inactive" {
		t.Errorf("the file policy is %q on the third gateway, want inactive", got)
	}

	// A rollback on the second gateway reaches the first.
	resp, raw = apiRequest(t, http.MethodPost, baseB+"/admin/policies/rollback", itAdminKey, "")
	if resp.StatusCode != http.StatusOK {
		t.Fatalf("rollback: status %d, body %s", resp.StatusCode, raw)
	}
	waitFor(t, 15*time.Second, "the first gateway following the rollback", func() bool {
		l := itPolicies(t, baseA)
		return l.Active != nil && *l.Active == v1
	})
	var active string
	var activations int
	queryRow(`select version from policies where status = 'active'`)(&active)
	queryRow(`select count(*) from policy_activations`)(&activations)
	if active != v1 || activations != 3 {
		t.Errorf("database: active %s after %d activations, want %s after 3 (startup, admin, rollback)", active, activations, v1)
	}
}
