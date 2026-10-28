//go:build integration

package postgres

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"os"
	"reflect"
	"sync"
	"testing"
	"time"

	"github.com/omorsi45/routeforge/internal/accounting"
	"github.com/omorsi45/routeforge/internal/policy"
	"github.com/omorsi45/routeforge/internal/router"
	"github.com/omorsi45/routeforge/internal/store/postgres/pgtest"
)

const committedPolicyFile = "../../../trainer/artifacts/policy.json"

func testContext(t *testing.T) context.Context {
	t.Helper()
	ctx, cancel := context.WithTimeout(context.Background(), time.Minute)
	t.Cleanup(cancel)
	return ctx
}

func openStore(t *testing.T, dsn string, o Options) *Store {
	t.Helper()
	s, err := Open(testContext(t), dsn, o)
	if err != nil {
		t.Fatalf("Open: %v", err)
	}
	t.Cleanup(s.Close)
	return s
}

func scanOne(t *testing.T, s *Store, query string, args []any, dest ...any) {
	t.Helper()
	if err := s.pool.QueryRow(testContext(t), query, args...).Scan(dest...); err != nil {
		t.Fatalf("%s: %v", query, err)
	}
}

func count(t *testing.T, s *Store, query string, args ...any) int {
	t.Helper()
	var n int
	scanOne(t, s, query, args, &n)
	return n
}

func TestMigrationsAreIdempotentOnReopen(t *testing.T) {
	dsn := pgtest.Start(t)

	// Several gateways may start together against an empty database: the
	// advisory lock lets exactly one apply the migrations.
	const replicas = 4
	var wg sync.WaitGroup
	stores := make([]*Store, replicas)
	errs := make([]error, replicas)
	for i := range replicas {
		wg.Add(1)
		go func() {
			defer wg.Done()
			stores[i], errs[i] = Open(context.Background(), dsn, Options{})
		}()
	}
	wg.Wait()
	for i, err := range errs {
		if err != nil {
			t.Fatalf("replica %d: Open: %v", i, err)
		}
	}
	for _, s := range stores {
		s.Close()
	}

	s := openStore(t, dsn, Options{})
	var version int
	var applied time.Time
	scanOne(t, s, `select version, applied_at from schema_migrations`, nil, &version, &applied)
	if version != 1 {
		t.Errorf("schema_migrations version = %d, want 1", version)
	}
	if n := count(t, s, `select count(*) from schema_migrations`); n != 1 {
		t.Fatalf("schema_migrations has %d rows after %d concurrent starts, want 1", n, replicas)
	}

	s.Close()
	s = openStore(t, dsn, Options{})
	var again time.Time
	scanOne(t, s, `select applied_at from schema_migrations where version = 1`, nil, &again)
	if !again.Equal(applied) {
		t.Errorf("applied_at moved from %v to %v: the migration ran again on reopen", applied, again)
	}
	if n := count(t, s, `select count(*) from schema_migrations`); n != 1 {
		t.Errorf("schema_migrations has %d rows after reopening, want 1", n)
	}
	for _, table := range []string{"policies", "policy_activations", "usage_records", "routing_decisions"} {
		if n := count(t, s, `select count(*) from information_schema.tables where table_schema = 'public' and table_name = $1`, table); n != 1 {
			t.Errorf("table %s is missing", table)
		}
	}
}

func usageRecord(id, tenant, day string, cost int64) accounting.UsageRecord {
	return accounting.UsageRecord{
		ID:               id,
		RequestID:        "req_" + id,
		DecisionID:       "dec_" + id,
		Tenant:           tenant,
		Day:              day,
		Model:            "gemini-2.5-pro",
		Attempt:          2,
		Outcome:          "partial",
		Partial:          true,
		PromptTokens:     11,
		CompletionTokens: 7,
		UsageSource:      accounting.SourceEstimated,
		CostNanos:        cost,
		CostSource:       accounting.SourceReported,
		CreatedAt:        time.Date(2026, 10, 28, 12, 30, 45, 123456000, time.UTC), // microseconds: what timestamptz keeps
	}
}

func TestUsageWriteAndTotals(t *testing.T) {
	s := openStore(t, pgtest.Start(t), Options{FlushInterval: time.Hour}) // only Totals can flush
	ctx := testContext(t)

	for _, r := range []accounting.UsageRecord{
		usageRecord("u1", "tenant-a", "20261028", 100),
		usageRecord("u2", "tenant-a", "20261028", 250),
		usageRecord("u3", "tenant-a", "20261028", 7),
		usageRecord("u4", "tenant-a", "20261029", 1_000_000),
		usageRecord("u5", "tenant-b", "20261028", 9_000_000_000),
	} {
		if err := s.Write(ctx, r); err != nil {
			t.Fatalf("Write(%s): %v", r.ID, err)
		}
	}
	if s.Full() {
		t.Error("Full with 5 pending records")
	}

	got, err := s.Totals(ctx, "tenant-a", "20261028")
	if err != nil {
		t.Fatalf("Totals: %v", err)
	}
	if want := (accounting.UsageTotals{Records: 3, CostNanos: 357}); got != want {
		t.Errorf("Totals(tenant-a, 20261028) = %+v, want %+v: Totals must see rows that were still queued", got, want)
	}
	for _, c := range []struct {
		tenant, day string
		want        accounting.UsageTotals
	}{
		{"tenant-a", "20261029", accounting.UsageTotals{Records: 1, CostNanos: 1_000_000}},
		{"tenant-b", "20261028", accounting.UsageTotals{Records: 1, CostNanos: 9_000_000_000}},
		{"tenant-c", "20261028", accounting.UsageTotals{}},
	} {
		got, err := s.Totals(ctx, c.tenant, c.day)
		if err != nil || got != c.want {
			t.Errorf("Totals(%s, %s) = %+v, %v; want %+v", c.tenant, c.day, got, err, c.want)
		}
	}

	// Every column survives.
	var rec accounting.UsageRecord
	scanOne(t, s, `select id, request_id, decision_id, tenant, day, model, attempt, outcome, partial,
		prompt_tokens, completion_tokens, usage_source, cost_nanos, cost_source, created_at
		from usage_records where id = 'u1'`, nil,
		&rec.ID, &rec.RequestID, &rec.DecisionID, &rec.Tenant, &rec.Day, &rec.Model, &rec.Attempt, &rec.Outcome, &rec.Partial,
		&rec.PromptTokens, &rec.CompletionTokens, &rec.UsageSource, &rec.CostNanos, &rec.CostSource, &rec.CreatedAt)
	want := usageRecord("u1", "tenant-a", "20261028", 100)
	if !rec.CreatedAt.Equal(want.CreatedAt) {
		t.Errorf("stored created_at = %v, want %v", rec.CreatedAt, want.CreatedAt)
	}
	rec.CreatedAt = want.CreatedAt // compared above; the Location may differ
	if rec != want {
		t.Errorf("stored record = %+v, want %+v", rec, want)
	}

	// A record that is written twice (a flush that is retried after an unknown
	// outcome) is stored once.
	if err := s.Write(ctx, usageRecord("u1", "tenant-a", "20261028", 100)); err != nil {
		t.Fatal(err)
	}
	if got, err := s.Totals(ctx, "tenant-a", "20261028"); err != nil || got.Records != 3 || got.CostNanos != 357 {
		t.Errorf("Totals after a repeated write = %+v, %v; want 3 records and 357", got, err)
	}
}

func TestCloseFlushesPendingRows(t *testing.T) {
	dsn := pgtest.Start(t)
	s := openStore(t, dsn, Options{FlushInterval: time.Hour})
	for i := range 5 {
		if err := s.Write(testContext(t), usageRecord(fmt.Sprintf("c%d", i), "tenant-a", "20261028", 1)); err != nil {
			t.Fatal(err)
		}
	}
	s.Close()

	again := openStore(t, dsn, Options{})
	if n := count(t, again, `select count(*) from usage_records`); n != 5 {
		t.Errorf("%d usage records after Close, want the 5 that were pending", n)
	}
	if err := s.Write(testContext(t), usageRecord("late", "tenant-a", "20261028", 1)); err == nil {
		t.Error("Write after Close succeeded")
	}
}

func TestBatchedWritesLandInBulk(t *testing.T) {
	dsn := pgtest.Start(t)
	s := openStore(t, dsn, Options{})
	const writers, each = 8, 150
	var wg sync.WaitGroup
	for w := range writers {
		wg.Add(1)
		go func() {
			defer wg.Done()
			for i := range each {
				if err := s.Write(context.Background(), usageRecord(fmt.Sprintf("w%d-%d", w, i), "tenant-a", "20261028", 3)); err != nil {
					t.Errorf("Write: %v", err)
					return
				}
			}
		}()
	}
	wg.Wait()
	got, err := s.Totals(testContext(t), "tenant-a", "20261028")
	if err != nil {
		t.Fatal(err)
	}
	if want := (accounting.UsageTotals{Records: writers * each, CostNanos: 3 * writers * each}); got != want {
		t.Errorf("Totals = %+v, want %+v", got, want)
	}
}

func testDecision(id, tenant string) *router.Decision {
	lambda := 0.25
	raw, p, lcb := 1.5, 0.75, 0.5
	n, k := 120, 90
	cost, p95 := 0.000123, 812.5
	return &router.Decision{
		ID:               id,
		Tenant:           tenant,
		CreatedAt:        time.Date(2026, 10, 28, 12, 30, 45, 123456789, time.UTC),
		PolicyVersion:    "7ae71d0e6deb",
		Strategy:         router.StrategyQuality,
		Profile:          "balanced",
		Chosen:           "gemini-2.5-flash",
		Reason:           "cheapest_above_threshold",
		Lambda:           &lambda,
		GuaranteeApplies: true,
		DecidedInUS:      314,
		Candidates: []router.Candidate{
			{Model: "gemini-2.5-pro", Eligible: true, IneligibleReasons: []string{}, Raw: &raw, P: &p, LCB: &lcb, BlockN: &n, BlockK: &k, ExpectedCostUSD: &cost, PredictedP95MS: &p95},
			{Model: "deepseek-v3.1-terminus", Eligible: false, IneligibleReasons: []string{"breaker_open"}},
		},
		Fallbacks: []string{"gemini-2.5-pro"},
		Attempts:  []router.Attempt{{Model: "gemini-2.5-flash", Outcome: router.OutcomeOK, DurationMS: 12.345678}},
	}
}

func TestDecisionPutAndGet(t *testing.T) {
	dsn := pgtest.Start(t)
	s := openStore(t, dsn, Options{FlushInterval: time.Hour})
	ctx := testContext(t)

	want := testDecision("dec_1", "tenant-a")
	if err := s.Put(want); err != nil {
		t.Fatalf("Put: %v", err)
	}

	// Served from the ring, before anything reaches the table.
	got, found, err := s.Get(ctx, "tenant-a", "dec_1")
	if err != nil || !found {
		t.Fatalf("Get = %v, %v, %v; want the decision", got, found, err)
	}
	if !reflect.DeepEqual(got, want) {
		t.Errorf("ring decision = %+v, want %+v", got, want)
	}

	// A second gateway on the same database has an empty ring, so it reads the
	// table. Closing the first one writes its queue.
	s.Close()
	other := openStore(t, dsn, Options{})
	got, found, err = other.Get(ctx, "tenant-a", "dec_1")
	if err != nil || !found {
		t.Fatalf("Get from the table = %v, %v, %v; want the decision", got, found, err)
	}
	if !reflect.DeepEqual(got, want) {
		gotJSON, _ := json.Marshal(got)
		wantJSON, _ := json.Marshal(want)
		t.Errorf("table decision = %s, want %s", gotJSON, wantJSON)
	}

	// Tenant scoped and unknown ids not found, from the table (dec_1) and from
	// the ring (dec_2, which is still queued).
	if err := other.Put(testDecision("dec_2", "tenant-a")); err != nil {
		t.Fatal(err)
	}
	for round := range 2 { // the second round finds dec_2 in the table as well
		for _, c := range []struct{ tenant, id string }{
			{"tenant-b", "dec_1"},
			{"tenant-b", "dec_2"},
			{"tenant-a", "dec_missing"},
		} {
			if d, found, err := other.Get(ctx, c.tenant, c.id); err != nil || found || d != nil {
				t.Errorf("round %d: Get(%s, %s) = %v, %v, %v; want not found", round, c.tenant, c.id, d, found, err)
			}
		}
		if _, found, err := other.Get(ctx, "tenant-a", "dec_2"); err != nil || !found {
			t.Errorf("round %d: Get(tenant-a, dec_2) = %v, %v; want found", round, found, err)
		}
		if err := other.batch.Flush(ctx); err != nil {
			t.Fatal(err)
		}
	}
	if n := count(t, other, `select count(*) from routing_decisions where tenant = 'tenant-a'`); n != 2 {
		t.Errorf("%d decisions in the table, want 2", n)
	}
}

func TestDecisionStoredColumns(t *testing.T) {
	s := openStore(t, pgtest.Start(t), Options{})
	d := testDecision("dec_cols", "tenant-a")
	if err := s.Put(d); err != nil {
		t.Fatal(err)
	}
	if err := s.batch.Flush(testContext(t)); err != nil {
		t.Fatal(err)
	}
	var tenant, policyVersion, strategy, profile, chosen, reason, recordID string
	var createdAt time.Time
	scanOne(t, s, `select tenant, created_at, policy_version, strategy, profile, chosen, reason, record->>'id'
		from routing_decisions where id = 'dec_cols'`, nil,
		&tenant, &createdAt, &policyVersion, &strategy, &profile, &chosen, &reason, &recordID)
	if tenant != "tenant-a" || policyVersion != d.PolicyVersion || strategy != d.Strategy || profile != d.Profile ||
		chosen != d.Chosen || reason != d.Reason || recordID != d.ID {
		t.Errorf("columns = %q %q %q %q %q %q %q", tenant, policyVersion, strategy, profile, chosen, reason, recordID)
	}
	if !createdAt.Equal(d.CreatedAt.Truncate(time.Microsecond)) {
		t.Errorf("created_at = %v, want %v", createdAt, d.CreatedAt)
	}
}

// smallPolicy loads the committed policy cut down to its first trees, after
// mutate has edited its generic tree, with its version recomputed.
func smallPolicy(t *testing.T, mutate func(tree map[string]any)) *policy.Policy {
	t.Helper()
	data, err := os.ReadFile(committedPolicyFile)
	if err != nil {
		t.Fatal(err)
	}
	dec := json.NewDecoder(bytes.NewReader(data))
	dec.UseNumber() // number text is part of the version hash
	var tree map[string]any
	if err := dec.Decode(&tree); err != nil {
		t.Fatal(err)
	}
	models, _ := tree["models"].([]any)
	for _, raw := range models {
		predictor, _ := raw.(map[string]any)["predictor"].(map[string]any)
		trees, _ := predictor["trees"].([]any)
		predictor["trees"] = trees[:3]
	}
	if mutate != nil {
		mutate(tree)
	}
	render := func() []byte {
		out, err := json.Marshal(tree)
		if err != nil {
			t.Fatal(err)
		}
		return out
	}
	tree["version"] = "000000000000"
	version, err := policy.CanonicalVersion(render())
	if err != nil {
		t.Fatal(err)
	}
	tree["version"] = version
	p, err := policy.Load(render())
	if err != nil {
		t.Fatal(err)
	}
	return p
}

func smallPolicyPair(t *testing.T) (a, b *policy.Policy) {
	t.Helper()
	a = smallPolicy(t, nil)
	b = smallPolicy(t, func(tree map[string]any) {
		models, _ := tree["models"].([]any)
		predictor, _ := models[0].(map[string]any)["predictor"].(map[string]any)
		predictor["init"] = json.Number("0.123")
	})
	if a.Version == b.Version {
		t.Fatalf("test policies share version %s", a.Version)
	}
	return a, b
}

func statusOf(t *testing.T, s *Store) map[string]string {
	t.Helper()
	rows, err := s.List(testContext(t))
	if err != nil {
		t.Fatalf("List: %v", err)
	}
	out := make(map[string]string, len(rows))
	for _, r := range rows {
		out[r.Version] = r.Status
	}
	return out
}

type activationRow struct {
	Version, Previous, Actor string
}

func activations(t *testing.T, s *Store) []activationRow {
	t.Helper()
	rows, err := s.pool.Query(testContext(t), `select version, coalesce(previous_version, ''), actor from policy_activations order by id`)
	if err != nil {
		t.Fatal(err)
	}
	defer rows.Close()
	var out []activationRow
	for rows.Next() {
		var r activationRow
		if err := rows.Scan(&r.Version, &r.Previous, &r.Actor); err != nil {
			t.Fatal(err)
		}
		out = append(out, r)
	}
	if err := rows.Err(); err != nil {
		t.Fatal(err)
	}
	return out
}

func TestPolicySaveActivateAndRollbackHistory(t *testing.T) {
	s := openStore(t, pgtest.Start(t), Options{})
	ctx := testContext(t)
	a, b := smallPolicyPair(t)

	if v, ok, err := s.ActiveVersion(ctx); err != nil || ok || v != "" {
		t.Fatalf("ActiveVersion of an empty database = %q, %v, %v; want none", v, ok, err)
	}
	for _, p := range []*policy.Policy{a, b} {
		if err := s.SavePolicy(ctx, p); err != nil {
			t.Fatalf("SavePolicy(%s): %v", p.Version, err)
		}
	}
	if got := statusOf(t, s); !reflect.DeepEqual(got, map[string]string{a.Version: "inactive", b.Version: "inactive"}) {
		t.Fatalf("statuses after saving = %v, want both inactive", got)
	}
	var predictorType string
	var createdAt time.Time
	scanOne(t, s, `select predictor_type, created_at from policies where version = $1`, []any{a.Version}, &predictorType, &createdAt)
	if predictorType != a.PredictorType || createdAt.UTC().Format(time.RFC3339) != a.CreatedAt {
		t.Errorf("predictor_type, created_at = %q, %v; want %q, %q", predictorType, createdAt, a.PredictorType, a.CreatedAt)
	}

	// Forward activations: the replaced policy is retired.
	if err := s.Activate(ctx, a.Version, "", policy.StatusRetired, "alice"); err != nil {
		t.Fatalf("Activate(a): %v", err)
	}
	if v, ok, err := s.ActiveVersion(ctx); err != nil || !ok || v != a.Version {
		t.Fatalf("ActiveVersion = %q, %v, %v; want %s", v, ok, err, a.Version)
	}
	if err := s.Activate(ctx, b.Version, a.Version, policy.StatusRetired, "bob"); err != nil {
		t.Fatalf("Activate(b): %v", err)
	}
	if got := statusOf(t, s); !reflect.DeepEqual(got, map[string]string{a.Version: "retired", b.Version: "active"}) {
		t.Fatalf("statuses after a then b = %v", got)
	}

	// A rollback is an activation of the older version: the replaced one is
	// inactive, not retired.
	if err := s.Activate(ctx, a.Version, b.Version, policy.StatusInactive, "carol"); err != nil {
		t.Fatalf("Activate(a) as a rollback: %v", err)
	}
	if got := statusOf(t, s); !reflect.DeepEqual(got, map[string]string{a.Version: "active", b.Version: "inactive"}) {
		t.Fatalf("statuses after the rollback = %v", got)
	}
	want := []activationRow{
		{a.Version, "", "alice"},
		{b.Version, a.Version, "bob"},
		{a.Version, b.Version, "carol"},
	}
	if got := activations(t, s); !reflect.DeepEqual(got, want) {
		t.Errorf("activation history = %+v, want %+v", got, want)
	}

	// Saving a policy again changes nothing: it must not reset the status.
	if err := s.SavePolicy(ctx, a); err != nil {
		t.Fatalf("SavePolicy(a) again: %v", err)
	}
	if got := statusOf(t, s); !reflect.DeepEqual(got, map[string]string{a.Version: "active", b.Version: "inactive"}) {
		t.Errorf("statuses after saving a again = %v", got)
	}
	if n := count(t, s, `select count(*) from policies`); n != 2 {
		t.Errorf("%d policies, want 2", n)
	}

	rows, err := s.List(ctx)
	if err != nil || len(rows) != 2 {
		t.Fatalf("List = %+v, %v", rows, err)
	}
	if rows[0].Version != a.Version || rows[0].PredictorType != a.PredictorType || rows[0].AddedAt.IsZero() || rows[0].CreatedAt.IsZero() {
		t.Errorf("List()[0] = %+v, want the policy added first, with times", rows[0])
	}
}

func TestPolicyActivateUnknownVersion(t *testing.T) {
	s := openStore(t, pgtest.Start(t), Options{})
	ctx := testContext(t)
	a, _ := smallPolicyPair(t)
	if err := s.SavePolicy(ctx, a); err != nil {
		t.Fatal(err)
	}
	if err := s.Activate(ctx, a.Version, "", policy.StatusRetired, "alice"); err != nil {
		t.Fatal(err)
	}

	err := s.Activate(ctx, "ffffffffffff", a.Version, policy.StatusRetired, "mallory")
	if !errors.Is(err, policy.ErrUnknownVersion) {
		t.Fatalf("Activate of an unknown version = %v, want policy.ErrUnknownVersion", err)
	}
	// The failed activation left no trace: a stays active, nothing was logged.
	if got := statusOf(t, s); !reflect.DeepEqual(got, map[string]string{a.Version: "active"}) {
		t.Errorf("statuses = %v, want a still active", got)
	}
	if got := activations(t, s); len(got) != 1 {
		t.Errorf("activation history = %+v, want only alice's", got)
	}
	if _, err := s.LoadRaw(ctx, "ffffffffffff"); !errors.Is(err, policy.ErrUnknownVersion) {
		t.Errorf("LoadRaw of an unknown version = %v, want policy.ErrUnknownVersion", err)
	}
}

func TestOnlyOnePolicyIsActive(t *testing.T) {
	s := openStore(t, pgtest.Start(t), Options{})
	ctx := testContext(t)
	a, b := smallPolicyPair(t)
	for _, p := range []*policy.Policy{a, b} {
		if err := s.SavePolicy(ctx, p); err != nil {
			t.Fatal(err)
		}
	}
	if err := s.Activate(ctx, a.Version, "", policy.StatusRetired, "alice"); err != nil {
		t.Fatal(err)
	}
	// The database refuses a second active row on its own.
	if _, err := s.pool.Exec(ctx, `update policies set status = 'active' where version = $1`, b.Version); err == nil {
		t.Fatal("two policies were active at once")
	}
	// Activating while the registry's idea of the previous policy is stale
	// still ends with exactly one active row.
	if err := s.Activate(ctx, b.Version, "", policy.StatusRetired, "bob"); err != nil {
		t.Fatalf("Activate(b) with a stale previous: %v", err)
	}
	if got := statusOf(t, s); !reflect.DeepEqual(got, map[string]string{a.Version: "retired", b.Version: "active"}) {
		t.Errorf("statuses = %v, want a retired and b active", got)
	}
}

func TestPolicyRawBytesRoundTrip(t *testing.T) {
	s := openStore(t, pgtest.Start(t), Options{})
	ctx := testContext(t)
	p, err := policy.LoadFile(committedPolicyFile)
	if err != nil {
		t.Fatal(err)
	}
	if err := s.SavePolicy(ctx, p); err != nil {
		t.Fatalf("SavePolicy: %v", err)
	}

	raw, err := s.LoadRaw(ctx, p.Version)
	if err != nil {
		t.Fatalf("LoadRaw: %v", err)
	}
	if !bytes.Equal(raw, p.Raw) {
		t.Fatalf("LoadRaw returned %d bytes that differ from the %d bytes saved", len(raw), len(p.Raw))
	}
	version, err := policy.CanonicalVersion(raw)
	if err != nil || version != p.Version {
		t.Errorf("CanonicalVersion(raw) = %q, %v; want %q", version, err, p.Version)
	}
	again, err := policy.Load(raw)
	if err != nil || again.Version != p.Version {
		t.Errorf("Load(raw) = %v, %v; want version %s", again, err, p.Version)
	}

	// The jsonb column is queryable but not the source of truth: it rewrites
	// the text, which would change the version hash.
	var artifactText, artifactVersion string
	scanOne(t, s, `select artifact::text, artifact->>'version' from policies where version = $1`, []any{p.Version}, &artifactText, &artifactVersion)
	if artifactVersion != p.Version {
		t.Errorf("artifact->>'version' = %q, want %q", artifactVersion, p.Version)
	}
	if artifactText == string(p.Raw) {
		t.Error("the jsonb text equals the raw bytes: this test no longer shows why the raw bytes are stored")
	}
}

func TestSavePolicyRejectsABadCreatedAt(t *testing.T) {
	s := openStore(t, pgtest.Start(t), Options{})
	a, _ := smallPolicyPair(t)
	bad := *a
	bad.CreatedAt = "last tuesday"
	err := s.SavePolicy(testContext(t), &bad)
	var ve *policy.ValidationError
	if !errors.As(err, &ve) || ve.Path != "created_at" {
		t.Fatalf("SavePolicy = %v, want a *policy.ValidationError at created_at", err)
	}
	if n := count(t, s, `select count(*) from policies`); n != 0 {
		t.Errorf("%d policies stored, want none", n)
	}
}

func TestPingAndClose(t *testing.T) {
	s := openStore(t, pgtest.Start(t), Options{})
	if err := s.Ping(testContext(t)); err != nil {
		t.Fatalf("Ping: %v", err)
	}
	s.Close()
	s.Close() // closing twice is fine
	if err := s.Ping(testContext(t)); err == nil {
		t.Error("Ping after Close succeeded")
	}
}

func TestOpenErrorsDoNotLeakTheDSN(t *testing.T) {
	const secret = "s3cret-pass-word"
	for name, dsn := range map[string]string{
		"unparsable":  "postgres://routeforge:" + secret + "@localhost:notaport/db",
		"unreachable": "postgres://routeforge:" + secret + "@127.0.0.1:1/db?sslmode=disable&connect_timeout=2",
	} {
		t.Run(name, func(t *testing.T) {
			_, err := Open(testContext(t), dsn, Options{})
			if err == nil {
				t.Fatal("Open succeeded")
			}
			if bytes.Contains([]byte(err.Error()), []byte(secret)) {
				t.Errorf("error %q contains the password", err)
			}
		})
	}
}
