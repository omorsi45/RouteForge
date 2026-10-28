// Package postgres keeps usage records, routing decisions and policies in
// PostgreSQL. Open applies the embedded migrations; usage records and
// decisions are written in batches by a bounded queue, so a request never waits
// for the database.
package postgres

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"log/slog"
	"strings"
	"sync"
	"time"

	"github.com/jackc/pgx/v5"
	"github.com/jackc/pgx/v5/pgxpool"

	"github.com/omorsi45/routeforge/internal/accounting"
	"github.com/omorsi45/routeforge/internal/policy"
	"github.com/omorsi45/routeforge/internal/router"
	"github.com/omorsi45/routeforge/internal/store"
)

// closeTimeout bounds how long Close waits to write what is still queued.
const closeTimeout = 15 * time.Second

// rollbackTimeout bounds the rollback of an abandoned transaction.
const rollbackTimeout = 5 * time.Second

// decisionRingSize is how many decisions Store keeps in memory for reads, as
// the in-memory store does.
const decisionRingSize = 10000

// Options configures a Store. A zero field takes its default.
type Options struct {
	// BatchSize is how many queued rows are written at once. Default 500.
	BatchSize int
	// FlushInterval is how long the oldest queued row waits for company before
	// the queue is written anyway. Default 200 ms.
	FlushInterval time.Duration
	// QueueSize is how many rows may be queued or being written before Full
	// reports true; it is accounting.usage_queue. Default 10,000.
	QueueSize int
	Logger    *slog.Logger // nil logs nothing
}

// Store implements accounting.UsageStore, store.DecisionStore and the policy
// persistence of policy.Registry on PostgreSQL. It is safe for concurrent use.
type Store struct {
	pool   *pgxpool.Pool
	log    *slog.Logger
	batch  *batcher[row]
	ring   *store.Ring
	secret string // the database password, kept out of error text

	closeOnce sync.Once
}

var (
	_ accounting.UsageStore = (*Store)(nil)
	_ store.DecisionStore   = (*Store)(nil)
)

// row is one queued write: a usage record or a decision.
type row struct {
	usage    *accounting.UsageRecord
	decision *decisionRow
}

type decisionRow struct {
	id, tenant                                       string
	createdAt                                        time.Time
	policyVersion, strategy, profile, chosen, reason string
	record                                           []byte // the decision as JSON
}

// Open connects to the database at dsn, applies the embedded migrations and
// starts the batching writer. The error never contains the DSN or the password.
func Open(ctx context.Context, dsn string, o Options) (*Store, error) {
	cfg, err := pgxpool.ParseConfig(dsn)
	if err != nil {
		// pgx's own text may quote the connection string, so none of it is used.
		return nil, errors.New("postgres: the DSN could not be parsed")
	}
	secret := cfg.ConnConfig.Password
	pool, err := pgxpool.NewWithConfig(ctx, cfg)
	if err != nil {
		return nil, fmt.Errorf("postgres: create the connection pool: %w", redact(err, secret))
	}
	if err := pool.Ping(ctx); err != nil {
		pool.Close()
		return nil, fmt.Errorf("postgres: connect: %w", redact(err, secret))
	}
	if err := migrate(ctx, pool); err != nil {
		pool.Close()
		return nil, fmt.Errorf("postgres: %w", redact(err, secret))
	}

	logger := o.Logger
	if logger == nil {
		logger = slog.New(slog.DiscardHandler)
	}
	s := &Store{pool: pool, log: logger, ring: store.NewRing(decisionRingSize), secret: secret}
	s.batch = newBatcher(batcherOptions{
		BatchSize:     o.BatchSize,
		FlushInterval: o.FlushInterval,
		QueueSize:     o.QueueSize,
		Logger:        logger,
	}, s.flushRows)
	return s, nil
}

// Close writes the rows still queued and closes the connections. It keeps
// trying while the database is down, until closeTimeout; the rows still queued
// then are dropped and logged with their count. It is safe to call more than
// once. Failures are logged.
func (s *Store) Close() {
	s.closeOnce.Do(func() {
		ctx, cancel := context.WithTimeout(context.Background(), closeTimeout)
		defer cancel()
		if err := s.batch.Close(ctx); err != nil {
			s.log.Error("postgres writer did not flush on close", "error", redact(err, s.secret))
		}
		s.pool.Close()
	})
}

// Ping checks that the database answers.
func (s *Store) Ping(ctx context.Context) error {
	if err := s.pool.Ping(ctx); err != nil {
		return fmt.Errorf("postgres ping: %w", redact(err, s.secret))
	}
	return nil
}

// rollback abandons tx. It runs on a context of its own, because the caller's
// may be the reason the transaction is being abandoned. After a commit it does
// nothing.
func rollback(ctx context.Context, tx pgx.Tx) {
	ctx, cancel := context.WithTimeout(context.WithoutCancel(ctx), rollbackTimeout)
	defer cancel()
	_ = tx.Rollback(ctx)
}

// redactedError is an error whose text has a secret blanked out.
type redactedError struct {
	msg string
	err error
}

func (e *redactedError) Error() string { return e.msg }
func (e *redactedError) Unwrap() error { return e.err }

// redact returns err, with secret blanked out of its text when it appears.
func redact(err error, secret string) error {
	if err == nil || secret == "" || !strings.Contains(err.Error(), secret) {
		return err
	}
	return &redactedError{msg: strings.ReplaceAll(err.Error(), secret, "xxxxx"), err: err}
}

const insertUsage = `insert into usage_records
	(id, request_id, decision_id, tenant, day, model, attempt, outcome, partial,
	 prompt_tokens, completion_tokens, usage_source, cost_nanos, cost_source, created_at)
	values ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14, $15)
	on conflict (id) do nothing`

const insertDecision = `insert into routing_decisions
	(id, tenant, created_at, policy_version, strategy, profile, chosen, reason, record)
	values ($1, $2, $3, $4, $5, $6, $7, $8, $9)
	on conflict (id) do nothing`

// flushRows writes a batch in one transaction. A row whose id is already stored
// is skipped, so a batch that is tried again after an unknown outcome adds
// nothing twice.
func (s *Store) flushRows(ctx context.Context, rows []row) error {
	tx, err := s.pool.Begin(ctx)
	if err != nil {
		return fmt.Errorf("begin: %w", redact(err, s.secret))
	}
	defer rollback(ctx, tx)

	var batch pgx.Batch
	for _, r := range rows {
		switch {
		case r.usage != nil:
			u := r.usage
			batch.Queue(insertUsage, u.ID, u.RequestID, u.DecisionID, u.Tenant, u.Day, u.Model, u.Attempt, u.Outcome, u.Partial,
				u.PromptTokens, u.CompletionTokens, u.UsageSource, u.CostNanos, u.CostSource, u.CreatedAt)
		case r.decision != nil:
			d := r.decision
			batch.Queue(insertDecision, d.id, d.tenant, d.createdAt, d.policyVersion, d.strategy, d.profile, d.chosen, d.reason, d.record)
		}
	}
	results := tx.SendBatch(ctx, &batch)
	for range batch.Len() {
		if _, err := results.Exec(); err != nil {
			_ = results.Close()
			return fmt.Errorf("insert: %w", redact(err, s.secret))
		}
	}
	if err := results.Close(); err != nil {
		return fmt.Errorf("insert: %w", redact(err, s.secret))
	}
	if err := tx.Commit(ctx); err != nil {
		return fmt.Errorf("commit: %w", redact(err, s.secret))
	}
	return nil
}

// Write implements accounting.UsageStore. It queues the record and returns; the
// queue accepts a record even when Full, so that a request admitted before the
// queue filled up can still bill itself. It fails only after Close.
func (s *Store) Write(_ context.Context, r accounting.UsageRecord) error {
	if err := s.batch.Add(row{usage: &r}); err != nil {
		return fmt.Errorf("write usage record %s: %w", r.ID, err)
	}
	return nil
}

// Full implements accounting.UsageStore: the queue holds as many rows as
// Options.QueueSize allows.
func (s *Store) Full() bool { return s.batch.Full() }

// Totals implements accounting.UsageStore. It writes the queued rows first, so
// the answer includes every record written before the call.
func (s *Store) Totals(ctx context.Context, tenant, day string) (accounting.UsageTotals, error) {
	if err := s.batch.Flush(ctx); err != nil {
		return accounting.UsageTotals{}, fmt.Errorf("write queued usage records: %w", err)
	}
	var t accounting.UsageTotals
	err := s.pool.QueryRow(ctx,
		`select count(*), coalesce(sum(cost_nanos), 0)::bigint from usage_records where tenant = $1 and day = $2`,
		tenant, day).Scan(&t.Records, &t.CostNanos)
	if err != nil {
		return accounting.UsageTotals{}, fmt.Errorf("usage totals of %q on %s: %w", tenant, day, redact(err, s.secret))
	}
	return t, nil
}

// Put implements store.DecisionStore. The decision is kept in the in-memory
// ring at once and queued for the table; when the queue is full it stays in the
// ring only and Put returns an error, because a decision is worth less than a
// usage record.
func (s *Store) Put(d *router.Decision) error {
	if d == nil {
		return errors.New("store decision: decision is nil")
	}
	record, err := json.Marshal(d)
	if err != nil {
		return fmt.Errorf("store decision %s: encode: %w", d.ID, err)
	}
	if err := s.ring.Put(d); err != nil {
		return err
	}
	err = s.batch.TryAdd(row{decision: &decisionRow{
		id: d.ID, tenant: d.Tenant, createdAt: d.CreatedAt,
		policyVersion: d.PolicyVersion, strategy: d.Strategy, profile: d.Profile, chosen: d.Chosen, reason: d.Reason,
		record: record,
	}})
	if err != nil {
		return fmt.Errorf("store decision %s: %w", d.ID, err)
	}
	return nil
}

// Get implements store.DecisionStore: the ring first, then the table. A
// decision of another tenant is not found.
func (s *Store) Get(ctx context.Context, tenant, id string) (*router.Decision, bool, error) {
	if d, found, err := s.ring.Get(ctx, tenant, id); err != nil || found {
		return d, found, err
	}
	var record []byte
	err := s.pool.QueryRow(ctx, `select record from routing_decisions where id = $1 and tenant = $2`, id, tenant).Scan(&record)
	if errors.Is(err, pgx.ErrNoRows) {
		return nil, false, nil
	}
	if err != nil {
		return nil, false, fmt.Errorf("read decision %s: %w", id, redact(err, s.secret))
	}
	var d router.Decision
	if err := json.Unmarshal(record, &d); err != nil {
		return nil, false, fmt.Errorf("decode decision %s: %w", id, err)
	}
	return &d, true, nil
}

// PolicyRow is one stored policy, without its document.
type PolicyRow struct {
	Version       string
	PredictorType string
	CreatedAt     time.Time // from the artifact
	AddedAt       time.Time // when it was stored
	Status        string    // policy.StatusInactive, StatusActive or StatusRetired
}

// SavePolicy stores p as inactive: its exact bytes in artifact_raw, which are
// the source of truth, and a jsonb copy for queries. A policy whose version is
// already stored is left as it is, status included.
func (s *Store) SavePolicy(ctx context.Context, p *policy.Policy) error {
	if p == nil || len(p.Raw) == 0 {
		return errors.New("save policy: the policy has no document bytes")
	}
	created, err := time.Parse(time.RFC3339, p.CreatedAt)
	if err != nil {
		return &policy.ValidationError{Path: "created_at", Problem: fmt.Sprintf("must be an RFC 3339 timestamp, got %q", p.CreatedAt)}
	}
	_, err = s.pool.Exec(ctx,
		`insert into policies (version, artifact, artifact_raw, predictor_type, created_at, status)
		 values ($1, $2, $3, $4, $5, 'inactive')
		 on conflict (version) do nothing`,
		p.Version, p.Raw, p.Raw, p.PredictorType, created)
	if err != nil {
		return fmt.Errorf("save policy %s: %w", p.Version, redact(err, s.secret))
	}
	return nil
}

// Activate makes version the active policy in one transaction: the policy that
// is active now takes previousStatus (policy.StatusRetired after an
// activation, policy.StatusInactive after a rollback), version becomes
// active and an activation row records previous and actor. previous is what
// the caller believes was active; whichever row really is active is demoted, so
// that one policy is active whatever the caller saw. It returns an error
// wrapping policy.ErrUnknownVersion for a version that was never saved, and
// then changes nothing.
func (s *Store) Activate(ctx context.Context, version, previous, previousStatus, actor string) error {
	if previousStatus != policy.StatusInactive && previousStatus != policy.StatusRetired {
		return fmt.Errorf("activate policy %s: the replaced policy cannot take status %q", version, previousStatus)
	}
	tx, err := s.pool.Begin(ctx)
	if err != nil {
		return fmt.Errorf("activate policy %s: %w", version, redact(err, s.secret))
	}
	defer rollback(ctx, tx)

	if _, err := tx.Exec(ctx, `update policies set status = $1 where status = 'active' and version <> $2`, previousStatus, version); err != nil {
		return fmt.Errorf("activate policy %s: demote the active policy: %w", version, redact(err, s.secret))
	}
	tag, err := tx.Exec(ctx, `update policies set status = 'active' where version = $1`, version)
	if err != nil {
		return fmt.Errorf("activate policy %s: %w", version, redact(err, s.secret))
	}
	if tag.RowsAffected() == 0 {
		return fmt.Errorf("activate policy: %w: %s", policy.ErrUnknownVersion, version)
	}
	if _, err := tx.Exec(ctx,
		`insert into policy_activations (version, previous_version, actor) values ($1, nullif($2, ''), $3)`,
		version, previous, actor); err != nil {
		return fmt.Errorf("activate policy %s: record the activation: %w", version, redact(err, s.secret))
	}
	if err := tx.Commit(ctx); err != nil {
		return fmt.Errorf("activate policy %s: %w", version, redact(err, s.secret))
	}
	return nil
}

// ActiveVersion returns the version of the active policy, and false when none
// is active.
func (s *Store) ActiveVersion(ctx context.Context) (string, bool, error) {
	var version string
	err := s.pool.QueryRow(ctx, `select version from policies where status = 'active'`).Scan(&version)
	if errors.Is(err, pgx.ErrNoRows) {
		return "", false, nil
	}
	if err != nil {
		return "", false, fmt.Errorf("read the active policy: %w", redact(err, s.secret))
	}
	return version, true, nil
}

// LoadRaw returns the exact bytes a policy was saved with, so that its version
// hash still verifies. A version that was never saved gives an error wrapping
// policy.ErrUnknownVersion.
func (s *Store) LoadRaw(ctx context.Context, version string) ([]byte, error) {
	var raw []byte
	err := s.pool.QueryRow(ctx, `select artifact_raw from policies where version = $1`, version).Scan(&raw)
	if errors.Is(err, pgx.ErrNoRows) {
		return nil, fmt.Errorf("load policy: %w: %s", policy.ErrUnknownVersion, version)
	}
	if err != nil {
		return nil, fmt.Errorf("load policy %s: %w", version, redact(err, s.secret))
	}
	return raw, nil
}

// List returns every stored policy, oldest stored first.
func (s *Store) List(ctx context.Context) ([]PolicyRow, error) {
	rows, err := s.pool.Query(ctx, `select version, predictor_type, created_at, added_at, status from policies order by added_at, version`)
	if err != nil {
		return nil, fmt.Errorf("list policies: %w", redact(err, s.secret))
	}
	defer rows.Close()
	var out []PolicyRow
	for rows.Next() {
		var r PolicyRow
		if err := rows.Scan(&r.Version, &r.PredictorType, &r.CreatedAt, &r.AddedAt, &r.Status); err != nil {
			return nil, fmt.Errorf("list policies: %w", redact(err, s.secret))
		}
		out = append(out, r)
	}
	if err := rows.Err(); err != nil {
		return nil, fmt.Errorf("list policies: %w", redact(err, s.secret))
	}
	return out, nil
}
