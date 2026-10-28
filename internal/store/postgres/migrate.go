package postgres

import (
	"context"
	"embed"
	"fmt"
	"io/fs"
	"path"
	"slices"
	"strconv"
	"strings"

	"github.com/jackc/pgx/v5"
	"github.com/jackc/pgx/v5/pgxpool"
)

// migrationLockID is the advisory lock that serializes migrations, so that
// gateways starting together against an empty database do not race to create
// the same tables.
const migrationLockID = 727274

//go:embed migrations/*.sql
var migrationFiles embed.FS

// migration is one embedded SQL file, applied in one transaction.
type migration struct {
	version int
	name    string
	sql     string
}

// loadMigrations reads the .sql files of dir. A file name is a version
// number, an underscore and a description, such as 0001_init.sql. The versions
// must run 1, 2, 3 and so on, with no gap and no repeat; the result is in
// version order.
func loadMigrations(fsys fs.FS, dir string) ([]migration, error) {
	entries, err := fs.ReadDir(fsys, dir)
	if err != nil {
		return nil, fmt.Errorf("read migrations: %w", err)
	}
	var ms []migration
	for _, e := range entries {
		if e.IsDir() || !strings.HasSuffix(e.Name(), ".sql") {
			continue
		}
		version, err := parseMigrationName(e.Name())
		if err != nil {
			return nil, err
		}
		data, err := fs.ReadFile(fsys, path.Join(dir, e.Name()))
		if err != nil {
			return nil, fmt.Errorf("read migration %s: %w", e.Name(), err)
		}
		ms = append(ms, migration{version: version, name: e.Name(), sql: string(data)})
	}
	slices.SortFunc(ms, func(a, b migration) int { return a.version - b.version })
	for i, m := range ms {
		if i > 0 && m.version == ms[i-1].version {
			return nil, fmt.Errorf("migration %s: version %d is also used by %s", m.name, m.version, ms[i-1].name)
		}
		if m.version != i+1 {
			return nil, fmt.Errorf("migration %s: want version %d next: versions start at 1 and have no gaps", m.name, i+1)
		}
	}
	return ms, nil
}

// parseMigrationName returns the version of a file named like 0001_init.sql.
func parseMigrationName(name string) (int, error) {
	prefix, description, ok := strings.Cut(strings.TrimSuffix(name, ".sql"), "_")
	if !ok || description == "" {
		return 0, fmt.Errorf("migration %s: want a name like 0001_description.sql", name)
	}
	version, err := strconv.Atoi(prefix)
	if prefix == "" || strings.Trim(prefix, "0123456789") != "" || err != nil || version < 1 {
		return 0, fmt.Errorf("migration %s: the part before the underscore must be a version number from 1", name)
	}
	return version, nil
}

// migrate applies the embedded migrations that the database has not recorded
// yet, in order, each in its own transaction, under the advisory lock. Opening
// a database that is up to date changes nothing.
func migrate(ctx context.Context, pool *pgxpool.Pool) (err error) {
	ms, err := loadMigrations(migrationFiles, "migrations")
	if err != nil {
		return err
	}
	conn, err := pool.Acquire(ctx)
	if err != nil {
		return fmt.Errorf("acquire a connection: %w", err)
	}
	// The lock belongs to the session, so it has to be released on the
	// connection that took it.
	if _, err := conn.Exec(ctx, `select pg_advisory_lock($1)`, migrationLockID); err != nil {
		conn.Release()
		return fmt.Errorf("take the migration lock: %w", err)
	}
	defer func() {
		bg := context.WithoutCancel(ctx)
		if _, unlockErr := conn.Exec(bg, `select pg_advisory_unlock($1)`, migrationLockID); unlockErr != nil {
			// Do not hand a connection that may still hold the lock back to the
			// pool: closing the session releases it.
			_ = conn.Hijack().Close(bg)
			if err == nil {
				err = fmt.Errorf("release the migration lock: %w", unlockErr)
			}
			return
		}
		conn.Release()
	}()

	applied, err := appliedVersions(ctx, conn)
	if err != nil {
		return err
	}
	for _, m := range ms {
		if applied[m.version] {
			continue
		}
		if err := apply(ctx, conn, m); err != nil {
			return fmt.Errorf("migration %s: %w", m.name, err)
		}
	}
	return nil
}

// appliedVersions returns the versions recorded in schema_migrations. A
// database that has no such table yet has applied none.
func appliedVersions(ctx context.Context, conn *pgxpool.Conn) (map[int]bool, error) {
	var exists bool
	if err := conn.QueryRow(ctx, `select to_regclass('schema_migrations') is not null`).Scan(&exists); err != nil {
		return nil, fmt.Errorf("look for schema_migrations: %w", err)
	}
	applied := make(map[int]bool)
	if !exists {
		return applied, nil
	}
	rows, err := conn.Query(ctx, `select version from schema_migrations`)
	if err != nil {
		return nil, fmt.Errorf("read schema_migrations: %w", err)
	}
	defer rows.Close()
	for rows.Next() {
		var v int
		if err := rows.Scan(&v); err != nil {
			return nil, fmt.Errorf("read schema_migrations: %w", err)
		}
		applied[v] = true
	}
	if err := rows.Err(); err != nil {
		return nil, fmt.Errorf("read schema_migrations: %w", err)
	}
	return applied, nil
}

// apply runs one migration and records its version in the same transaction.
// 0001_init.sql creates schema_migrations itself, so the record comes after the
// SQL. The SQL file holds several statements; with no arguments pgx sends it as
// one simple query.
func apply(ctx context.Context, conn *pgxpool.Conn, m migration) error {
	tx, err := conn.Begin(ctx)
	if err != nil {
		return fmt.Errorf("begin: %w", err)
	}
	defer rollback(ctx, tx)
	if err := execMigration(ctx, tx, m); err != nil {
		return err
	}
	return tx.Commit(ctx)
}

func execMigration(ctx context.Context, tx pgx.Tx, m migration) error {
	if _, err := tx.Exec(ctx, m.sql); err != nil {
		return err
	}
	if _, err := tx.Exec(ctx, `insert into schema_migrations (version) values ($1)`, m.version); err != nil {
		return fmt.Errorf("record the version: %w", err)
	}
	return nil
}
