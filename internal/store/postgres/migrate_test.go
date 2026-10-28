package postgres

import (
	"strings"
	"testing"
	"testing/fstest"
)

func TestEmbeddedMigrationsAreOrdered(t *testing.T) {
	ms, err := loadMigrations(migrationFiles, "migrations")
	if err != nil {
		t.Fatalf("loadMigrations: %v", err)
	}
	if len(ms) == 0 {
		t.Fatal("no migrations are embedded")
	}
	for i, m := range ms {
		if m.version != i+1 {
			t.Errorf("migration %d has version %d, want %d: versions start at 1 and have no gaps", i, m.version, i+1)
		}
		if strings.TrimSpace(m.sql) == "" {
			t.Errorf("migration %d (%s) is empty", m.version, m.name)
		}
	}
	if ms[0].name != "0001_init.sql" {
		t.Errorf("first migration is %q, want 0001_init.sql", ms[0].name)
	}
}

func TestInitMigrationCreatesTheSpecTables(t *testing.T) {
	ms, err := loadMigrations(migrationFiles, "migrations")
	if err != nil {
		t.Fatal(err)
	}
	sql := strings.ToLower(ms[0].sql)
	for _, want := range []string{
		"create table schema_migrations",
		"create table policies",
		"artifact jsonb not null",
		"artifact_raw bytea not null",
		"create unique index policies_one_active",
		"create table policy_activations",
		"create table usage_records",
		"create index usage_records_tenant_day",
		"create table routing_decisions",
		"create index routing_decisions_tenant_created",
	} {
		if !strings.Contains(sql, want) {
			t.Errorf("0001_init.sql does not contain %q", want)
		}
	}
}

func TestLoadMigrationsOrdersByVersion(t *testing.T) {
	fsys := fstest.MapFS{
		"m/0010_later.sql":  {Data: []byte("select 10;")},
		"m/0002_second.sql": {Data: []byte("select 2;")},
		"m/0001_first.sql":  {Data: []byte("select 1;")},
		"m/README.md":       {Data: []byte("not a migration")},
	}
	ms, err := loadMigrations(fsys, "m")
	if err == nil {
		t.Fatalf("loadMigrations accepted a gap between versions 2 and 10: %+v", ms)
	}
	if !strings.Contains(err.Error(), "0010_later.sql") {
		t.Errorf("error %q does not name the migration after the gap", err)
	}

	delete(fsys, "m/0010_later.sql")
	ms, err = loadMigrations(fsys, "m")
	if err != nil {
		t.Fatalf("loadMigrations: %v", err)
	}
	if len(ms) != 2 || ms[0].version != 1 || ms[1].version != 2 || ms[1].sql != "select 2;" {
		t.Errorf("migrations = %+v, want versions 1 and 2 in order", ms)
	}
}

func TestLoadMigrationsRejectsBadNames(t *testing.T) {
	for name, files := range map[string]fstest.MapFS{
		"no version":        {"m/init.sql": {Data: []byte("select 1;")}},
		"version zero":      {"m/0000_zero.sql": {Data: []byte("select 1;")}},
		"duplicate version": {"m/0001_a.sql": {Data: []byte("select 1;")}, "m/0001_b.sql": {Data: []byte("select 2;")}},
		"no description":    {"m/0001_.sql": {Data: []byte("select 1;")}},
	} {
		t.Run(name, func(t *testing.T) {
			if ms, err := loadMigrations(files, "m"); err == nil {
				t.Fatalf("loadMigrations accepted %s: %+v", name, ms)
			}
		})
	}
}

func TestAdvisoryLockID(t *testing.T) {
	if migrationLockID != 727274 {
		t.Errorf("migrationLockID = %d, want 727274", migrationLockID)
	}
}
