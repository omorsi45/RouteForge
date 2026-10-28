# 0004: Tenants in YAML

Status: accepted

## Context

The gateway needs tenants: names, API key digests, a routing profile, deadlines, output caps, rate limits, daily budgets and the models each may use. They could live in PostgreSQL with an admin API to manage them, or in the configuration file.

Most of that data is small and changes rarely. It decides who may spend money, so changes to it should be reviewed. The first release has to run on a single instance with no database at all.

## Decision

Tenants are declared in the YAML configuration.

- Only the SHA-256 digest of an API key is stored. `routeforge keygen` prints a key and its digest, and the key is shown once.
- A tenant may list several digests, which makes rotation a two-step edit.
- `routeforge validate-config` checks every tenant, and CI runs it on every shipped config. It rejects duplicate names, duplicate digests, an unknown profile, an unknown allowed model, a rate limit above its cap and a budget that is not a whole number of nano-USD.
- Admin keys are separate digests under `admin`. An admin key may not also be a tenant key.
- The admin API manages policies only. Tenants, and experiment metadata (versioned JSON under `bench/results/`), do not live in PostgreSQL.
- What a tenant has spent is state, not configuration. It lives in the limiter backend, in memory or in Redis.

## Consequences

- Tenants are reviewable and diffable, and a bad change fails validation before it ships.
- There is no table to bootstrap and no seed secret in a database.
- Applying a change needs a restart and a copy of the file on every replica. There is no hot reload of tenants.
- There is no self-service key management and no per-tenant admin UI. An operator edits a file.
- A deployment with many tenants, or with tenants that sign up at run time, would need a store and an API for them. That would supersede this decision.
