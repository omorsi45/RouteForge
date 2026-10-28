-- The first schema. Policies keep their exact bytes next to
-- the jsonb copy: jsonb rewrites number text, which would change the version
-- hash.

create table schema_migrations (
    version int primary key,
    applied_at timestamptz not null default now()
);

create table policies (
    version text primary key,
    artifact jsonb not null,
    artifact_raw bytea not null,
    predictor_type text not null,
    created_at timestamptz not null,
    added_at timestamptz not null default now(),
    status text not null check (status in ('inactive', 'active', 'retired'))
);

-- At most one policy is active.
create unique index policies_one_active on policies (status) where status = 'active';

create table policy_activations (
    id bigserial primary key,
    version text not null references policies(version),
    previous_version text,
    actor text not null,
    activated_at timestamptz not null default now()
);

create table usage_records (
    id text primary key,
    request_id text not null,
    decision_id text not null,
    tenant text not null,
    day text not null,
    model text not null,
    attempt int not null,
    outcome text not null,
    partial boolean not null,
    prompt_tokens int not null,
    completion_tokens int not null,
    usage_source text not null,
    cost_nanos bigint not null,
    cost_source text not null,
    created_at timestamptz not null
);

create index usage_records_tenant_day on usage_records (tenant, day);

create table routing_decisions (
    id text primary key,
    tenant text not null,
    created_at timestamptz not null,
    policy_version text not null,
    strategy text not null,
    profile text not null,
    chosen text not null,
    reason text not null,
    record jsonb not null
);

create index routing_decisions_tenant_created on routing_decisions (tenant, created_at);
