# Operations

How to configure, run, deploy and observe RouteForge, and how to reproduce its results. It states configured defaults and the constants of the design. It states no measured result: those are in [benchmarks.md](benchmarks.md).

- [Commands](#commands)
- [Configuration reference](#configuration-reference)
- [API details](#api-details)
- [Keys, tenants and the admin API](#keys-tenants-and-the-admin-api)
- [Deployment](#deployment)
- [Storage and limiter backends](#storage-and-limiter-backends)
- [Accounting](#accounting)
- [Reliability](#reliability)
- [Observability](#observability)
- [Tests that need extra setup](#tests-that-need-extra-setup)
- [Reproducing the results](#reproducing-the-results)
- [Known limits](#known-limits)

## Commands

| Command | Does |
| --- | --- |
| `routeforge serve --config <file> [--listen <addr>]` | Runs the gateway. `--listen` overrides `server.listen`. |
| `routeforge validate-config --config <file>` | Loads the config, the price table and, when `policy.path` is set, the policy. Prints `config OK` with counts, or every problem with its YAML path. `make validate-configs` runs it on every shipped config. |
| `routeforge keygen` | Prints a new API key and its SHA-256 digest. The key is shown once. |
| `routeforge healthcheck [--url <url>] [--timeout <d>]` | Probes a running gateway (default `http://127.0.0.1:8080/healthz`, 2s) and exits non-zero if it is not healthy. The container image uses it, because the image has no curl. |
| `routeforge version` | Prints the version, the Go version, the platform and the commit. |

On SIGINT or SIGTERM the gateway drains. `/readyz` answers 503 with `{"status":"draining"}`, in-flight requests get up to `server.shutdown_timeout` to finish, and then the usage records and decisions still queued are written and the last spans are exported.

There is no config reload. Restart the gateway to apply a change. Logs are JSON lines on stderr.

## Configuration reference

The file is one YAML document. Unknown keys are an error. Durations are Go durations such as `500ms` or `2s`. Relative file paths (`policy.path`, `pricing.file`, `providers[].replay.file`, `providers[].replay.latency_profiles`) resolve against the directory of the config file, so a config validates from any working directory.

Zero means "use the default" for every numeric field that has one. The default replaces an explicit `0`, so a value of 0 cannot be set for them. For example `queue_size: 0` (no waiting line), `latency_prior.jitter: 0` and `retry_budget_min: 0` cannot be expressed. The fields where 0 has its own meaning are the tenant limits `rpm`, `tpm` and `daily_budget_usd` (0 is unlimited) and the settings of the fake provider, where 0 is no fault.

### `server`

| Key | Default | Meaning |
| --- | --- | --- |
| `listen` | `:8080` | Listen address. |
| `read_header_timeout` | `5s` | Time allowed to read request headers. |
| `idle_timeout` | `120s` | Idle keep-alive connection timeout. |
| `max_header_bytes` | `1048576` | Largest request header (1 MiB). |
| `max_body_bytes` | `2097152` | Largest request body (2 MiB). A larger one gets 413 `request_too_large`. |
| `default_deadline` | `120s` | Largest deadline of any request. |
| `shutdown_timeout` | `30s` | How long a drain waits for in-flight requests. |
| `max_inflight` | `1024` | Chat and explain requests served at once. More get 503 `overloaded`. |

There is no global write timeout, because streams outlive any fixed value. Each request has its own deadline.

### `defaults`

Values inherited by tenants that leave them unset.

| Key | Default | Meaning |
| --- | --- | --- |
| `profile` | `balanced` | Routing profile: `strict`, `balanced` or `economy`. |
| `max_deadline` | `server.default_deadline` | Tenant deadline cap. |
| `max_output_tokens` | `4096` | Tenant output cap. |

### `tenants[]`

At least one tenant is required.

| Key | Default | Meaning |
| --- | --- | --- |
| `name` | required | Lowercase letters, digits, `_` and `-`, up to 64 characters, unique. |
| `api_key_sha256` | required | One or more SHA-256 digests of API keys, as 64 lowercase hex characters, unique across tenants and admin keys. |
| `profile` | `defaults.profile` | The profile when the request has no `X-RouteForge-Profile`. |
| `max_deadline` | `defaults.max_deadline` | Largest deadline for this tenant. |
| `max_output_tokens` | `defaults.max_output_tokens` | Cap on `max_tokens`. The capped value is always sent upstream. |
| `allow_strategy_override` | `false` | Whether the tenant may send `X-RouteForge-Strategy`. |
| `allowed_models` | empty | Configured models the tenant may use. Empty allows all. |
| `rpm` | `0` | Requests per minute. 0 is unlimited. At most 100,000,000. |
| `tpm` | `0` | Tokens per minute (estimated prompt tokens plus the output cap). 0 is unlimited. At most 100,000,000. |
| `daily_budget_usd` | `0` | Spend allowed per UTC day. 0 is no budget. A whole number of nano-USD (at most nine decimals) and at most 2^50 nano-USD, about 1.1 million USD. |

### `providers[]`

At least one provider is required.

| Key | Default | Meaning |
| --- | --- | --- |
| `name` | required | Same rules as a tenant name, unique. |
| `type` | required | `openai`, `anthropic`, `fake` or `replay`. `openai` also covers any OpenAI-compatible server, such as Ollama, vLLM or Groq. |
| `base_url` | `https://api.openai.com/v1` for `openai`, `https://api.anthropic.com` for `anthropic` | Absolute http or https URL. Not used by `fake` and `replay`. Provider URLs come only from config. |
| `api_key_env` | none | Name of the environment variable that holds the provider key. Required for `anthropic`. If set, the variable must be non-empty at startup. The key is never logged and is redacted from errors. |
| `max_tokens_param` | `max_completion_tokens` for `openai` | `max_completion_tokens` or `max_tokens`, the parameter the upstream expects. Use `max_tokens` for Ollama, vLLM and Groq. |
| `fake` | | Settings of a `fake` provider, below. |
| `replay` | | Settings of a `replay` provider, below. |

`providers[].fake` makes a deterministic provider that can inject faults. A reply is a sequence of words, one per token. Its latency is `ttft` plus `per_token` for each word, with jitter. Output and faults depend only on the seed, the request and the order of calls.

| Key | Default | Meaning |
| --- | --- | --- |
| `seed` | `0` | Seed of the words, the jitter and the fault draws. |
| `ttft` | `0` | Time to first token. |
| `per_token` | `0` | Time per word. |
| `jitter` | `0` | Fraction of uniform noise on each wait, in [0, 1). |
| `output_tokens` | `16` | Words in each reply, at most the request's `max_tokens`. |
| `error_rate` | `0` | Fraction of calls that fail, in [0, 1]. |
| `error_kind` | `unavailable` | Failure kind: `rate_limited`, `unavailable`, `timeout`, `bad_request`, `auth`, `context_length` or `internal`. |
| `retry_after` | `0` | `Retry-After` sent with a `rate_limited` failure. |
| `disconnect_rate` | `0` | Fraction of streams cut short. |
| `disconnect_after` | `1` | Words delivered before a cut stream ends. |
| `omit_usage_rate` | `0` | Fraction of replies that leave usage out. |
| `over_report_tokens` | `0` | Completion tokens added to the usage a reply reports. |

`providers[].replay` serves recorded outputs with simulated latency. It is used by the benchmarks and needs `make data` first.

| Key | Default | Meaning |
| --- | --- | --- |
| `file` | required | Recorded records, a JSON-lines file. |
| `latency_profiles` | required | A latency profile file such as `bench/latency_profiles.yaml`. |
| `time_scale` | `1` | Multiplies the simulated latency, in (0, 1]. |
| `seed` | `20261026` | Seed of the latency jitter. |

### `models[]`

At least one model is required.

| Key | Default | Meaning |
| --- | --- | --- |
| `name` | required | The name clients send. Up to 128 characters from letters, digits and `. _ : / -`. Unique. `routeforge-auto` is reserved. |
| `provider` | required | A `providers[].name`. |
| `upstream_model` | `name` | The model id sent to the provider. |
| `policy_name` | `name` | The model's name in the policy and in the price file. Unique. Quality, static, strongest and cheapest routing consider only models whose `policy_name` is in the active policy. |
| `strongest` | `false` | Marks the model that the router uses when no policy is active. With a policy, the policy's `strongest_model` decides. |
| `input_per_mtok`, `output_per_mtok` | none | USD per 1M tokens. Set both or neither. Both override the price file. |
| `context_window` | `0` | Context size in tokens. 0 means unknown and disables the `context_window` check. The shipped configs leave it unset because no verified window sizes are recorded in the repository. |
| `latency_prior.ttft_ms` | `500` | Time to first token assumed before any call is observed. |
| `latency_prior.decode_tokens_per_s` | `50` | Decode speed assumed before any call is observed. |
| `latency_prior.jitter` | `0.1` | Noise of the prior, in [0, 1). |
| `max_concurrency` | `64` | Requests running at once on this model. |
| `queue_size` | `128` | Requests that may wait for a slot. |
| `queue_timeout` | `2s` | Longest wait for a slot. |

Every model must be priced. A model uses its own `input_per_mtok` and `output_per_mtok` when both are set. Otherwise its price comes from `pricing.file` by `policy_name`. A model with neither fails `validate-config`.

### `reliability`

| Key | Default | Meaning |
| --- | --- | --- |
| `max_attempts` | `3` | Upstream calls per request, the first included. Between 1 and 10. |
| `backoff_base` | `100ms` | Base of the full-jitter exponential backoff. |
| `backoff_cap` | `2s` | Cap of the backoff. |
| `retry_budget_ratio` | `0.2` | Retries allowed per request in the window, in (0, 1]. |
| `retry_budget_window` | `10s` | Sliding window of the retry budget. |
| `retry_budget_min` | `3` | Retries allowed on top of the ratio. See [Reliability](#reliability). |
| `breaker.consecutive_failures` | `5` | Failures in a row that open a model's breaker. |
| `breaker.error_rate` | `0.5` | Error rate that opens it, once `min_requests` requests are in the window. In (0, 1]. |
| `breaker.min_requests` | `10` | Requests in the window before the error rate counts. |
| `breaker.window` | `30s` | Window of the error rate. |
| `breaker.open_for` | `15s` | How long an open breaker refuses requests. |
| `breaker.half_open_probes` | `2` | Probes a half-open breaker admits at once. |

### Other sections

| Key | Default | Meaning |
| --- | --- | --- |
| `policy.path` | none | The trained policy to load and activate at startup. Without it the registry starts empty. |
| `pricing.file` | none | The price table, `deploy/prices.yaml` in the shipped configs. |
| `accounting.fail_mode` | `closed` | `closed` or `open`. See [Accounting](#accounting). |
| `accounting.usage_queue` | `10000` | Usage records that may wait to be written before the gateway stops admitting work. Applies to the PostgreSQL writer. The in-memory store has no queue and ignores it. |
| `storage.type` | `memory` | `memory` or `postgres`. |
| `storage.dsn_env` | none | Name of the environment variable that holds the PostgreSQL connection string. Required for `postgres`. |
| `limiter.type` | `memory` | `memory` or `redis`. |
| `limiter.url_env` | none | Name of the environment variable that holds the Redis URL. Required for `redis`. |
| `admin.api_key_sha256` | none | SHA-256 digests of admin keys. Without any, the `/admin` routes answer 404. |

The environment variables a deployment may need are the ones named by `api_key_env`, `dsn_env` and `url_env`, plus `OTEL_EXPORTER_OTLP_ENDPOINT` for tracing. A connection string is read at startup and appears in no log and no error. An error names the variable.

### Prices

`deploy/prices.yaml` holds `effective_date`, `source`, `currency` (only `USD`) and the list price of each model in USD per 1M tokens. A price must be a whole number of nano-USD per token, which means at most three decimals. The loader rejects others.

## API details

### Chat completions

`POST /v1/chat/completions` takes an OpenAI Chat Completions body.

- Supported: `model`, `messages` (content as a string or an array of `{"type":"text"}` parts; `developer` counts as `system`), `max_tokens` or `max_completion_tokens`, `temperature` (0 to 2), `top_p` (0 to 1), `stop` (a string or up to four), `stream`, `stream_options.include_usage`, `user`, `n` of 1 and `response_format` of type `text`.
- Not supported: tools, `tool_choice`, `functions`, `function_call`, `logprobs`, `top_logprobs`, `audio`, modalities other than text, non-text content parts and `tool_calls` in messages. They get 400 `unsupported_parameter`. Other unknown fields are ignored. Trailing data after the JSON value is rejected.
- `model` is `routeforge-auto` or a configured model name that the tenant may use.

Request headers:

| Header | Meaning |
| --- | --- |
| `Authorization: Bearer <key>` | The tenant's API key. |
| `X-RouteForge-Profile` | `strict`, `balanced` or `economy`. Default: the tenant's profile. |
| `X-RouteForge-Deadline-Ms` | A positive integer. Lowers the deadline; it never raises it above `min(tenant.max_deadline, server.default_deadline)`. |
| `X-RouteForge-Strategy` | `quality`, `strongest`, `cheapest`, `static` or `latency`. Only for tenants with `allow_strategy_override`, otherwise 403. |
| `X-Request-Id` | Kept if it matches `[A-Za-z0-9._-]{1,128}`, else the gateway makes one. It is on every response. |

Response headers, sent before the body: `X-RouteForge-Model` (the model that served the request), `X-RouteForge-Decision-Id`, `X-RouteForge-Policy-Version` (the version, or `none`), `X-RouteForge-Reason` and `X-Request-Id`.

A non-streaming response is the OpenAI shape plus `routeforge: {decision_id, reason, attempts, cost_usd, cost_source}`. `cost_source` is `reported` when every attempt was priced from usage the provider reported, and `estimated` otherwise.

A streaming response is server-sent events, `data: <chunk>`, ending with `data: [DONE]`. With `include_usage` the final usage chunk is forwarded with `choices: []`. Streams do not carry `cost_usd` or `cost_source`, because the chunk format has no place for them. The usage record and `GET /admin/accounting` have them. If the upstream fails after the first byte has reached the client, the gateway sends `data: {"error":{...,"code":"upstream_stream_interrupted"}}` and closes without `[DONE]`. If the request deadline passes, the event carries `deadline_exceeded`.

### Other endpoints

- `GET /v1/models` lists `routeforge-auto` and the configured models the tenant may use.
- `POST /v1/routing/explain` takes a chat body and returns the decision. It calls no provider, reserves nothing and stores nothing, and it shares the in-flight limit.
- `GET /v1/routing/decisions/{id}` returns a stored decision of the calling tenant.
- `GET /healthz` says the process is up. `GET /readyz` says the gateway can serve: a policy is active (when `policy.path` is set), and PostgreSQL and Redis answer when they are configured. A failing backend shows a fixed text, `storage unavailable` or `limiter unavailable`, and the detail goes to the log.
- `GET /metrics` serves Prometheus metrics.

### Error codes

Errors use `{"error":{"message","type","code"}}`. A 429 or 503 below always carries `Retry-After` of at least one second.

| Status | Code | When |
| --- | --- | --- |
| 400 | `invalid_request` | A malformed body, header or field, an unknown model or a strategy that is not available. |
| 400 | `unsupported_parameter` | A feature the gateway does not implement. |
| 400 | `context_length_exceeded` | The upstream reported the context window exceeded. |
| 401 | `invalid_api_key` | A missing or unknown key. The answer names no tenant. |
| 403 | `strategy_override_forbidden` | The tenant may not override the strategy. |
| 403 | `model_not_allowed` | The tenant may not use the model. |
| 404 | `not_found` | An unknown or foreign decision, an unknown policy version, or an `/admin` route with no admin key configured. |
| 409 | `no_previous_policy` | A rollback with nothing to roll back to. |
| 413 | `request_too_large` | The body or an uploaded policy is over its limit. |
| 429 | `rate_limited` | The tenant's RPM or TPM bucket, or the upstream's rate limit. |
| 429 | `budget_exceeded` | A reservation was refused at an attempt. `Retry-After` is the time to the next UTC midnight. |
| 500 | `internal_error` | A failure of the gateway's own. |
| 502 | `upstream_error` | The upstream failed. The kind is in the message, never the upstream text. |
| 502 | `upstream_stream_interrupted` | Sent as an event when a stream breaks after the first byte. |
| 503 | `no_eligible_model` | Every candidate is ineligible. The reasons are in the message. A model whose reservation does not fit the remaining budget is ineligible with the reason `budget`. |
| 503 | `overloaded` | The in-flight limit is reached, or every candidate's queue is full. |
| 503 | `accounting_unavailable` | A budget, limiter or usage backend cannot be used and `fail_mode` is `closed`, or the usage queue is full. |
| 503 | `storage_unavailable` | The policy store did not take an admin write. |
| 504 | `deadline_exceeded` | The request deadline passed. |

## Keys, tenants and the admin API

Tenants and their key digests are declared in the YAML config, not in a database ([ADR 0004](adr/0004-tenants-in-yaml.md)).

**Making a key.** `routeforge keygen` prints `key: rf_...` and `sha256: ...`. Put the digest in the tenant's `api_key_sha256` and give the key to the client. The gateway stores and compares digests only.

**Rotating a key.** A tenant may list several digests. Add the new digest beside the old, restart the gateway, move the client to the new key, then remove the old digest and restart again. Two keys are valid in between.

**Admin keys.** `admin.api_key_sha256` lists the digests of admin keys. An admin key may not also be a tenant key. Admin routes share the listener with the tenant API, so restrict `/admin` at your proxy and keep the admin key out of client hands. Without an admin key the admin routes answer 404.

| Route | Does |
| --- | --- |
| `GET /admin/policies` | Lists stored policies (version, creation time, status, predictor type) and the activation log. |
| `POST /admin/policies` | Validates and stores a policy document, inactive. Up to 8 MiB. Answers 201 with its version. A policy with a model that has no configured model is rejected. |
| `POST /admin/policies/{version}/activate` | Makes a stored policy active. In-flight requests keep their decisions. |
| `POST /admin/policies/rollback` | Re-activates the version that was active before the current one. |
| `GET /admin/accounting?tenant=<name>&day=<yyyymmdd>` | Spend, reservations, overshoot and usage totals of one tenant on one UTC day, in USD and in nano-USD. The day defaults to today. |

## Deployment

### The image

`make docker` runs `docker build -f deploy/Dockerfile --build-arg VERSION=$(git describe --always --dirty) -t routeforge:local .`. The build uses a pinned `golang:1.27.3-bookworm` image with `GOTOOLCHAIN=local`, `CGO_ENABLED=0` and `-trimpath`. The runtime is the pinned `gcr.io/distroless/static-debian12:nonroot` image, so it has no shell and runs as `nonroot`. The base images are pinned by tag and digest. The image holds the binary, the trained policy, the price table and `deploy/configs/compose.yaml` under `/etc/routeforge/`, in the layout of the repository so that the same relative paths resolve. It exposes 8080 and its `HEALTHCHECK` runs `routeforge healthcheck`. To serve another config, mount it and override the entrypoint.

### The compose stack

`docker compose -f deploy/docker-compose.yml up --build` starts four services:

- `routeforge`, with a read-only root filesystem and `no-new-privileges`, using PostgreSQL storage and the Redis limiter. The connection strings reach it through `ROUTEFORGE_POSTGRES_DSN` and `ROUTEFORGE_REDIS_URL`.
- `postgres` 17.11, with the database password `routeforge-local-only`, a placeholder for a throwaway database.
- `redis` 7.4.12 with append-only persistence.
- `prometheus` v3.15.0, scraping `routeforge:8080/metrics` every 5 seconds.

All images are pinned by tag and digest. The gateway (8080) and Prometheus (9090) are published on `127.0.0.1` only, because `/metrics` has no authentication and the demo key is public. The providers in the compose config are fakes. Do not reuse the stack's keys or password anywhere shared.

`scripts/compose-smoke.sh` (`make compose-smoke`) builds and starts the stack, checks `/healthz` and `/readyz`, sends one routed request with the demo key and checks that the policy version header has 12 hex digits and that `/metrics` lists `routeforge_requests_total`. It removes the stack, volumes included, on the way out.

The image build and the compose smoke test were run on the development host (Docker Desktop on Windows, through Git Bash). `.github/workflows/integration.yml` builds the image and runs the smoke test on every push and pull request. Unit tests never need Docker.

### Putting it in front of real traffic

- **TLS.** The gateway serves plain HTTP. Terminate TLS at a proxy or load balancer.
- **Streaming.** The proxy must not buffer responses. The gateway sends `X-Accel-Buffering: no`, and a proxy has to allow responses that last as long as a request deadline.
- **`/metrics`.** It has no authentication. Expose it to the metrics system only, and block it from the public side.
- **`/admin`.** Expose it to operators only.
- **Real providers.** Start from `deploy/configs/providers.yaml`. Its tenant digest matches no key, so replace it with one from `keygen`. Never run the public demo key against real provider credentials.

## Storage and limiter backends

Both backends are optional. `memory` is the default for each, for development and tests, and it suffices for a single instance.

**PostgreSQL** (`storage.type: postgres`) keeps usage records, routing decisions and policies. The gateway opens the database at startup and applies its embedded migrations, tracked in a `schema_migrations` table, so an unreachable database stops startup. The tables are `policies`, `policy_activations`, `usage_records` and `routing_decisions`. Each policy is stored as `artifact` JSONB and as `artifact_raw` bytes, and the bytes are the source of truth because JSONB rewrites number text, which would change the version hash. At most one policy is active. Usage records and decisions go through a bounded batching writer (a batch of 500 rows or 200 ms). Every replica polls the active policy version every 5 seconds and follows an activation made on another. When PostgreSQL is down, an admin write fails with 503 `storage_unavailable` and `/readyz` reports `storage unavailable`.

**Redis** (`limiter.type: redis`) keeps the token buckets and the daily budgets so that replicas share them. Both are updated by Lua scripts, which makes each step atomic. Keys are `rf:rl:{tenant}:rpm`, `rf:rl:{tenant}:tpm` and `rf:budget:{tenant}:{yyyymmdd}`, and they expire after 48 hours. The scripts read the server clock with `TIME` and rely on effects replication, so Redis 5.0 or newer is required. The client does not retry a command, because the scripts are not idempotent and a retry could count a reservation twice. Its dial timeout is 2 seconds and its read and write timeouts are 1 second. Redis connects on first use, and `/readyz` reports `limiter unavailable` until it answers.

Circuit breakers, latency estimates and queues are always per replica.

## Accounting

All money is an `int64` count of nano-USD (1e-9 USD) ([ADR 0003](adr/0003-nano-usd-integer-accounting.md)).

- **Rate limits.** Each tenant has an RPM and a TPM token bucket. A request takes one RPM token and its estimated prompt tokens plus its output cap from TPM. A refusal is 429 `rate_limited`.
- **Budgets.** A tenant's budget is per UTC day. Before every attempt the gateway reserves the worst-case cost: prompt bytes at the input price plus the output cap at the output price. Prompt bytes bound prompt tokens for byte-level tokenizers, because every token is at least one byte. An attempt starts only if `spent + reserved + the new reservation <= budget`.
- **A refused reservation** ends the request with 429 `budget_exceeded` on the first attempt, and stops failover on a later one. The router also treats a model whose reservation does not fit as ineligible, so a request that fits on no model gets 503 `no_eligible_model` with the reason `budget`.
- **Settlement** happens on every outcome of every attempt, a cancellation and a panic included. It replaces the reservation with the actual charge and writes a usage record. A partial stream is settled with the tokens delivered. An attempt that fails before delivering output is settled at zero with the cost source `estimated`.
- **Usage source.** `reported` means the provider sent usage. Otherwise the gateway estimates `ceil(bytes / 4)` tokens for the prompt and for the delivered output, and the source is `estimated`. RouteForge never sees invoices: a cost is provider-reported (or estimated) tokens priced with the local table.
- **Actual charges are never clipped.** If a provider reports more tokens than the limits sent, the full cost is charged. `routeforge_budget_overshoot_usd_total` adds, at each settlement, `max(0, spent_after - max(budget, spent_before))`, so its total equals `max(0, spend - budget)` for a tenant and day.
- **Fail modes.** With `accounting.fail_mode: closed` (the default), a budget or limiter backend that cannot be reached makes the gateway answer 503 `accounting_unavailable`, with `Retry-After`, before any upstream call. With `open` the request goes through unmetered and the failure is logged, at most once a minute per backend. Use `open` only if an unmetered request is better than a refused one.
- **A full usage queue** is 503 `accounting_unavailable` in both modes, because dropping a settled attempt's record would make spend differ from the sum of usage records. The cap is soft: records of requests that were already admitted are always accepted.

The fault benchmark checks four invariants through `GET /admin/accounting`: (a) for every tenant, settled spend equals the sum of its usage records; (b) the reserved amount returns to 0 once all requests finish; (c) when providers honor the limits sent, spend stays within the budget; (d) when a provider over-reports usage, spend beyond the budget equals the overshoot. [benchmarks.md](benchmarks.md) has the outcome.

## Reliability

- **Deadline.** A request's deadline is the smallest of `X-RouteForge-Deadline-Ms`, the tenant's `max_deadline` and `server.default_deadline`. The context carries it to every attempt, and a client that disconnects cancels the upstream call. The connection stays writable for 5 seconds past the deadline so that a 504 or a terminal event can still be sent.
- **Retries.** At most `max_attempts` calls per request. Only `rate_limited`, `unavailable` and `timeout` failures are retried, and only before any byte has reached the client. The wait is full-jitter exponential backoff, raised to the provider's `Retry-After` when that is longer, and a wait that does not fit in the time left ends the request. Attempts after the first move down the decision's fallback list. A model whose breaker is open or whose queue is full or times out is skipped, which does not cost an attempt.
- **Retry budget.** Within the sliding window, the gateway allows `floor(retry_budget_ratio * requests in the window) + retry_budget_min` retries. The floor (3 by default) is deliberate: with a literal 20% a quiet gateway would allow no retry at all, and an isolated request could not fail over. The cost is up to `retry_budget_min` extra retries per window.
- **Circuit breakers.** One per model. Open after `consecutive_failures` failures in a row, or when the error rate reaches `error_rate` over at least `min_requests` requests in `window`. Open for `open_for`, then half-open with `half_open_probes` probes, and closed again on a probe's success. Only `unavailable`, `timeout`, `rate_limited` and `internal` failures count. A bad request, an auth failure and a client that left say nothing about the model.
- **Queues.** Each model has `max_concurrency` slots and a first-in, first-out line of `queue_size`. A full line makes the model ineligible for the decision. A wait ends at `queue_timeout`.
- **In-flight limit.** `max_inflight` covers chat and explain requests, and the slot is taken before authentication.
- **Partial streams.** Once the first byte is sent, a failure is not retried. The gateway sends the terminal error event, settles the attempt with the tokens delivered and writes a usage record with `partial` set.

## Observability

### Metrics

`GET /metrics` serves these series, with the Go runtime and process collectors. No label carries prompt or completion text.

| Metric | Labels | Meaning |
| --- | --- | --- |
| `routeforge_requests_total` | `model`, `strategy`, `profile`, `policy_version`, `status` | Chat requests. 499 is a client that left. See [routing.md](routing.md#routing-metrics). |
| `routeforge_route_decisions_total` | `model`, `reason`, `strategy`, `policy_version` | Decisions that chose a model. Explain requests count here. |
| `routeforge_route_decision_seconds` | none | Time to decide. |
| `routeforge_gateway_overhead_seconds` | `stream` | Time the gateway adds: the request's duration minus its upstream attempts. For a stream, the time to the first byte minus the wait for the first upstream event. |
| `routeforge_upstream_ttft_seconds` | `model` | Time to the first event of a streaming attempt. |
| `routeforge_upstream_duration_seconds` | `model`, `outcome` | Duration of an attempt. |
| `routeforge_retries_total` | `model`, `kind` | Retries, by the model that failed and the failure kind. |
| `routeforge_failovers_total` | none | Attempts made on another model than the one before. |
| `routeforge_breaker_state` | `model` | 0 closed, 1 half-open, 2 open. |
| `routeforge_queue_depth` | `model` | Requests waiting for a slot. |
| `routeforge_inflight` | none | Requests being served. |
| `routeforge_rate_limited_total` | `kind` | Requests refused by `rpm` or `tpm`. |
| `routeforge_budget_rejections_total` | none | Attempts refused for lack of budget. |
| `routeforge_cost_usd_total` | `model`, `cost_source` | Settled cost. |
| `routeforge_tokens_total` | `model`, `direction`, `usage_source` | Settled tokens. |
| `routeforge_budget_overshoot_usd_total` | `tenant` | Spend beyond a tenant's budget. |
| `routeforge_predicted_success` | `model` | Histogram of the calibrated probability of the chosen model. The drift signal. |
| `routeforge_policy_info` | `version` | 1 for the active policy version. |

### Tracing

Tracing is on only when `OTEL_EXPORTER_OTLP_ENDPOINT` is set to the base URL of an OTLP/HTTP collector; the gateway appends `/v1/traces`. The other `OTEL_EXPORTER_OTLP_*` variables are read by the exporter. Without the variable, tracing costs nothing. Spans are `http.request`, then `route.decide` and one `provider.attempt` per upstream call, with the model, attempt number, decision id, policy version and outcome as attributes. An incoming `traceparent` header is not continued, and nothing is injected upstream. A span has no prompt text.

### Logs

One JSON line per request with the request id, method, path, status, `client_gone`, duration, tenant, model and decision id. Bodies, prompts and completions are never logged. A client that left before its response was complete is logged and counted with status 499.

## Tests that need extra setup

`make test-integration` runs `go test -tags integration`. The PostgreSQL tests start an embedded PostgreSQL 17 server. The first run downloads its pinned binary archive, which the library keeps in a cache under the user's home directory, and later runs reuse it. The server's data directory must be on a filesystem that keeps Unix permissions. CI runs these tests too.

`make vet` does not compile files that carry the `integration` tag, so run `go vet -tags integration ./...` after changing one. The Redis unit tests use an in-process miniredis. Unit tests never need network, keys, Docker or a model.

## Reproducing the results

Every number in the README results block and in [benchmarks.md](benchmarks.md) is rendered from files that are committed, by a program. The steps, in order:

| Step | Reads | Writes |
| --- | --- | --- |
| `make py-setup` | `trainer/requirements*.txt` | `trainer/.venv` |
| `make data` | The pinned LLMRouterBench archive, downloaded and checksum-verified. | `data/raw/`, `data/work/` (outcomes, splits), `trainer/artifacts/replay/`, `data/manifest.json` |
| `make train` | The train, calibration and certification splits. | `trainer/artifacts/policy.json`, `testdata/parity/features.jsonl`, `bench/limits.json`, `trainer/artifacts/reports/training.{json,md}` |
| `make evaluate` | The test split, once. | `trainer/artifacts/reports/evaluation.{json,md}` and reliability plots |
| `make bench-replay RUN_ID=<id>` | `bench/configs/replay.yaml`, the replay files | `bench/results/<id>/replay.json` |
| `make bench-load RUN_ID=<id>` | `bench/configs/load-zero.yaml`, `load-profiled.yaml` | `bench/results/<id>/load-zero.json`, `load-profiled.json` |
| `make bench-faults RUN_ID=<id>` | `bench/configs/faults.yaml` | `bench/results/<id>/faults.json` |
| `make report RUN_ID=<id>` | The result files and `evaluation.json` | `docs/benchmarks.md` and the results block of `README.md` |
| `make demo` | `deploy/configs/policy-fake.yaml`, `examples/prompts.json` | `examples/demo-output.md` |

The trainer commands pin BLAS and OpenMP to one thread, which keeps runs deterministic. `make data`, `make train` and `make evaluate` do not run in CI. The scripts under `scripts/` start the gateway on a loopback port, wait for `/readyz`, run the benchmark and stop the gateway.

What to know when reading the benchmark results:

- The benchmarks ran in WSL2 Ubuntu 24.04 on the development host. The environment block of each result file (operating system, CPU, memory, Go and Python versions, git commit, config hash, dataset revision and policy version) is the source, and [benchmarks.md](benchmarks.md) shows it.
- Quality and cost come from replaying recorded outputs. Gateway performance comes from fake providers with SIMULATED latency profiles, so it measures the gateway, not any provider.
- The load configs set generous queue and in-flight limits, because the load benchmark measures gateway overhead and not admission limits. The fault config uses faster simulated decode priors (4000 tokens per second) so that its 2 second deadline leaves models eligible. Its header says so.
- The overhead target is pre-declared for one scenario: the profiled open loop at 200 requests per second. The load tables also show closed-loop runs at high concurrency. In those, the benchmark client and the gateway ran on the same machine and shared its CPUs, so those rows show overhead under saturation. Read them in the load tables of [benchmarks.md](benchmarks.md).
- Replay parity covers the strategies that are pure functions of the request and the policy: `quality` with each profile, `strongest`, `cheapest` and `static`. The gateway's `latency` strategy adapts to observed latencies (see [routing.md](routing.md#latency-profiles-are-simulated)) while the offline router uses the frozen priors, so its agreement is reported separately.
- Targets are the hypotheses. Whether each was met is in the generated block and tables.

## Known limits

- The PostgreSQL writer retries a failed batch three times with a doubling wait (100 ms base) and then drops it with an error log. During a long outage usage rows can be lost, and then spend no longer equals the sum of usage records.
- Incoming `traceparent` headers are not continued.
- Configuration is read once. A change needs a restart.
- Breakers, latency estimates and queues are per replica. Only budgets and rate limits are shared, and only through Redis.
- A pinned request is not scored, so its decision has no policy version and no probabilities.
- There is no TLS in the process, no UI, no multi-region support and no Kubernetes manifests.
- Extraction quality is unmeasured.
