# Architecture

RouteForge has two halves that meet at one file. A Python trainer learns, calibrates and certifies a routing policy offline and exports it as JSON. A Go gateway loads that file and serves requests with it. This page covers the request path, the packages, how a policy gets from the trainer into the gateway and where each property of the system is enforced. [routing.md](routing.md) covers the routing method and [operations.md](operations.md) covers running it.

## Request path

```
client (OpenAI SDK)
  -> api        : in-flight slot, auth, headers, body limit, deadline, request id
  -> admission  : usage queue check, rate limits (RPM, TPM)
  -> router     : features -> per-model calibrated p and lower bound -> eligibility
                  -> decision with fallbacks
  -> executor   : per model: breaker -> bounded queue -> budget reservation
                  -> provider attempt -> retry or failover
  -> provider   : openai | anthropic | fake | replay
  -> accounting : settle every attempt with reported or estimated usage, write a usage record
  -> telemetry  : spans, metrics, decision record
```

In order, for `POST /v1/chat/completions`:

1. The gateway takes a slot of the global in-flight limit (`server.max_inflight`). It does this before authentication, so a gateway at its limit refuses with 503 `overloaded` at the lowest possible cost. `POST /v1/routing/explain` shares the limit.
2. The bearer key is hashed with SHA-256 and looked up among the tenants' digests. Then the profile, strategy and deadline headers are checked, and the body is decoded within `server.max_body_bytes`. The feature text and the prompt size are computed once here and reused by routing, accounting and the replay provider.
3. Admission asks the usage store whether it is full and charges the tenant's RPM and TPM buckets. Nothing is reserved yet.
4. `router.Decide` reads the active policy once, scores every candidate, applies eligibility and picks a model, with an ordered fallback list. It returns a decision record that has no prompt text.
5. The executor works down the chosen model and its fallbacks. Each attempt passes the model's circuit breaker, takes a slot of the model's bounded queue and reserves its worst-case cost against the tenant's daily budget before the provider is called.
6. Every attempt ends exactly once, whatever happened. The end records the attempt on the decision, gives the breaker its verdict, feeds the latency estimators when the attempt succeeded, replaces the reservation with the actual charge, writes a usage record and frees the queue slot.
7. A non-streaming response is returned as JSON. A streaming response is relayed as server-sent events. The decision is stored just before the response is complete, so a client that has read the end of a response can fetch its decision.

## Packages

| Package | Responsibility |
| --- | --- |
| `cmd/routeforge` | The gateway binary: `serve`, `validate-config`, `keygen`, `version`, `healthcheck`. Wires the packages below together. |
| `cmd/routeforge-bench` | The benchmark driver: `replay`, `load`, `faults`, `report`, `demo`. |
| `internal/config` | YAML configuration: strict decoding, defaults, validation, path resolution. |
| `internal/auth` | API key generation, SHA-256 digests and tenant lookup. |
| `internal/api` | The HTTP server, OpenAI wire types, request validation, streaming, error mapping, the admin API, request ids and the access log. |
| `internal/policy` | The policy artifact: parsing, validation, the version hash, feature extraction, predictors, calibration lookup, the static rules and the registry with activation and rollback. |
| `internal/router` | The routing engine: strategies, eligibility, the decision record, reasons. |
| `internal/executor` | Attempts against providers, with retries, failover, breakers, queues and metering. |
| `internal/reliability` | Circuit breakers, the retry budget, backoff, bounded queues and the in-flight limit. |
| `internal/health` | Per-model latency estimators that feed the latency check and the `latency` strategy. |
| `internal/accounting` | The price table, exact integer cost, budgets, rate limits and usage records. |
| `internal/store` | The decision ring, plus the `postgres` and `redisstore` backends behind small interfaces. |
| `internal/provider` | The provider contract, error kinds and SSE helpers, with the `openai`, `anthropic`, `fake` and `replay` implementations. |
| `internal/telemetry` | Prometheus metrics and OpenTelemetry tracing. |
| `internal/bench` | Replay, load and fault runs, environment capture and report rendering. |
| `trainer/routeforge_train` | The Python trainer, calibrator, certifier and evaluator. |

## From the trainer to the gateway

```
LLMRouterBench records
  -> make data      outcomes.jsonl, splits (train, calibration, certification, test),
                    replay files, data/manifest.json
  -> make train     fit predictors, calibrate, select, certify, export
                    -> trainer/artifacts/policy.json
                       testdata/parity/features.jsonl
                       bench/limits.json
                       trainer/artifacts/reports/training.{json,md}
  -> make evaluate  score the test split once
                    -> trainer/artifacts/reports/evaluation.{json,md}, reliability plots
```

`make train` reads the train, calibration and certification splits, each checked against the manifest checksum, and never opens the test split. `make evaluate` is the only command that reads it. The trainer reads only the generic `outcomes.jsonl` format, so a pool other than the shipped one can be trained from its own outcomes.

The policy file is canonical JSON. Its `version` is the first 12 hex digits of the SHA-256 of its canonical content. The Go loader recomputes that hash from the bytes it was given and rejects a mismatch. It also rejects an unknown schema version, a missing model, calibration blocks that do not tile 0 to 1 or whose `p` falls, non-finite numbers and a profile whose lambda is not on the grid or the sentinel.

Two shared rules keep the halves from drifting. The feature definitions are written twice, in `trainer/routeforge_train/features.py` and `internal/policy/features.go`, and both are tested against `testdata/parity/features.jsonl`, which the trainer writes. The decision rule exists twice as well, in `routing.py` and `internal/router/engine.go`, and the tests replay the fixture decisions through Go. [ADR 0002](adr/0002-json-policy-artifact-with-parity.md) explains the choice.

The gateway gets a policy in one of three ways: `policy.path` in the config at startup, `POST /admin/policies` followed by an activation, or, with PostgreSQL storage, the active policy another replica activated.

## Hot swap and rollback

The registry holds every policy the gateway knows and one active pointer. `Active()` is lock-free. A request reads the pointer once in `router.Decide` and scores and routes against that one policy version, so activating or rolling back while requests are in flight does not change their decisions. The next decision uses the new policy.

- A policy is added inactive. Adding it fails if one of its models has no configured model whose `policy_name` matches, so a policy that the gateway cannot serve never becomes active.
- Activation retires the policy it replaces. Rollback re-activates the version that was active before the current one, and the version it replaces becomes inactive. A second rollback switches back. Activating the active version changes nothing.
- With PostgreSQL storage, the registry writes through to the database before it changes in memory, so a failed write leaves the active policy as it was. The raw bytes of each policy are stored next to the JSONB copy, because JSONB rewrites number text and that would change the version hash. Every replica polls the active version every 5 seconds and follows an activation made elsewhere.
- Every activation is recorded with its actor (`startup`, `admin` or `poller`) in the activation log that `GET /admin/policies` returns, and the `routeforge_policy_info` gauge follows the active version.

## Where each guarantee is enforced

| Property | Enforced in |
| --- | --- |
| Only a tenant with a known key gets in. An unknown key gets 401 with no tenant detail. | `internal/api` (`authenticate`) against `internal/auth` digests. Keys are never stored, only digests. |
| Bodies, headers and request ids are bounded. Trailing JSON data is rejected. | `internal/api/server.go`, `request.go` and `provider.DecodeJSON`. |
| A request has a deadline and a departing client cancels the upstream call. | `internal/api` (`requestBudget`, connection deadlines) and the context passed to every attempt. |
| The gateway sheds load before it queues work. | The in-flight limit in `internal/api`, the per-model queues in `internal/reliability`. A full queue makes a model ineligible for the decision. |
| A model that cannot serve the request is not chosen. | `internal/router` eligibility: `breaker_open`, `queue_full`, `context_window`, `latency_slo`, `budget`. |
| Retries stay bounded and never follow a first byte. | `internal/executor` (at most `reliability.max_attempts`, a retry budget, a backoff that must fit the deadline, no retry once a byte reached the client). |
| An unhealthy model is isolated. | The per-model circuit breaker in `internal/reliability`, consulted by the router and by the executor. |
| Money is exact. | `internal/accounting` keeps every amount as an `int64` count of nano-USD. [ADR 0003](adr/0003-nano-usd-integer-accounting.md). |
| A failure, retry or partial stream cannot bypass the budget. | Each attempt reserves before it calls and settles in a deferred path on every outcome, including cancellation and panic. A partial stream is settled with the tokens delivered. |
| A broken accounting backend does not mean free traffic. | `accounting.fail_mode: closed` (the default) returns 503 `accounting_unavailable` before any upstream call. |
| The routing threshold carries a statistical certificate. | The trainer's Learn-then-Test step (`certify.py`) sets each profile's lambda, the engine applies it, and the decision's `guarantee_applies` says whether every pool model was eligible. The sentence of the guarantee is in [routing.md](routing.md). |
| Go and Python compute the same features, scores and decisions. | The shared parity fixture, tested on both sides. |
| Prompt and completion text are never logged or stored. | Decision records, usage records, metric labels and span attributes have no field for it, and the access log writes none. |
| Quality and cost claims are reproducible. | Every number in the README results block and in `docs/benchmarks.md` is rendered from committed result files, and a test fails if the committed documents differ from the render. |

## Where state lives

| State | Memory (default) | Shared backend |
| --- | --- | --- |
| Usage records | In process, the last records plus running totals. | PostgreSQL `usage_records`, through a batching writer. |
| Routing decisions | A ring of 10,000 decisions. | PostgreSQL `routing_decisions`. |
| Policies and the activation log | In process. | PostgreSQL `policies` and `policy_activations`. |
| Rate limit buckets and daily budgets | In process, per replica. | Redis, shared by replicas, updated by Lua scripts. |
| Circuit breakers, latency estimates, queues | In process. | Always per replica. |
| Tenants and API key digests | The YAML config. | Always the config file. |
| Experiment results | Versioned JSON under `bench/results/`. | Always the repository. |
