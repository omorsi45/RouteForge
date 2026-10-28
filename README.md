# RouteForge

RouteForge is an OpenAI-compatible LLM gateway written in Go. For each request it picks the cheapest model whose calibrated, conservatively bounded probability of answering well clears a threshold, and escalates to a stronger model when none does. Every decision is recorded and can be explained.

## Why

Sending every request to the strongest model is safe and expensive. Routing by price alone is cheap and loses quality on the requests that need a stronger model. Gateways already ship fallbacks, retries, budgets and rate limits, and RouteForge has those too. What it adds is a router that has to justify its choices:

- Per-model success probabilities are learned from recorded outcomes and calibrated on held-out data. Each comes with a one-sided 95% Wilson lower bound over the calibration requests that scored alike.
- The threshold of each risk profile (strict, balanced, economy) is certified with Learn-then-Test on a separate split, against the regret of routing away from the strongest model.
- `POST /v1/routing/explain` shows why a request would go to a model, with the calibration counts behind the probability.
- The offline evaluation compares the router with always-strongest, always-cheapest, keyword rules, a latency-aware router and a content-blind random router that has the same model mix, and tests the differences statistically.

The only guarantee RouteForge states is a marginal one:

With probability at least 1 - delta over the draw of the certification set, the expected regret rate on future traffic exchangeable with it is at most alpha, when all pool models are eligible. It says nothing about any single request.

The targets are 25% lower cost per successful task than always-strongest and p95 gateway overhead under 20 ms. They are hypotheses. The results are reported whether they pass or fail, and the hypothesis H1 has pre-declared falsification criteria in [docs/routing.md](docs/routing.md).

## Results

<!-- results:begin -->
H1: falsified (static_oracle_family)

- Cost per success reduction versus always_strongest: 47.5% (95% CI 44.6% to 50.3%). Target of 25.0% or more: pass.
- Regret rate: 0.043 against alpha 0.050 (95% CI 0.034 to 0.053).
- Gateway replay parity: pass (deterministic agreement 1.000).
- p95 gateway overhead < 20 ms: pass (1.42 ms, profiled open loop 200 rps)
- Accounting invariants: holds.

Run `2026-10-28-wsl`. Full results, environment and commands: [docs/benchmarks.md](docs/benchmarks.md).
<!-- results:end -->

Where the numbers come from, and what they do not show:

- Quality and cost come from replaying recorded LLMRouterBench outputs through the router, not from live provider calls.
- Gateway performance comes from fake providers with SIMULATED latency profiles. The load tables also include closed-loop runs at high concurrency, where the benchmark client and the gateway shared one host's CPUs. Those rows show overhead under saturation.
- The benchmark prompts' templates let a classifier recognize the dataset, which inflates in-distribution gains. Leave-one-dataset-out results are reported next to them for that reason.
- Extraction quality is unmeasured.
- RouteForge never sees invoices. A cost is provider-reported (or estimated) tokens priced with the local table in `deploy/prices.yaml`.

The full tables, the environment of each run and the commands are in [docs/benchmarks.md](docs/benchmarks.md); the load tables there are the ones to read for the saturation caveat above. The trainer's own reports are [training.md](trainer/artifacts/reports/training.md) and [evaluation.md](trainer/artifacts/reports/evaluation.md) in `trainer/artifacts/reports/`. A walk through the demo prompts is in [examples/demo-output.md](examples/demo-output.md).

## Quick start

You need Go 1.27.3 for the demo, or Docker for the compose stack. Neither needs a provider key. Both download dependencies or images the first time.

Run the demo. It builds the gateway, starts it on `127.0.0.1:18081` with the trained policy over fake providers, explains and sends the prompts of `examples/prompts.json`, and writes `examples/demo-output.md`:

```sh
make demo
```

Or start the stack with the gateway, PostgreSQL, Redis and Prometheus:

```sh
docker compose -f deploy/docker-compose.yml up --build
```

Add `-d --wait` to run it in the background, and use `down -v` to remove it. The gateway listens on `127.0.0.1:8080` and Prometheus on `127.0.0.1:9090`. Without Docker, `go run ./cmd/routeforge serve --config deploy/configs/policy-fake.yaml` serves the same policy on `:8080` with in-memory storage.

The providers in these configs are fakes, and the key `rf_local_demo_only` is public. Use both on loopback only. Make your own keys with `go run ./cmd/routeforge keygen`. The examples below read the key from a variable:

```sh
export ROUTEFORGE_API_KEY=rf_local_demo_only
```

Send a request to `routeforge-auto`, which routes:

```sh
curl -sS http://127.0.0.1:8080/v1/chat/completions \
  -H "Authorization: Bearer $ROUTEFORGE_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{"model":"routeforge-auto","messages":[{"role":"user","content":"What is the capital of Australia?"}],"max_tokens":64}'
```

The response is the OpenAI shape plus a `routeforge` object with the decision id, the reason, the number of attempts and the cost. The response headers `X-RouteForge-Model`, `X-RouteForge-Decision-Id`, `X-RouteForge-Policy-Version` and `X-RouteForge-Reason` say what served it. The fake providers answer with short synthetic text.

Ask how a request would be routed, without calling any provider. `X-RouteForge-Profile` picks `strict`, `balanced` or `economy`:

```sh
curl -sS http://127.0.0.1:8080/v1/routing/explain \
  -H "Authorization: Bearer $ROUTEFORGE_API_KEY" \
  -H "Content-Type: application/json" \
  -H "X-RouteForge-Profile: strict" \
  -d '{"model":"routeforge-auto","messages":[{"role":"user","content":"What is the capital of Australia?"}],"max_tokens":64}'
```

Each candidate in the answer shows its calibrated `p`, the lower bound `lcb`, the calibration block it fell into (`block_k` of `block_n`), its expected cost and why it is eligible or not. [docs/routing.md](docs/routing.md) explains how to read it. To use real providers, start from `deploy/configs/providers.yaml` and see [docs/operations.md](docs/operations.md).

## API surface

| Method and path | Key | Purpose |
| --- | --- | --- |
| `POST /v1/chat/completions` | tenant | Chat completions, streaming or not. `model` is `routeforge-auto` or a configured model name, which pins it. |
| `GET /v1/models` | tenant | `routeforge-auto` and the configured models the tenant may use. |
| `POST /v1/routing/explain` | tenant | The routing decision for a chat body. No provider call, nothing stored. |
| `GET /v1/routing/decisions/{id}` | tenant | A stored decision of the same tenant. |
| `GET /admin/policies` | admin | Stored policies and the activation log. |
| `POST /admin/policies` | admin | Upload and validate a policy. It is stored inactive. |
| `POST /admin/policies/{version}/activate` | admin | Make a stored policy the active one. |
| `POST /admin/policies/rollback` | admin | Re-activate the policy that was active before. |
| `GET /admin/accounting` | admin | Spend, reservations, overshoot and usage totals of one tenant and day. |
| `GET /healthz` | none | The process is up. |
| `GET /readyz` | none | A policy is loaded and the storage and limiter backends answer. |
| `GET /metrics` | none | Prometheus metrics. Restrict it at your proxy. |

Errors use OpenAI's `{"error":{"message","type","code"}}` envelope. Tools, function calling, images, audio, embeddings and the Responses API are not supported, and a request that uses them gets a 400 with `unsupported_parameter`. The error codes and every configuration key are in [docs/operations.md](docs/operations.md).

## Reproduce

The offline numbers come from the trainer and the gateway numbers from the benchmark driver. Run these from the repository root. They need bash, Go and Python 3.14, and the first one downloads the pinned LLMRouterBench archive (about 1.3 GB, checksum-verified):

```sh
make py-setup data train evaluate
make bench-replay RUN_ID=<id>
make bench-load RUN_ID=<id>
make bench-faults RUN_ID=<id>
make report RUN_ID=<id>
```

`make report` regenerates `docs/benchmarks.md` and the results block above from the files in `bench/results/<id>/`. Nothing in the block is typed by hand. [docs/operations.md](docs/operations.md) lists what each step reads and writes.

## Project layout

```
cmd/routeforge/        gateway binary: serve, validate-config, keygen, version, healthcheck
cmd/routeforge-bench/  benchmark driver: replay, load, faults, report, demo
internal/api/          HTTP server, OpenAI wire types, streaming, admin API
internal/policy/       policy artifact, features, predictors, calibration, registry
internal/router/       strategies, eligibility, decision records
internal/executor/     attempts, retries, failover
internal/reliability/  circuit breakers, retry budget, queues, in-flight limit
internal/accounting/   prices, exact cost, budgets, rate limits, usage records
internal/store/        decision ring, PostgreSQL and Redis backends
internal/provider/     OpenAI-compatible, Anthropic, fake and replay providers
internal/telemetry/    Prometheus metrics, OpenTelemetry tracing
internal/bench/        benchmark runs and report rendering
trainer/               Python trainer, calibrator, certifier and evaluator
bench/                 benchmark configs, latency profiles, committed results
deploy/                Dockerfile, compose stack, example configs, price table
docs/                  architecture, routing, operations, benchmarks, decision records
examples/              demo prompts and generated demo output
```

## Data

The training and replay data are recorded outputs from LLMRouterBench. Its data license is unstated, so the repository never commits the records. `make data` downloads them, and `data/raw/`, `data/work/` and `trainer/artifacts/replay/` are gitignored. What is committed is derived: the policy file, the aggregate reports and the parity fixtures.

## License

Apache License 2.0. See [LICENSE](LICENSE).
