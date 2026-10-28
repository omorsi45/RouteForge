# Contributing

## Build and test

Go 1.27.3 and GNU make are required for the gateway, Python 3.14 for the trainer. The Makefile needs bash.

- `make check test test-race vuln` is the Go gate and mirrors CI.
- `make check` runs gofmt, `go vet`, `go build`, `go mod tidy -diff` and `make validate-configs`.
- `make validate-configs` runs `routeforge validate-config` on every file in `deploy/configs/` and `bench/configs/`. It also loads the price table and the policy each config names.
- `make test-integration` runs the tests with the `integration` build tag. They start an embedded PostgreSQL 17 server; the first run downloads its binary (see `docs/operations.md`). `make vet` does not compile those files, so run `go vet -tags integration ./...` after changing one.
- `make py-setup py-check py-test` is the Python gate: venv in `trainer/.venv`, exact pins installed with `--no-deps`, then `pip check`, ruff and pytest.
- `make docker` builds the image and `make compose-smoke` starts the compose stack and probes it. Both need Docker, and `.github/workflows/integration.yml` runs them.

## Data and benchmarks

None of these run in CI.

- `make data train evaluate` downloads LLMRouterBench, trains the policy and writes the offline evaluation. It is slow and needs the network once for the download.
- `make label-extraction EXTRACTION_MODELS=<models>` labels the extraction workload with local Ollama models.
- `make bench-replay`, `make bench-load` and `make bench-faults`, each with `RUN_ID=<id>`, write result files to `bench/results/<id>/`.
- `make report RUN_ID=<id>` renders `docs/benchmarks.md` and the results block of `README.md` from those files.
- `make demo` runs the routing demo and writes `examples/demo-output.md`.

## Generated files

`docs/benchmarks.md`, the part of `README.md` between the `results:begin` and `results:end` markers, and `examples/demo-output.md` are generated. Do not edit them by hand. `TestCommittedBenchmarksRegenerate` fails when they differ from what the committed result files render.

## Git

- Stage exact paths. Never `git add -A` or `git add .`.
- Never pass `--no-verify`. If the secret-scanning hook flags a fixture, make the fixture low-entropy and obviously fake.
- Never commit `data/raw/`, `data/work/`, `trainer/artifacts/replay/` or `bench/results/*/raw/`. They hold third-party data or raw logs.

## Commits

- Imperative mood, sentence case, no prefixes, no trailing period, under 72 characters.

## Tests

- Tests never need network access, API keys, Docker or a real model.
- Exercise upstream HTTP with `httptest` and fixtures under each package's `testdata/`.
- Fake credentials are low-entropy, for example `test-key-tenant-a`.
- Compute digests of test keys at runtime with `auth.HashKey` instead of pasting hex.
- Write the failing test first and watch it fail for the reason you expect.

## Dependencies

- Pin exact versions.
- Prefer the standard library first.

## Writing

- No em dashes in any file. `TestDocsHaveNoEmDash` scans the documents.
- Never log or store prompt or completion text.
- Hand-written documents state no measured result. A number or a verdict comes from a generated file, and the document links to it.
- The only guarantee RouteForge states is the marginal one in `docs/routing.md`.
