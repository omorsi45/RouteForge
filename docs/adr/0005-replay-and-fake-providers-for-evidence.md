# 0005: Replay and fake providers for evidence

Status: accepted

## Context

The project has no provider API keys and a budget of no inference spend, and it still has to show two kinds of evidence: how well routing trades quality for cost, and how the gateway behaves under load and under failure. The tests must not need a network, a key, Docker or a model.

Real calls would also make results unrepeatable. A provider's latency, errors and answers change from day to day.

## Decision

Use recorded and simulated providers, and say so wherever a number appears.

- Quality and cost evidence comes from replaying recorded LLMRouterBench outputs. The replay provider serves the recorded answer for a prompt and a model, keyed by the SHA-256 of the feature text plus the model name, with the recorded token usage. It never truncates. The benchmark tenant's output cap comes from the largest recorded completion, written by the trainer to `bench/limits.json`.
- Gateway performance and fault behavior come from fake providers: deterministic, seeded and able to inject errors, timeouts, rate limits with `Retry-After`, cut streams, missing usage and over-reported usage.
- Latency in both comes from `bench/latency_profiles.yaml`, a fixed rule over the models' price ranks, marked SIMULATED. It is not a measurement of any provider.
- The OpenAI-compatible and Anthropic adapters are tested against `httptest` servers and fixtures.
- Each run writes JSON under `bench/results/<run-id>/` with its environment: operating system, CPU, memory, Go and Python versions, git commit, config hash, dataset revision and policy version. `docs/benchmarks.md` and the README results block are rendered from those files by a program, and a test fails if the committed documents differ from the render.
- The replay benchmark sends the test prompts through the running gateway and compares the model that served each one with the model the trainer chose. Strategies that are pure functions of the request and the policy are held to a threshold. The `latency` strategy adapts to observed attempts, so its agreement is reported separately.

## Consequences

- Anyone can reproduce every number without a key or spend, and the files that back a claim are in the repository.
- A replay result is not a live result. It shows how routing would have chosen among recorded answers. It cannot show how a provider behaves today.
- Gateway numbers describe the gateway against fake providers on one host. They say nothing about a provider's latency or about a network. When the benchmark client and the gateway share a host, high-concurrency rows reflect that sharing.
- The benchmark prompt templates let a classifier recognize the dataset, so leave-one-dataset-out results are reported beside the in-distribution ones.
- Costs are the dataset's recorded cost field, which is provider usage times list price, or the local price table. They are not invoices.
- Extraction quality is unmeasured until its labels are produced with local models.
- Real-provider behavior is covered only by the adapters' fixture tests until someone runs the gateway with real credentials.
