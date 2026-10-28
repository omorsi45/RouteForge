H1: falsified (static_oracle_family)

- Cost per success reduction versus always_strongest: 20.0% (95% CI 15.0% to 25.0%). Target of 25.0% or more: fail.
- Regret rate: 0.040 against alpha 0.100 (95% CI 0.011 to 0.099).
- Gateway replay parity: pass (deterministic agreement 1.000).
- p95 gateway overhead < 20 ms: pass (12.50 ms, profiled open loop 200 rps)
- Accounting invariants: holds.

Run `golden-run`. Full results, environment and commands: [docs/benchmarks.md](docs/benchmarks.md).
