# RouteForge

<!-- results:begin -->
H1: falsified (static_oracle_family)

- Cost per success reduction versus always_strongest: 47.5% (95% CI 44.6% to 50.3%). Target of 25.0% or more: pass.
- Regret rate: 0.043 against alpha 0.050 (95% CI 0.034 to 0.053).
- Gateway replay parity: pass (deterministic agreement 1.000).
- p95 gateway overhead < 20 ms: pass (1.42 ms, profiled open loop 200 rps)
- Accounting invariants: holds.

Run `2026-10-28-wsl`. Full results, environment and commands: [docs/benchmarks.md](docs/benchmarks.md).
<!-- results:end -->
