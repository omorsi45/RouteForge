# RouteForge trainer

Offline trainer, calibrator and evaluator for RouteForge routing policies. It fits predictors on
recorded LLMRouterBench outcomes, certifies a routing threshold for each risk profile, exports the
policy file the Go gateway loads, and scores the held-out test split. It never calls a model
provider and never runs in production.

## Setup

Python 3.14 and `make` are required. From the repository root:

```sh
make py-setup           # creates trainer/.venv, installs the exact pins, runs pip check
make py-check py-test   # ruff and pytest; hermetic, no network, no model
```

## Commands

Run these from the repository root. None of them runs in CI. `data`, `train` and `evaluate` build
on each other, so run them in that order. Trainer commands pin BLAS and OpenMP to one thread, which
keeps runs deterministic and avoids thread oversubscription on many-core hosts.

- `make data`: downloads the pinned LLMRouterBench archive, verifies its checksum, extracts the
  pool members, then builds outcomes, stratified splits, replay files and `data/manifest.json`.
- `make train`: fits and calibrates the predictors, certifies the profiles, and exports the
  policy, the parity fixtures, the bench limits and the training report.
- `make evaluate`: scores the test split offline and writes the evaluation report and plots.
- `make label-extraction EXTRACTION_MODELS=llama3.2:3b,qwen2.5:7b`: labels the text-json
  extraction workload with local Ollama models and writes `requests.jsonl` and `outcomes.jsonl`
  under `data/work/extraction/`. Needs `ollama serve` running and the models pulled. Nothing is
  written if any call fails.

## What is committed

These paths are committed, so a reader can check them without rerunning anything:

- `data/manifest.json`: sources, checksums, per-dataset counts and the split sizes.
- `deploy/prices.yaml`: list prices per model, read by the trainer and the gateway.
- `bench/latency_profiles.yaml`: simulated latency profiles, a reference only.
- `bench/limits.json`: output token limits for the benchmark harness.
- `testdata/parity/features.jsonl`: pinned features, scores and decisions the Go tests replay.
- `trainer/artifacts/policy.json`: the exported routing policy.
- `trainer/artifacts/reports/`: training and evaluation reports as JSON and Markdown, and plots.

These paths are gitignored, because the LLMRouterBench data license is unstated and raw records,
splits and replay files must not be redistributed:

- `data/raw/`: the downloaded archive members and the text-json source file.
- `data/work/`: outcomes, splits, candidate predictors and extraction labels.
- `trainer/artifacts/replay/`: recorded model outputs for the test prompts, used by the bench.

## Formats other components read

- Policy (`trainer/artifacts/policy.json`): canonical JSON with a content-derived `version`, the
  feature spec, one predictor and calibration per pool model, the static rules and the certified
  lambda of each profile.
- Parity fixtures (`testdata/parity/features.jsonl`): one JSON line per synthetic request.
- Replay files (`trainer/artifacts/replay/`): one JSON line per test prompt and model, plus the
  test prompts.
- Bench limits (`bench/limits.json`): the policy version and the recorded output token caps.

## Guarantee

With probability at least 1 - delta over the draw of the certification set, the expected regret rate on future traffic exchangeable with it is at most alpha, when all pool models are eligible. It says nothing about any single request.

## Extraction

Extraction quality is unmeasured until `make label-extraction` has been run against local Ollama models; no extraction results are reported.
