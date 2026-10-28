# 0002: A JSON policy artifact with parity tests

Status: accepted

## Context

The trainer is Python, with scikit-learn, and the gateway is Go. The gateway has to score a request in microseconds, load a new policy without a restart, and reach the same decision the trainer reached offline. The options were to embed a Python runtime in the gateway, to call a scoring service over the network, or to export the fitted model as data and reimplement scoring in Go.

## Decision

The trainer exports one JSON file, the policy, and Go reimplements the scoring.

- The file has `schema_version` 1. Its `version` is the first 12 hex digits of the SHA-256 of its canonical content.
- It holds the feature specification, one predictor and one calibration per pool model, the expected output tokens per model, the strongest and cheapest models, the static keyword rules and the certified profiles.
- Logistic regression is exported as sparse and dense weights with an intercept. Gradient boosting is exported as flat tree arrays read from scikit-learn's public `tree_` attributes. Trees are evaluated on float32-rounded inputs, as scikit-learn does.
- The Go loader decodes strictly. It rejects an unknown schema version, a model that is missing, non-finite numbers, calibration blocks that do not tile 0 to 1 or whose probabilities fall, a lambda that is not on the grid or the sentinel, and a version hash that does not match the bytes.
- The trainer writes `testdata/parity/features.jsonl`: prompts with the expected bucket indices, values, dense vector, per-model raw and calibrated scores and decisions. Go tests require identical indices, values within 1e-12, scores within 1e-9, and identical calibration blocks, decisions and reasons.
- On the parity path, products that feed a sum are explicit conversions so that no compiler fuses a multiply and an add, because Go may fuse where Python never does. Norms are summed sequentially in ascending index order.
- Only logistic regression and gradient boosting are served. The kNN and keyword-rule baselines are evaluated offline and never exported as predictors.

## Consequences

- The gateway has no Python dependency at run time, and a policy can be uploaded, activated and rolled back through the admin API.
- A policy is a plain file. It can be reviewed, versioned in git and loaded from a database, and its version identifies exactly the bytes that were scored.
- Features and the decision rule exist in two languages. A change to either must change both sides and the fixture, and the parity tests are the guard.
- Floating-point discipline is part of the contract. A refactor that reorders a sum can break parity without changing a result by much.
- The file is large, because it carries weight and tree arrays. PostgreSQL stores the raw bytes next to a JSONB copy, because JSONB rewrites number text and that would change the version hash.
- Adding a predictor type means a new exporter, a new Go scorer, new fixtures and a schema decision.
