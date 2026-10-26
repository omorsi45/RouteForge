# Contributor rules

## Build and test

- `make check test test-race vuln` is the full gate and mirrors CI.
- `make check` runs gofmt, vet, build, `go mod tidy -diff` and validates the example configs.

## Git

- Use `/usr/bin/git` only.
- Stage exact paths. Never `git add -A` or `git add .`.
- Never pass `--no-verify`. If the secret-scanning hook flags a fixture, make the fixture low-entropy and obviously fake.

## Commits

- Imperative mood, sentence case, no prefixes, no trailing period, under 72 characters.

## Tests

- Tests never need network access, API keys, Docker or a real model.
- Exercise upstream HTTP with `httptest` and fixtures under each package's `testdata/`.
- Fake credentials are low-entropy, for example `test-key-tenant-a`.
- Compute digests of test keys at runtime with `auth.HashKey` instead of pasting hex.

## Dependencies

- Pin exact versions.
- Prefer the standard library first.

## Writing

- No em dashes in any file.
- Never log prompt or completion text.

## Local-only files

- Never commit `docs/superpowers/` or `.superpowers/`.
