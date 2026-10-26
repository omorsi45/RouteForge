GO ?= go
GOVULNCHECK_VERSION := v1.9.0
EXAMPLE_CONFIGS := $(wildcard deploy/configs/*.yaml)

.PHONY: check fmt-check vet build tidy-check validate-configs test test-race vuln

check: fmt-check vet build tidy-check validate-configs

fmt-check:
	@out="$$(gofmt -l .)"; if [ -n "$$out" ]; then echo "gofmt needed:"; echo "$$out"; exit 1; fi

vet:
	$(GO) vet ./...

build:
	$(GO) build ./...

tidy-check:
	$(GO) mod tidy -diff

validate-configs:
	@for f in $(EXAMPLE_CONFIGS); do $(GO) run ./cmd/routeforge validate-config --config $$f || exit 1; done

test:
	$(GO) test -count=1 ./...

test-race:
	$(GO) test -race -count=1 ./...

vuln:
	$(GO) run golang.org/x/vuln/cmd/govulncheck@$(GOVULNCHECK_VERSION) ./...
