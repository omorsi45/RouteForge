GO ?= go
GOVULNCHECK_VERSION := v1.9.0
EXAMPLE_CONFIGS := $(wildcard deploy/configs/*.yaml bench/configs/*.yaml)
PYTHON ?= python3
VENV := trainer/.venv
VPY := $(VENV)/bin/python
TRAIN := OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONPATH=trainer $(VPY) -m routeforge_train --root .
EXTRACTION_MODELS ?=

.PHONY: check fmt-check vet build tidy-check validate-configs test test-race vuln py-setup py-check py-test data train evaluate label-extraction

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

py-setup:
	$(PYTHON) -m venv $(VENV)
	$(VPY) -m pip install --disable-pip-version-check --no-deps -r trainer/requirements.txt -r trainer/requirements-dev.txt
	$(VPY) -m pip check

py-check:
	cd trainer && .venv/bin/ruff check . && .venv/bin/ruff format --check .

py-test:
	cd trainer && OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 .venv/bin/python -m pytest

data:
	$(TRAIN) data

train:
	$(TRAIN) train

evaluate:
	$(TRAIN) evaluate

label-extraction:
	@test -n "$(EXTRACTION_MODELS)" || { echo "set EXTRACTION_MODELS, for example: make label-extraction EXTRACTION_MODELS=llama3.2:3b"; exit 2; }
	$(TRAIN) label-extraction --models "$(EXTRACTION_MODELS)"
