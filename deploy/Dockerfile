# RouteForge gateway image. Build from the repository root:
#   docker build -f deploy/Dockerfile --build-arg VERSION=$(git describe --always --dirty) -t routeforge:local .
#
# The base images are pinned by tag and digest. The image runs as the nonroot
# user of the distroless static image and holds the gateway binary, the trained
# policy, the price table and deploy/configs/compose.yaml. Relative paths in a
# config resolve against the directory of the config file, so the files sit
# under /etc/routeforge in the layout of the repository.

FROM golang:1.27.3-bookworm@sha256:d9b6a1beb9474adc2a1ce842573639b44b2a0dd49b6669f51d3821baf9ec37c9 AS build
ARG VERSION=dev
# Never download another toolchain: the Go of this image must satisfy go.mod.
ENV GOTOOLCHAIN=local
WORKDIR /src
COPY go.mod go.sum ./
RUN go mod download
COPY cmd ./cmd
COPY internal ./internal
RUN CGO_ENABLED=0 go build -trimpath -ldflags "-s -w -X main.version=${VERSION}" -o /out/routeforge ./cmd/routeforge

FROM gcr.io/distroless/static-debian12:nonroot@sha256:afa5c872c891853ca7fcf1f12c3edb23f7eeef36189728842dd51042ff57f7ab
COPY --from=build /out/routeforge /routeforge
COPY trainer/artifacts/policy.json /etc/routeforge/trainer/artifacts/policy.json
COPY deploy/prices.yaml /etc/routeforge/deploy/prices.yaml
COPY deploy/configs/compose.yaml /etc/routeforge/deploy/configs/compose.yaml
USER nonroot:nonroot
EXPOSE 8080
HEALTHCHECK --interval=10s --timeout=3s --retries=3 CMD ["/routeforge", "healthcheck", "--url", "http://127.0.0.1:8080/healthz"]
ENTRYPOINT ["/routeforge", "serve", "--config", "/etc/routeforge/deploy/configs/compose.yaml"]
