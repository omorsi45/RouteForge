#!/usr/bin/env bash
# Run the routing demo.
#
# Usage: scripts/demo.sh
#
# The script builds bin/routeforge and bin/routeforge-bench, starts the gateway
# with deploy/configs/policy-fake.yaml on 127.0.0.1:18081, waits up to 30
# seconds for /readyz and runs the demo, which writes examples/demo-output.md.
# It then prints that file. It refuses to start when something already answers
# on the port, and stops the gateway with SIGTERM whichever way it ends. The key
# the demo sends, rf_local_demo_only, is the public key of the demo config and
# is meant for loopback runs only. It needs Go and bash, and nothing else.
set -euo pipefail

[ "$#" -eq 0 ] || {
  echo "usage: scripts/demo.sh" >&2
  exit 2
}

cd "$(dirname "${BASH_SOURCE[0]}")/.."
port=18081
addr="127.0.0.1:$port"
config=deploy/configs/policy-fake.yaml
output=examples/demo-output.md

"${GO:-go}" build -o bin/routeforge ./cmd/routeforge
"${GO:-go}" build -o bin/routeforge-bench ./cmd/routeforge-bench

# Something that already answers on the port would be shown in place of the
# gateway that is started for the demo.
if bin/routeforge healthcheck --url "http://$addr/healthz" --timeout 1s >/dev/null 2>&1; then
  echo "demo.sh: something is already serving on $addr" >&2
  exit 1
fi

log="$(mktemp "${TMPDIR:-/tmp}/routeforge-demo.XXXXXX")"
gateway_pid=""
cleanup() {
  if [ -n "$gateway_pid" ]; then
    kill -TERM "$gateway_pid" 2>/dev/null || true
    wait "$gateway_pid" 2>/dev/null || true
    gateway_pid=""
  fi
  rm -f "$log"
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

bin/routeforge serve --config "$config" --listen "$addr" >"$log" 2>&1 &
gateway_pid=$!
ready=""
for ((i = 0; i < 60; i++)); do
  if ! kill -0 "$gateway_pid" 2>/dev/null; then
    gateway_pid=""
    echo "demo.sh: the gateway exited before it was ready:" >&2
    cat "$log" >&2
    exit 1
  fi
  if bin/routeforge healthcheck --url "http://$addr/readyz" --timeout 1s >/dev/null 2>&1; then
    ready=1
    break
  fi
  sleep 0.5
done
if [ -z "$ready" ]; then
  echo "demo.sh: the gateway was not ready within 30 seconds:" >&2
  cat "$log" >&2
  exit 1
fi

bin/routeforge-bench demo \
  --base-url "http://$addr" \
  --key rf_local_demo_only \
  --root . \
  --out "$output"
cat "$output"
