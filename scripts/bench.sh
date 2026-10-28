#!/usr/bin/env bash
# Run one benchmark against a gateway that this script starts and stops.
#
# Usage: scripts/bench.sh replay <run-id>
#
# The script builds bin/routeforge and bin/routeforge-bench, starts the gateway
# with the config of the benchmark on 127.0.0.1:18080, waits up to 30 seconds
# for /readyz, runs the benchmark and stops the gateway with SIGTERM, whichever
# way it ends. Its exit code is the benchmark's. The result goes to
# bench/results/<run-id>/ and the gateway log to bench/results/<run-id>/raw/,
# which git ignores. It needs Go and bash, and nothing else.
set -euo pipefail

usage() {
  echo "usage: scripts/bench.sh replay <run-id>" >&2
  exit 2
}

[ "$#" -eq 2 ] || usage
kind="$1"
run_id="$2"
case "$kind" in
  replay) ;;
  *) usage ;;
esac
if ! [[ "$run_id" =~ ^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$ ]]; then
  echo "bench.sh: invalid run id '$run_id'" >&2
  exit 2
fi

cd "$(dirname "${BASH_SOURCE[0]}")/.."
port=18080
addr="127.0.0.1:$port"
config="bench/configs/$kind.yaml"
raw_dir="bench/results/$run_id/raw"
gateway_log="$raw_dir/gateway-$kind.log"
mkdir -p "$raw_dir"

"${GO:-go}" build -o bin/routeforge ./cmd/routeforge
"${GO:-go}" build -o bin/routeforge-bench ./cmd/routeforge-bench

# Something that already answers on the port would be benchmarked in place of
# the gateway this script starts.
if bin/routeforge healthcheck --url "http://$addr/healthz" --timeout 1s >/dev/null 2>&1; then
  echo "bench.sh: something is already serving on $addr" >&2
  exit 1
fi

gateway_pid=""
stop_gateway() {
  if [ -n "$gateway_pid" ]; then
    kill -TERM "$gateway_pid" 2>/dev/null || true
    wait "$gateway_pid" 2>/dev/null || true
    gateway_pid=""
  fi
}
trap stop_gateway EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

bin/routeforge serve --config "$config" --listen "$addr" >"$gateway_log" 2>&1 &
gateway_pid=$!

ready=""
for ((i = 0; i < 60; i++)); do
  if ! kill -0 "$gateway_pid" 2>/dev/null; then
    gateway_pid=""
    echo "bench.sh: the gateway exited before it was ready; see $gateway_log" >&2
    exit 1
  fi
  if bin/routeforge healthcheck --url "http://$addr/readyz" --timeout 1s >/dev/null 2>&1; then
    ready=1
    break
  fi
  sleep 0.5
done
if [ -z "$ready" ]; then
  echo "bench.sh: the gateway was not ready within 30 seconds; see $gateway_log" >&2
  exit 1
fi

# The benchmark's exit code is the script's, and the gateway is stopped on the
# way out by the EXIT trap.
set +e
bin/routeforge-bench "$kind" \
  --base-url "http://$addr" \
  --key test-key-bench \
  --config "$config" \
  --root . \
  --run-id "$run_id" \
  --workers 16
code=$?
set -e
exit "$code"
