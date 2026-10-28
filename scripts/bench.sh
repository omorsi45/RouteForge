#!/usr/bin/env bash
# Run one benchmark.
#
# Usage: scripts/bench.sh <replay|load|faults> <run-id>
#
# The script builds bin/routeforge and bin/routeforge-bench, then:
#   replay  starts the gateway with bench/configs/replay.yaml on 127.0.0.1:18080,
#           waits up to 30 seconds for /readyz and runs the replay benchmark;
#   faults  does the same with bench/configs/faults.yaml and runs the fault
#           benchmark;
#   load    runs the load benchmark for bench/configs/load-zero.yaml and then
#           bench/configs/load-profiled.yaml. The benchmark starts and stops its
#           own gateway for every scenario, on the same port.
# A gateway the script started is stopped with SIGTERM whichever way the script
# ends. Its exit code is the benchmark's. Results go to bench/results/<run-id>/,
# and the gateway logs to bench/results/<run-id>/raw/, which git ignores. It
# needs Go and bash, and nothing else.
set -euo pipefail

usage() {
  echo "usage: scripts/bench.sh <replay|load|faults> <run-id>" >&2
  exit 2
}

[ "$#" -eq 2 ] || usage
kind="$1"
run_id="$2"
case "$kind" in
  replay | load | faults) ;;
  *) usage ;;
esac
if ! [[ "$run_id" =~ ^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$ ]]; then
  echo "bench.sh: invalid run id '$run_id'" >&2
  exit 2
fi

cd "$(dirname "${BASH_SOURCE[0]}")/.."
port=18080
addr="127.0.0.1:$port"
raw_dir="bench/results/$run_id/raw"
mkdir -p "$raw_dir"

"${GO:-go}" build -o bin/routeforge ./cmd/routeforge
"${GO:-go}" build -o bin/routeforge-bench ./cmd/routeforge-bench

# Something that already answers on the port would be benchmarked in place of
# the gateway that is started for the run.
if bin/routeforge healthcheck --url "http://$addr/healthz" --timeout 1s >/dev/null 2>&1; then
  echo "bench.sh: something is already serving on $addr" >&2
  exit 1
fi

# The gateway and the benchmark each hold a connection per request in flight, and
# the default limit of 1024 open files is below what the highest load asks for.
if [ "$kind" != replay ]; then
  want=65536
  hard="$(ulimit -Hn)"
  if [ "$hard" != unlimited ] && [ "$hard" -lt "$want" ]; then
    want="$hard"
  fi
  if [ "$(ulimit -n)" -lt "$want" ]; then
    ulimit -n "$want" 2>/dev/null || echo "bench.sh: could not raise the open file limit; it is $(ulimit -n)" >&2
  fi
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

# start_gateway CONFIG starts the gateway and waits for it to be ready.
start_gateway() {
  local config="$1" log="$raw_dir/gateway-$kind.log" ready=""
  bin/routeforge serve --config "$config" --listen "$addr" >"$log" 2>&1 &
  gateway_pid=$!
  for ((i = 0; i < 60; i++)); do
    if ! kill -0 "$gateway_pid" 2>/dev/null; then
      gateway_pid=""
      echo "bench.sh: the gateway exited before it was ready; see $log" >&2
      exit 1
    fi
    if bin/routeforge healthcheck --url "http://$addr/readyz" --timeout 1s >/dev/null 2>&1; then
      ready=1
      break
    fi
    sleep 0.5
  done
  if [ -z "$ready" ]; then
    echo "bench.sh: the gateway was not ready within 30 seconds; see $log" >&2
    exit 1
  fi
}

# The benchmark's exit code is the script's, and a gateway this script started
# is stopped on the way out by the EXIT trap.
code=0
case "$kind" in
  replay)
    start_gateway bench/configs/replay.yaml
    set +e
    bin/routeforge-bench replay \
      --base-url "http://$addr" \
      --key test-key-bench \
      --config bench/configs/replay.yaml \
      --root . \
      --run-id "$run_id" \
      --workers 16
    code=$?
    set -e
    ;;
  faults)
    start_gateway bench/configs/faults.yaml
    set +e
    bin/routeforge-bench faults \
      --base-url "http://$addr" \
      --key test-key-bench \
      --over-key test-key-bench-over \
      --admin-key test-admin-key \
      --config bench/configs/faults.yaml \
      --root . \
      --run-id "$run_id"
    code=$?
    set -e
    ;;
  load)
    for name in load-zero load-profiled; do
      set +e
      bin/routeforge-bench load \
        --gateway-bin bin/routeforge \
        --config "bench/configs/$name.yaml" \
        --root . \
        --run-id "$run_id"
      code=$?
      set -e
      [ "$code" -eq 0 ] || break
    done
    ;;
esac
exit "$code"
