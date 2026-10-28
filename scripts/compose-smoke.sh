#!/usr/bin/env bash
# Smoke test for the compose stack in deploy/docker-compose.yml.
#
# Usage: scripts/compose-smoke.sh
#
# The script builds the image and starts the gateway with Postgres, Redis and
# Prometheus, then checks, in this order:
#   1. /healthz answers 200;
#   2. /readyz answers 200 (retried for up to 60 seconds, because the
#      migrations and the first policy load take a moment);
#   3. one routeforge-auto chat request with the public demo key answers 200
#      and carries an X-RouteForge-Policy-Version header of 12 hex digits;
#   4. /metrics lists routeforge_requests_total.
# The stack is always removed, volumes included, on the way out. When a check
# failed, the last 50 lines of the gateway log are printed first. The exit code
# is 0 only if every check passed.
#
# Environment:
#   DOCKER    the Docker CLI to run (default: docker)
#   BASE_URL  where the compose file publishes the gateway
#             (default: http://127.0.0.1:8080)
# It needs bash, curl and Docker with the compose plugin.
set -euo pipefail

DOCKER="${DOCKER:-docker}"
BASE_URL="${BASE_URL:-http://127.0.0.1:8080}"
# The public key of the demo tenant in deploy/configs/compose.yaml.
demo_key="rf_local_demo_only"

cd "$(dirname "${BASH_SOURCE[0]}")/.."
compose=("$DOCKER" compose -f deploy/docker-compose.yml)
work="$(mktemp -d)"

cleanup() {
  local code=$?
  trap - EXIT
  if [ "$code" -ne 0 ]; then
    echo "compose-smoke: failed with exit code $code; the last 50 lines of the gateway log follow" >&2
    { "${compose[@]}" logs routeforge --no-color 2>&1 | tail -50; } >&2 || true
  fi
  "${compose[@]}" down -v || echo "compose-smoke: compose down failed" >&2
  rm -rf "$work"
  exit "$code"
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

fail() {
  echo "compose-smoke: FAIL: $*" >&2
  exit 1
}

pass() {
  echo "compose-smoke: ok: $*"
}

# status_of URL prints the HTTP status of a GET, or 000 when nothing answers.
status_of() {
  curl -sS -o /dev/null -w '%{http_code}' --max-time 5 "$1" 2>/dev/null || true
}

# wait_for_200 PATH SECONDS asks once a second until PATH answers 200 and fails
# when it has not within SECONDS.
wait_for_200() {
  local path="$1" seconds="$2" status="" i
  for ((i = 0; i < seconds; i++)); do
    status="$(status_of "$BASE_URL$path")"
    if [ "$status" = 200 ]; then
      pass "$path answered 200"
      return 0
    fi
    sleep 1
  done
  fail "$path answered $status, want 200 within ${seconds}s"
}

"${compose[@]}" up -d --build --wait

wait_for_200 /healthz 60
wait_for_200 /readyz 60

request='{"model":"routeforge-auto","messages":[{"role":"user","content":"Reply with one short sentence."}],"max_tokens":32}'
status="$(curl -sS -o "$work/chat.json" -D "$work/chat.headers" -w '%{http_code}' --max-time 30 \
  -H "Authorization: Bearer $demo_key" \
  -H 'Content-Type: application/json' \
  -d "$request" "$BASE_URL/v1/chat/completions")" || fail "the chat request did not complete"
if [ "$status" != 200 ]; then
  fail "the chat request answered $status, want 200: $(head -c 300 "$work/chat.json")"
fi
version="$(tr -d '\r' <"$work/chat.headers" | grep -i '^x-routeforge-policy-version:' | sed 's/^[^:]*: *//' || true)"
if ! [[ "$version" =~ ^[0-9a-f]{12}$ ]]; then
  fail "X-RouteForge-Policy-Version is '$version', want 12 hex digits"
fi
pass "the chat request answered 200 under policy $version"

curl -fsS --max-time 5 -o "$work/metrics.txt" "$BASE_URL/metrics" || fail "/metrics did not answer 200"
if ! grep -q '^routeforge_requests_total' "$work/metrics.txt"; then
  fail "/metrics does not list routeforge_requests_total"
fi
pass "/metrics lists routeforge_requests_total"

echo "compose-smoke: all checks passed"
