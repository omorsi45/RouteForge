package redisstore

import "github.com/redis/go-redis/v9"

// Every operation is one script, so Redis runs it without interleaving: two
// gateway replicas can never both pass the same check. Numbers are kept as
// integers below 2^53, where a Lua double is exact, and are written back with
// %.0f so no digit is lost to the default number formatting. Scripts run
// through Script.Run: EVALSHA first, EVAL when the server has no copy.

// allowScript is a token bucket whose tokens are counted in micro-tokens
// (tokens x 1e6), refilled from the server clock in microseconds, so every
// replica sees one clock and the arithmetic stays integral.
//
//	KEYS[1]  bucket hash (fields tokens, ts_us)
//	ARGV[1]  n, tokens wanted
//	ARGV[2]  perMinute, the capacity; refill is perMinute/60 tokens a second
//	ARGV[3]  key expiry in seconds
//
// Returns {allowed (1 or 0), retry-after in whole seconds}. The rules are those
// of accounting.MemoryLimiter: a bucket starts full; a clock that goes back
// refills nothing and is not followed back; a denial takes nothing; a request
// above the capacity waits a minute; any other wait is rounded up to a whole
// second (never zero, because the shortfall is at least one micro-token).
//
// One microsecond refills perMinute/60 micro-tokens. The refill is rounded
// down to whole micro-tokens, and the bucket's timestamp moves only when
// something was added (or the bucket is full), so calls closer together than
// one micro-token never lose the time between them. A span of a minute or more
// is capped, which fills any bucket and keeps the product below 2^53 for rates
// up to maxPerMinute.
var allowScript = redis.NewScript(`
local n = tonumber(ARGV[1])
local rate = tonumber(ARGV[2])
local capacity = rate * 1000000

local t = redis.call('TIME')
local now = tonumber(t[1]) * 1000000 + tonumber(t[2])

local v = redis.call('HMGET', KEYS[1], 'tokens', 'ts_us')
local tokens = tonumber(v[1])
local ts = tonumber(v[2])
if tokens == nil or ts == nil then
  tokens = capacity
  ts = now
else
  local elapsed = now - ts
  if elapsed > 0 then
    local refill = math.floor(math.min(elapsed, 60000000) * rate / 60)
    tokens = tokens + refill
    if refill > 0 or tokens >= capacity then
      ts = now
    end
  end
end
tokens = math.min(tokens, capacity)

local allowed = 0
local retry = 0
if n > rate then
  retry = 60
elseif tokens >= n * 1000000 then
  tokens = tokens - n * 1000000
  allowed = 1
else
  retry = math.ceil((n * 1000000 - tokens) * 60 / capacity)
end

redis.call('HSET', KEYS[1], 'tokens', string.format('%.0f', tokens), 'ts_us', string.format('%.0f', ts))
redis.call('EXPIRE', KEYS[1], ARGV[3])
return {allowed, retry}
`)

// reserveScript holds an amount back when spent + reserved + amount stays
// within the limit and holds nothing otherwise, like MemoryBudgets.Reserve.
//
//	KEYS[1]  budget hash (fields spent, reserved, overshoot)
//	ARGV[1]  amount, ARGV[2] limit, ARGV[3] key expiry in seconds
//
// Returns 1 when the amount was reserved, else 0. A rejection writes nothing,
// so it does not renew the expiry.
var reserveScript = redis.NewScript(`
local v = redis.call('HMGET', KEYS[1], 'spent', 'reserved', 'overshoot')
local spent = tonumber(v[1]) or 0
local reserved = tonumber(v[2]) or 0
local overshoot = tonumber(v[3]) or 0
local amount = tonumber(ARGV[1])

if spent + reserved + amount > tonumber(ARGV[2]) then
  return 0
end

redis.call('HSET', KEYS[1],
  'spent', string.format('%.0f', spent),
  'reserved', string.format('%.0f', reserved + amount),
  'overshoot', string.format('%.0f', overshoot))
redis.call('EXPIRE', KEYS[1], ARGV[3])
return 1
`)

// settleScript returns a reservation, charges the actual cost and adds the
// overshoot of this charge, max(0, spent_after - max(limit, spent_before))
// (Ruling R7), like MemoryBudgets.Settle. A limit near the int64 maximum, which
// the accountant uses for a tenant without a budget, is not exact as a double
// and does not need to be: no spend gets near it.
//
//	KEYS[1]  budget hash (fields spent, reserved, overshoot)
//	ARGV[1]  reserved, ARGV[2] actual, ARGV[3] limit, ARGV[4] key expiry in seconds
//
// Returns the overshoot this charge added.
var settleScript = redis.NewScript(`
local v = redis.call('HMGET', KEYS[1], 'spent', 'reserved', 'overshoot')
local before = tonumber(v[1]) or 0
local reserved = tonumber(v[2]) or 0
local total = tonumber(v[3]) or 0

local spent = before + tonumber(ARGV[2])
reserved = reserved - tonumber(ARGV[1])
local over = math.max(0, spent - math.max(tonumber(ARGV[3]), before))
total = total + over

redis.call('HSET', KEYS[1],
  'spent', string.format('%.0f', spent),
  'reserved', string.format('%.0f', reserved),
  'overshoot', string.format('%.0f', total))
redis.call('EXPIRE', KEYS[1], ARGV[4])
return over
`)
