# 0003: Integer nano-USD accounting

Status: accepted

## Context

The accounting has invariants that tests check exactly, under fault injection:

- for every tenant, settled spend equals the sum of its usage records;
- the reserved amount returns to zero once all requests finish;
- when providers honor the limits sent, spend stays within the budget;
- when a provider reports more usage than the limits allow, the spend beyond the budget equals the overshoot metric.

Sums of floating-point money drift with the order of addition, so those equalities cannot be exact. Rounding to cents would lose the cost of a cheap request. The options were float64 USD, a decimal library, integer micro-USD and integer nano-USD.

## Decision

Every money amount is an `int64` count of nano-USD (1e-9 USD).

- A price of `x` USD per 1M tokens is `x * 1000` nano-USD per token, and it must be a whole number. The price loader rejects a price with more than three decimals.
- A tenant's `daily_budget_usd` must be a whole number of nano-USD and at most 2^50 nano-USD, about 1.1 million USD, the range in which the conversion from USD to nano-USD is exact in float64. Config validation and `accounting.USDToNanos` apply the same rule, and a test checks that they agree.
- Cost is `prompt tokens * input price + completion tokens * output price`, in integers. A reservation is `prompt bytes * input price + output cap * output price`, in integers.
- Rate-limit rates are capped at 100,000,000 per minute so that the Redis Lua scripts, which use doubles, stay exact. Budgets in Redis are integer nano-USD, and the cost of a usage record in PostgreSQL is a `bigint`.
- Amounts become USD floats only at the edges: the response's `cost_usd`, the metrics and the admin view.

## Consequences

- The invariants are exact equalities, and the fault benchmark checks them through `GET /admin/accounting`, which returns both USD and nano-USD.
- The overshoot is defined so that its total equals `max(0, spend - budget)`: each settlement adds `max(0, spent_after - max(budget, spent_before))`.
- A model priced with four or more decimals per 1M tokens cannot be loaded, and would need a smaller unit.
- The budget cap is a real limit. A larger one would need budgets expressed in integer USD.
- Expected cost in the router stays in float64 USD, because it is an estimate that has to match the trainer's float arithmetic, and no money moves on it.
- The costs the gateway reports are provider-reported or estimated tokens priced with the local table. RouteForge never sees invoices.
