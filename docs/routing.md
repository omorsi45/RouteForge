# Routing

This page explains how RouteForge chooses a model, how the threshold behind that choice is certified, and how to read the explanation it returns. It describes the method as built. It states no measured result: those are in the generated [benchmarks](benchmarks.md) and in the trainer's reports, [training.md](../trainer/artifacts/reports/training.md) and [evaluation.md](../trainer/artifacts/reports/evaluation.md).

A request for `routeforge-auto` is routed. A request that names a configured model is pinned to it: it skips scoring and keeps authentication, rate limits, budgets, queues and breakers.

## Features

Python and Go compute the same features. The definitions live in `trainer/routeforge_train/features.py` and `internal/policy/features.go` and are pinned by `testdata/parity/features.jsonl`.

- Feature text: the message contents joined in order with `\n`. For array content, the text parts are joined with `\n`. Role names are not part of it. A content part that is not an object with a string `text` is rejected, and a lone UTF-16 surrogate escape becomes U+FFFD in both languages.
- Tokens: the text is read as UTF-8 bytes, `A` to `Z` are lowercased, bytes in `[a-z0-9]` are token bytes and every other byte separates tokens. Lowercasing is ASCII only, to avoid Unicode case-mapping differences between languages.
- Hashed features: unigrams and bigrams (two tokens joined by one space byte), hashed with FNV-1a 32-bit into 2^18 buckets. A bucket's value is `1 + ln(count)`, and the values are L2-normalized. The norm is summed sequentially in ascending index order and each value is divided by it.
- Dense features: `ln(1 + bytes)`, `ln(1 + tokens)`, the digit-byte fraction, the uppercase-byte fraction of the original text, the count of three backticks, the number of lines, the count of `?` and the longest token length. They are standardized with the mean and standard deviation exported in the policy.
- The predictors use the hashed buckets that appear in at least 5 training documents (at most 20,000, most frequent first) and the eight dense features.

Parity rules: Go must produce identical bucket indices, values within 1e-12 and scores within 1e-9. Calibration block, decision and reason must be identical. Nothing on a parity path may use a fused multiply-add, because Go may fuse where Python never does.

## Predictors

There is one binary classifier per pool model, and its target is whether that model succeeded on the prompt.

- Logistic regression with an L2 penalty. The strength `C` is chosen per model by 5-fold cross-validation on the train split, by log loss. Weights are exported rounded to 6 significant digits.
- Gradient boosting: scikit-learn `GradientBoostingClassifier` with 200 estimators, depth 3, learning rate 0.1, subsample 0.8 and a fixed seed. The trees are exported as flat arrays and compared on float32-rounded inputs, as scikit-learn does.
- kNN (cosine on the hashed vector, k = 25) and keyword rules are offline baselines. kNN raw scores go through the same calibration and certification so that the comparison is like for like. Neither is ever served.

The served predictor is logistic regression or gradient boosting. It is chosen on the calibration split only, by running the same certification procedure for each candidate and comparing cost per success at its lambda. The training report records the choice and the numbers of the candidate that lost.

## Calibration and lower bounds

A raw score is not a probability, so each model gets a calibration fitted on the calibration split:

1. Isotonic regression over the raw scores.
2. Adjacent blocks are merged until every block has at least 40 calibration examples.
3. Each block stores its raw-score range `[lo, hi)`, `n` examples, `k` successes, `p = k / n` and `lcb`, the one-sided 95% Wilson lower bound (z = 1.645).

Serving finds the block of a raw score by binary search. There is no interpolation, so a score that falls in a block gets that block's `p` and `lcb`. Platt scaling is evaluated offline for comparison with Brier score, expected calibration error and reliability diagrams.

The lower bound is what makes the router conservative. A block with a high `p` and few examples has a lower `lcb` than one with the same `p` and many. An explanation reads like this: on the calibration set, `k` of `n` requests scored like this succeeded on model X, and the 95% lower bound is `lcb`.

## The decision rule

Strategy `quality`, the default for `routeforge-auto`:

1. Candidates are the configured models whose `policy_name` is a model of the active policy and which the tenant may use, in pool order. Pool order is `gpt-5`, `claude-sonnet-4`, `gemini-2.5-pro`, `gemini-2.5-flash`, `deepseek-v3.1-terminus`, `qwen3-235b-a22b-2507`, and it breaks every tie.
2. A candidate is ineligible, with a recorded reason, when:
   - `breaker_open`: its circuit breaker is open;
   - `queue_full`: all its slots are busy and its waiting line is at its limit;
   - `context_window`: the estimated prompt tokens plus the reserved output tokens exceed its `context_window` (checked only when the config sets one);
   - `latency_slo`: its predicted p95 latency exceeds the time left before the deadline;
   - `budget`: the reservation for this request on this model exceeds what remains of the tenant's daily budget.
3. Every candidate is scored: calibrated `p`, `lcb` and the expected cost `(ceil(bytes / 4) * input_per_mtok + expected_output_tokens * output_per_mtok) / 1e6`, computed in that order in float64. `expected_output_tokens` is the median completion length of that model in the train split, exported in the policy.
4. Eligible candidates are stably ordered by expected cost, cheapest first. Equal costs keep pool order. For example `gpt-5` and `gemini-2.5-pro` have equal list prices in `deploy/prices.yaml`, so their expected costs tie. The first candidate with `lcb >= lambda` wins, with reason `cheapest_passing`. `lambda` is the certified threshold of the request's profile (strict, balanced or economy).
5. If none passes, the policy's `strongest_model` wins when it is eligible, otherwise the eligible candidate with the highest `p`. The reason is `escalated_no_model_passed`. Because a Wilson bound is always below 1, a lambda of 1.00 sends everything to the strongest model, so the top of the grid is always safe.
6. If no candidate is eligible, the request gets 503 `no_eligible_model` with each model's reasons, or 503 `overloaded` when every candidate's queue was full. `POST /v1/routing/explain` returns the decision together with that error.
7. Fallbacks, tried in order when an attempt fails: the other passing candidates by cost, then the other eligible candidates by descending `p`.
8. `guarantee_applies` is true only when every pool model was a candidate and eligible, because certification assumed the full pool.

If the policy has a model with no configured model, the policy is rejected when it is added, so the registry never holds one the gateway cannot serve.

### Other strategies

A tenant with `allow_strategy_override` can send `X-RouteForge-Strategy`. These exist for benchmarks and comparisons. They use the same eligibility.

| Strategy | Picks | Reason |
| --- | --- | --- |
| `quality` | The rule above. | `cheapest_passing`, `escalated_no_model_passed` |
| `strongest` | The policy's strongest model. | `strongest` |
| `cheapest` | The lowest expected cost. | `cheapest` |
| `static` | A keyword classifier names the family of the prompt, and each family has the cheapest model whose train accuracy on it is within 2 points of the best. | `static_family` |
| `latency` | The lowest predicted p95 latency. | `lowest_latency` |

When a strategy's own pick is ineligible, the eligible candidate with the highest `p` is chosen and the reason is `strategy_pick_ineligible`. Without an active policy the gateway behaves like a plain router: only pinned requests and `strongest` (the first allowed model marked `strongest: true` in the config) work, and any other strategy is a 400.

## Certification with Learn-then-Test

The threshold is not tuned by hand. The trainer certifies it.

A regret event for a prompt is that the routed model fails while the strongest model succeeds. The regret rate `R(lambda)` is the mean of that event. Each profile has a target `alpha` (strict 0.02, balanced 0.05, economy 0.10) and a confidence parameter `delta` of 0.05.

1. On the certification split, the quality rule is simulated with all models eligible for every lambda of the grid 1.00, 0.99, ..., 0.00.
2. Fixed-sequence testing goes from 1.00 downward. At each lambda the p-value is `P(Binomial(n, alpha) <= k)`, where `k` is the regret count. If it is at most `delta`, the null hypothesis "`R(lambda) > alpha`" is rejected and the test continues. At the first non-rejection it stops.
3. The certified lambda is the last rejected one. If even 1.00 is not rejected, which can only happen when `n` is below `ln(delta) / ln(1 - alpha)`, the profile gets the sentinel 1.01, meaning always strongest, and the training report says so.
4. The policy stores `n`, the regret count and the one-sided 95% Clopper-Pearson upper bound of the regret rate at the certified lambda.

The order of operations keeps selection away from certification data. The predictors are fitted on the train split and calibrated on the calibration split. The served predictor is chosen on the calibration split. Only that winner is certified, on the certification split, with the full `delta`. The test split is written once and read only by the final evaluation command.

This is the guarantee, and the only one RouteForge states:

With probability at least 1 - delta over the draw of the certification set, the expected regret rate on future traffic exchangeable with it is at most alpha, when all pool models are eligible. It says nothing about any single request.

It is marginal, over traffic that looks like the certification data. It does not hold when a pool model was ineligible, which is what `guarantee_applies` tells you per decision. It does not extend to traffic of another kind, which is why leave-one-dataset-out results exist: there the guarantee is not expected to hold.

## Latency profiles are SIMULATED

The `latency_slo` check and the `latency` strategy use a predicted p95 latency per model: the time to first token plus the expected output tokens times the time per token. The gateway estimates both with an exponentially weighted mean and variance of the successful attempts it has observed (weight 0.1, p95 as the mean plus 1.645 standard deviations), starting from the model's `latency_prior` in the config.

For benchmarks, the priors and the fake and replay providers use `bench/latency_profiles.yaml`, which is marked `simulated: true`. It applies a fixed rule. Pool models are ranked by output price ascending, rank `r` is 0 for the cheapest, the time to first token is `200 + 100 r` ms, the decode speed is `120 - 15 r` tokens per second, and jitter is plus or minus 10% from a seeded random generator. These numbers are synthetic. They are not measurements of any provider.

Under this rule the cheapest pool model also has the lowest simulated time to first token and the highest simulated decode speed. The offline latency router therefore picks the same model as always-cheapest, unless a model's expected output length is long enough to outweigh the speed advantage of a cheaper one. That follows from the rule, not from a measurement. The generated tables show the row of the latency router. The gateway's `latency` strategy adapts to observed attempts while the offline one uses the frozen priors, so their decisions can differ once traffic flows. The replay report shows that agreement separately from the strategies that are pure functions of the request and the policy. The latency router is a reference only and is not part of H1.

## The decision record

Every decision is a record with no prompt text. It is kept in a ring of 10,000 decisions in memory and, with PostgreSQL storage, in `routing_decisions`. `GET /v1/routing/decisions/{id}` returns it to its own tenant; an unknown id and another tenant's id get the same 404.

| Field | Meaning |
| --- | --- |
| `id`, `tenant`, `created_at` | Identity and time. The id starts with `dec_`. |
| `policy_version` | The policy that scored the request. Empty for a pinned request or when no policy is active. |
| `strategy`, `profile` | The strategy used (`pinned` for a named model) and the profile. |
| `chosen`, `reason` | The model and why: see the table above, plus `pinned`. |
| `lambda` | The profile's certified threshold. Null unless the strategy is `quality`. |
| `guarantee_applies` | Whether every pool model was a candidate and eligible. |
| `decided_in_us` | How long the decision took, in microseconds. |
| `candidates` | One entry per candidate, below. |
| `fallbacks` | Model names in failover order. |
| `attempts` | The upstream attempts, below. Empty for an explain request. |

A candidate has `model`, `eligible`, `ineligible_reasons`, `raw` (the uncalibrated score), `p`, `lcb`, `block_n`, `block_k`, `expected_cost_usd` and `predicted_p95_ms`. A value the decision did not compute is null. An attempt has `model`, `outcome` (`ok`, `error`, `canceled` or `partial`), `error_kind` and `duration_ms`, a float with microsecond resolution.

## Reading `/v1/routing/explain`

`POST /v1/routing/explain` takes the same body and headers as a chat request and returns the decision without calling any provider. The decision is returned, not stored. Reading one:

1. `reason`. `cheapest_passing` says the chosen model was the cheapest eligible candidate whose `lcb` reached `lambda`. `escalated_no_model_passed` says no candidate's `lcb` reached `lambda`, so the policy's strongest model was chosen even though its own `lcb` may be below `lambda`.
2. `lambda` against each candidate's `lcb`. The gap shows how close a cheaper model came.
3. `block_k` of `block_n`. They are the calibration counts behind `p` and `lcb`: `block_k` of `block_n` calibration requests that scored like this one succeeded on that model.
4. `expected_cost_usd`. It is the policy's estimate from the model's typical output length, not a quote. The cost of a served request is in the `routeforge` object of its response and in its usage record, and the two are not comparable.
5. `ineligible_reasons`. A model that is not eligible has at least one reason from the list above, and it explains why a cheaper model was skipped.
6. `fallbacks`, the failover order.
7. `guarantee_applies`. When it is false, the certified guarantee does not cover this decision.

## Hypothesis H1

H1 is a hypothesis, not a result. On a held-out test split from the same distribution as calibration, for the balanced profile (alpha 0.05), the quality-aware policy has lower cost per successful task than always-strongest (target: at least 25% lower), keeps its regret rate at or below alpha, and beats the best static per-family rule and the share-matched random router on cost per successful task at equal or better accuracy.

It is predicted that gains grow with the within-family variance of cheap-model success, with the price spread between models and with the predictor's within-family AUROC, and that they vanish when cheap models almost always or almost never suffice.

H1 is falsified, and reported as such, if any of these holds for the balanced profile and the served predictor:

1. The paired bootstrap 95% interval of the cost-per-success difference against the share-matched random router is not entirely below zero.
2. The same interval against the static rule with the oracle family (an upper bound for rules) is not entirely below zero.
3. The lower end of the 95% interval of the realized test regret rate is above alpha.
4. The pooled leave-one-dataset-out reduction in cost per success has a 95% interval that is not above zero, so the gains disappear on held-out datasets.

The 25% target is reported separately and does not decide falsification. The share-matched random router assigns the quality policy's realized model counts without looking at the content, averaged over 1,000 seeds. The bootstrap is paired and stratified by dataset, with 10,000 resamples. The evaluation also reports an exact McNemar test against always-strongest, Holm correction across baselines and a TOST test for no accuracy drop with a pre-declared margin of 2 points. The verdict is in the generated block of the [README](../README.md), in [benchmarks.md](benchmarks.md) and in [evaluation.md](../trainer/artifacts/reports/evaluation.md).

## What the evidence rests on

- Quality and cost come from replaying recorded LLMRouterBench outputs through the router. No live provider was called. Offline costs are the dataset's recorded cost field, which is provider usage times list price.
- Gateway performance comes from fake providers with the SIMULATED latency profiles above.
- Benchmark prompts come from templates that let a classifier recognize the dataset. That is realistic for application traffic but inflates in-distribution gains, so the evaluation reports leave-one-dataset-out results next to the in-distribution ones.
- Extraction quality is unmeasured. The extraction workload and `make label-extraction` exist, but until labels exist no extraction result is reported.
- RouteForge never sees invoices. A reported cost is provider-reported tokens priced with the local table, and an estimated cost uses `ceil(bytes / 4)` tokens.
- There is no online learning. There is no live labeled traffic, and the offline data is full-information, so supervised learning dominates. A bandit or a canary would need logged propensities and online quality feedback, which is future work. The only drift signal is the `routeforge_predicted_success` histogram.

## Routing metrics

`routeforge_requests_total` counts chat requests only. Its `model` label, like the `model` field of the access log, is the model that served the request: after a failover that is the fallback, the same model `X-RouteForge-Model` names. A request that no model served (every attempt failed, or none was made) keeps the model the decision chose. `routeforge_route_decisions_total` always counts the chosen model, so the two differ by the failovers. The `status` label is the HTTP status, with 499 for a client that left. An explain request is counted only in the decision metrics (`routeforge_route_decisions_total`, `routeforge_route_decision_seconds` and `routeforge_predicted_success`). [operations.md](operations.md) lists the metrics.
