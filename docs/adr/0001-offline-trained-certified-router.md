# 0001: An offline-trained, certified router

Status: accepted

## Context

RouteForge exists to send each request to the cheapest model that is likely to answer it well, and to show evidence for that. Three approaches were considered.

- A lightweight adaptive router: a library or sidecar with a classifier and a threshold, and no serving concerns. It is quick to build, but it skips everything the project is meant to show about serving and reliability.
- A production-style gateway whose router is trained offline, calibrated and certified.
- An online-learning platform with bandits, canaries, shadow traffic and drift-triggered retraining. It is the most ambitious, but there is no live labeled traffic to validate its core claims, and the offline data records every model's outcome on every prompt, so supervised learning dominates. Its claims would be a list of features nobody can check.

The central claim has to be measurable for no inference spend, on one machine, from recorded outputs.

## Decision

Build the gateway with an offline-trained, calibrated and certified router.

- Python owns training, calibration, certification and evaluation. Go does eligibility, scoring and escalation in process, in microseconds. They meet at a versioned policy file ([ADR 0002](0002-json-policy-artifact-with-parity.md)).
- There is one binary success classifier per pool model, logistic regression or gradient boosting. Its raw score is mapped to a calibration block, which stores how many calibration requests succeeded, and the router uses the one-sided 95% Wilson lower bound of that block.
- The router picks the cheapest eligible model whose lower bound reaches the threshold of the request's profile, and escalates to the strongest model when none does.
- The threshold of each profile is certified with Learn-then-Test on a split that no earlier step used, against the regret of routing away from the strongest model.
- Every decision is recorded with the numbers behind it, and `POST /v1/routing/explain` shows them.
- The claims are tested against honest baselines, including a content-blind random router with the same model mix, with paired statistics and leave-one-dataset-out runs.

## Consequences

- Every quality and cost claim can be rechecked offline from committed files.
- A decision can be explained in terms a reader can check: this many of that many similar calibration requests succeeded.
- The only guarantee is marginal. It holds over traffic exchangeable with the certification data, and only when every pool model is eligible. It says nothing about any single request. `guarantee_applies` on each decision says whether the second condition held.
- The policy does not adapt after it is exported. Retraining is a batch job, and drift shows only as the `routeforge_predicted_success` histogram.
- Benchmark prompt templates let a classifier recognize the dataset, which is realistic for application traffic and inflates in-distribution gains. The evaluation reports leave-one-dataset-out results next to them.
- Online learning, canaries and shadow evaluation are future work. They would need logged propensities and online quality feedback.
- The hypothesis H1 is stated with pre-declared falsification criteria ([routing.md](../routing.md)), and its result is reported whichever way it falls.
