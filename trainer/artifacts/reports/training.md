# RouteForge training report

## Provenance

- Dataset: NPULH/LLMRouterBench
- Revision: 0e5af1b84bf73437a01a1849c0f1d2468baa93fc
- Archive sha256: b79f8cde1a6f029c2efa663a3a3b6f7748defb22341fe59f328cebef6648c8f1
- Git SHA: 41f9629e8a4bda56119e4cb91e2a813a81cce770
- Policy version: 7ae71d0e6deb
- Created at: 2026-10-28T00:09:12Z
- Environment: matplotlib 3.11.2, numpy 2.5.4, platform Linux-6.18.33.2-microsoft-standard-WSL2-x86_64-with-glibc2.39, python 3.14.10, scipy 1.18.1, sklearn 1.9.1

## Data

Pool, in tie-break order: gpt-5, claude-sonnet-4, gemini-2.5-pro, gemini-2.5-flash, deepseek-v3.1-terminus, qwen3-235b-a22b-2507.

| Dataset | Family | Prompts in | Kept | duplicate_row | missing_model | prompt_mismatch | Label rule | train | calibration | certification | test |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| aime | reasoning | 60 | 60 | 0 | 0 | 0 | score >= 1.0 | 30 | 12 | 9 | 9 |
| arenahard | instruction | 750 | 749 | 0 | 0 | 1 | score >= 0.5 | 374 | 149 | 112 | 114 |
| gpqa | reasoning | 198 | 198 | 0 | 0 | 0 | score >= 1.0 | 99 | 39 | 29 | 31 |
| hle | qa | 2158 | 2157 | 0 | 0 | 1 | score >= 1.0 | 1078 | 431 | 323 | 325 |
| livecodebench | code | 1055 | 1055 | 0 | 0 | 0 | score >= 1.0 | 527 | 211 | 158 | 159 |
| livemathbench | reasoning | 121 | 121 | 0 | 0 | 0 | score >= 1.0 | 60 | 24 | 18 | 19 |
| mmlupro | reasoning | 3000 | 3000 | 0 | 0 | 0 | score >= 1.0 | 1500 | 600 | 450 | 450 |
| simpleqa | qa | 4326 | 4326 | 0 | 0 | 0 | score >= 1.0 | 2163 | 865 | 648 | 650 |

Prompts per split: train 5831, calibration 2331, certification 1747, test 1757.

## Predictors

- Hashed features: 262144 buckets; vocabulary of 20000 indices (document frequency at least 5, at most 20000).
- Logistic regression weights are rounded to 6 significant digits.
- Gradient boosting parameters: learning_rate 0.1, max_depth 3, n_estimators 200, random_state 20261026, subsample 0.8.
- kNN: k = 25, offline only.
- Strongest model: gemini-2.5-pro. Cheapest model: qwen3-235b-a22b-2507.

| Model | LR C | LR CV log loss | Expected output tokens |
| --- | --- | --- | --- |
| gpt-5 | 3.0 | 0.552 | 1312 |
| claude-sonnet-4 | 3.0 | 0.433 | 348 |
| gemini-2.5-pro | 1.0 | 0.551 | 2150 |
| gemini-2.5-flash | 3.0 | 0.522 | 426 |
| deepseek-v3.1-terminus | 3.0 | 0.497 | 284 |
| qwen3-235b-a22b-2507 | 3.0 | 0.530 | 313 |

## Selection (calibration split, balanced)

Calibration prompts: 2331. Profile: balanced.

| Predictor | Lambda | Accuracy | Total cost | Cost per success | Regret rate |
| --- | --- | --- | --- | --- | --- |
| lr | 0.65 | 61.6% | $76.5878 | $0.0533 | 4.0% |
| gbm | 0.67 | 61.2% | $73.0671 | $0.0512 | 4.2% |

Winner: gbm

## Certification

### Served (gbm)

| Profile | Alpha | Lambda | n | Regret count | Regret UCB |
| --- | --- | --- | --- | --- | --- |
| strict | 0.02 | 0.81 | 1747 | 17 | 0.015 |
| balanced | 0.05 | 0.67 | 1747 | 65 | 0.046 |
| economy | 0.10 | 0.50 | 1747 | 152 | 0.099 |

### Offline comparison (lr, never exported)

| Profile | Alpha | Lambda | n | Regret count | Regret UCB |
| --- | --- | --- | --- | --- | --- |
| strict | 0.02 | 0.83 | 1747 | 20 | 0.017 |
| balanced | 0.05 | 0.67 | 1747 | 66 | 0.046 |
| economy | 0.10 | 0.43 | 1747 | 153 | 0.100 |

### Offline comparison (knn, never exported)

| Profile | Alpha | Lambda | n | Regret count | Regret UCB |
| --- | --- | --- | --- | --- | --- |
| strict | 0.02 | 0.82 | 1747 | 18 | 0.015 |
| balanced | 0.05 | 0.61 | 1747 | 63 | 0.044 |
| economy | 0.10 | 0.48 | 1747 | 151 | 0.098 |

## Costs

Offline costs are the dataset-recorded cost field (provider usage times list price per LLMRouterBench). The gateway price table (deploy/prices.yaml) uses list prices. Files whose recorded total differs from the list-price recomputation by more than 1% are listed below.

| Model | Recorded (all rows) | List (all rows) | Recorded (kept rows) | List (kept rows) |
| --- | --- | --- | --- | --- |
| gpt-5 | $330.3665 | $369.9775 | $330.2808 | $369.8917 |
| claude-sonnet-4 | $90.3975 | $90.3975 | $90.3481 | $90.3481 |
| gemini-2.5-pro | $649.9691 | $649.9713 | $649.8753 | $649.8775 |
| gemini-2.5-flash | $101.1147 | $103.4212 | $101.1000 | $103.4065 |
| deepseek-v3.1-terminus | $9.7093 | $9.7093 | $9.7068 | $9.7068 |
| qwen3-235b-a22b-2507 | $8.9077 | $8.5665 | $8.9056 | $8.5643 |

Pairs whose recorded total differs from the list-price total by more than 1%:

| Dataset | Model | Recorded | List | Ratio |
| --- | --- | --- | --- | --- |
| aime | qwen3-235b-a22b-2507 | $0.3090 | $0.2315 | 1.335 |
| livemathbench | qwen3-235b-a22b-2507 | $0.2339 | $0.2419 | 0.967 |
| gpqa | qwen3-235b-a22b-2507 | $0.0657 | $0.0635 | 1.035 |
| mmlupro | qwen3-235b-a22b-2507 | $1.8037 | $1.8636 | 0.968 |
| livecodebench | gemini-2.5-flash | $6.1552 | $8.4592 | 0.728 |
| livecodebench | qwen3-235b-a22b-2507 | $1.0443 | $1.0115 | 1.032 |
| simpleqa | qwen3-235b-a22b-2507 | $0.2021 | $0.2069 | 0.976 |
| hle | gpt-5 | $137.5793 | $177.1824 | 0.776 |
| hle | qwen3-235b-a22b-2507 | $4.5136 | $4.1875 | 1.078 |
| arenahard | qwen3-235b-a22b-2507 | $0.7334 | $0.7580 | 0.968 |

## Guarantee

With probability at least 1 - delta over the draw of the certification set, the expected regret rate on future traffic exchangeable with it is at most alpha, when all pool models are eligible. It says nothing about any single request.
