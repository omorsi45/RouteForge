# RouteForge offline evaluation

## Provenance

- Dataset: NPULH/LLMRouterBench
- Revision: 0e5af1b84bf73437a01a1849c0f1d2468baa93fc
- Archive sha256: b79f8cde1a6f029c2efa663a3a3b6f7748defb22341fe59f328cebef6648c8f1
- Git SHA: 41f9629e8a4bda56119e4cb91e2a813a81cce770
- Policy version: 7ae71d0e6deb
- Served predictor: gbm
- Created at: 2026-10-28T00:11:30Z
- Test split: 1757 prompts, sha256 240c607cc2f7e41f9913af6957bba2c73cc2232011d3b5afa0971af748d65486
- Test prompts per dataset: aime 9, arenahard 114, gpqa 31, hle 325, livecodebench 159, livemathbench 19, mmlupro 450, simpleqa 650
- Paired bootstrap: 10000 resamples stratified by dataset; share-matched random averages 1000 seeds

## H1 (balanced, gbm)

- Target: cost per success at least 25.0% below always_strongest. Reduction: 47.5% [44.6%, 50.3%] (95% CI). Target met: yes.
- Regret rate: 4.3% (75 of 1757), 95% CI [3.4%, 5.3%], alpha 5.0%. CI lower bound above alpha: no.
- Cost per success versus share-matched random: -$0.0043 [-$0.0068, -$0.0018]. Beats it: yes.
- Cost per success versus the static oracle family: $0.0045 [$0.0001, $0.0088]. Beats it: no.
- LODO pooled cost per success reduction: 74.9% [72.2%, 77.4%]. Gains disappear: no.

H1: falsified (static_oracle_family)

## Strategies

Test split, recorded costs. Regret is measured against the policy's strongest model.

### strict

| Strategy | Accuracy | Total cost | Cost per success | Regret rate | Escalation rate |
| --- | --- | --- | --- | --- | --- |
| always_strongest | 59.9% | $95.0076 | $0.0902 | 0.0% | n/a |
| always_cheapest | 54.6% | $1.2849 | $0.0013 | 15.0% | n/a |
| static_keyword | 59.1% | $48.5205 | $0.0467 | 10.4% | n/a |
| static_oracle_family | 59.5% | $44.8313 | $0.0429 | 10.1% | n/a |
| latency | 54.6% | $1.2849 | $0.0013 | 15.0% | n/a |
| quality/lr/strict | 59.9% | $72.5427 | $0.0689 | 1.5% | 72.9% |
| quality/gbm/strict | 60.2% | $78.6197 | $0.0743 | 0.9% | 61.3% |
| quality/knn/strict | 60.7% | $72.5454 | $0.0681 | 1.1% | 68.5% |
| share_matched_random/strict | 58.8% | $81.7653 | $0.0791 | 2.9% | n/a |
| oracle | 77.2% | $12.1652 | $0.0090 | 0.0% | n/a |

### balanced

| Strategy | Accuracy | Total cost | Cost per success | Regret rate | Escalation rate |
| --- | --- | --- | --- | --- | --- |
| always_strongest | 59.9% | $95.0076 | $0.0902 | 0.0% | n/a |
| always_cheapest | 54.6% | $1.2849 | $0.0013 | 15.0% | n/a |
| static_keyword | 59.1% | $48.5205 | $0.0467 | 10.4% | n/a |
| static_oracle_family | 59.5% | $44.8313 | $0.0429 | 10.1% | n/a |
| latency | 54.6% | $1.2849 | $0.0013 | 15.0% | n/a |
| quality/lr/balanced | 59.9% | $53.6663 | $0.0510 | 3.9% | 50.1% |
| quality/gbm/balanced | 59.8% | $49.7549 | $0.0474 | 4.3% | 49.5% |
| quality/knn/balanced | 58.1% | $49.7913 | $0.0488 | 4.8% | 51.1% |
| share_matched_random/balanced | 56.9% | $51.6589 | $0.0517 | 7.3% | n/a |
| oracle | 77.2% | $12.1652 | $0.0090 | 0.0% | n/a |

### economy

| Strategy | Accuracy | Total cost | Cost per success | Regret rate | Escalation rate |
| --- | --- | --- | --- | --- | --- |
| always_strongest | 59.9% | $95.0076 | $0.0902 | 0.0% | n/a |
| always_cheapest | 54.6% | $1.2849 | $0.0013 | 15.0% | n/a |
| static_keyword | 59.1% | $48.5205 | $0.0467 | 10.4% | n/a |
| static_oracle_family | 59.5% | $44.8313 | $0.0429 | 10.1% | n/a |
| latency | 54.6% | $1.2849 | $0.0013 | 15.0% | n/a |
| quality/lr/economy | 58.8% | $40.9188 | $0.0396 | 9.1% | 22.0% |
| quality/gbm/economy | 58.1% | $42.2087 | $0.0413 | 9.3% | 24.8% |
| quality/knn/economy | 58.3% | $41.9870 | $0.0410 | 9.5% | 22.3% |
| share_matched_random/economy | 56.1% | $28.1764 | $0.0286 | 10.7% | n/a |
| oracle | 77.2% | $12.1652 | $0.0090 | 0.0% | n/a |

## Model shares (balanced)

| Strategy | gpt-5 | claude-sonnet-4 | gemini-2.5-pro | gemini-2.5-flash | deepseek-v3.1-terminus | qwen3-235b-a22b-2507 |
| --- | --- | --- | --- | --- | --- | --- |
| always_strongest | 0.0% | 0.0% | 100.0% | 0.0% | 0.0% | 0.0% |
| always_cheapest | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% | 100.0% |
| static_keyword | 99.8% | 0.0% | 0.0% | 0.0% | 0.0% | 0.2% |
| static_oracle_family | 93.5% | 0.0% | 0.0% | 0.0% | 0.0% | 6.5% |
| latency | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% | 100.0% |
| quality/lr/balanced | 5.4% | 0.2% | 52.2% | 0.4% | 3.3% | 38.5% |
| quality/gbm/balanced | 2.7% | 0.1% | 52.2% | 0.6% | 6.6% | 37.7% |
| quality/knn/balanced | 0.9% | 0.1% | 55.3% | 0.0% | 8.0% | 35.8% |
| share_matched_random/balanced | 2.7% | 0.1% | 52.2% | 0.6% | 6.6% | 37.7% |
| oracle | 8.8% | 1.5% | 5.6% | 9.3% | 17.4% | 57.5% |

## Per dataset (balanced)

### aime

9 test prompts.

| Strategy | Accuracy | Total cost | Cost per success | Regret rate | Escalation rate |
| --- | --- | --- | --- | --- | --- |
| always_strongest | 77.8% | $1.5057 | $0.2151 | 0.0% | n/a |
| always_cheapest | 88.9% | $0.0432 | $0.0054 | 0.0% | n/a |
| static_keyword | 77.8% | $0.7183 | $0.1026 | 0.0% | n/a |
| static_oracle_family | 77.8% | $0.7183 | $0.1026 | 0.0% | n/a |
| latency | 88.9% | $0.0432 | $0.0054 | 0.0% | n/a |
| quality/lr/balanced | 88.9% | $0.7542 | $0.0943 | 0.0% | 22.2% |
| quality/gbm/balanced | 88.9% | $0.0432 | $0.0054 | 0.0% | 0.0% |
| quality/knn/balanced | 88.9% | $0.0432 | $0.0054 | 0.0% | 0.0% |
| share_matched_random/balanced | 81.5% | $0.8061 | $0.1099 | 0.8% | n/a |
| oracle | 88.9% | $0.0240 | $0.0030 | 0.0% | n/a |

### arenahard

114 test prompts.

| Strategy | Accuracy | Total cost | Cost per success | Regret rate | Escalation rate |
| --- | --- | --- | --- | --- | --- |
| always_strongest | 85.1% | $4.8621 | $0.0501 | 0.0% | n/a |
| always_cheapest | 85.1% | $0.1081 | $0.0011 | 11.4% | n/a |
| static_keyword | 78.9% | $3.7973 | $0.0422 | 14.9% | n/a |
| static_oracle_family | 85.1% | $0.1081 | $0.0011 | 11.4% | n/a |
| latency | 85.1% | $0.1081 | $0.0011 | 11.4% | n/a |
| quality/lr/balanced | 86.8% | $1.5495 | $0.0157 | 7.9% | 17.5% |
| quality/gbm/balanced | 89.5% | $1.1941 | $0.0117 | 7.0% | 6.1% |
| quality/knn/balanced | 84.2% | $3.0775 | $0.0321 | 5.3% | 52.6% |
| share_matched_random/balanced | 84.7% | $2.6941 | $0.0279 | 5.9% | n/a |
| oracle | 99.1% | $0.1626 | $0.0014 | 0.0% | n/a |

### gpqa

31 test prompts.

| Strategy | Accuracy | Total cost | Cost per success | Regret rate | Escalation rate |
| --- | --- | --- | --- | --- | --- |
| always_strongest | 77.4% | $2.9678 | $0.1237 | 0.0% | n/a |
| always_cheapest | 51.6% | $0.0111 | $0.0007 | 32.3% | n/a |
| static_keyword | 77.4% | $1.3560 | $0.0565 | 12.9% | n/a |
| static_oracle_family | 77.4% | $1.3560 | $0.0565 | 12.9% | n/a |
| latency | 51.6% | $0.0111 | $0.0007 | 32.3% | n/a |
| quality/lr/balanced | 74.2% | $0.4184 | $0.0182 | 9.7% | 3.2% |
| quality/gbm/balanced | 74.2% | $0.0396 | $0.0017 | 9.7% | 0.0% |
| quality/knn/balanced | 71.0% | $0.0317 | $0.0014 | 16.1% | 0.0% |
| share_matched_random/balanced | 67.6% | $1.6152 | $0.0771 | 13.2% | n/a |
| oracle | 96.8% | $0.3206 | $0.0107 | 0.0% | n/a |

### hle

325 test prompts.

| Strategy | Accuracy | Total cost | Cost per success | Regret rate | Escalation rate |
| --- | --- | --- | --- | --- | --- |
| always_strongest | 21.8% | $38.6538 | $0.5444 | 0.0% | n/a |
| always_cheapest | 7.7% | $0.6341 | $0.0254 | 17.8% | n/a |
| static_keyword | 21.8% | $20.0787 | $0.2828 | 12.0% | n/a |
| static_oracle_family | 21.8% | $20.0787 | $0.2828 | 12.0% | n/a |
| latency | 7.7% | $0.6341 | $0.0254 | 17.8% | n/a |
| quality/lr/balanced | 21.8% | $38.4079 | $0.5410 | 0.0% | 99.7% |
| quality/gbm/balanced | 21.8% | $38.6538 | $0.5444 | 0.0% | 100.0% |
| quality/knn/balanced | 21.8% | $38.6538 | $0.5444 | 0.0% | 100.0% |
| share_matched_random/balanced | 15.5% | $21.0631 | $0.4184 | 8.4% | n/a |
| oracle | 40.0% | $6.8077 | $0.0524 | 0.0% | n/a |

### livecodebench

159 test prompts.

| Strategy | Accuracy | Total cost | Cost per success | Regret rate | Escalation rate |
| --- | --- | --- | --- | --- | --- |
| always_strongest | 82.4% | $21.4462 | $0.1637 | 0.0% | n/a |
| always_cheapest | 63.5% | $0.1530 | $0.0015 | 22.6% | n/a |
| static_keyword | 91.2% | $7.5894 | $0.0523 | 2.5% | n/a |
| static_oracle_family | 91.2% | $7.5894 | $0.0523 | 2.5% | n/a |
| latency | 63.5% | $0.1530 | $0.0015 | 22.6% | n/a |
| quality/lr/balanced | 79.2% | $6.3048 | $0.0500 | 7.5% | 9.4% |
| quality/gbm/balanced | 75.5% | $3.9268 | $0.0327 | 13.2% | 4.4% |
| quality/knn/balanced | 66.7% | $1.9376 | $0.0183 | 17.6% | 1.9% |
| share_matched_random/balanced | 74.4% | $11.5355 | $0.0975 | 10.0% | n/a |
| oracle | 93.7% | $3.1294 | $0.0210 | 0.0% | n/a |

### livemathbench

19 test prompts.

| Strategy | Accuracy | Total cost | Cost per success | Regret rate | Escalation rate |
| --- | --- | --- | --- | --- | --- |
| always_strongest | 57.9% | $1.4920 | $0.1356 | 0.0% | n/a |
| always_cheapest | 73.7% | $0.0412 | $0.0029 | 0.0% | n/a |
| static_keyword | 78.9% | $0.5723 | $0.0382 | 0.0% | n/a |
| static_oracle_family | 78.9% | $0.5723 | $0.0382 | 0.0% | n/a |
| latency | 73.7% | $0.0412 | $0.0029 | 0.0% | n/a |
| quality/lr/balanced | 73.7% | $0.0592 | $0.0042 | 0.0% | 0.0% |
| quality/gbm/balanced | 73.7% | $0.0400 | $0.0029 | 0.0% | 0.0% |
| quality/knn/balanced | 73.7% | $0.0412 | $0.0029 | 0.0% | 0.0% |
| share_matched_random/balanced | 66.1% | $0.8138 | $0.0648 | 0.0% | n/a |
| oracle | 84.2% | $0.0462 | $0.0029 | 0.0% | n/a |

### mmlupro

450 test prompts.

| Strategy | Accuracy | Total cost | Cost per success | Regret rate | Escalation rate |
| --- | --- | --- | --- | --- | --- |
| always_strongest | 86.7% | $17.8933 | $0.0459 | 0.0% | n/a |
| always_cheapest | 82.4% | $0.2647 | $0.0007 | 8.2% | n/a |
| static_keyword | 88.2% | $5.6804 | $0.0143 | 2.7% | n/a |
| static_oracle_family | 88.2% | $5.6804 | $0.0143 | 2.7% | n/a |
| latency | 82.4% | $0.2647 | $0.0007 | 8.2% | n/a |
| quality/lr/balanced | 82.9% | $0.9388 | $0.0025 | 7.3% | 2.4% |
| quality/gbm/balanced | 82.9% | $0.4025 | $0.0011 | 7.6% | 0.4% |
| quality/knn/balanced | 82.4% | $0.3146 | $0.0008 | 7.8% | 0.2% |
| share_matched_random/balanced | 85.0% | $9.6418 | $0.0252 | 3.6% | n/a |
| oracle | 93.6% | $0.8142 | $0.0019 | 0.0% | n/a |

### simpleqa

650 test prompts.

| Strategy | Accuracy | Total cost | Cost per success | Regret rate | Escalation rate |
| --- | --- | --- | --- | --- | --- |
| always_strongest | 49.5% | $6.1866 | $0.0192 | 0.0% | n/a |
| always_cheapest | 50.5% | $0.0295 | $0.0001 | 16.8% | n/a |
| static_keyword | 44.5% | $8.7281 | $0.0302 | 16.3% | n/a |
| static_oracle_family | 44.5% | $8.7281 | $0.0302 | 16.3% | n/a |
| latency | 50.5% | $0.0295 | $0.0001 | 16.8% | n/a |
| quality/lr/balanced | 52.2% | $5.2334 | $0.0154 | 1.8% | 78.0% |
| quality/gbm/balanced | 52.2% | $5.4550 | $0.0161 | 1.4% | 81.2% |
| quality/knn/balanced | 51.2% | $5.6917 | $0.0171 | 1.7% | 78.3% |
| share_matched_random/balanced | 47.8% | $3.4893 | $0.0112 | 9.0% | n/a |
| oracle | 75.4% | $0.8605 | $0.0018 | 0.0% | n/a |

## Leave-one-dataset-out

Each row retrains the served predictor without the named dataset, certifies it on the remaining certification prompts and routes that dataset's test prompts (balanced profile). Lambdas are strict / balanced / economy.

| Held-out dataset | n | Strongest | Lambdas | Quality accuracy | Strongest accuracy | Quality cost per success | Strongest cost per success | Quality regret rate |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| aime | 9 | gemini-2.5-pro | 0.86 / 0.69 / 0.44 | 88.9% | 77.8% | $0.0054 | $0.2151 | 0.0% |
| arenahard | 114 | gpt-5 | 0.84 / 0.64 / 0.44 | 79.8% | 79.8% | $0.0444 | $0.0428 | 0.9% |
| gpqa | 31 | gpt-5 | 0.85 / 0.67 / 0.49 | 51.6% | 77.4% | $0.0007 | $0.0565 | 25.8% |
| hle | 325 | gemini-2.5-pro | 0.84 / 0.72 / 0.50 | 8.9% | 21.8% | $0.2010 | $0.5444 | 17.2% |
| livecodebench | 159 | gemini-2.5-pro | 0.84 / 0.63 / 0.45 | 62.9% | 82.4% | $0.0045 | $0.1637 | 22.0% |
| livemathbench | 19 | gemini-2.5-pro | 0.87 / 0.65 / 0.46 | 73.7% | 57.9% | $0.0029 | $0.1356 | 0.0% |
| mmlupro | 450 | gemini-2.5-pro | 0.72 / 0.63 / 0.46 | 84.7% | 86.7% | $0.0128 | $0.0459 | 4.9% |
| simpleqa | 650 | gpt-5 | 0.83 / 0.82 / 0.39 | 47.4% | 44.5% | $0.0227 | $0.0302 | 6.6% |

Pooled (balanced, 1757 test prompts): cost per success reduction versus always_strongest 74.9% [72.2%, 77.4%] (95% CI).

## Calibration

Isotonic blocks are the served calibration; Platt scaling, fit on the calibration split, is the offline comparison. Lower Brier and ECE are better.

### lr

Reliability plot: [reliability-lr.png](reliability-lr.png)

| Model | AUROC | AUROC code | AUROC instruction | AUROC qa | AUROC reasoning | Brier isotonic | Brier Platt | ECE isotonic | ECE Platt |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| gpt-5 | 0.808 | 0.646 | 0.516 | 0.653 | 0.619 | 0.175 | 0.175 | 0.044 | 0.049 |
| claude-sonnet-4 | 0.882 | 0.661 | 0.598 | 0.671 | 0.641 | 0.129 | 0.130 | 0.028 | 0.042 |
| gemini-2.5-pro | 0.786 | 0.685 | 0.562 | 0.665 | 0.612 | 0.184 | 0.184 | 0.034 | 0.045 |
| gemini-2.5-flash | 0.817 | 0.681 | 0.628 | 0.676 | 0.606 | 0.172 | 0.173 | 0.034 | 0.040 |
| deepseek-v3.1-terminus | 0.849 | 0.619 | 0.711 | 0.631 | 0.633 | 0.153 | 0.153 | 0.041 | 0.042 |
| qwen3-235b-a22b-2507 | 0.808 | 0.713 | 0.673 | 0.773 | 0.606 | 0.174 | 0.175 | 0.041 | 0.040 |

### gbm

Reliability plot: [reliability-gbm.png](reliability-gbm.png)

| Model | AUROC | AUROC code | AUROC instruction | AUROC qa | AUROC reasoning | Brier isotonic | Brier Platt | ECE isotonic | ECE Platt |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| gpt-5 | 0.807 | 0.675 | 0.587 | 0.649 | 0.604 | 0.176 | 0.175 | 0.033 | 0.037 |
| claude-sonnet-4 | 0.882 | 0.648 | 0.621 | 0.689 | 0.607 | 0.131 | 0.131 | 0.034 | 0.038 |
| gemini-2.5-pro | 0.784 | 0.730 | 0.573 | 0.649 | 0.619 | 0.184 | 0.184 | 0.033 | 0.045 |
| gemini-2.5-flash | 0.813 | 0.695 | 0.584 | 0.659 | 0.602 | 0.173 | 0.173 | 0.035 | 0.029 |
| deepseek-v3.1-terminus | 0.853 | 0.637 | 0.634 | 0.649 | 0.651 | 0.151 | 0.151 | 0.037 | 0.039 |
| qwen3-235b-a22b-2507 | 0.807 | 0.692 | 0.725 | 0.767 | 0.608 | 0.175 | 0.174 | 0.040 | 0.033 |

### knn

Reliability plot: [reliability-knn.png](reliability-knn.png)

| Model | AUROC | AUROC code | AUROC instruction | AUROC qa | AUROC reasoning | Brier isotonic | Brier Platt | ECE isotonic | ECE Platt |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| gpt-5 | 0.806 | 0.625 | 0.544 | 0.651 | 0.656 | 0.178 | 0.183 | 0.033 | 0.075 |
| claude-sonnet-4 | 0.876 | 0.636 | 0.482 | 0.684 | 0.636 | 0.136 | 0.136 | 0.036 | 0.042 |
| gemini-2.5-pro | 0.777 | 0.666 | 0.532 | 0.665 | 0.600 | 0.186 | 0.192 | 0.046 | 0.076 |
| gemini-2.5-flash | 0.802 | 0.613 | 0.517 | 0.655 | 0.583 | 0.179 | 0.181 | 0.038 | 0.051 |
| deepseek-v3.1-terminus | 0.836 | 0.628 | 0.577 | 0.620 | 0.617 | 0.158 | 0.160 | 0.034 | 0.050 |
| qwen3-235b-a22b-2507 | 0.798 | 0.660 | 0.449 | 0.779 | 0.645 | 0.179 | 0.180 | 0.043 | 0.032 |

## Statistics

Differences are quality minus baseline, from a paired bootstrap stratified by dataset (95% percentile intervals). Raw p-values are two-sided; Holm adjusts across the six baselines of a profile, separately for accuracy and for cost per success.

### strict

Quality strategy: quality/gbm/strict.

| Baseline | Accuracy diff | p | Holm p | Cost per success diff | p | Holm p | Cost per success reduction |
| --- | --- | --- | --- | --- | --- | --- | --- |
| always_strongest | +0.3 pp [-0.4 pp, +1.0 pp] | 0.4496 | 0.8991 | -$0.0159 [-$0.0181, -$0.0138] | 0.0002 | 0.0012 | 17.6% [15.4%, 19.9%] |
| always_cheapest | +5.6 pp [+3.4 pp, +7.9 pp] | 0.0002 | 0.0012 | $0.0730 [$0.0683, $0.0779] | 0.0002 | 0.0012 | -5451.8% [-5908.1%, -5033.3%] |
| static_keyword | +1.1 pp [-0.9 pp, +3.2 pp] | 0.2794 | 0.8381 | $0.0276 [$0.0226, $0.0325] | 0.0002 | 0.0012 | -59.0% [-72.3%, -46.1%] |
| static_oracle_family | +0.7 pp [-1.3 pp, +2.7 pp] | 0.4800 | 0.8991 | $0.0314 [$0.0265, $0.0362] | 0.0002 | 0.0012 | -73.2% [-88.4%, -58.9%] |
| latency | +5.6 pp [+3.4 pp, +7.9 pp] | 0.0002 | 0.0012 | $0.0730 [$0.0683, $0.0779] | 0.0002 | 0.0012 | -5451.8% [-5908.1%, -5033.3%] |
| share_matched_random/strict | +1.4 pp [+0.7 pp, +2.1 pp] | 0.0002 | 0.0012 | -$0.0048 [-$0.0069, -$0.0027] | 0.0002 | 0.0012 | 6.1% [3.4%, 8.7%] |

- McNemar versus always_strongest: quality only 21, always_strongest only 16, exact p 0.5114.
- TOST accuracy versus always_strongest, margin +/-2.0 pp: difference +0.3 pp, 90% CI [-0.3 pp, +0.9 pp], non-inferior yes (p 0.0001), equivalent yes (p 0.0001).
- Regret rate 0.9% (16 of 1757), 95% exact CI [0.5%, 1.5%], alpha 2.0%.

### balanced

Quality strategy: quality/gbm/balanced.

| Baseline | Accuracy diff | p | Holm p | Cost per success diff | p | Holm p | Cost per success reduction |
| --- | --- | --- | --- | --- | --- | --- | --- |
| always_strongest | -0.2 pp [-1.5 pp, +1.2 pp] | 0.8341 | 1.0000 | -$0.0428 [-$0.0461, -$0.0396] | 0.0002 | 0.0012 | 47.5% [44.6%, 50.3%] |
| always_cheapest | +5.1 pp [+3.2 pp, +7.1 pp] | 0.0002 | 0.0012 | $0.0460 [$0.0422, $0.0501] | 0.0002 | 0.0012 | -3440.2% [-3760.9%, -3144.9%] |
| static_keyword | +0.7 pp [-1.4 pp, +2.8 pp] | 0.5385 | 1.0000 | $0.0006 [-$0.0038, $0.0050] | 0.7773 | 0.7773 | -1.4% [-11.2%, 7.7%] |
| static_oracle_family | +0.3 pp [-1.8 pp, +2.4 pp] | 0.8087 | 1.0000 | $0.0045 [$0.0001, $0.0088] | 0.0448 | 0.0896 | -10.5% [-21.4%, -0.3%] |
| latency | +5.1 pp [+3.2 pp, +7.1 pp] | 0.0002 | 0.0012 | $0.0460 [$0.0422, $0.0501] | 0.0002 | 0.0012 | -3440.2% [-3760.9%, -3144.9%] |
| share_matched_random/balanced | +2.9 pp [+1.7 pp, +4.1 pp] | 0.0002 | 0.0012 | -$0.0043 [-$0.0068, -$0.0018] | 0.0016 | 0.0048 | 8.4% [3.4%, 13.3%] |

- McNemar versus always_strongest: quality only 72, always_strongest only 75, exact p 0.8690.
- TOST accuracy versus always_strongest, margin +/-2.0 pp: difference -0.2 pp, 90% CI [-1.3 pp, +1.0 pp], non-inferior yes (p 0.0029), equivalent yes (p 0.0029).
- Regret rate 4.3% (75 of 1757), 95% exact CI [3.4%, 5.3%], alpha 5.0%.

### economy

Quality strategy: quality/gbm/economy.

| Baseline | Accuracy diff | p | Holm p | Cost per success diff | p | Holm p | Cost per success reduction |
| --- | --- | --- | --- | --- | --- | --- | --- |
| always_strongest | -1.8 pp [-3.7 pp, +0.1 pp] | 0.0630 | 0.1890 | -$0.0489 [-$0.0525, -$0.0454] | 0.0002 | 0.0012 | 54.2% [51.1%, 57.2%] |
| always_cheapest | +3.5 pp [+2.2 pp, +4.8 pp] | 0.0002 | 0.0012 | $0.0400 [$0.0362, $0.0440] | 0.0002 | 0.0012 | -2988.6% [-3276.4%, -2724.2%] |
| static_keyword | -1.0 pp [-3.2 pp, +1.3 pp] | 0.4102 | 0.4380 | -$0.0054 [-$0.0098, -$0.0011] | 0.0154 | 0.0308 | 11.6% [2.3%, 20.2%] |
| static_oracle_family | -1.4 pp [-3.5 pp, +0.7 pp] | 0.2190 | 0.4380 | -$0.0016 [-$0.0058, $0.0027] | 0.4960 | 0.4960 | 3.6% [-6.6%, 13.1%] |
| latency | +3.5 pp [+2.2 pp, +4.8 pp] | 0.0002 | 0.0012 | $0.0400 [$0.0362, $0.0440] | 0.0002 | 0.0012 | -2988.6% [-3276.4%, -2724.2%] |
| share_matched_random/economy | +2.0 pp [+0.9 pp, +3.1 pp] | 0.0004 | 0.0016 | $0.0127 [$0.0100, $0.0157] | 0.0002 | 0.0012 | -44.6% [-53.6%, -35.6%] |

- McNemar versus always_strongest: quality only 132, always_strongest only 164, exact p 0.0714.
- TOST accuracy versus always_strongest, margin +/-2.0 pp: difference -1.8 pp, 90% CI [-3.4 pp, -0.2 pp], non-inferior no (p 0.4222), equivalent no (p 0.4222).
- Regret rate 9.3% (164 of 1757), 95% exact CI [8.0%, 10.8%], alpha 10.0%.

## Costs

Offline costs are the dataset-recorded cost field (provider usage times list price per LLMRouterBench). The gateway price table (deploy/prices.yaml) uses list prices. Files whose recorded total differs from the list-price recomputation by more than 1% are listed below.

| Model | Recorded (test rows) | List (test rows) |
| --- | --- | --- |
| gpt-5 | $48.6147 | $53.9863 |
| claude-sonnet-4 | $14.0281 | $14.0281 |
| gemini-2.5-pro | $95.0076 | $95.0098 |
| gemini-2.5-flash | $14.5808 | $14.8982 |
| deepseek-v3.1-terminus | $1.4304 | $1.4304 |
| qwen3-235b-a22b-2507 | $1.2849 | $1.2545 |

Pairs whose recorded total differs from the list-price total by more than 1%:

| Dataset | Model | Recorded | List | Ratio |
| --- | --- | --- | --- | --- |
| aime | qwen3-235b-a22b-2507 | $0.0432 | $0.0324 | 1.335 |
| arenahard | qwen3-235b-a22b-2507 | $0.1081 | $0.1117 | 0.967 |
| gpqa | qwen3-235b-a22b-2507 | $0.0111 | $0.0108 | 1.030 |
| hle | gpt-5 | $20.0787 | $25.4503 | 0.789 |
| hle | qwen3-235b-a22b-2507 | $0.6341 | $0.6064 | 1.046 |
| livecodebench | gemini-2.5-flash | $0.9243 | $1.2418 | 0.744 |
| livecodebench | qwen3-235b-a22b-2507 | $0.1530 | $0.1470 | 1.041 |
| livemathbench | qwen3-235b-a22b-2507 | $0.0412 | $0.0426 | 0.967 |
| mmlupro | qwen3-235b-a22b-2507 | $0.2647 | $0.2735 | 0.968 |
| simpleqa | qwen3-235b-a22b-2507 | $0.0295 | $0.0301 | 0.978 |

## Notes

- With probability at least 1 - delta over the draw of the certification set, the expected regret rate on future traffic exchangeable with it is at most alpha, when all pool models are eligible. It says nothing about any single request.
- Benchmark prompt templates let a classifier recognize the dataset, which inflates in-distribution gains; leave-one-dataset-out results are reported next to them.
- The latency-aware router uses SIMULATED latency profiles (bench/latency_profiles.yaml) and is a reference only, not part of the H1 test.

## Extraction

Extraction quality is unmeasured: make label-extraction has not been run.
