# NB 11 — Bias / Fairness Robustness Analysis — Summary findings
## 1. Overall masking flip rate per variant
Flip rate = fraction of identity-laden samples where T1 prediction changes when identity tokens are masked.
- **MVP 4-lower**: 0.3534
- **MVP 4-IW**: 0.3539
- **MVP 4-IW-CC**: 0.3635
- **MVP 4-IW-CC-S**: 0.3691

## 2. Communities with highest flip rate (most identity-sensitive)
- MVP 4-IW: **Homosexual** at 0.4614
- MVP 4-IW-CC: **Homosexual** at 0.4487
- MVP 4-IW-CC-S: **Homosexual** at 0.4595
- MVP 4-lower: **Homosexual** at 0.4419

## 3. Communities with lowest flip rate (least identity-sensitive)
- MVP 4-IW: **African** at 0.2506
- MVP 4-IW-CC: **Refugee** at 0.2751
- MVP 4-IW-CC-S: **Refugee** at 0.2794
- MVP 4-lower: **Refugee** at 0.2808

## 4. MVP 4-IW-CC-S vs MVP 4-lower per-community T1 accuracy deltas
Given bias-off ablation (Δ = +0.0002 NotHate F1), expect minimal differences.

| Community | Δ T1 acc (IWCCS − MVP4-lower) |
|---|---:|
| Islam | +0.0083 |
| African | -0.0006 |
| Jewish | +0.0022 |
| Refugee | +0.0201 |
| Homosexual | +0.0024 |
| Women | +0.0006 |
| Arab | -0.0032 |
| Asian | -0.0027 |
| Caucasian | +0.0073 |
| Hispanic | +0.0130 |

Mean absolute Δ: 0.0060

## 5. Cross-community swap asymmetries
Top asymmetric reciprocal swap pairs (|forward − reverse| flip rate):

| Pair | Variant | Asymmetry |
|---|---|---:|
| jew ↔ muslim | MVP 4-IW | 0.1429 |
| christian ↔ muslim | MVP 4-IW-CC | 0.1429 |
| christian ↔ muslim | MVP 4-IW | 0.1429 |
| jew ↔ muslim | MVP 4-IW-CC-S | 0.1429 |
| christian ↔ muslim | MVP 4-lower | 0.1429 |
| muslim ↔ christian | MVP 4-lower | 0.1429 |
| muslim ↔ christian | MVP 4-IW-CC | 0.1429 |
| muslim ↔ christian | MVP 4-IW | 0.1429 |

## 6. Per-community performance disparities
| Variant | Best community | Worst community | Disparity |
|---|---|---|---:|
| MVP 4-lower | Homosexual (0.7285) | Refugee (0.6447) | 0.0838 |
| MVP 4-IW | Homosexual (0.7314) | Women (0.6445) | 0.0869 |
| MVP 4-IW-CC | Homosexual (0.7305) | Women (0.6461) | 0.0844 |
| MVP 4-IW-CC-S | Homosexual (0.7310) | Women (0.6474) | 0.0836 |
