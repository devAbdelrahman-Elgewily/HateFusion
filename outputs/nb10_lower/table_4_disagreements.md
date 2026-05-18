# Table 4 — Cross-variant disagreement analysis (T1 fused_pred)

For each pair (A, B) of variants, samples where they disagree on fused_pred, split by which side is correct (they cannot both be correct since they disagree).

| Pair (A vs B) | n disagreements | A correct | B correct | A%·right | B%·right |
|---|---:|---:|---:|---:|---:|
| mvp4_lower vs mvp4_iw | 172 | 81 | 91 | 47.09% | 52.91% |
| mvp4_lower vs mvp4_iwcc | 205 | 95 | 110 | 46.34% | 53.66% |
| mvp4_lower vs mvp4_iwccs | 181 | 80 | 101 | 44.20% | 55.80% |
| mvp4_lower vs mvp4_iwccs_biasoff | 182 | 81 | 101 | 44.51% | 55.49% |
| mvp4_iw vs mvp4_iwcc | 201 | 98 | 103 | 48.76% | 51.24% |
| mvp4_iw vs mvp4_iwccs | 201 | 95 | 106 | 47.26% | 52.74% |
| mvp4_iw vs mvp4_iwccs_biasoff | 202 | 96 | 106 | 47.52% | 52.48% |
| mvp4_iwcc vs mvp4_iwccs | 116 | 55 | 61 | 47.41% | 52.59% |
| mvp4_iwcc vs mvp4_iwccs_biasoff | 115 | 55 | 60 | 47.83% | 52.17% |
| mvp4_iwccs vs mvp4_iwccs_biasoff | 1 | 1 | 0 | 100.00% | 0.00% |
