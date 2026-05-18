# NB08 vs NB10 — preprocessing audit

## Headline finding

The 0.5685 (NB08) vs 0.6154 (NB10) discrepancy in MVP 4 T2 NotHate F1 is caused by **case-preprocessing** of the tokenizer input. NB08 fed `tweet_text` as-is (mixed-case); NB09+ and NB10 lowercased before tokenization. The cardiffnlp Twitter-RoBERTa tokenizer is case-sensitive, so the same model produces different predictions on the same data depending on which case-preprocessing is applied.

## Probe results (all on 8,411 t2_valid rows, fp16 autocast, same checkpoint per variant)

| Run | Case mode | NotHate F1 | Racist F1 | Sexist F1 | Homophobe F1 | Religion F1 | OtherHate F1 | Macro F1 |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| MVP 4 (NB08 canonical) | mixed | 0.5681 | 0.4688 | 0.4052 | 0.7420 | 0.1685 | 0.6054 | 0.4930 |
| MVP 4 (NB10 setup) | lower | 0.6154 | 0.4442 | 0.4041 | 0.7389 | 0.1562 | 0.5979 | 0.4928 |
| MVP 4-IW-CC-S (OOD probe, id-bias=0) | mixed | 0.5603 | 0.4405 | 0.3973 | 0.7003 | 0.0679 | 0.6151 | 0.4636 |
| MVP 4-IW-CC-S (canonical, id-bias=0) | lower | 0.5789 | 0.4575 | 0.4054 | 0.7347 | 0.0885 | 0.6208 | 0.4810 |


**Note on iwccs probes:** the identity-bias term `λ_id · identity_mask · (1 + α · vader_neg_centered)` was zeroed (identity_mask=0, vader_neg_centered=0) so the IWCCS model behaves like a plain gated fusion. This isolates the case-preprocessing effect from the IW mechanism. The non-bias IWCCS numbers are NOT the canonical NB09c result; they're a controlled probe.
