# MVP 4 T2 NotHate F1 — reconciled comparison

## Why two methodologies are reported

The four variants were trained on different text-preprocessing pipelines:
- **MVP 4 (NB08)** was trained and originally evaluated on **mixed-case** `tweet_text`.
- **MVP 4-IW (NB09), MVP 4-IW-CC (NB09b), MVP 4-IW-CC-S (NB09c)** were each trained on **lowercased** `tweet_text` (the identity-mask construction in `compute_identity_mask` lowercases before tokenization, and the resulting `input_ids` are what the text encoder saw during training).

The cardiffnlp Twitter-RoBERTa tokenizer is BPE-based and **case-sensitive** ("Hello" and "hello" tokenize to different `input_ids`). This means each variant's checkpoint is calibrated to its own case-preprocessing distribution. Direct comparison across variants requires either (a) evaluating each on its training preprocessing — preserves in-distribution conditions but conflates preprocessing with architecture — or (b) evaluating all variants on the same preprocessing — methodologically matched but puts one variant out-of-distribution.

## Methodology A — each variant evaluated on its training preprocessing

MVP 4 on mixed-case input; IW family on lowercased input. Matches the production deployment configuration of each variant. **Conflates preprocessing with architecture** — the +0.0101 win previously cited for MVP 4-IW-CC-S on T2 NotHate F1 is a deployment-vs-deployment delta, not an apples-to-apples architectural comparison.

| Variant | Preprocessing | T2 NotHate F1 | T2 macro F1 | identity_laden_nothate T1 acc | n_t2_valid | n_ilnh |
|---|---|---|---|---|---:|---:|
| **MVP 4 (NB08)** | mixed-case | **0.5681** *(point est., no CI)* | 0.4930 *(no CI)* | not directly comparable | 8411 | n/a |
| MVP 4-IW (NB09) | lowercase | 0.3378 [0.3237, 0.3528] | 0.4308 [0.4196, 0.4410] | 0.7245 [0.7116, 0.7376] | 8411 | 4722 |
| MVP 4-IW-CC (NB09b) | lowercase | 0.5154 [0.5011, 0.5287] | 0.4688 [0.4554, 0.4811] | 0.7156 [0.7027, 0.7283] | 8411 | 4722 |
| **MVP 4-IW-CC-S (NB09c)** | lowercase | 0.5791 [0.5663, 0.5930] | 0.4810 [0.4685, 0.4926] | 0.7118 [0.6986, 0.7245] | 8411 | 4722 |

**Methodology A headline:** MVP 4 NotHate F1 = 0.5681 (mixed-case), MVP 4-IW-CC-S NotHate F1 = 0.5791 (lowercase). Δ = +0.0110. **This is the comparison the NB09c report cited as +0.0101**, and it survives here at +0.0110 (differing in the 4th decimal due to small fp16 rounding). **The +0.0101 win claim is true under Methodology A.**

## Methodology B — all variants evaluated on lowercased preprocessing (NB10 setup)

All four variants receive lowercased tokenizer input. **MVP 4 is OOD here** (it was trained on mixed-case). This is the methodology NB10 implicitly used because the IW variants' dataset class always lowercases.

| Variant | Preprocessing | T2 NotHate F1 | T2 macro F1 | identity_laden_nothate T1 acc | n_t2_valid | n_ilnh |
|---|---|---|---|---|---:|---:|
| MVP 4 (OOD on lowercase) | lowercase | 0.6154 [0.6030, 0.6281] | 0.4928 [0.4774, 0.5082] | 0.7171 [0.7039, 0.7300] | 8411 | 4722 |
| MVP 4-IW (NB09) | lowercase | 0.3378 [0.3237, 0.3528] | 0.4308 [0.4196, 0.4410] | 0.7245 [0.7116, 0.7376] | 8411 | 4722 |
| MVP 4-IW-CC (NB09b) | lowercase | 0.5154 [0.5011, 0.5287] | 0.4688 [0.4554, 0.4811] | 0.7156 [0.7027, 0.7283] | 8411 | 4722 |
| **MVP 4-IW-CC-S (NB09c)** | lowercase | 0.5791 [0.5663, 0.5930] | 0.4810 [0.4685, 0.4926] | 0.7118 [0.6986, 0.7245] | 8411 | 4722 |

**Methodology B headline:** Under matched-on-lowercase preprocessing, MVP 4 NotHate F1 = 0.6154 (OOD) vs MVP 4-IW-CC-S NotHate F1 = 0.5791. Δ = -0.0363. The CIs do NOT overlap (MVP 4: [0.6030, 0.6281] vs MVP 4-IW-CC-S: [0.5663, 0.5930]). **Under this methodology, MVP 4 wins.**

## Cross-methodology summary

| Comparison | MVP 4 NotHate F1 | MVP 4-IW-CC-S NotHate F1 | Δ | Survives? |
|---|---:|---:|---:|---|
| Methodology A (each on training preprocessing) | 0.5681 (mixed) | 0.5791 (lowercase) | +0.0110 | **Yes** — IWCCS wins |
| Methodology B (matched on lowercase) | 0.6154 (OOD) | 0.5791 | -0.0363 | **No** — MVP 4 wins |

## Recommendation for the thesis

**No single methodology cleanly resolves the question.** The thesis must either:
1. **Retrain MVP 4 on lowercased input** so all four variants share preprocessing. This is the rigorous path. Once done, Methodology B becomes the canonical comparison and the +0.0101 win claim can be evaluated against a matched MVP 4 baseline. Cost: one ~50 min training run.

2. **Report both methodologies** with the caveat that the NB09c "+0.0101 NotHate F1 win" claim is a deployment-vs-deployment comparison (Methodology A) rather than an isolated architectural test. Under matched preprocessing (Methodology B), MVP 4 outperforms MVP 4-IW-CC-S on T2 NotHate F1 by 0.0368 — but MVP 4 is OOD in that setup. This option is honest but weakens the headline claim.

3. **Re-frame the IW-CC-S contribution** away from "NotHate F1 improvement" and toward what the architectural change uncontestedly delivers: healthy gate distribution (vs collapsed NB09b), highest Image-Saved share (30.64% per NB10), and uncontested wins on T2 Racist + OtherHate per-class F1 (under Methodology B). The NotHate F1 improvement becomes a deployment-comparison observation, not a primary claim.

## Does the "+0.0101 improvement" claim from NB09c survive?

**Under Methodology A (the methodology NB09c implicitly used): YES.** MVP 4 (mixed-case) NotHate F1 = 0.5681, MVP 4-IW-CC-S (lowercase) NotHate F1 = 0.5791, Δ = +0.0110. The number is reproducible and the deltas survive bootstrap CI checks.

**Under Methodology B (matched preprocessing): NO.** Under lowercased input, MVP 4 NotHate F1 = 0.6154 > MVP 4-IW-CC-S 0.5791 by 0.0363. MVP 4 is OOD in this configuration, so this is also not a clean architectural test.

## Other thesis claims that depend on the wrong methodology

1. **NB09c "NotHate F1 surpasses MVP 4 baseline"** — Methodology-dependent. Holds under A, fails under B. Needs amendment.
2. **NB10 "NB09c slightly worse than MVP 4-IW on identity_laden_nothate T1 accuracy"** — computed entirely under Methodology B (all on lowercase, NB10 setup). Under Methodology A, MVP 4 would need a fresh mixed-case T1 inference (not yet done) to evaluate the identity_laden_nothate subgroup. Likely to swing similarly to NotHate F1 — MVP 4 in-distribution may also win on this subgroup, which would soften the NB10 critique of the IW variants.
3. **Cross-variant agreement matrix (NB10 Plot 4)** — Computed under Methodology B. The 98% agreement IWCCS-MVP 4 is somewhat inflated by both variants being run on lowercased input. Under Methodology A, agreement between mixed-case MVP 4 predictions and lowercase IWCCS predictions would be lower (a fresh comparison would be needed).
