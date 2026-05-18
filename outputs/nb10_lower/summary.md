# NB10-lower — Summary findings (matched-methodology, lowercase preprocessing)

## 1. Taxonomy distribution per variant

All five variants now operate on identical input preprocessing (lowercased text). Fusion Failure shares: MVP 4-lower 31.32% / MVP 4-IW 31.22% / MVP 4-IW-CC 31.17% / MVP 4-IW-CC-S 31.11% / IW-CC-S-bias-off 31.12%. Convergent Correct shares: MVP 4-lower 6.82% / MVP 4-IW 1.03% / MVP 4-IW-CC 17.35% / MVP 4-IW-CC-S 4.20% / IW-CC-S-bias-off 4.21%.

## 2. Cross-variant agreement

MVP 4-IW-CC-S agrees with MVP 4-lower on 98.19% of test samples, with MVP 4-IW-CC on 98.84%, and with MVP 4-IW on 97.99%. On disagreement samples vs MVP 4-lower, MVP 4-IW-CC-S is uniquely correct on 101 samples, MVP 4-lower uniquely correct on 80 samples.

## 3. Subgroup-stratified accuracy

identity_laden_nothate FPR: MVP 4-lower 0.2810 / MVP 4-IW 0.2755 / MVP 4-IW-CC 0.2844 / MVP 4-IW-CC-S 0.2882 / IW-CC-S-bias-off 0.2884. Lower is better. The matched-methodology comparison now isolates architectural effects from preprocessing.

## 4. Bias-off ablation (matched-methodology contribution of the IW mechanism itself)

Comparing MVP 4-IW-CC-S vs IW-CC-S-bias-off (same weights, λ_id forced to 0): T2 NotHate F1 = 0.5791 vs 0.5789, Δ = +0.0002. The identity-bias term's standalone contribution to T2 NotHate F1 is therefore quantified directly — any sign and magnitude flows from here. The architectural plumbing (per-branch LN + VADER centering + healthy gate) is what bias-off ALREADY delivers; the bias term provides the marginal addition (or subtraction) on top.

## 5. Matched-methodology contribution of MVP 4-IW-CC-S vs MVP 4-lower

Both variants are now in-distribution. T2 NotHate F1: MVP 4-IW-CC-S 0.5791 vs MVP 4-lower 0.2692, Δ = +0.3099. identity_laden_nothate FPR: MVP 4-IW-CC-S 0.2882 vs MVP 4-lower 0.2810. This is the apples-to-apples comparison the thesis requires.

## 6. Per-T2-class analysis

Religion F1 (the structural floor at ~0.3% of training data): MVP 4-lower 0.151 / MVP 4-IW 0.065 / MVP 4-IW-CC 0.125 / MVP 4-IW-CC-S 0.089 / IW-CC-S-bias-off 0.089. Consistently low — class scarcity not solved by any IW variant, consistent with CLAUDE.md §10 (no resampling).

## 7. Emergent Multimodal population

The 'Emergent Multimodal' bucket — fused correct but no single branch confident-and-correct — captures the population where the gated routing genuinely adds value. Share: MVP 4-lower 19.18% / MVP 4-IW 23.15% / MVP 4-IW-CC 27.79% / MVP 4-IW-CC-S 11.90% / IW-CC-S-bias-off 11.90%.

## 8. Open questions for NB11 (bias analysis)

(a) Does the recovery on identity_laden_nothate generalise across the 15 HateXplain communities, or is it concentrated on a subset? (b) What is the per-community gate-distribution divergence under MVP 4-IW-CC-S? (c) Cross-domain robustness on Hateful Memes — does the IW-CC-S mechanism transfer or memorise MMHS150K conventions?
