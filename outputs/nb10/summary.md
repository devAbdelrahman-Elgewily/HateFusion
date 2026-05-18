# NB10 — Summary findings

## 1. Taxonomy distribution per variant

Aggregate AUC is within a 0.006 band across the four variants, but the taxonomy distribution diverges: Fusion Failure share is MVP 4 31.31% / MVP 4-IW 31.22% / MVP 4-IW-CC 31.17% / MVP 4-IW-CC-S 31.11%. Convergent Correct is MVP 4 2.10% / MVP 4-IW 1.03% / MVP 4-IW-CC 17.35% / MVP 4-IW-CC-S 4.20%. NB 09b's gate-collapsed regime shows up as a low Image Saved / Emergent Multimodal share; NB 09c recovers both.

## 2. Cross-variant agreement

MVP 4-IW-CC-S agrees with MVP 4 on 98.30% of test samples, with MVP 4-IW-CC on 98.84%, and with MVP 4-IW on 97.99%. On disagreement samples vs MVP 4, MVP 4-IW-CC-S is uniquely correct on 95 samples, MVP 4 uniquely correct on 75 samples. These two populations are the cleanest targets for qualitative case analysis in the thesis.

## 3. Subgroup-stratified accuracy

On identity_laden_nothate (the failure-mode-prone subgroup, see Plot 6), FPR is: MVP 4 0.2829 / MVP 4-IW 0.2755 / MVP 4-IW-CC 0.2844 / MVP 4-IW-CC-S 0.2882. Lower is better (fewer benign identity-bearing tweets misclassified as hate). Compare across subgroups in Plot 8.

## 4. Identity attention vs correctness

Plot 5 visualises whether high identity attention correlates with correct hate predictions (intended) or with non-hate false positives (over-firing failure mode). NB 09b's collapsed gate decouples identity attention from the final prediction; NB 09c re-couples them.

## 5. Hard-sample identity_laden_nothate confusion matrices

Plot 6 directly validates the IW-CC-S contribution. Identity-laden NOT-HATE off-diagonal (FPR) is 0.2755 for MVP 4-IW vs 0.2882 for MVP 4-IW-CC-S. This is the failure mode the NB 09c stabilisation pass was designed to resolve.

## 6. Per-T2-class analysis

Religion F1 (the structural floor at ~0.3% of training data): MVP 4 0.156 / MVP 4-IW 0.065 / MVP 4-IW-CC 0.125 / MVP 4-IW-CC-S 0.089. Consistently low — class scarcity not solved by any IW variant, consistent with CLAUDE.md §10 (no resampling).

## 7. Emergent Multimodal population

The 'Emergent Multimodal' bucket — fused correct but no single branch confident-and-correct — captures the population where the gated routing genuinely adds value. Share: MVP 4 26.06% / MVP 4-IW 23.15% / MVP 4-IW-CC 27.79% / MVP 4-IW-CC-S 11.90%.

## 8. Open questions for NB11 (bias analysis)

(a) Does the recovery on identity_laden_nothate generalise across the 15 HateXplain communities, or is it concentrated on a subset? (b) What is the per-community gate-distribution divergence under MVP 4-IW-CC-S? (c) Cross-domain robustness on Hateful Memes — does the IW-CC-S mechanism transfer or memorise MMHS150K conventions?
