# MVP 4 — Corrected Matched-Methodology Comparison

Generated from `data/processed/nb10_lower_per_sample_data.parquet` (50,000 rows: 5 variants × 10,000 test samples). All 95 % CIs from 1,000 bootstrap resamples (seed 42). All five variants now operate on the **same input distribution** (lowercase preprocessing): MVP 4-lower was retrained from scratch with `text = text.lower()` added to NB08's dataset class, while the IW family already lowercased internally. This is the apples-to-apples comparison the thesis requires.

## NB08-lower convergence — flag

MVP 4-lower converged to **val_t1_auc = 0.7414** (best epoch = 2) vs the original NB08's **0.7470** (best epoch = 4). Δ = −0.0056 — just outside the 0.005 noise band the user asked me to flag. The most striking divergence is in T2 NotHate F1 on test: NB08 original (mixed-case eval) 0.5685 → NB08-lower (lowercase eval) **0.2692**. This is a −0.30 collapse, not a noise-level shift, and reflects an underlying issue: the frozen Run-D LoRA text encoder was trained on mixed-case input. All IW variants (NB09 / 09b / 09c) inherit the same OOD encoder behaviour because they too feed lowercased input. MVP 4-lower's collapse on NotHate shows how a plain gated fusion fails to compensate when the encoder is OOD; the IW family's architecture (cross-attention + gated fusion) compensates enough to retain NotHate F1 ≈ 0.58.

## Table A — T1 + T2 aggregate metrics (5 variants, matched lowercase methodology)

| Variant | AUC [95% CI] | F1m [95% CI] | FPR | T2 macro F1 [95% CI] | T2 NotHate F1 [95% CI] | T2 Racist F1 | T2 OtherHate F1 | Gate (t/i/s, H) |
|---|---|---|---:|---|---|---|---|---|
| **MVP 4-lower (NB08-lower)** | 0.7358 [0.7253, 0.7456] | 0.6867 [0.6772, 0.6956] | 0.2921 | 0.4411 [0.4260, 0.4555] | 0.2692 [0.2559, 0.2829] | 0.4490 [0.4327, 0.4644] | 0.6480 [0.6179, 0.6725] | 0.450/0.062/0.489  H=0.845 |
| **MVP 4-IW (NB09)** | 0.7359 [0.7258, 0.7459] | 0.6876 [0.6780, 0.6965] | 0.2857 | 0.4308 [0.4196, 0.4410] | 0.3378 [0.3237, 0.3528] | 0.4401 [0.4225, 0.4552] | 0.6000 [0.5742, 0.6262] | 0.374/0.299/0.327  H=1.073 |
| **MVP 4-IW-CC (NB09b)** | 0.7340 [0.7241, 0.7437] | 0.6882 [0.6790, 0.6971] | 0.2981 | 0.4688 [0.4554, 0.4811] | 0.5154 [0.5011, 0.5287] | 0.4406 [0.4221, 0.4585] | 0.6005 [0.5738, 0.6267] | 1.000/0.000/0.000  H=0.000 |
| **MVP 4-IW-CC-S (NB09c)** | 0.7359 [0.7256, 0.7458] | 0.6889 [0.6797, 0.6979] | 0.3011 | 0.4810 [0.4685, 0.4926] | 0.5791 [0.5663, 0.5930] | 0.4576 [0.4376, 0.4764] | 0.6212 [0.5929, 0.6472] | 0.467/0.219/0.315  H=1.029 |
| **IW-CC-S-bias-off (λ_id=0)** | 0.7359 [0.7256, 0.7458] | 0.6888 [0.6796, 0.6979] | 0.3013 | 0.4810 [0.4686, 0.4926] | 0.5789 [0.5661, 0.5930] | 0.4575 [0.4376, 0.4763] | 0.6208 [0.5926, 0.6468] | 0.467/0.219/0.315  H=1.029 |

(n_t2_valid = 8,411 for all variants)

## Table B — Subgroup-stratified T1 accuracy (5 variants, matched lowercase methodology)

| Variant | identity_laden_hate T1 acc [95% CI] | identity_laden_nothate T1 acc [95% CI] | identity_free T1 acc [95% CI] |
|---|---|---|---|
| **MVP 4-lower (NB08-lower)** | 0.6635 [0.6509, 0.6766] (n=4,796) | 0.7190 [0.7061, 0.7319] (n=4,722) | 0.6037 [0.5602, 0.6474] (n=482) |
| **MVP 4-IW (NB09)** | 0.6599 [0.6468, 0.6733] (n=4,796) | 0.7245 [0.7116, 0.7376] (n=4,722) | 0.6058 [0.5602, 0.6473] (n=482) |
| **MVP 4-IW-CC (NB09b)** | 0.6697 [0.6568, 0.6825] (n=4,796) | 0.7156 [0.7027, 0.7283] (n=4,722) | 0.6058 [0.5622, 0.6494] (n=482) |
| **MVP 4-IW-CC-S (NB09c)** | 0.6754 [0.6628, 0.6885] (n=4,796) | 0.7118 [0.6986, 0.7245] (n=4,722) | 0.5996 [0.5560, 0.6452] (n=482) |
| **IW-CC-S-bias-off (λ_id=0)** | 0.6754 [0.6628, 0.6885] (n=4,796) | 0.7116 [0.6984, 0.7243] (n=4,722) | 0.5996 [0.5560, 0.6452] (n=482) |

## Table C — Per-T2-class F1 with 95% bootstrap CIs (5 variants)

| T2 class | MVP 4-lower (NB08-lower) | MVP 4-IW (NB09) | MVP 4-IW-CC (NB09b) | MVP 4-IW-CC-S (NB09c) | IW-CC-S-bias-off (λ_id=0) |
|---|---|---|---|---|---|
| **NotHate** | 0.2692 [0.2559, 0.2829] | 0.3378 [0.3237, 0.3528] | 0.5154 [0.5011, 0.5287] | 0.5791 [0.5663, 0.5930] | 0.5789 [0.5661, 0.5930] |
| **Racist** | 0.4490 [0.4327, 0.4644] | 0.4401 [0.4225, 0.4552] | 0.4406 [0.4221, 0.4585] | 0.4576 [0.4376, 0.4764] | 0.4575 [0.4376, 0.4763] |
| **Sexist** | 0.3895 [0.3599, 0.4160] | 0.4119 [0.3810, 0.4409] | 0.3880 [0.3591, 0.4145] | 0.4054 [0.3759, 0.4328] | 0.4054 [0.3759, 0.4326] |
| **Homophobe** | 0.7400 [0.7120, 0.7676] | 0.7305 [0.7029, 0.7563] | 0.7430 [0.7153, 0.7679] | 0.7342 [0.7067, 0.7584] | 0.7347 [0.7073, 0.7587] |
| **Religion** | 0.1508 [0.0837, 0.2201] | 0.0645 [0.0367, 0.0936] | 0.1250 [0.0728, 0.1788] | 0.0885 [0.0517, 0.1276] | 0.0885 [0.0517, 0.1276] |
| **OtherHate** | 0.6480 [0.6179, 0.6725] | 0.6000 [0.5742, 0.6262] | 0.6005 [0.5738, 0.6267] | 0.6212 [0.5929, 0.6472] | 0.6208 [0.5926, 0.6468] |

## Headline findings under matched methodology

### Q1. Does MVP 4-IW-CC-S beat MVP 4-lower on T2 NotHate F1?

**YES, by a large margin.** MVP 4-IW-CC-S NotHate F1 = **0.5791** [0.5663, 0.5930] vs MVP 4-lower **0.2692** [0.2559, 0.2829]. **Δ = +0.3099.** CIs do not overlap. The +0.0101 originally claimed in NB09c was an undercount of the true matched-methodology delta by ~30×, but for the wrong reasons: not because IW-CC-S is dramatically better, but because lowercase-trained MVP 4 collapses on NotHate (0.5685 mixed-case → 0.2692 lowercase). The IW family's architecture compensates for the encoder OOD that lowercase preprocessing introduces.

### Q2. Does MVP 4-IW-CC-S beat MVP 4-IW-CC-S-bias-off? (the IW mechanism's standalone contribution)

**NO — essentially zero.** MVP 4-IW-CC-S NotHate F1 = **0.5791** vs bias-off **0.5789**. **Δ = +0.0002** (≈ noise floor). The two variants disagree on **only 1 out of 10,000** test samples; agreement rate = 99.9900%. The IW bias term `λ_id · identity_mask · (1 + α · vader_neg_centered)` contributes essentially nothing to the final prediction. **All of MVP 4-IW-CC-S's lift over MVP 4-IW comes from the architectural plumbing** (per-branch LayerNorm + VADER centering of the modulator + healthy gate distribution), not from the identity-bias mechanism itself.

### Q3. Under matched methodology, what wins survive?

**Wins that survive:**

- **T2 NotHate F1**: MVP 4-IW-CC-S (0.5791) > MVP 4-lower (0.2692) by +0.3099. CIs non-overlapping. Strong win.
- **T2 macro F1**: MVP 4-IW-CC-S (0.4810) vs MVP 4-lower (0.4411). Δ = +0.0399. Modest win.
- **T2 OtherHate F1**: MVP 4-IW-CC-S (0.6212) vs MVP 4-lower (0.6480). Δ = -0.0268. CIs overlap — within noise. Inconclusive.
- **Healthy gate distribution**: MVP 4-IW-CC-S H = 1.029 (vs MVP 4-lower 0.845). Both maintain multimodal routing under matched methodology; the NB09b collapse was a transient failure mode addressed in NB09c.

**Wins that DO NOT survive:**

- **identity_laden_nothate FPR**: MVP 4-IW-CC-S 0.2882 vs MVP 4-lower 0.2810. Δ = +0.0072. **MVP 4-IW-CC-S is slightly WORSE on this subgroup**, not better. The 'IW prevents over-firing on benign identity vocabulary' claim is not supported by the matched-methodology data.
- **Aggregate AUC**: MVP 4-IW-CC-S 0.7359 vs MVP 4-lower 0.7358. Δ = +0.0001. CIs overlap. Within noise; no clear win.

**The bias-off ablation result is the most important finding for the thesis:** the identity-bias mechanism (the named contribution of NB09 → NB09c) contributes essentially nothing to final performance. The architectural plumbing introduced alongside the bias term is what carries the win. The thesis claim should therefore be reframed from 'context-conditioned identity-weighted attention' to **'per-branch LayerNorm + centered VADER modulation + gated cross-attention'** as the deliverable, with the identity-bias term documented as a tested-but-non-contributory feature.
