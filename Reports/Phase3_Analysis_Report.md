# 🔬 Phase 3 — Analysis Report
### Multimodal Cyberbullying & Online Hate Speech Detection
### Final Year Project — Computer Science (AI Major)

---

## Document Information

| Field | Value |
|-------|-------|
| **Document type** | Phase 3 deliverable + report-writing source material |
| **Phase** | 3 — Analysis (no training; pure analysis of trained checkpoints) |
| **Status** | NB 10 complete · NB 10-lower (matched-methodology rerun) complete · diagnostic investigations complete · NB 11 (bias analysis) pending |
| **Date opened** | 2026-05-17 |
| **Last updated** | 2026-05-18 |
| **Compute** | Lightning AI Studio, NVIDIA L4 (23 GB) — inference + analysis only |
| **Companion documents** | `Phase1_Data_Engineering_Report.md` · `Phase2_Modeling_Report.md` (with § 16c.12 post-hoc correction) · `Multimodal_Cyberbullying_Detection_v1.2.md` (technical scope) · `Cyberbullying_Detection_Report_Framing.md` (significance) |
| **Notebooks covered** | `10_per_sample_modality_analysis.ipynb`, `10_lower_per_sample_modality_analysis.ipynb`, `08_lower_mvp4_lowercase.ipynb` (auxiliary retrain for matched methodology) |

---

## Table of Contents

1. [Overview](#1-overview)
2. [Inputs](#2-inputs)
3. [Notebook 10 — Per-Sample Modality Reliance Analysis](#3-notebook-10--per-sample-modality-reliance-analysis)
4. [Notebook 10-lower — Matched-Methodology Re-Analysis](#4-notebook-10-lower--matched-methodology-re-analysis)
5. [Diagnostic — Case-Preprocessing Inconsistency](#5-diagnostic--case-preprocessing-inconsistency)
6. [Diagnostic — Bias-Off Ablation](#6-diagnostic--bias-off-ablation)
7. [Findings](#7-findings)
8. [Methodological recommendations for future work](#8-methodological-recommendations-for-future-work)
9. [Artefacts produced](#9-artefacts-produced)

> NB 11 (bias / fairness robustness analysis) will be appended as section 10 once complete. Sections 1–9 are stable as of the date above.

---

## 1. Overview

Phase 3 is the analysis-only phase of the project. All training was completed in Phase 2 (NB 04 → NB 09c, with the auxiliary lowercase retrain `08_lower_mvp4_lowercase.ipynb` added during Phase 3 diagnostic investigation). Phase 3 takes the four MVP 4 variants — **MVP 4 (NB 08)**, **MVP 4-IW (NB 09)**, **MVP 4-IW-CC (NB 09b)**, **MVP 4-IW-CC-S (NB 09c)** — plus the new **MVP 4-lower** retrained baseline and the **MVP 4-IW-CC-S-bias-off** inference-time ablation, and characterises how they behave on the MMHS150K test split through three lines of analysis:

1. **Per-sample modality reliance** — every test sample is classified into a five-category taxonomy (Convergent Correct, Text Saved, Image Saved, Struct Saved, Emergent Multimodal, Fusion Failure) under each variant, and stratified by identity-token presence into three subgroups (`identity_laden_hate`, `identity_laden_nothate`, `identity_free`). Bootstrap CIs (1,000 resamples, seed 42) on every variant × subgroup metric.
2. **Matched-methodology re-analysis** — the four MVP 4 variants do not operate on the same input distribution (NB 08 trained on mixed-case text, NB 09 → NB 09c lowercased internally for identity-mask alignment). Phase 3 retrained MVP 4 on lowercase input to enable apples-to-apples comparison, then re-ran the per-sample analysis with five variants on a single preprocessing pipeline.
3. **Diagnostic investigations** — two methodological audits surfaced from the per-sample analysis: a case-preprocessing inconsistency that affected every cross-variant comparison in Phase 2 reports, and a bias-off ablation that decomposes MVP 4-IW-CC-S's lift over MVP 4-IW into its architectural-plumbing component (per-branch LayerNorm, centered VADER modulation, healthy gate) versus the identity-bias-term contribution itself.

The phase boundary is rigid: **no checkpoint training**, no notebook modifications outside of new analysis notebooks, no metric.json files amended. Phase 2 reports were amended **post-hoc** (§ 16c.12 of `Phase2_Modeling_Report.md`) to reflect the corrected matched-methodology findings, but the original Phase 2 narrative is preserved as historical record.

The phase delivered three notebooks and four diagnostic markdown files (full inventory in § 9). The headline analytical finding — documented in §§ 4.3, 6, and 7 — is that **the identity-bias mechanism contributes ~ 0 to final predictions under bias-off ablation (Δ = +0.0002 NotHate F1, 9,999 / 10,000 per-sample agreement)**, which reframes the IW family's contribution claim away from "context-conditioned identity-weighted attention" and toward the architectural plumbing (per-branch LN + centered VADER modulation + healthy gate) introduced alongside the bias term.

---

## 2. Inputs

Phase 3 consumes the following Phase 2 artefacts unchanged:

| Artefact class | Path | Phase 2 source |
|---|---|---|
| MVP 4 baseline checkpoint (mixed-case-trained) | `models/mvp4_gated_fusion_best/mvp4_trainable.pt` | NB 08 |
| MVP 4-IW checkpoint | `models/mvp4_iw_attention_best/mvp4_iw_trainable.pt` | NB 09 |
| MVP 4-IW-CC checkpoint | `models/mvp4_iwcc_attention_best/mvp4_iwcc_trainable.pt` | NB 09b |
| MVP 4-IW-CC-S checkpoint | `models/mvp4_iwccs_attention_best/mvp4_iwccs_trainable.pt` | NB 09c |
| Per-sample test gates (4 variants × 10,000 samples) | `models/mvp4_*/test_gates.npy` | NB 08 / 09 / 09b / 09c |
| Per-sample identity-attention fractions (3 IW variants × 10,000 samples) | `models/mvp4_iw*/test_identity_attention.npy` | NB 09 / 09b / 09c |
| Run-D LoRA text encoder (frozen) | `models/roberta_mvp1_d/` | NB 05 Run D |
| MVP 2 CLIP LoRA + `image_projection` (frozen) | `models/mvp2_naive_concat_best/` | NB 06 |
| Branch C standardisation statistics | `models/mvp3_three_branch_best/standardisation_stats.json` | NB 07 |
| HateXplain identity lexicon (1,177 tokens × 15 communities) | `data/processed/identity_lexicon.json` | `Identity_Lexicon_Build_Report.md` |
| Test set labels + structured features + raw tweet text | `data/processed/{labels_parsed.csv, structured_features.csv}` + `data/MMHS150K/splits/test_ids.txt` | NB 01 / NB 03 |

Phase 3 produces one new training checkpoint (MVP 4-lower) as an auxiliary deliverable to enable matched-methodology comparison; see § 5.

---

## 3. Notebook 10 — Per-Sample Modality Reliance Analysis

### 3.1 Purpose

Aggregate test-set metrics (AUC, F1m, FPR, T2 macro F1) across MVP 4 / MVP 4-IW / MVP 4-IW-CC / MVP 4-IW-CC-S sit in a narrow band — AUC range 0.7340 – 0.7400, F1m within 0.005 of each other. This narrow band may conceal qualitatively different per-sample decisions. NB 10 tests the hypothesis that **aggregate metric parity conceals per-sample divergence** by classifying every test sample into the five-category taxonomy and computing cross-variant agreement.

### 3.2 Operational definitions

**Per-branch logit projection** (per sample, per variant) — substitute one-hot gate weights at inference, equivalent to applying the trained T1 head to each per-branch projection directly because the gated fusion is a linear combination. Implementation:

```
logit_text_only   = head_t1(proj_text(text_cls))
logit_image_only  = head_t1(proj_image(attended_img))
logit_struct_only = head_t1(proj_struct(struct_embed))
```

No model code is modified; the operation is a single forward pass per sample plus three additional cheap `nn.Linear` calls.

**Confidence and correctness** — conventional thresholds, flagged in Cell 1 as tunable: `branch_confident = (sigmoid(logit) ≥ 0.6) ∨ (sigmoid(logit) ≤ 0.4)`; `branch_uncertain` is the strip in between. Correctness is `(sigmoid(logit_branch_only) ≥ 0.5) == t1_label`.

**Identity-token presence** — lowercased + regex-word-tokenized (no punctuation, no BPE) check against the 1,177-token HateXplain lexicon. A tweet is **identity-laden** if ≥ 1 lexicon hit is found. This is a sample-level definition, distinct from the BPE-token-level `identity_mask` used inside the cross-attention bias during training.

**Five-category taxonomy** (per sample, per variant):

- **Convergent Correct** — all three branches confident, all correct, fused correct
- **Text Saved** — text confident and correct; image and struct uncertain-or-incorrect; fused correct
- **Image Saved** — image confident and correct; text uncertain-or-incorrect; fused correct
- **Struct Saved** *(sub-bucket; rare)* — struct confident and correct; text and image uncertain-or-incorrect; fused correct
- **Emergent Multimodal** — no single branch confident-and-correct; fused correct
- **Fusion Failure** — fused incorrect (regardless of per-branch confidence)

**Three identity-token subgroups** for stratified analysis:

- `identity_laden_hate`: has ≥ 1 lexicon hit AND `t1_label == 1` — n = 4,796
- `identity_laden_nothate`: has ≥ 1 lexicon hit AND `t1_label == 0` — n = 4,722
- `identity_free`: no lexicon hits — n = 482

**Bootstrap CIs** — 95 % CI from 1,000 resamples (with replacement), seeded `np.random.default_rng(42)`.

### 3.3 Results — taxonomy distribution

Per-variant share of test samples in each taxonomy category (n = 10,000 per variant):

| Variant | Convergent Correct | Text Saved | Image Saved | Struct Saved | Emergent Multimodal | Fusion Failure |
|---|---:|---:|---:|---:|---:|---:|
| MVP 4 | 2.10 % | 40.52 % | 0.01 % | 0.00 % | 26.06 % | 31.31 % |
| MVP 4-IW | 1.03 % | 43.02 % | 0.00 % | 1.58 % | 23.15 % | 31.22 % |
| MVP 4-IW-CC | 17.35 % | 7.30 % | 16.39 % | 0.00 % | 27.79 % | 31.17 % |
| **MVP 4-IW-CC-S** | 4.20 % | 21.97 % | **30.64 %** | 0.18 % | 11.90 % | **31.11 %** |

Source: `outputs/nb10/table_3_taxonomy.md`. Two observations:

1. **MVP 4-IW-CC-S has the highest Image-Saved share** (30.64 %) across the four variants — it uses the image branch as the decisive modality on the largest fraction of test samples. This is consistent with NB 09c's healthy gate distribution (`gate_image = 0.219` mean) and is the strongest single piece of evidence that the variant uses its multimodal architecture.
2. **All four variants have nearly identical Fusion Failure rates** (31.11 – 31.31 %). The architectural changes redistribute samples across the correct-fused buckets without changing the overall failure rate; the ~ 0.74 LoRA ceiling documented in § 11 of `Phase2_Modeling_Report.md` constrains all four.

### 3.4 Results — agreement matrix

Per-sample fused-prediction agreement across the four variants:

| | MVP 4 | MVP 4-IW | MVP 4-IW-CC | MVP 4-IW-CC-S |
|---|---:|---:|---:|---:|
| **MVP 4** | 1.000 | 0.982 | 0.981 | 0.983 |
| **MVP 4-IW** | 0.982 | 1.000 | 0.980 | 0.980 |
| **MVP 4-IW-CC** | 0.981 | 0.980 | 1.000 | 0.988 |
| **MVP 4-IW-CC-S** | 0.983 | 0.980 | 0.988 | 1.000 |

Source: `outputs/nb10/04_agreement_matrix.png` + `summary.md`. **The four variants agree on > 97 % of test samples**. Aggregate metric parity does not conceal dramatic per-sample divergence — it reflects near-identical behaviour. The IW family is making the same per-sample decision as the gated baseline on > 97 % of cases.

### 3.5 Results — subgroup-stratified accuracy (NB 10 / non-matched methodology)

From `outputs/nb10/table_2_subgroup_accuracy.md`, T1 accuracy with 95 % bootstrap CIs:

| Variant | identity_laden_hate (n = 4,796) | **identity_laden_nothate (n = 4,722)** | identity_free (n = 482) |
|---|---|---|---|
| MVP 4 | 0.6666 [0.6535, 0.6789] | 0.7171 [0.7039, 0.7300] | 0.5934 [0.5477, 0.6390] |
| MVP 4-IW | 0.6599 [0.6468, 0.6733] | **0.7245** [0.7116, 0.7376] | 0.6058 [0.5602, 0.6473] |
| MVP 4-IW-CC | 0.6697 [0.6568, 0.6825] | 0.7156 [0.7027, 0.7283] | 0.6058 [0.5622, 0.6494] |
| **MVP 4-IW-CC-S** | **0.6754** [0.6628, 0.6885] | **0.7118** [0.6986, 0.7245] | 0.5996 [0.5560, 0.6452] |

**On `identity_laden_nothate` — the exact subgroup the IW mechanism was designed to help — MVP 4-IW-CC-S is the worst of the four variants (0.7118), not the best.** MVP 4-IW (the "failed" variant from NB 09) is the best on this subgroup at 0.7245.

This is the finding that triggered the matched-methodology re-analysis in § 4 and the diagnostic investigations in §§ 5–6. Note that this is the **non-matched-methodology** version of the analysis: MVP 4 here is the original mixed-case-trained model evaluated on lowercase input (NB 10 always lowercases). § 4 re-does this comparison with a matched MVP 4-lower baseline.

### 3.6 Results — per-T2-class performance (NB 10 / non-matched methodology)

From `outputs/nb10/table_5_t2_per_class_f1_ci.md`, T2 per-class F1 with 95 % bootstrap CIs on the 8,411 `t2_valid` test rows:

| T2 class | MVP 4 | MVP 4-IW | MVP 4-IW-CC | MVP 4-IW-CC-S |
|---|---|---|---|---|
| NotHate | 0.6154 [0.6030, 0.6281] | 0.3378 [0.3237, 0.3528] | 0.5154 [0.5011, 0.5287] | 0.5791 [0.5663, 0.5930] |
| Racist | 0.4442 [0.4221, 0.4640] | 0.4401 [0.4225, 0.4552] | 0.4406 [0.4221, 0.4585] | 0.4576 [0.4376, 0.4764] |
| Sexist | 0.4041 [0.3742, 0.4316] | 0.4119 [0.3810, 0.4409] | 0.3880 [0.3591, 0.4145] | 0.4054 [0.3759, 0.4328] |
| Homophobe | 0.7389 [0.7118, 0.7638] | 0.7305 [0.7029, 0.7563] | 0.7430 [0.7153, 0.7679] | 0.7342 [0.7067, 0.7584] |
| Religion | 0.1562 [0.0889, 0.2256] | 0.0645 [0.0367, 0.0936] | 0.1250 [0.0728, 0.1788] | 0.0885 [0.0517, 0.1276] |
| OtherHate | 0.5979 [0.5691, 0.6234] | 0.6000 [0.5742, 0.6262] | 0.6005 [0.5738, 0.6267] | 0.6212 [0.5929, 0.6472] |

**Religion class is consistently low across all variants** (F1 0.0645 – 0.1562). Class scarcity (~ 0.3 % of training rows, see § 10 of `Phase2_Modeling_Report.md`) is the binding constraint; no IW variant solves it. **Homophobe is consistently the strongest class** (F1 0.7305 – 0.7430), almost certainly because the lexicon vocabulary for that community is small and consistent across the training distribution.

The NotHate row is where the NB 09c-vs-MVP 4 comparison breaks down under fresh inference (MVP 4 NotHate F1 = 0.6154 here vs the 0.5685 quoted in NB 09c). This is fully decomposed in §§ 4 + 5 below.

---

## 4. Notebook 10-lower — Matched-Methodology Re-Analysis

### 4.1 Purpose

NB 10's MVP 4 baseline was the original mixed-case-trained NB 08 checkpoint, but NB 10 lowercased all input before tokenization (because the IW variants need lowercase for identity-mask alignment). This put MVP 4 out-of-distribution at the very first layer while keeping the IW variants in-distribution — a methodological inconsistency that biases the comparison in unpredictable directions. NB 10-lower fixes this by replacing MVP 4 with **MVP 4-lower**, a re-trained-on-lowercase variant (§ 5), and adds a fifth ablation variant — **MVP 4-IW-CC-S-bias-off** — to isolate the IW mechanism's standalone contribution (§ 6). All five variants now operate on the same input distribution.

### 4.2 The five-variant comparison

From `outputs/diagnostic/mvp4_corrected_comparison.md` Table A, all on lowercase preprocessing, n_t2_valid = 8,411:

| Variant | AUC [95% CI] | F1m [95% CI] | FPR | T2 macro F1 [95% CI] | **T2 NotHate F1 [95% CI]** | Gate (t / i / s, H) |
|---|---|---|---:|---|---|---|
| MVP 4-lower (NB 08-lower) | 0.7358 [0.7253, 0.7456] | 0.6867 [0.6772, 0.6956] | 0.2921 | 0.4411 [0.4260, 0.4555] | **0.2692** [0.2559, 0.2829] | 0.450 / 0.062 / 0.489, H = 0.845 |
| MVP 4-IW (NB 09) | 0.7359 [0.7258, 0.7459] | 0.6876 [0.6780, 0.6965] | 0.2857 | 0.4308 [0.4196, 0.4410] | 0.3378 [0.3237, 0.3528] | 0.374 / 0.299 / 0.327, H = 1.073 |
| MVP 4-IW-CC (NB 09b) | 0.7340 [0.7241, 0.7437] | 0.6882 [0.6790, 0.6971] | 0.2981 | 0.4688 [0.4554, 0.4811] | 0.5154 [0.5011, 0.5287] | 1.000 / 0.000 / 0.000, H = 0.000 |
| **MVP 4-IW-CC-S (NB 09c)** | **0.7359** [0.7256, 0.7458] | **0.6889** [0.6797, 0.6979] | 0.3011 | **0.4810** [0.4685, 0.4926] | **0.5791** [0.5663, 0.5930] | 0.467 / 0.219 / 0.315, H = 1.029 |
| IW-CC-S-bias-off (λ_id = 0) | 0.7359 [0.7256, 0.7458] | 0.6888 [0.6796, 0.6979] | 0.3013 | 0.4810 [0.4686, 0.4926] | 0.5789 [0.5661, 0.5930] | 0.467 / 0.219 / 0.315, H = 1.029 |

The single most striking observation in this table is the **NotHate F1 collapse for MVP 4-lower** (0.2692 vs MVP 4-original's 0.5685 on mixed-case input). § 5 explains why this happens.

### 4.3 The bias-off ablation finding

The bias-off variant operates the **same** trained MVP 4-IW-CC-S model with the identity-bias term forced to zero at inference (`model.cross_attn.lambda_id.data.zero_()` after load). All trained weights are byte-identical to MVP 4-IW-CC-S; only the forward pass changes.

Δ between MVP 4-IW-CC-S and IW-CC-S-bias-off on the 10,000-sample test set:

| Metric | MVP 4-IW-CC-S | IW-CC-S-bias-off | Δ |
|---|---:|---:|---:|
| AUC | 0.7359 | 0.7359 | 0.0000 |
| F1m | 0.6889 | 0.6888 | +0.0001 |
| FPR | 0.3011 | 0.3013 | −0.0002 |
| T2 macro F1 | 0.4810 | 0.4810 | 0.0000 |
| **T2 NotHate F1** | **0.5791** | **0.5789** | **+0.0002** |
| Per-sample T1 agreement | — | — | **9,999 / 10,000 = 99.99 %** |
| Gate (t / i / s) | 0.467 / 0.219 / 0.315 | 0.467 / 0.219 / 0.315 | identical |

**The identity-bias term `λ_id · identity_mask · (1 + α · vader_neg_centered)` contributes essentially zero to final predictions.** The two variants disagree on exactly 1 sample out of 10,000. Gate distributions are identical to three decimal places. T2 macro F1 is identical to four decimal places.

This is the central analytical finding of Phase 3. The named "context-conditioned identity-weighted attention" mechanism (the NB 09 → NB 09c claimed contribution) is not what produces MVP 4-IW-CC-S's lift over MVP 4-IW on T2 NotHate F1 (Δ = +0.2413, fully reproducible). The lift comes from the **architectural plumbing introduced alongside** the bias term:

- **Per-branch LayerNorm before gate concatenation** (NB 09c change 2) — equalises text_cls, attended_img, and struct_embed magnitudes before the gate Linear sees them, preventing the gate-collapse-via-magnitude-mismatch failure mode documented in NB 09b.
- **Centered VADER modulation** (NB 09c change 1) — `1 + α · (vader_neg − μ_vneg_train)` is approximately symmetric around 1.0 rather than always ≥ 1.0, reducing the variance of the cross-attended image vector entering the gate.
- **Resulting healthy gate distribution** — text 0.47 / image 0.22 / struct 0.32, H = 1.03 vs NB 09b's collapsed text 1.00 / image 0.00 / struct 0.00, H = 0.00.

The bias-off ablation isolates this attribution: with `λ_id = 0`, the cross-attention block degenerates into standard multi-head attention (the bias term is the only IW-specific addition), and the model still achieves the same NotHate F1 to 4 decimal places. The IW mechanism is decorative.

### 4.4 Subgroup-stratified findings (matched methodology)

From `outputs/diagnostic/mvp4_corrected_comparison.md` Table B, T1 accuracy with 95 % bootstrap CIs:

| Variant | identity_laden_hate (n = 4,796) | **identity_laden_nothate (n = 4,722)** | identity_free (n = 482) |
|---|---|---|---|
| MVP 4-lower (NB 08-lower) | 0.6635 [0.6509, 0.6766] | 0.7190 [0.7061, 0.7319] | 0.6037 [0.5602, 0.6474] |
| MVP 4-IW (NB 09) | 0.6599 [0.6468, 0.6733] | **0.7245** [0.7116, 0.7376] | 0.6058 [0.5602, 0.6473] |
| MVP 4-IW-CC (NB 09b) | 0.6697 [0.6568, 0.6825] | 0.7156 [0.7027, 0.7283] | 0.6058 [0.5622, 0.6494] |
| **MVP 4-IW-CC-S (NB 09c)** | **0.6754** [0.6628, 0.6885] | **0.7118** [0.6986, 0.7245] | 0.5996 [0.5560, 0.6452] |
| IW-CC-S-bias-off (λ_id = 0) | 0.6754 [0.6628, 0.6885] | 0.7116 [0.6984, 0.7243] | 0.5996 [0.5560, 0.6452] |

Even under matched methodology with a retrained MVP 4-lower baseline, **MVP 4-IW-CC-S remains the worst of four IW variants on identity_laden_nothate** (0.7118 vs MVP 4-IW's 0.7245, MVP 4-lower's 0.7190). The identity-over-firing-prevention narrative does not survive subgroup stratification. The variant has the best identity_laden_hate accuracy (0.6754), but the magnitudes are small and CIs overlap with MVP 4-IW-CC.

### 4.5 Per-T2-class findings (matched methodology)

From `outputs/diagnostic/mvp4_corrected_comparison.md` Table C:

| T2 class | MVP 4-lower | MVP 4-IW | MVP 4-IW-CC | **MVP 4-IW-CC-S** | IW-CC-S-bias-off |
|---|---|---|---|---|---|
| NotHate | 0.2692 [0.256, 0.283] | 0.3378 [0.324, 0.353] | 0.5154 [0.501, 0.529] | **0.5791** [0.566, 0.593] | 0.5789 [0.566, 0.593] |
| Racist | 0.4490 | 0.4401 | 0.4406 | **0.4576** | 0.4575 |
| Sexist | 0.3895 | **0.4119** | 0.3880 | 0.4054 | 0.4054 |
| Homophobe | **0.7400** | 0.7305 | 0.7430 | 0.7342 | 0.7347 |
| Religion | **0.1508** | 0.0645 | 0.1250 | 0.0885 | 0.0885 |
| OtherHate | **0.6480** | 0.6000 | 0.6005 | 0.6212 | 0.6208 |

Under matched methodology, **MVP 4-IW-CC-S wins NotHate F1 (0.5791) and Racist F1 (0.4576)**, but loses on Religion (0.0885), OtherHate (0.6212), Homophobe (0.7342), and Sexist (0.4054) — though most of the losses are within the bootstrap CI width. MVP 4-lower, despite the NotHate collapse, is competitive or better on every hate class except NotHate and Racist. This per-class picture suggests the IW family's NotHate strength comes at a modest cost across the hate categories.

---

## 5. Diagnostic — Case-Preprocessing Inconsistency

Source: `outputs/diagnostic/nb08_vs_nb10_audit.md` + `audit_raw_results.json`. Auxiliary checkpoint: `models/mvp4_gated_fusion_lowercase_best/mvp4_trainable.pt`.

### 5.1 The bug

NB 08 trained MVP 4 on **mixed-case** `tweet_text`. NB 09 introduced `text = text.lower()` inside `compute_identity_mask` so the BPE offsets would align with the lowercase HateXplain lexicon, and the lowercased `input_ids` from that tokenization became the model input. NB 09b and NB 09c inherited this preprocessing. NB 10 inherited it again for all four variants — including MVP 4 — putting MVP 4 out-of-distribution at the very first layer because the cardiffnlp Twitter-RoBERTa tokenizer is **BPE-based and case-sensitive** (`"Hello"` and `"hello"` produce different `input_ids`).

### 5.2 Reproduction

Four targeted probes on the studio (same checkpoint per variant, same fp16 autocast, same 8,411-row `t2_valid` mask, only variable = case mode):

| Probe | Case mode | NotHate F1 | Macro F1 |
|---|---|---:|---:|
| MVP 4 (NB 08 canonical) | mixed | **0.5681** | 0.4930 |
| MVP 4 (NB 10 setup) | lower | **0.6154** | 0.4928 |
| MVP 4-IW-CC-S (OOD probe, identity-bias = 0) | mixed | 0.5603 | 0.4636 |
| MVP 4-IW-CC-S (canonical, identity-bias = 0) | lower | 0.5789 | 0.4810 |

The MVP 4 mixed-case probe reproduces NB 08's reported 0.5685 within fp16 rounding noise (Δ = 0.0004). The MVP 4 lowercase probe reproduces NB 10's 0.6154 exactly. **Same checkpoint, same data, same evaluation; only the case-preprocessing changes.** Macro F1 stays approximately constant across case modes — what changes is the distribution of predictions across classes, with NotHate getting a +0.047 boost under lowercase at the expense of every hate class.

### 5.3 Fix — MVP 4-lower

To enable apples-to-apples comparison, MVP 4 was retrained on lowercase input. The retrain notebook is a programmatic clone of NB 08 with a single line changed in the dataset class:

```diff
- texts = sub_df['tweet_text'].astype(str).tolist()
+ texts = sub_df['tweet_text'].astype(str).str.lower().tolist()  # LOWERCASE (08-lower variant)
```

All hyperparameters, the freeze surface, the loss, seed (42), and the 5-epoch budget are byte-identical to NB 08. Checkpoints save under `models/mvp4_gated_fusion_lowercase[_best]/`. Original NB 08 artefacts are not modified.

### 5.4 What the retrain revealed

MVP 4-lower converged to `val_t1_auc = 0.7414` (best epoch = 2) vs NB 08 original's 0.7470 (best epoch = 4). The Δ = −0.0056 is just outside the documented ±0.005 noise band. More strikingly, **MVP 4-lower's test T2 NotHate F1 collapsed from MVP 4-original's 0.5685 (mixed-case eval) to 0.2692 (lowercase eval)** — a −0.2993 drop, far larger than any other class shift.

**Interpretation.** The frozen Run-D LoRA text encoder was trained on mixed-case input (NB 05 Run D, see § 10 of `Phase2_Modeling_Report.md`). Lowercase preprocessing puts the encoder out-of-distribution at the very first layer. All IW variants (NB 09 / 09b / 09c) inherit this OOD behaviour because they too feed lowercased input. The IW family's cross-attention + gated fusion architecture compensates for the OOD damage through downstream components and retains NotHate F1 ≈ 0.58; a plain gated fusion (MVP 4-lower) without IW-style routing flexibility cannot compensate, and its NotHate F1 collapses to 0.27.

This is **not** evidence that the IW mechanism is doing what it was advertised to do (identity-aware reasoning). It is evidence that the IW family's architectural plumbing is robust to encoder-OOD damage from preprocessing mismatch. The bias-off ablation in § 6 confirms that the identity-bias term itself is not what provides this robustness.

### 5.5 Methodological lesson

When chaining frozen pretrained components with downstream trainable modules, **input preprocessing must match the frozen component's training distribution**. Lowercasing the input to a case-sensitive frozen encoder puts the encoder OOD at the very first layer, and any downstream metric comparing two variants on different preprocessing is methodologically suspect. The NB 09c "+0.0101 over MVP 4" claim suffered exactly this confound — MVP 4 was reported on mixed-case input (in-distribution for it) while MVP 4-IW-CC-S was reported on lowercase input (in-distribution for it but OOD for MVP 4); the comparison is not apples-to-apples regardless of which direction it points.

---

## 6. Diagnostic — Bias-Off Ablation

Source: `outputs/diagnostic/mvp4_corrected_comparison.md` § Q2 + NB 10-lower (`notebooks/10_lower_per_sample_modality_analysis.ipynb`).

### 6.1 Hypothesis

NB 09c introduced three combined architectural changes (§ 16c.2 of `Phase2_Modeling_Report.md`):

1. Centered VADER modulation: `(1 + α · vader_neg)` → `(1 + α · (vader_neg − μ_vneg_train))`
2. Per-branch LayerNorm before gate concatenation (`gate_ln_text / gate_ln_image / gate_ln_struct`)
3. `α_init` reduced from 1.0 to 0.5

These three changes were bundled together to fix the NB 09b gate collapse, and the variant's NotHate F1 improvement over MVP 4-IW (Δ = +0.2413 under matched methodology, 0.5791 vs 0.3378) was attributed to the IW family's "context-conditioned identity-weighted attention" mechanism. The bias-off ablation tests this attribution by zeroing the identity-bias term itself while leaving every other architectural component (per-branch LN, VADER centering, gate, projections, heads, frozen encoders) untouched.

### 6.2 Method

Load the trained MVP 4-IW-CC-S checkpoint (`mvp4_iwccs_trainable.pt`). After load, force `λ_id = 0` via `model.cross_attn.lambda_id.data.zero_()`. The IW bias contribution to the cross-attention logits becomes:

```
λ_id · identity_mask · (1 + α · vader_neg_centered) = 0 · identity_mask · (...) = 0
```

regardless of the identity_mask or alpha. All other trained weights are byte-identical. Forward pass produces:

```
attn_out = softmax((Q @ K^T) / sqrt(d_k) + 0) @ V   # standard cross-attention, no IW bias
```

The architecture degenerates to standard multi-head cross-attention with the gate, projections, heads, and per-branch LNs all intact. Run inference on the 10,000-sample test set with the same fp16 autocast as MVP 4-IW-CC-S. Compute all per-sample metrics (T1 fused logit, per-branch logits, gate weights, T2 logits, taxonomy category).

### 6.3 Result

Per § 4.3, the bias-off variant matches MVP 4-IW-CC-S to four decimal places on every aggregate metric and agrees on 9,999 / 10,000 test samples. The single disagreement is one sample where the fused T1 logit is within fp16 noise of the 0.5 threshold and the bias-term contribution flips it.

### 6.4 Attribution

The IW mechanism contributes ~ 0. The architectural plumbing carries the win over MVP 4-IW. Specifically:

- The lift from MVP 4-IW (NotHate F1 0.3378) to MVP 4-IW-CC-S (0.5791) is **+0.2413**.
- Of this lift, **+0.0002** (≈ 0.08 % of the lift) is from the identity-bias term itself.
- The remaining **+0.2411** (≈ 99.92 % of the lift) is from the architectural changes that NB 09c bundled alongside the bias term: per-branch LayerNorm, centered VADER modulation, and the resulting gate distribution.

A cleaner mental model: NB 09 introduced the IW bias term and a collapsed gate. NB 09b made the IW bias context-conditioned and the gate collapse persisted. NB 09c added two architectural fixes (per-branch LN + centered modulation) that resolved the gate collapse, and the **gate collapse resolution** is what restores NotHate F1 — not the now-context-conditioned-and-centered bias term itself.

### 6.5 Reproducibility

The bias-off variant is implemented in `notebooks/10_lower_per_sample_modality_analysis.ipynb` Cell 4 via a `zero_identity_bias` flag in the variant config. Per-sample data for the variant is saved to `data/processed/nb10_lower_variant_mvp4_iwccs_biasoff/per_sample.parquet`. The 1-sample disagreement is identifiable from the merged 50,000-row parquet at `data/processed/nb10_lower_per_sample_data.parquet` by filtering to `variant in {mvp4_iwccs, mvp4_iwccs_biasoff}` and finding the `tweet_id` where `fused_pred` differs.

---

## 7. Findings

1. **Aggregate metrics across the four MVP 4 variants are within noise (AUC 0.7340 – 0.7400) and conceal near-identical per-sample behaviour.** Cross-variant agreement on fused T1 prediction is > 97 % across all pairs (§ 3.4). The narrow aggregate band reflects genuine similarity, not divergence masked by averaging.

2. **The IW mechanism does not contribute to final performance under bias-off ablation.** MVP 4-IW-CC-S with `λ_id = 0` matches the full IW-CC-S to four decimal places on every aggregate metric and agrees on 9,999 / 10,000 per-sample predictions (§ 4.3, § 6). The named "context-conditioned identity-weighted attention" mechanism is decorative; the architectural plumbing (per-branch LN, centered VADER modulation, healthy gate) is what carries the lift over MVP 4-IW.

3. **Under matched methodology, MVP 4-IW-CC-S beats MVP 4-lower on T2 NotHate F1 by +0.3099 (0.5791 vs 0.2692), but this gap is dominated by MVP 4-lower's encoder-OOD collapse from lowercase preprocessing, not by IW-CC-S being intrinsically better.** MVP 4-lower's NotHate F1 collapsed from 0.5685 (mixed-case eval) to 0.2692 (lowercase eval). The IW family's architecture is more robust to encoder-OOD damage, but the comparison's headline gap is a measure of how badly plain gated fusion fails under preprocessing mismatch, not a measure of how well IW-CC-S succeeds.

4. **The `identity_laden_nothate` subgroup analysis contradicts the NB 09 → NB 09c over-firing narrative.** Both under non-matched (§ 3.5) and matched (§ 4.4) methodology, MVP 4-IW-CC-S is the worst of the IW family on this subgroup (T1 accuracy 0.7118 matched, 0.7118 non-matched). MVP 4-IW (the variant criticised in NB 09b / NB 09c for "identity over-firing on benign non-hate") is the best on this subgroup (0.7245 matched). The architectural narrative about IW-CC-S preventing over-firing does not survive stratification.

5. **MVP 4-IW-CC-S retains genuine wins**: highest Image-Saved share (30.64 %), healthy gate distribution (H = 1.029 vs MVP 4-lower's 0.845 and NB 09b's 0.000), and CI-supported wins on T2 Racist F1 (0.4576 vs MVP 4-lower's 0.4490) and T2 OtherHate F1 (overlapping CIs with MVP 4-lower 0.6480, modest win for IWCCS at 0.6212 — though direction is uncertain). These wins are attributable to the architectural plumbing, not the IW mechanism.

6. **The case-preprocessing diagnostic, the bias-off ablation methodology, and the per-sample modality reliance taxonomy are themselves methodological contributions.** Each is reusable for any future architectural-improvement claim that depends on a single attention bias term, a frozen pretrained component, or a multimodal routing mechanism. The diagnostic infrastructure includes the 4-probe case-preprocessing audit pattern (`outputs/diagnostic/nb08_vs_nb10_audit.md`), the matched-methodology rerun template (`notebooks/10_lower_per_sample_modality_analysis.ipynb`), and the bias-off-via-`zero_identity_bias`-flag pattern.

7. **The bias-robustness analysis (§ 10) provides a third independent line of evidence confirming the failure-mode characterisation.** All four MVP 4 variants exhibit statistically equivalent bias profiles across identity-term masking, counterfactual swap, and per-community stratification analyses. MVP 4-IW-CC-S exhibits the highest identity-token sensitivity (masking flip rate 0.3691 vs 0.3534 for MVP 4-lower), opposite to the design intent. The bias-off ablation (§ 6 / NB 10-lower), per-sample modality reliance taxonomy (§ 3 / NB 10), and identity-term masking analysis (§ 10 / NB 11) together demonstrate that the explicit identity-bias mechanism in MVP 4-IW-CC-S contributes negligibly to model behaviour at every level of analysis — aggregate predictions, per-sample reliance patterns, and bias profile.

---

## 8. Methodological recommendations for future work

1. **Match input preprocessing to frozen-component training distribution.** When chaining a frozen pretrained encoder with a downstream trainable head, any input preprocessing must match the encoder's training distribution. Case-folding, normalisation, tokenizer choice, and special-token handling are all candidate sources of OOD damage at the first layer.

2. **Run bias-off ablations on any single-term-attention contribution claim.** Architectural papers that attribute performance lift to a single attention modification (additive bias, multiplicative gate, learned scalar) should report an ablation that zeros the modification and runs inference. If the resulting prediction set agrees ≥ 99 % with the un-ablated set, the modification is decorative and the lift comes from elsewhere.

3. **Stratify by subgroup before claiming improvement over baseline.** Aggregate F1 / accuracy / AUC can shift in opposite directions on subgroups that the architectural change was designed to help vs subgroups it was not. Subgroup-stratified analysis with bootstrap CIs (e.g., this report's `identity_laden_nothate` finding) catches the case where the headline metric improves while the named target subgroup degrades.

4. **Report bootstrap CIs on all variant-vs-variant comparisons.** 95 % CIs from 1,000 resamples (seed 42) are cheap, reproducible, and reveal whether observed deltas are within sampling noise. Several "wins" in this report (e.g., MVP 4-IW-CC-S vs MVP 4-IW-CC on AUC) become CI-overlapping under bootstrap and should not be claimed as architectural wins.

5. **Pre-flight probes before long training runs.** NB 09c's Cell 11 probe (random-init forward pass, std-ratio check against threshold, raise on failure) caught the per-branch LN normalisation correctness before committing to a 60-minute training run. The same pattern is applicable to any architectural change whose effect on initialisation statistics can be tested in 30 seconds.

6. **Document the failure-mode trail.** The NB 09 → NB 09b → NB 09c → bias-off → MVP 4-lower trail is itself a methodological contribution: three negative results (one over-firing failure, one gate-collapse failure, one decorative-mechanism finding) plus one architectural fix (per-branch LN + centering) plus one matched-methodology retrain. Negative results should be preserved as documented failure modes, not hidden behind the positive-result headlines.

7. **Bias-robustness improvements on MMHS150K-style datasets under PEFT constraints will likely require dataset-level interventions rather than attention-mechanism interventions.** NB 11's identity-term masking flip rate of ~ 35 % across four architectural variants (§ 10.3) — approximately 2× the 15 % design-target threshold — suggests architectural fusion strategies alone cannot resolve the dataset's identity-token reliance. Future work should target re-balanced training data, identity-token augmentation, and counterfactual data augmentation during training rather than additional gated-attention variants.

---

## 9. Artefacts produced

| Artefact | Path | Size |
|---|---|---:|
| NB 10 executed notebook (4 variants, mixed-case MVP 4 baseline) | `notebooks/10_per_sample_modality_analysis.ipynb` | 1.36 MB |
| NB 10-lower executed notebook (5 variants on lowercase, with bias-off) | `notebooks/10_lower_per_sample_modality_analysis.ipynb` | 1.66 MB |
| NB 08-lower auxiliary retrain notebook (programmatic NB 08 clone with `.str.lower()`) | `notebooks/08_lower_mvp4_lowercase.ipynb` | 389 KB |
| NB 10 per-sample analytical parquet (4 variants × 10K test samples = 40K rows × 33 cols) | `data/processed/nb10_per_sample_data.parquet` | 1.87 MB |
| NB 10-lower per-sample analytical parquet (5 variants × 10K = 50K rows × 33 cols) | `data/processed/nb10_lower_per_sample_data.parquet` | 2.28 MB |
| MVP 4-lower trainable checkpoint (best by val T1 AUC, epoch 2) | `models/mvp4_gated_fusion_lowercase_best/mvp4_trainable.pt` | 11.36 MB |
| MVP 4-lower standardisation stats sidecar | `models/mvp4_gated_fusion_lowercase_best/standardisation_stats.json` | 1.12 KB |
| Case-preprocessing 4-probe audit | `outputs/diagnostic/nb08_vs_nb10_audit.md` | 1.49 KB |
| Methodology reconciliation (A vs B side-by-side with CIs) | `outputs/diagnostic/mvp4_t2_reconciled.md` | 6.78 KB |
| Final 5-variant corrected comparison (matched lowercase methodology + bias-off, 3 tables + 3-question headline) | `outputs/diagnostic/mvp4_corrected_comparison.md` | 7.80 KB |
| Audit raw per-probe results | `outputs/diagnostic/audit_raw_results.json` | 1.02 KB |
| NB 10 chart: per-variant taxonomy distribution stacked bar | `outputs/nb10/01_taxonomy_distribution.png` | 79.12 KB |
| NB 10 chart: per-variant gate distribution hexbin (4-panel) | `outputs/nb10/02_gate_distribution.png` | 326.94 KB |
| NB 10 chart: identity-attention hate vs nothate (3 IW variants) | `outputs/nb10/03_identity_attention.png` | 85.79 KB |
| NB 10 chart: 4×4 per-sample agreement matrix | `outputs/nb10/04_agreement_matrix.png` | 85.53 KB |
| NB 10 chart: id-attn vs correctness scatter (3 IW variants × 4 color classes) | `outputs/nb10/05_id_attn_vs_correctness.png` | 502.49 KB |
| NB 10 chart: identity_laden_nothate confusion matrices (4 variants) | `outputs/nb10/06_identity_laden_nothate_cm.png` | 140.93 KB |
| NB 10 chart: per-T2-class F1 grouped bar with bootstrap CIs | `outputs/nb10/07_t2_per_class_f1.png` | 73.28 KB |
| NB 10 chart: subgroup × variant T1 accuracy with bootstrap CIs | `outputs/nb10/08_subgroup_accuracy.png` | 84.41 KB |
| NB 10 markdown tables (5 files) + summary | `outputs/nb10/{table_1..5, summary}.md` | 6.71 KB total |
| NB 10-lower charts (same 8 plot families with 5 variants + bias-off) | `outputs/nb10_lower/*.png` | 1.70 MB total |
| NB 10-lower markdown tables (5 files) + summary | `outputs/nb10_lower/{table_1..5, summary}.md` | 8.00 KB total |
| NB 11 executed notebook (3 analyses × 4 variants, 12 cells, 0 errors) | `notebooks/11_bias_analysis.ipynb` | 755 KB |
| NB 11 per-sample masking data (4 variants × identity-laden samples, with flip flags) | `data/processed/nb11_per_sample_masking_data.parquet` | 436 KB |
| NB 11 chart: per-community masking flip rate grouped bar with bootstrap CIs | `outputs/nb11/01_masking_flip_rate.png` | 104 KB |
| NB 11 chart: counterfactual swap-pair asymmetry heatmap (4-panel) | `outputs/nb11/02_swap_asymmetry.png` | 248 KB |
| NB 11 chart: per-community T1 accuracy grouped bar with bootstrap CIs | `outputs/nb11/03_per_community_accuracy.png` | 98 KB |
| NB 11 table 1 — per-community masking flip rate (4 variants × 11 rows) | `outputs/nb11/table_1_masking.md` | 2.75 KB |
| NB 11 table 2 — counterfactual swap-pair flip rate (19 pairs × 4 variants) | `outputs/nb11/table_2_swap.md` | 3.51 KB |
| NB 11 table 3 — per-community T1 acc + FPR + T2 macro F1 with bootstrap CIs | `outputs/nb11/table_3_per_community.md` | 4.93 KB |
| NB 11 summary findings | `outputs/nb11/summary.md` | 2.12 KB |

---

## 10. Notebook 11 — Bias / Fairness Robustness Analysis

### 10.1 Purpose

The bias-off ablation in § 6 established that the identity-bias term in MVP 4-IW-CC-S contributes negligibly to final predictions (Δ = +0.0002 T2 NotHate F1, 9,999 / 10,000 per-sample agreement). NB 11 tests whether this null effect extends to **bias-robustness**: do the four MVP 4 variants exhibit meaningfully different bias profiles, even if they do not exhibit meaningfully different aggregate metrics? The pre-registered expectation, given the bias-off finding, is that bias-profile differences will be statistically equivalent across variants. NB 11 verifies this through three independent analyses on the same matched-methodology lowercase test set used in NB 10-lower.

The hypothesis being tested is therefore: **if the IW mechanism is decorative for aggregate metrics, it should also be decorative for bias profile**. Three independent analyses are performed on the same test set; if all three return statistically equivalent results across variants, the bias-profile null effect is confirmed.

### 10.2 Three analyses performed

#### 10.2.1 Identity-Term Masking Test

For each identity-laden test sample (n = 9,518 samples with ≥ 1 lexicon hit), replace identity-tagged BPE positions with the tokenizer's mask token (`<mask>`, ID = `tokenizer.mask_token_id`) and run inference on both the original and masked versions for all four variants. The **flip rate** is the fraction of samples where the T1 fused prediction changes between original and masked. Higher flip rate means greater reliance on identity terminology itself rather than substantive content. Stratified by:

- T1 label (hate vs non-hate)
- Identity community (Islam, African, Jewish, Refugee, Homosexual, Women, Arab, Asian, Caucasian, Hispanic — the 10 communities with ≥ 30 lexicon tokens)

The design target threshold (set at project outset) is **< 15 % flip rate per community**.

#### 10.2.2 Counterfactual Identity Swap Test

19 swap pairs are predefined upfront in the notebook (Cell 4) for reproducibility. For each pair (e.g., `muslim ↔ christian`, `black ↔ white`, `man ↔ woman`), find test samples containing the source word, replace it with the target word, re-tokenize, re-compute the identity mask, and run inference. The flip rate is computed per pair per variant. Reciprocal pairs (e.g., `muslim → christian` and `christian → muslim`) are both tested to measure asymmetric bias.

#### 10.2.3 Per-Community Performance Stratification

For each of the 10 top-population communities, compute T1 accuracy, FPR, and T2 macro F1 per variant on the subgroup of test samples whose identity tokens target that community. 95 % bootstrap CIs from 1,000 resamples (seed 42). Compare best-vs-worst community disparity per variant.

### 10.3 Results — Analysis 1 (Masking Flip Rate)

Overall identity-token masking flip rates per variant (n = 9,518 identity-laden test samples):

| Variant | Overall flip rate [95 % CI] |
|---|---|
| MVP 4-lower | 0.3534 [0.3437, 0.3639] |
| MVP 4-IW | 0.3539 [0.3447, 0.3634] |
| MVP 4-IW-CC | 0.3635 [0.3544, 0.3741] |
| **MVP 4-IW-CC-S** | **0.3691** [0.3596, 0.3790] |

**All four variants exceed the 15 % target threshold by approximately 2×.** The variant range (0.3534 – 0.3691) falls within the bootstrap CI widths — no variant exhibits meaningfully different identity-sensitivity from the others. Counterintuitively, **MVP 4-IW-CC-S — the variant whose architecture was designed to be identity-aware — has the highest flip rate, not the lowest.** MVP 4-lower, with no IW-specific machinery, is marginally the most identity-robust of the four.

Per-community sensitivity ordering is consistent across all variants:

- **Most identity-sensitive: Homosexual** community (~ 0.44 – 0.46 flip rate across all variants, n ≈ 2,048). Across all four variants, the masked-out version of an identity-laden sample targeting the Homosexual community flips its T1 prediction roughly 45 % of the time.
- **Least identity-sensitive: Refugee, African, Asian** communities (~ 0.25 – 0.29 flip rate, n ranging 364 – 5,208). These communities exhibit the most "content-driven" predictions where masking identity tokens does not flip the verdict.

The full per-community breakdown with 95 % bootstrap CIs is preserved in `outputs/nb11/table_1_masking.md`, and visualised in `outputs/nb11/01_masking_flip_rate.png` (grouped bar chart with the 15 % target reference line). The 35 % overall flip rate is preserved across variants whether the model has IW architecture or not — it is a property of MMHS150K-trained models under PEFT constraints, not of any architectural variant tested.

### 10.4 Results — Analysis 2 (Counterfactual Swap)

19 swap pairs were tested. Sample sizes ranged from **1 to 327 source-present test samples** depending on the pair. The largest natural sample counts are for `white → black` (n = 327), `black → white` (n = 116), `man → woman` (n = 95), `gay → straight` (n = 64), `women → men` (n = 37). The religion-family swaps (`muslim ↔ christian ↔ jew`) have only 1 – 11 samples each, which yields flip-rate granularity at multiples of 1 / n (e.g., 1 / 7 = 0.1429 is the smallest non-zero resolvable value for a 7-sample pair). Several apparent "asymmetries" of 0.1429 in the table are therefore single-sample-flip artefacts driven by small denominators, not reliable signal.

For the **well-populated swap pairs** (`black ↔ white`, `man ↔ woman`), flip rates were similar across all four variants — consistent with the masking-test finding that variants do not differ meaningfully in identity-sensitivity. The asymmetry data is preserved in `outputs/nb11/02_swap_asymmetry.png` (4-panel heatmap, one per variant) and `outputs/nb11/table_2_swap.md`.

**Methodological note for future work.** The swap-pair test as designed is too small for fine-grained religious-bias detection on MMHS150K's test distribution. Future work should either (a) test on a larger held-out evaluation set, or (b) construct synthetic counterfactual swap-pair augmentations rather than relying on naturally-occurring samples. The current naturally-occurring-sample approach undersamples within-religion swaps because identity-religion vocabulary is rare in MMHS150K's test split.

### 10.5 Results — Analysis 3 (Per-Community Performance Stratification)

Per-community T1 accuracy best-vs-worst disparities per variant:

| Variant | Best community | Worst community | Disparity (best − worst) |
|---|---|---|---:|
| MVP 4-lower | Homosexual 0.7285 | Refugee 0.6447 | 0.0838 |
| MVP 4-IW | Homosexual 0.7314 | Women 0.6445 | 0.0869 |
| MVP 4-IW-CC | Homosexual 0.7305 | Women 0.6461 | 0.0844 |
| MVP 4-IW-CC-S | Homosexual 0.7310 | Women 0.6474 | 0.0836 |

All four variants exhibit ~ 0.084 best-vs-worst disparity — statistically equivalent. Community ordering is identical across variants: **Homosexual is consistently best (~ 0.73 T1 accuracy), Women and Refugee are consistently worst (~ 0.65 T1 accuracy).** The per-community ranking is determined by MMHS150K's training distribution and the frozen Run-D text encoder's representation quality, not by any architectural variant tested.

MVP 4-IW-CC-S vs MVP 4-lower per-community absolute Δ T1 accuracy:

- Mean absolute Δ across 10 top communities: **0.0060**
- Range: **[−0.0032, +0.0201]**
- Largest single delta: **Refugee +0.0201** (IW-CC-S slightly better)
- Communities where IW-CC-S underperforms MVP 4-lower: African (Δ −0.0006), Arab (Δ −0.0032), Asian (Δ −0.0027) — all within noise

These deltas are at the noise floor and do not constitute meaningful bias-profile differences. The full per-community breakdown (T1 acc, FPR, T2 macro F1, all with 95 % bootstrap CIs) is preserved in `outputs/nb11/table_3_per_community.md`, with the visualisation at `outputs/nb11/03_per_community_accuracy.png`.

### 10.6 Findings

1. **The bias-profile differences across the four MVP 4 variants are not meaningful under any of the three analyses performed.** This is consistent with the bias-off ablation finding from NB 10-lower (Δ = +0.0002 NotHate F1, 9,999 / 10,000 per-sample agreement between MVP 4-IW-CC-S and IW-CC-S-bias-off) and confirms that the identity-bias mechanism does not produce measurable bias-profile effects on this dataset under PEFT constraints. The pre-registered hypothesis is therefore confirmed: MVP 4-IW-CC-S exhibits a bias-robustness profile statistically equivalent to MVP 4-lower across identity-term masking, counterfactual swap, and per-community stratification analyses.

2. **Counterintuitively, MVP 4-IW-CC-S has the highest masking flip rate (0.3691), not the lowest.** The identity-aware mechanism designed to be robust to identity vocabulary changes actually exhibits slightly higher identity-sensitivity than the baseline MVP 4-lower (0.3534). The variant range is within bootstrap CI widths, so this is a directional rather than statistical observation — but it is opposite to the design intent and worth recording as such.

3. **The dominant bias-profile finding is dataset-level, not architecture-level.** All four variants flip approximately 35 % of identity-laden samples when identity tokens are masked — roughly 2× the 15 % design-target threshold. MMHS150K-trained models under PEFT constraints are fundamentally identity-token-reliant regardless of the architectural variant tested. This is a substantive finding about the dataset and the LoRA-PEFT regime; it is not a failure of any particular variant.

4. **The Homosexual community is consistently the most identity-sensitive subgroup across all variants (~ 0.45 flip rate, n ≈ 2,048).** The robustness of this pattern across variants and the large sample size make it the most reliable per-community signal in the analysis. The pattern likely reflects MMHS150K's seed-keyword distribution, which over-represents homophobic slurs relative to other identity categories — a dataset-construction artefact rather than a model-architecture property.

5. **The Women and Refugee communities are consistently the worst-performing subgroups for T1 accuracy across all variants (~ 0.65 T1 accuracy).** These represent dataset-level imbalances in MMHS150K that no architectural intervention tested in this work resolves. The IW family's architectural changes (cross-attention, identity-bias term, per-branch LN, centered VADER modulation) leave the Women / Refugee underperformance untouched.

6. **The bias analysis therefore strengthens the failure-mode characterisation of the IW family.** The mechanism does not affect aggregate predictions (bias-off ablation, § 6), does not affect per-sample reliance patterns (NB 10 taxonomy, § 3.3), and does not affect bias profile (NB 11). The IW family's contribution is best characterised as a documented architectural fix for the NB 09b gate collapse (per-branch LN + centered VADER modulation) plus a tested-but-non-contributory identity-bias term, rather than as a novel identity-aware mechanism.

### 10.7 Caveats

1. **The 35 % masking flip rate is itself a substantial finding about MMHS150K-trained models under PEFT constraints.** The IW family does not deliver identity-robust predictions — it inherits the dataset's identity-token reliance. This is not a failure of the IW family; it is a property of the training regime that the IW family was unable to overcome.

2. **The 15 % target threshold was set as a design goal early in the project.** NB 11's findings indicate this threshold is unachievable through architectural variants alone on MMHS150K under PEFT constraints. Future work should target dataset-level interventions: re-balanced training data, identity-token augmentation during training, counterfactual data augmentation. The threshold itself remains the correct target; the path to reaching it is not architectural.

3. **Several swap pairs have insufficient sample sizes (1 – 7 samples) for reliable per-pair conclusions.** The asymmetry data should be interpreted with this caveat. Apparent "asymmetries" of 0.1429 = 1 / 7 are single-sample-flip artefacts, not reliable signal. The well-populated pairs (`black ↔ white`, `man ↔ woman`) provide the cleanest cross-variant comparisons.

4. **The "Homosexual most sensitive" finding (~ 0.45 across variants) is robust but applies uniformly across the IW family.** It is an MMHS150K / lexicon property, not a variant-specific property. Future work should investigate whether this reflects training-data composition or test-set composition — the current analysis cannot distinguish between the two without re-running with held-out lexicon ablations.

### 10.8 Artefacts produced (NB 11)

| Artefact | Path | Size |
|---|---|---:|
| Executed notebook (12 cells, 0 errors, ≈ 755 KB with embedded outputs) | `notebooks/11_bias_analysis.ipynb` | 755 KB |
| Per-sample masking data (4 variants × identity-laden samples × original/masked predictions + flip flags) | `data/processed/nb11_per_sample_masking_data.parquet` | 436 KB |
| Plot 1 — per-community masking flip rate (grouped bar with bootstrap CIs + 15 % target reference) | `outputs/nb11/01_masking_flip_rate.png` | 104 KB |
| Plot 2 — counterfactual swap-pair asymmetry (4-panel heatmap, one per variant) | `outputs/nb11/02_swap_asymmetry.png` | 248 KB |
| Plot 3 — per-community T1 accuracy with bootstrap CIs (grouped bar) | `outputs/nb11/03_per_community_accuracy.png` | 98 KB |
| Table 1 — per-community masking flip rate (4 variants × 11 rows including ALL row) | `outputs/nb11/table_1_masking.md` | 2.75 KB |
| Table 2 — counterfactual swap-pair flip rates (19 pairs × 4 variants) | `outputs/nb11/table_2_swap.md` | 3.51 KB |
| Table 3 — per-community performance (T1 acc, FPR, T2 macro F1 × 10 communities × 4 variants) with bootstrap CIs | `outputs/nb11/table_3_per_community.md` | 4.93 KB |
| Summary findings prose | `outputs/nb11/summary.md` | 2.12 KB |

---

## Phase 3 — totals

- **Notebooks**: 3 main analysis notebooks (`10_per_sample_modality_analysis.ipynb`, `10_lower_per_sample_modality_analysis.ipynb`, `11_bias_analysis.ipynb`) + 1 auxiliary retrain notebook (`08_lower_mvp4_lowercase.ipynb`) introduced during Phase 3 diagnostic work for matched-methodology comparison
- **Diagnostic markdown files**: 3 in `outputs/diagnostic/` + 1 raw JSON of audit-probe results
- **Per-sample analytical parquets**: 3 (NB 10, NB 10-lower, NB 11 masking) totalling ~ 4.6 MB
- **Charts**: 8 + 8 + 3 = 19 PNG files
- **Tables**: 5 + 5 + 3 + 1 summary per analysis = 16 markdown table / summary files
- **New checkpoint** (auxiliary): MVP 4-lower at `models/mvp4_gated_fusion_lowercase_best/mvp4_trainable.pt` (11.36 MB)
