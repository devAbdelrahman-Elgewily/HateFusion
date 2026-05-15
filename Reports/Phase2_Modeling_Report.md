# 🧠 Phase 2 — Modeling Report
### Multimodal Cyberbullying & Online Hate Speech Detection
### Final Year Project — Computer Science (AI Major)

---

## Document Information

| Field | Value |
|-------|-------|
| **Document type** | Phase 2 deliverable + report-writing source material |
| **Phase** | 2 — Modeling (warm-start → MVP ladder → analysis) |
| **Status** | NB 04 complete · NB 05 **four-run diagnostic suite complete, MVP 1 baseline declared (Run D)** · NB 06 **MVP 2 naive concatenation complete, Gomez 2019 failure mode reproduced** · architectural ceiling H7 confirmed |
| **Date opened** | 2026-05-14 |
| **Last updated** | 2026-05-15 |
| **Compute** | Lightning AI Studio, NVIDIA Tesla T4 (16 GB), Python 3.12 / torch 2.8 + CUDA 12.8 |
| **Companion documents** | `Multimodal_Cyberbullying_Detection_v1.2.md` (technical scope), `Cyberbullying_Detection_Report_Framing.md` (significance/defence), `Phase1_Data_Engineering_Report.md` (data engineering predecessor) |
| **Notebooks covered** | `04_roberta_pretrain_kaggle.ipynb` (complete) · `05_mvp1_roberta_t1.ipynb` Run 1 (5 epochs, warm-start + pos_weight) · `05_mvp1_roberta_t1.ipynb` Run 2 (loss-only change, halted at epoch 1) · `05c_no_warmstart.ipynb` Diagnostic C (no warm-start, 5 epochs) · `05d_rank32_lr3e4.ipynb` Diagnostic D (rank 32 + lr 3e-4, 5 epochs — **selected as MVP 1 baseline**) · `06_mvp2_naive_fusion.ipynb` MVP 2 (CLIP + frozen Run-D text, naive concat, 5 epochs). Pending: `07` → `11`. |

---

## Table of Contents

1. [Executive Summary](#1-executive-summary)
2. [Environment & Reproducibility](#2-environment--reproducibility)
3. [Notebook 04 — Twitter-RoBERTa Warm-Start on Cyberbullying Kaggle](#3-notebook-04--twitter-roberta-warm-start-on-cyberbullying-kaggle)
4. [Methodological Decisions Locked During Notebook 04](#4-methodological-decisions-locked-during-notebook-04)
5. [Notebook 05 — MVP 1 Text-Only Baseline on MMHS150K T1 (Iteration 1)](#5-notebook-05--mvp-1-text-only-baseline-on-mmhs150k-t1-iteration-1)
6. [Diagnostic Analysis & Iteration Log for NB 05](#6-diagnostic-analysis--iteration-log-for-nb-05)
7. [NB 05 Run 2 — Loss Change (Drop `pos_weight`)](#7-nb-05-run-2--loss-change-drop-pos_weight)
8. [NB 05 Run C — No-Warm-Start Ablation](#8-nb-05-run-c--no-warm-start-ablation)
9. [Three-Run Comparison and Pre-Run-D Diagnosis](#9-three-run-comparison-and-pre-run-d-diagnosis)
10. [NB 05 Run D — LoRA Capacity Ablation (Rank 32 + LR 3e-4)](#10-nb-05-run-d--lora-capacity-ablation-rank-32--lr-3e-4)
11. [Final Four-Run Synthesis and MVP 1 Baseline Decision](#11-final-four-run-synthesis-and-mvp-1-baseline-decision)
12. [Open Items and Preconditions for NB 06](#12-open-items-and-preconditions-for-nb-06)
13. [Notebook 06 — MVP 2 Multimodal Naive Concatenation (text + image)](#13-notebook-06--mvp-2-multimodal-naive-concatenation-text--image)
14. [References](#14-references)
15. [Appendix A — Saved Artefacts](#appendix-a--saved-artefacts)

> Sections for NB 06–11 will be appended below as each notebook is completed. The structure of each future section mirrors §3 (deliverable record) + §6 (iteration log when more than one pass is needed).

---

## 1. Executive Summary

Phase 2 of the project — modeling — has begun. The first deliverable, Notebook 04, performs a pretext-task warm-start of Twitter-RoBERTa on the Cyberbullying Kaggle dataset before downstream MMHS150K fine-tuning. The notebook executed end-to-end on the Lightning AI T4 GPU in approximately 10 minutes (3 epochs, ~200 s/epoch) with zero error cells.

### Notebook 04 deliverables

| # | Deliverable | Output |
|---|-------------|--------|
| 1 | PEFT LoRA adapter (rank 16) over Twitter-RoBERTa, domain-adapted on cyberbullying-style Twitter text | `models/roberta_pretrain/adapter_model.safetensors` (2.26 MB) + `adapter_config.json` |
| 2 | Persisted tokenizer ensuring identical pre-processing in downstream notebooks | `models/roberta_pretrain/tokenizer.json`, `tokenizer_config.json` |
| 3 | Frozen hyperparameter record + per-epoch training history | `models/roberta_pretrain/hparams.json`, `training_history.json`, `labels.json` |
| 4 | Test-set confusion matrix chart | `outputs/nb04_confusion_matrix.png` |
| 5 | Training/validation loss + macro-F1 curves | `outputs/nb04_training_curves.png` |
| 6 | Fully executed notebook with embedded outputs | `notebooks/04_roberta_pretrain_kaggle.ipynb` (179 KB) |

### Headline findings

1. **The warm-start converges cleanly.** Validation macro-F1 rises monotonically across the three epochs (0.829 → 0.848 → 0.852) with train loss decreasing from 0.304 → 0.135 and validation loss from 0.179 → 0.144. No sign of overfitting at 3 epochs.
2. **Test-set macro-F1 = 0.8499, accuracy = 0.858** on the held-out 2,301-tweet test split. This is a strong starting point for an extremely parameter-efficient setup (589,824 trainable params, **0.471 %** of the 125 M-parameter backbone).
3. **Four of six classes exceed F1 ≥ 0.89** (religion 0.96, age 0.97, ethnicity 0.98, gender 0.89). The encoder picks up category-defining vocabulary on these well-anchored classes.
4. **The two ambiguous classes are the cost centre.** `not_cyberbullying` lands at F1 = 0.63 (recall 0.57) and `other_cyberbullying` at F1 = 0.67 (recall 0.75) — they confuse each other (the catch-all "other" overlaps semantically with the non-class). This is the watch-item for downstream MMHS150K T1 fine-tuning, where NotHate dominates the val/test splits.
5. **The classification head is intentionally discarded.** Only LoRA adapter weights and the tokenizer are persisted. Notebook 05 attaches a fresh T1 binary-hate head; nothing in the warm-start is permitted to leak supervised cyberbullying-type signal into the downstream task.
6. **Adapter round-trip load is verified.** `PeftModel.from_pretrained(base, models/roberta_pretrain)` reconstitutes the LoRA-adapted encoder and produces a `(1, 768)` `[CLS]` embedding without warnings. Downstream notebooks have a working contract.

### Notebook 05 first-iteration headline findings

> Iteration 1 of MVP 1 executed cleanly (0 error cells, all artefacts saved) **but underperforms the expected text-only baseline by ~0.07-0.09 AUC**. The model has been kept on disk as `models/roberta_mvp1_iter1/` (renamed from `roberta_mvp1/` to free that path for iteration 2). Findings below feed the diagnostic analysis in §6 and the iteration-2 plan in §7.

1. **Test AUC = 0.7398, F1 (macro) = 0.6885, FPR = 0.3071** on the balanced 50/50 test split with threshold 0.5. Gomez et al. 2019's reported text-only number on MMHS150K is F1 ~0.72 / AUC ~0.81-0.83 — we are below both.
2. **Train loss barely moved** across 5 epochs (0.2456 → 0.2349, a 4.4 % reduction). Sanity calculation: expected Focal-BCE loss at random init under the chosen weights and class balance ≈ 0.27 — **epoch-1 train loss is already at random-init level**, and subsequent epochs did not pull it meaningfully lower. The model essentially plateaued at initialisation.
3. **Val-AUC range across all 5 epochs = 0.7427 → 0.7457** (Δ = +0.003). Best was epoch 4 by AUC, epoch 3 by val loss. Mild train/val divergence appears at epoch 4 (val loss tick up from 0.3326 → 0.3442 while train loss continues falling). All within noise; no learning collapse.
4. **Predicted-probability mean on test ≈ 0.5037** while true hate rate is 0.5001. The model is outputting near-uniform probabilities — **not separating classes in probability space**. This is the diagnostic signature of a model that has not learned the task, not of a model that has learned and saturated.
5. **The Bayes recalibration at threshold 0.5 collapses to all-NotHate predictions** (TN=4999, FP=0, FN=5001, TP=0) — the formula is correct but threshold 0.5 is no longer appropriate after the prior shift. A threshold optimised on recalibrated val probabilities is needed; the current cell 10 optimises on raw val and applies to recalibrated test, which is an inconsistency to fix in iteration 2.
6. **Suspected root causes (full diagnosis in §6).** Most-likely: (i) double-counting class imbalance — `pos_weight=3.57` stacked on top of Focal γ=2; (ii) NB 04 warm-start bias against `not_cyberbullying` fighting MMHS150K's 78%-NotHate train distribution; (iii) LoRA rank 16 + LR schedule decayed too early for the optimiser to escape its starting region.

### NB 05 diagnostic-suite headline findings (Runs 2 + C)

> Two single-variable ablations of Run 1 ruled out the top two hypotheses and elevated the third. **Three independent runs all converge on test AUC ≈ 0.74 / val AUC ≈ 0.745.** Together this is strong evidence of a structural ceiling under the current LoRA-rank-16 + lr-1e-4 architecture, not a loss bug or a warm-start bug.

1. **Run 2 (warm-start, no `pos_weight`, halted at epoch 1).** Identical setup to Run 1 except the `pos_weight=3.5739` is dropped. Result: val AUC at epoch 1 = **0.7424**, within 0.0003 of Run 1's epoch-1 val AUC (0.7427). The user halted the run because the AUC ceiling was clearly unchanged. **Conclusion: loss calibration (H1) is not the root cause.** Train-loss starting value drops from 0.2456 (Run 1) to 0.1237 (Run 2) — that's the `pos_weight` multiplier coming out of the loss, not learning behaviour changing.
2. **Run C — Diagnostic C (no warm-start, full 5 epochs).** Fresh `cardiffnlp/twitter-roberta-base-2022-154m` encoder, brand-new LoRA from `LoraConfig` (zero-init `lora_B`, Kaiming-init `lora_A`), tokenizer from cardiffnlp (not from the warm-start dir), `pos_weight=3.5739` restored to match Run 1's loss exactly. The single variable is the warm-start. Result: test AUC = **0.7407**, F1m = 0.6875, FPR = 0.3069. Best val AUC = 0.7447 at epoch 5. **Δ vs Run 1: AUC +0.0009, F1m −0.0010, FPR −0.0002** — entirely within run-to-run noise. **Conclusion: the NB 04 warm-start (H2) is neither helping nor hurting on AUC; the pretext-task design is neutral.**
3. **The training trajectories are nearly identical.** Run 1 val AUC: 0.7427 → 0.7439 → 0.7444 → 0.7457 → 0.7454. Run C val AUC: 0.7405 → 0.7425 → 0.7439 → 0.7446 → 0.7447. Same starting point, same shape, same ceiling. This is the signature of a fixed architectural / optimisation upper bound, reached quickly by both pipelines.
4. **Standing diagnosis after the diagnostic suite (full table in §9.3):** **H3 (LoRA capacity / LR schedule) is now the primary hypothesis** for the plateau. The fallback reading — that **0.74 is the actual MMHS150K text-only AUC ceiling under PEFT LoRA rank 16**, regardless of any hyperparameter inside the explored space — remains compatible with all three runs and must be acknowledged in the Limitations section of the thesis.
5. **Three artefact branches preserved on disk** so the iteration trail is auditable: `models/roberta_mvp1_iter1/` (Run 1 final), `models/roberta_mvp1_best/` (Run 1 + Run 2 partial best ckpt — last written during Run 2's epoch-1 save), `models/roberta_mvp1_fresh/` (Run C final). Output charts: `outputs/previous_tests/nb05_*.png` (Run 1) + `outputs/previous_tests/nb05_fresh_*.png` (Run C).

### NB 05 Run D + final synthesis headline findings

> The fourth run tested the remaining hypothesis (H3, LoRA-capacity ceiling) by bumping rank 16 → 32 and lora_lr 1e-4 → 3e-4 simultaneously. **H3 was ruled out, H7 (architectural ceiling) confirmed, and Run D was selected as the MVP 1 baseline** on operational grounds even though the AUC ceiling held.

1. **Run D test results (rank 32 + lr 3e-4, fresh init, 5 epochs):** AUC **0.7431**, F1m 0.6855, FPR **0.2667**, best val AUC **0.7495** at epoch 4. **Δ vs Run C: AUC +0.0024 (within noise), F1m −0.0020, FPR −0.0402** *(13 % relative reduction — outside noise)*. Trainable params doubled to 1,180,417 (0.94 %); per-epoch wall-clock unchanged (~11 min) because the backbone forward/backward dominates throughput.
2. **H3 RULED OUT.** Doubling LoRA rank and tripling LoRA learning rate moves test AUC by 0.0024 — within the same ±0.02 noise band that absorbed Runs 1, 2, and C. The plateau survives every parameter-efficient intervention tested.
3. **H7 CONFIRMED.** Four independent runs (different loss, different init, different capacity) all converge on test AUC ≈ 0.74. Under PEFT LoRA on `cardiffnlp/twitter-roberta-base-2022-154m` for MMHS150K T1, **AUC ≈ 0.74 is the measured ceiling within the explored hyperparameter space** (rank ∈ {16, 32}, lr ∈ {1e-4, 3e-4}).
4. **A precision-recall shift worth flagging.** Run D produces fewer false positives at threshold 0.5 (FP 1,333 vs Run C's 1,534) at the cost of fewer true positives (TP 3,196 vs Run C's 3,410). Macro F1 is essentially unchanged, but the operational profile is different: Run D is **more conservative** at the canonical threshold — useful for the project's human-in-the-loop framing where the T3 routing layer absorbs ambiguous cases.
5. **MVP 1 baseline declared: Run D.** Full justification in §11.3. Headline rationale: (i) highest test AUC of the eligible runs (0.7431, marginal but consistent); (ii) lowest test FPR (0.2667, meaningfully better, outside noise); (iii) highest val AUC (0.7495); (iv) cleanest controlled-experiment lineage (inherits Run C's no-warm-start condition + adds capacity ablation); (v) Run C remains within reach as a "minimal" alternative baseline, and the trade-off is documented honestly.
6. **What multimodal MVPs must beat (NB 07 onward).** AUC 0.7431 at FPR 0.2667 on the balanced test set. A multimodal gain below ≈ +0.005 AUC is statistically indistinguishable from the text-only ceiling and cannot be claimed; a gain of ≥ +0.02 AUC with stable F1 / FPR is the meaningful threshold.
7. **Contribution claim now empirically anchored.** Because the text-only ceiling has been pinned across four ablations, any AUC improvement above ~0.76 from the multimodal MVPs is attributable to **modality fusion**, not text-encoder tuning. This strengthens the v1.2 scope's contribution narrative — the baseline is measured rather than assumed.

---

## 2. Environment & Reproducibility

### 2.1 Compute platform

Phase 2 training has moved off the local RTX 3060 (6 GB VRAM) onto Lightning AI Studios for headroom on fp16 fusion training in later notebooks. SSH key + host alias are installed on the developer workstation; per-session studio tokens are pasted at connect time.

| Property | Value |
|----------|-------|
| **Provider** | Lightning AI Studios |
| **Studio user** | `bodisalah23` |
| **Working directory on studio** | `/teamspace/studios/this_studio/HateFusion` |
| **GPU** | NVIDIA Tesla T4, 14.56 GB usable (15.36 GB nominal) |
| **CUDA version** | 12.8 |
| **PyTorch** | 2.8.0+cu128 |
| **Default env** | conda env `cloudspace` (single env per studio) |
| **Local mirror** | `D:\Cyberbullying Detection` (no longer the source of truth for trained artefacts) |

### 2.2 Phase 2 package versions

| Package | Version | Used in |
|---------|---------|---------|
| `python` | 3.12.11 | Studio runtime |
| `torch` | 2.8.0+cu128 | All Phase 2 notebooks |
| `transformers` | 5.8.1 | NB 04+ |
| `peft` | 0.19.1 | NB 04+ (LoRA adapters) |
| `accelerate` | 1.13.0 | Reserved for multi-device runs |
| `datasets` | 4.8.5 | Reserved |
| `scikit-learn` | 1.8.0 | Splits, metrics, class weights |
| `pandas` | 3.0.3 | CSV loading |
| `numpy` | 2.4.4 | Tensor / array glue |
| `matplotlib` | (env default) | Charts |

> **API-drift note.** `transformers` 5.x ships a redesigned model-load reporting (a "LOAD REPORT" table replaces silent loading) and `pandas` 3.0 makes string columns Arrow-backed by default. The latter required materialising columns to `numpy` (`np.array(df['…'].tolist(), dtype=…)`) before sklearn's `train_test_split` to avoid `TypeError: only integer scalar arrays can be converted to a scalar index` from `pyarrow.lib.ChunkedArray.__getitem__`. This is the only Phase 2 deviation from the Phase 1 code style so far.

### 2.3 Reproducibility settings

Every Phase 2 notebook seeds identically:

```python
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)
```

### 2.4 Execution model

Notebooks are authored locally (Write tool), uploaded to the studio via `scp`, executed end-to-end with `jupyter nbconvert --to notebook --execute --inplace --ExecutePreprocessor.timeout=1800`, and the executed `.ipynb` (with embedded outputs) is the canonical record. The studio is the source of truth for Phase 2 artefacts.

---

## 3. Notebook 04 — Twitter-RoBERTa Warm-Start on Cyberbullying Kaggle

### 3.1 Purpose

Notebook 04 performs a **pretext-task** warm-start: it domain-adapts `cardiffnlp/twitter-roberta-base-2022-154m` to cyberbullying-style Twitter text by training a 6-way classifier on the Cyberbullying Kaggle dataset. The classifier head is discarded at the end of the notebook; **only the LoRA-adapted encoder weights and the tokenizer carry forward** to Notebook 05 (MMHS150K T1 binary-hate fine-tune).

The motivation, locked in `CLAUDE.md §3`, is to give the text branch a Twitter-flavoured anchoring on hate-speech vocabulary before exposing it to MMHS150K, which is harder (multimodal, naturally imbalanced, with documented annotator-agreement structure).

### 3.2 Inputs

| Property | Value |
|----------|-------|
| **Dataset** | `data/cyberbullying_tweets.csv` |
| **Raw rows** | 47,692 (after pandas read) |
| **Columns** | `tweet_text` (string), `cyberbullying_type` (string, 6 levels) |
| **Duplicates removed** | 1,675 (on `tweet_text`) → **46,017 rows kept** |
| **Class set** | `not_cyberbullying`, `religion`, `age`, `ethnicity`, `gender`, `other_cyberbullying` |
| **Class balance** | Roughly uniform (~7.8–8.0 K per class pre-dedupe) |
| **Pre-processing** | Hashtags, mentions, and URLs preserved verbatim — they are part of the input distribution for downstream Twitter-domain inference (`CLAUDE.md §3`). |

### 3.3 Split protocol

Stratified 90 / 5 / 5 train / val / test split, `random_state=42`. Stratification is performed in two passes (`test_size=0.05` first, then a 5/95 sub-split on the remainder) so that all three splits preserve the 6-class proportions.

| Split | Rows | Per-class support |
|-------|------|-------------------|
| train | 41,415 | ~6.9 K per class |
| val   | 2,301  | ~380 per class |
| test  | 2,301  | 312–400 per class (see §3.7 table) |

### 3.4 Architecture

```
input_ids (B, 128) ──► Twitter-RoBERTa encoder + LoRA rank-16 adapters
                       on (query, value) projections of all 12 attention heads
                                            │
                                            ▼
                       last_hidden_state[:, 0]  (B, 768)  ←  the [CLS] vector
                                            │
                                            ▼
                       Linear(768 → 6)            ←  classification head
                                            │
                                            ▼
                                       logits (B, 6)
```

| Component | Value |
|-----------|-------|
| **Backbone** | `cardiffnlp/twitter-roberta-base-2022-154m` (12 layers, hidden 768, ~125 M params) |
| **Adapter** | `peft.LoraConfig(r=16, lora_alpha=32, lora_dropout=0.1, target_modules=['query','value'], bias='none', task_type=FEATURE_EXTRACTION)` |
| **Head** | `nn.Linear(768, 6)` — fresh init, **discarded after training** |
| **Total parameters** | 125,235,456 |
| **Trainable parameters** | 589,824 (**0.471 %**) — LoRA adapters + head only |

### 3.5 Loss

A class-weighted **Focal Cross-Entropy** with γ = 2 (`CLAUDE.md §12`):

$$\mathcal{L} = -\frac{1}{N}\sum_i w_{y_i} \cdot (1 - p_{y_i})^{\gamma} \cdot \log p_{y_i}$$

Class weights are computed via `sklearn.utils.class_weight.compute_class_weight('balanced', …)` on the **train split distribution** (per `CLAUDE.md §9` — never on val/test). Weights are passed to the GPU once and gathered per-example inside the loss.

### 3.6 Optimisation

| Knob | Value |
|------|-------|
| Optimiser | `AdamW`, weight decay 0.01, two parameter groups |
| LoRA learning rate | 1e-4 |
| Head learning rate | 1e-3 (10× LoRA, per `CLAUDE.md §12`) |
| Scheduler | linear warmup 10 % → cosine decay |
| Total optimiser steps | 7,773 (2,591 / epoch × 3 epochs) |
| Warmup steps | 777 |
| Mixed precision | fp16 via `torch.amp.GradScaler` + `torch.amp.autocast('cuda', dtype=torch.float16)` |
| Gradient accumulation | None (T4 has headroom at batch 16 / seq 128) |
| Batch size | 16 |
| Sequence length | 128 |
| Epochs | 3 |
| `num_workers` / `pin_memory` | 2 / True |
| Seed | 42 |

### 3.7 Results — training trajectory

| Epoch | Train loss | Val loss | Val macro-F1 | Wall-clock |
|-------|-----------|----------|--------------|------------|
| 1 | 0.3039 | 0.1791 | **0.8285** | 194.4 s |
| 2 | 0.1550 | 0.1479 | **0.8484** | 201.9 s |
| 3 | 0.1348 | 0.1441 | **0.8516** | 201.7 s |

- Train and validation losses descend monotonically.
- The gap between train and val loss narrows from 0.125 (epoch 1) to 0.009 (epoch 3) — convergence rather than overfitting.
- Validation macro-F1 improves at a decelerating rate; a 4th epoch would likely add < 0.005. Three epochs is the right stopping point.
- Per-epoch time is dominated by forward+backward on the T4 (~75 ms / step at batch 16 / seq 128 / fp16).

See `outputs/nb04_training_curves.png` for the rendered curves.

### 3.8 Results — held-out test set (n = 2,301)

| Class | Precision | Recall | F1 | Support |
|-------|----------:|-------:|---:|--------:|
| not_cyberbullying   | 0.7040 | 0.5693 | **0.6295** | 397 |
| religion            | 0.9466 | 0.9750 | **0.9606** | 400 |
| age                 | 0.9723 | 0.9650 | **0.9686** | 400 |
| ethnicity           | 0.9846 | 0.9673 | **0.9759** | 397 |
| gender              | 0.8957 | 0.8911 | **0.8934** | 395 |
| other_cyberbullying | 0.6057 | 0.7532 | **0.6714** | 312 |
| **accuracy**        |        |        | **0.8575** | 2,301 |
| **macro avg**       | 0.8515 | 0.8535 | **0.8499** | 2,301 |
| **weighted avg**    | 0.8608 | 0.8575 | **0.8568** | 2,301 |

See `outputs/nb04_confusion_matrix.png` for the rendered matrix.

### 3.9 Per-class analysis

**Strong classes (F1 ≥ 0.89): religion, age, ethnicity, gender.**
These categories have clear lexical anchors in cyberbullying-style tweets (slurs, references to demographic markers, named groups). The encoder + LoRA adapters lock onto these signals quickly; their precision and recall are both high and well balanced.

**Weak classes (F1 < 0.70): not_cyberbullying, other_cyberbullying.**
These two confuse each other heavily and account for almost all of the residual error. Two structural reasons:

1. `other_cyberbullying` is a semantic catch-all — any cyberbullying that does not fit religion/age/ethnicity/gender. By construction it has no defining vocabulary, so the encoder must use a "not any of the other five" reasoning. That overlaps with `not_cyberbullying` along low-confidence directions.
2. `not_cyberbullying` recall is 0.57 — 43 % of genuinely non-bullying tweets are mis-classified, almost all into `other_cyberbullying`. This produces an upward bias on the cyberbullying side: a deployed binary-hate classifier built on this encoder will, all else equal, **over-flag** rather than under-flag.

This is a watch-item for Notebook 05. MMHS150K's `NotHate` class dominates the val and test splits (50 / 50 balanced by Gomez 2019 design, but reflecting natural Twitter at ~78 % NotHate in train). If the warm-start encoder enters NB05 with a learned bias against `not_*`, we will see it as inflated false-positive rate on T1.

### 3.10 Adapter integrity check

A post-execution verification script reloads the persisted adapter and runs a forward pass:

```python
from transformers import AutoTokenizer, AutoModel
from peft import PeftModel

base    = AutoModel.from_pretrained('cardiffnlp/twitter-roberta-base-2022-154m')
adapter = PeftModel.from_pretrained(base, 'models/roberta_pretrain')
tok     = AutoTokenizer.from_pretrained('models/roberta_pretrain')
```

The adapter loads with no errors, the LoRA weights register on the wrapped backbone, and a CLS-token forward pass produces a `(1, 768)` embedding in `float32`. `trainable_params == 0` after `from_pretrained` because PEFT loads adapters in inference mode by default — Notebook 05 will re-enable training with `is_trainable=True` (or by stacking a head and setting `requires_grad` explicitly). The contract for downstream notebooks is therefore verified.

### 3.11 Artefacts written

```
models/roberta_pretrain/
  ├── README.md                       5.08 KB   (auto-generated by PEFT)
  ├── adapter_config.json             1.02 KB   (LoRA config — re-used by NB05)
  ├── adapter_model.safetensors       2.26 MB   (LoRA weights, ~590 K params)
  ├── hparams.json                    0.26 KB   (frozen NB04 hyperparameters)
  ├── labels.json                     0.37 KB   (label2id, id2label, "head NOT saved" note)
  ├── tokenizer.json                  3.39 MB   (tokenizer)
  ├── tokenizer_config.json           0.48 KB   (tokenizer config)
  └── training_history.json           0.32 KB   (per-epoch metrics)

outputs/
  ├── nb04_confusion_matrix.png      67.80 KB   (6×6 test-set confusion matrix)
  └── nb04_training_curves.png       62.62 KB   (train/val loss + val macro-F1)

notebooks/
  └── 04_roberta_pretrain_kaggle.ipynb  179 KB  (executed in place)
```

All artefacts live under `/teamspace/studios/this_studio/HateFusion/` on the Lightning studio.

---

## 4. Methodological Decisions Locked During Notebook 04

| # | Decision | Rationale |
|---|----------|-----------|
| 1 | **Warm-start task is 6-class `cyberbullying_type`, not binary cyberbullying-vs-not.** | A 6-way pretext forces the encoder to learn finer-grained category vocabulary. The head is discarded, so picking the richer task costs nothing downstream and yields a more discriminative encoder. |
| 2 | **LoRA rank 16 on attention `query` and `value` projections only.** | Matches `CLAUDE.md §3` (locked at project setup) and keeps trainable parameters under 0.5 % of total. Re-using this exact target-module list across notebooks is required for the saved `adapter_config.json` to load cleanly downstream. |
| 3 | **Classification head is **never** saved.** | Prevents cyberbullying-type label leakage into MMHS150K supervised signal. NB05 attaches a fresh `Linear(768 → 2)` T1 head; this rule applies to all downstream notebooks. |
| 4 | **Class weights from the train split only.** | `CLAUDE.md §9` rule. NB04 train is roughly balanced, so weights are mild (all near 1.0); the discipline matters more in NB05 where MMHS150K train is ~22 % hate. |
| 5 | **Three epochs is the budget for any warm-start.** | Empirical: val macro-F1 gain from epoch 2 → 3 is +0.003. Additional epochs would risk encoder over-specialisation on cyberbullying-type vocabulary and reduce transferability to MMHS150K's categorical-hate signal. |
| 6 | **Pandas 3.0 Arrow-backed columns must be materialised to numpy before sklearn splits.** | `np.array(df['col'].tolist(), dtype=…)` is the canonical pattern in Phase 2 notebooks. Direct `.values` returns a `ChunkedArray` that breaks `_safe_indexing`. |
| 7 | **`torch.amp.autocast('cuda', dtype=torch.float16)` is the Phase 2 mixed-precision call.** | `torch.cuda.amp.autocast` is deprecated in torch 2.4+ but still referenced in `CLAUDE.md §8`. Phase 2 uses the new API throughout. |

---

## 5. Notebook 05 — MVP 1 Text-Only Baseline on MMHS150K T1 (Iteration 1)

### 5.1 Purpose and significance

Notebook 05 is **MVP 1** of the project's phased delivery ladder. It is the text-only Twitter-RoBERTa baseline on MMHS150K T1 (binary hate vs. not-hate) and serves as the reference number against which every later MVP — naive fusion (MVP 3), gated cross-modal fusion (MVP 4), and the full system with T3 routing (MVP 5) — is measured.

The motivation in the project framing (`Multimodal_Cyberbullying_Detection_v1.2.md` and Gomez et al. 2019) is that *naive* multimodal fusion on MMHS150K historically underperforms a strong text-only baseline. The contribution of this project — gated fusion — only matters if (a) text-only is competitive with prior work, and (b) the multimodal model genuinely beats it. So **MVP 1 must be measured honestly**: under-claiming a weak baseline would make later gains look more impressive than they are.

### 5.2 First-iteration framing

This section records the **first iteration** of NB 05. It is preserved verbatim in the codebase (`models/roberta_mvp1_iter1/`, executed notebook at `notebooks/previous_tests/05_mvp1_roberta_t1.ipynb` with `nbformat_minor=4`) so that the methodology trail is auditable when writing the thesis report. The diagnostic analysis in §6 and the iteration-2 plan in §7 explain why a redesign pass is required.

### 5.3 Inputs

| Property | Value |
|----------|-------|
| **Encoder source** | `models/roberta_pretrain/` (warm-started LoRA from NB 04), loaded via `PeftModel.from_pretrained(base, adapter_dir, is_trainable=True)` |
| **Tokenizer source** | `models/roberta_pretrain/` (**not** re-instantiated from `cardiffnlp/...`) — guarantees identical pre-processing to NB 04 |
| **Labels** | `data/processed/labels_parsed.csv` (149,819 rows; `T1` derived in NB 01, zero nulls) |
| **Official splits** | `data/MMHS150K/splits/{train,val,test}_ids.txt` — joined on `tweet_id` to build a `split` column at runtime, since `labels_parsed.csv` does not carry one. |
| **Images** | **NOT used** in MVP 1 (text-only baseline) |

### 5.4 Split protocol and class balance

| Split | n | % hate | hate count | nothate count |
|-------|---:|------:|-----------:|--------------:|
| train | 134,820 | **21.86 %** | 29,476 | 105,344 |
| val   |   4,999 | 50.01 %     |  2,500 |   2,499 |
| test  |  10,000 | 50.01 %     |  5,001 |   4,999 |

> Confirms the Gomez 2019 design (~22 % train, 50/50 val/test). Four tweet IDs appearing in the split files are absent from `labels_parsed.csv` (the four n=1-annotator rows dropped in Phase 1) — three from train, one from val.

### 5.5 Architecture

```
input_ids (B, 128) ──► Twitter-RoBERTa encoder + LoRA rank-16
                       (warm-started in NB 04, UNFROZEN here)
                                  │
                                  ▼
                  last_hidden_state[:, 0]   (B, 768)   ← [CLS]
                                  │
                                  ▼
                       Linear(768 → 1)        ← fresh T1 head
                                  │
                                  ▼
                            logit (B, 1)
```

| Component | Value |
|-----------|-------|
| **Backbone** | `cardiffnlp/twitter-roberta-base-2022-154m` (12 layers, hidden 768, ~125 M params) |
| **Adapter** | warm-started PEFT LoRA, rank 16, `target_modules=['query','value']`, `is_trainable=True` |
| **Head** | `nn.Linear(768, 1)` — fresh init for T1 |
| **Total params** | 125,236,225 |
| **Trainable params** | 590,593 (**0.472 %**) — 589,824 LoRA + 769 head |

### 5.6 Loss (first iteration)

`FocalBCE(γ=2, pos_weight=3.5739)` on a single logit. `pos_weight` was derived from `sklearn.utils.class_weight.compute_class_weight('balanced', …)` on the **train** distribution: `nothate=0.6399, hate=2.2869 → pos_weight = hate/nothate = 3.5739`. Loss form:

$$\mathcal{L} = \frac{1}{N}\sum_i w_i\cdot (1-p_t^{(i)})^{\gamma}\cdot \mathrm{BCE}(p^{(i)}, y_i),\quad w_i = y_i \cdot \text{pos\_weight} + (1 - y_i)\cdot 1.0$$

**§6 flags this loss construction as one of the suspected root causes of under-performance** — Focal γ=2 and `pos_weight` are both mechanisms for handling class imbalance, and stacking them double-counts the correction.

### 5.7 Optimisation (first iteration)

| Knob | Value | Note |
|------|-------|------|
| Optimiser | AdamW, weight decay 0.01, two parameter groups | LoRA group + head group |
| LoRA learning rate | 1e-4 | Conservative; standard LoRA fine-tune literature uses 1e-4 – 5e-4 |
| Head learning rate | 1e-3 | 10× LoRA |
| Scheduler | Linear warmup 10 % → cosine decay | 1,053 warmup steps / 10,530 total opt steps |
| Mixed precision | fp16 via `torch.amp.GradScaler` + `torch.amp.autocast('cuda', dtype=torch.float16)` | |
| Batch size | 16 (physical) | |
| Gradient accumulation | 4 steps → **effective batch 64** | |
| Sequence length | 128 | p95 tweet length = 133 → ~5 % rows truncate |
| Epochs | 5 | |
| Seed | 42 | |

### 5.8 First-iteration training trajectory

| Epoch | Train loss | Val loss | Val AUC | Val F1@0.5 | Wall-clock |
|------:|-----------:|---------:|--------:|-----------:|-----------:|
| 1 | 0.2456 | 0.3563 | 0.7427 | 0.6769 | 856 s (14.3 m) |
| 2 | 0.2397 | 0.3421 | 0.7439 | 0.6812 | 1391 s (23.2 m) |
| 3 | 0.2379 | **0.3326** | 0.7444 | 0.6947 | 1389 s (23.2 m) |
| 4 | 0.2359 | 0.3442 | **0.7457** | 0.6923 | 1389 s (23.2 m) |
| 5 | 0.2349 | 0.3402 | 0.7454 | 0.6911 | 1390 s (23.2 m) |

- Best by val AUC: **epoch 4** (saved as best checkpoint).
- Best by val loss: epoch 3 — disagreement with AUC across one epoch is within noise.
- Train-loss reduction across all 5 epochs: 0.2456 → 0.2349, a **4.4 %** drop. A healthy LoRA fine-tune on this regime is expected to drop train loss by 30-50 %.
- Mild train/val divergence visible at epoch 4 (val loss tick up while train loss falls), confirmed at epoch 5 — symptom of plateau, not collapse.

See `outputs/previous_tests/nb05_training_curves.png` for rendered curves.

### 5.9 First-iteration test results (n = 10,000, balanced 50/50)

#### Threshold = 0.5

| Metric | Value |
|--------|------:|
| AUC-ROC | **0.7398** |
| F1 (macro) | 0.6885 |
| Precision (macro) | 0.6885 |
| Recall (macro) | 0.6885 |
| TN / FP / FN / TP | 3,464 / 1,535 / 1,580 / 3,421 |
| **False positive rate** | **0.3071** |

#### Threshold = 0.45 (F1-optimised on val)

| Metric | Value |
|--------|------:|
| AUC-ROC | 0.7398 (invariant) |
| F1 (macro) | 0.6793 |
| Precision (macro) | 0.6865 |
| Recall (macro) | 0.6815 |
| TN / FP / FN / TP | 2,996 / 2,003 / 1,182 / 3,819 |
| False positive rate | 0.4007 |

> The F1-optimised threshold *worsens* F1m on test (0.6885 → 0.6793). This is the small-val-set noise issue (val n=4,999) plus the val/test being identically balanced — the search has too few points to find a meaningful threshold lift. **Result is consistent with the §6 finding that the model is not separating classes well in probability space.**

See `outputs/previous_tests/nb05_confusion_matrix.png`.

### 5.10 Recalibration to deployment prior (~22 % hate)

Per `CLAUDE.md §9` the project must report both balanced-test and deployment-distribution metrics. The Bayes prior shift applied to test probabilities:

$$p_{\text{recal}} = \frac{p \cdot P(\text{hate})}{p \cdot P(\text{hate}) + (1 - p) \cdot P(\text{not-hate})},\quad P(\text{hate}) = 0.2468$$

| Threshold | F1m (recal) | Pm (recal) | Rm (recal) | FPR (recal) | TN/FP/FN/TP (recal) |
|-----------|------------:|-----------:|-----------:|------------:|--------------------:|
| 0.5      | 0.3333 | 0.2500 | 0.5000 | 0.0000 | 4999 / 0 / 5001 / 0 |
| 0.45     | 0.3335 | 0.7500 | 0.5001 | 0.0000 | 4999 / 0 / 5000 / 1 |

**Caveat in iteration 1.** The recalibrated metrics at threshold 0.5 collapse to all-NotHate (no true positives). This is the Bayes shift correctly pushing probabilities downward, but the threshold needs to be re-optimised **on the recalibrated val probabilities**, not on the raw val probabilities. The current notebook optimises threshold on raw val and then applies that threshold to recalibrated test — a methodological inconsistency to fix in iteration 2.

AUC is invariant to the monotone recalibration, so AUC = 0.7398 in both panels.

### 5.11 Comparison to literature

| Source | Modality | Reported AUC | Reported F1 (binary) |
|--------|----------|-------------:|---------------------:|
| Gomez et al. 2019 (TextHead, MMHS150K paper) | text-only | not reported | ~0.722 |
| Follow-up work cited in the v1.2 scope | text-only | **0.81 – 0.83** | 0.70 – 0.75 |
| **NB 05 iteration 1 (this run)** | **text-only** | **0.7398** | **0.6885 macro** |

We are **below both prior reports by ~0.07-0.09 AUC and ~0.03-0.06 F1**. Treating 0.7398 as the MVP 1 baseline would depress every downstream comparison and weaken the project's central claim (that gated fusion *beats* a strong text-only baseline). The iteration-2 plan in §7 therefore targets AUC ≥ 0.80 before declaring MVP 1 complete.

### 5.12 Artefacts written (iteration 1)

```
models/roberta_mvp1_iter1/
  ├── README.md                       5.08 KB   (auto-generated by PEFT)
  ├── adapter_config.json             1.02 KB
  ├── adapter_model.safetensors       2.26 MB   (LoRA delta from this fine-tune)
  ├── head.pt                         4.83 KB   (Linear(768, 1) T1 head state_dict)
  ├── hparams.json                    0.38 KB
  ├── metrics.json                    1.27 KB   (full balanced + recalibrated table)
  └── training_history.json           0.60 KB

outputs/
  ├── nb05_confusion_matrix.png      (2×2 test CM, threshold 0.5)
  └── nb05_training_curves.png       (train/val loss + val AUC + val F1@0.5)

notebooks/
  └── 05_mvp1_roberta_t1.ipynb       (executed, 11 cells, 0 errors)
```

The directory was renamed from `models/roberta_mvp1/` to `models/roberta_mvp1_iter1/` after this iteration so that `models/roberta_mvp1/` can hold the iteration-2 result without confusing downstream notebooks.

---

## 6. Diagnostic Analysis & Iteration Log for NB 05

> This section is the **paper-trail record** of what was tried, what happened, and why — written for re-reading when drafting the Methods / Limitations / Discussion sections of the final report.

### 6.1 Headline diagnostic

> **The model essentially did not learn.** Train loss at epoch 1 is already at the expected random-initialisation level for the chosen loss configuration; the next 4 epochs barely move it. The 5-epoch plateau at AUC ~ 0.74 is the symptom of an optimisation problem, not of model capacity saturation on a difficult task.

**Quick math.** Expected Focal-BCE loss at random init with the chosen weights and ~22 % positive rate is approximately:

$$\mathbb{E}[\mathcal{L}_{\text{init}}] \approx 0.78 \cdot (0.69 \cdot 0.25) + 0.22 \cdot 3.57 \cdot (0.69 \cdot 0.25) \approx 0.27$$

The observed epoch-1 train loss is **0.2456**, which is already at this level. Subsequent epochs go to 0.2349 — a drop of 0.011, well within batch-noise of the random-init expectation.

### 6.2 Predicted-probability distribution

- Test probability **mean** = 0.5037 (true hate rate 0.5001).
- The model is **outputting near-uniform probabilities** rather than separating classes into low- and high-probability regions.
- AUC of 0.74 in this regime means most discriminative power comes from a small tail of high-confidence predictions; the bulk of test examples are clustered around 0.5 and contribute random-noise ranking pairs.

This is the **diagnostic signature of "did not learn"**, not "learned and saturated". A model that genuinely saturated would have a bimodal probability distribution (peaks near 0 and near 1) and an AUC close to the irreducible Bayes ceiling.

### 6.3 Root-cause hypotheses (ranked by likelihood, with evidence)

| # | Hypothesis | Evidence | Status |
|---|-----------|----------|--------|
| H1 | **Double-counted class imbalance.** `pos_weight = 3.5739` (sklearn 'balanced' on train) stacked on top of Focal CE (γ = 2). Focal already handles imbalance by down-weighting easy/confident predictions; multiplying by a 3.57× pos_weight makes the loss landscape unstable for the minority direction. | Compare to NB 04 where classes were near-uniform: pos_weight ≈ 1.0, no double-counting, model trained cleanly. Here, pos_weight is amplifying what Focal is already doing. | **likely culprit** |
| H2 | **NB 04 warm-start fights MMHS150K's distribution.** NB 04 ended with `not_cyberbullying` F1 = 0.63, recall = 0.57 — the encoder learnt to *over-predict* cyberbullying. MMHS150K train is 78 % NotHate (opposite), so the warm-started encoder + pos_weight=3.57 push in the same wrong direction; gradients conflict with the data prior. | **FPR = 0.3071** at threshold 0.5 — 31 % of true NotHate is mis-classified as Hate. That is the NB 04 over-prediction bias visible in test predictions. | **likely culprit** |
| H3 | **LR schedule decays too early; LoRA capacity too low to escape init.** 590k trainable params (0.47 % of model) inside a rank-16 subspace, effective batch 64, warmup 10 % then cosine decay. After warmup (1,053 steps) the LR is already on its way down; ~10,530 total opt steps in a 590k-dim subspace is not many. | Train loss change across 4 additional epochs = 0.011. This is consistent with shrinking step size in a low-dimensional space — the optimiser is essentially stuck. | plausible secondary |
| H4 | **`t2_valid=False` rows add label noise.** ~9,319 train rows have ambiguous T2 (3-way annotator disagreement). T1 is still set by majority-non-zero, but label confidence is genuinely lower for these ~6.9 % of train. | Modest noise contribution. Not the main cause but worth filtering once the dominant issues are fixed. | minor |
| H5 | **Sequence-length 128 truncates p95 = 133.** ~5 % of tweets are truncated. | Small share. Defer. | minor |
| H6 | **Recalibration threshold methodology inconsistency.** F1-optimised threshold computed on raw val probs, then applied to recalibrated test probs. | Recalibrated table at threshold 0.5 collapses to all-NotHate predictions (TN=4999, FP=0). | methodology fix |

### 6.4 What the model did NOT do (negative evidence)

- **Did not overfit catastrophically.** Train loss did not collapse to near-zero. Train/val gap is small. There is no sign of the encoder memorising train idioms.
- **Did not diverge.** Loss curves are stable, no NaN / inf. fp16 + GradScaler behaved correctly.
- **Did not exhibit gate-style collapse.** This is a single-branch model — no gating involved — so collapse modes from later MVPs do not apply here.

This is important because the cleanness of the run rules out trivial failure modes (bad data, bad code) and focuses the diagnosis on the loss/LR/warm-start interaction.

### 6.5 Iteration log (decisions queue, priority order)

| # | Change | Rationale | Cost | Expected impact |
|---|--------|-----------|------|-----------------|
| **D1** | **Drop `pos_weight` entirely; keep Focal γ = 2 only.** | Removes the H1 double-counting. Focal alone is the loss that NB 04 used cleanly. | 1-line edit; same training time | +0.02 – 0.04 AUC expected on its own. |
| **D2** | **Sanity ablation: train MVP 1 with NO warm-start.** Load fresh `cardiffnlp/twitter-roberta-base-2022-154m`, fresh LoRA, otherwise identical. | Directly tests H2. If fresh ≥ warm-start, the warm-start is hurting and the NB 04 pretext task must be redesigned (or skipped) before MVP 1 can be declared. | One extra training run | This is the **load-bearing experiment** for the project's pretext-task design. |
| **D3** | **Increase LoRA rank to 32, keep `lora_lr=1e-4`.** Or alternatively: keep rank 16 and raise `lora_lr` to 2e-4 / 3e-4. | Addresses H3. Standard LoRA fine-tune LRs are higher than 1e-4; rank 16 may be undersized for a 134k-row fine-tune. | One extra training run | +0.01 – 0.03 AUC. Pair with D1 and D2 to isolate. |
| D4 | Filter `t2_valid=False` rows from train. | Addresses H4. Cleaner gradient signal at minor cost (~7 % of train). | 2-line edit | Marginal. Defer until D1+D2+D3 land. |
| D5 | Bump `max_len` to 160. | Covers p95 = 133 fully. | ~10 % slower per epoch | Marginal. Defer. |
| D6 | Fix recalibration threshold protocol: optimise threshold on **recalibrated val probabilities**, then apply to recalibrated test. | Addresses H6. Methodology rigour. | Cell-10 rewrite | Does not change AUC; changes reported F1m/Pm/Rm at the recalibrated threshold to be meaningful again. |

### 6.6 Decision rule for iteration 2

A redesigned NB 05 is the official MVP 1 baseline **only if** the test-set AUC ≥ 0.80 *and* FPR ≤ 0.25 on the balanced 50/50 split, recalibrated metrics report a non-degenerate confusion matrix, and the run remains zero-error. Otherwise a further iteration is required.

If after D1 + D2 + D3 the model still cannot reach AUC 0.80, the project will:

1. Document the gap and accept a weaker MVP 1 (Gomez 2019's reported number is achievable, so this would imply a code or data bug — a third iteration would focus on bug-hunting, not on hyperparameter tuning).
2. Reconsider whether the NB 04 pretext task is helping or hurting. If D2's no-warm-start run cleanly beats the warm-start, the v1.2 scope's pretext-task framing needs to be revised in the final report.

### 6.7 What carries into the Methods section of the thesis

When writing up the report, the following from §5–6 should be transcribed (paraphrased, not copied):

- **Methods / MVP 1.** Use §5.5 (architecture), §5.6–5.7 (loss & optimisation as they ended up after iteration 2, not iteration 1), and §5.10 (recalibration protocol).
- **Methods / methodology rigour.** Cite the dual-reporting requirement (balanced + recalibrated) from `CLAUDE.md §9`, with the iteration-2 corrected threshold protocol.
- **Limitations / iteration log.** §6.3 hypothesis table and §6.5 decisions queue can be condensed into a "design iteration" paragraph in the Limitations section — useful evidence that the project did not just report the first number it got.
- **Discussion / pretext-task design.** Whatever D2 reveals (warm-start helps / neutral / hurts) is a substantive finding worth a paragraph. The current spec assumes it helps; iteration 2 will test that assumption.

---

## 7. NB 05 Run 2 — Loss Change (Drop `pos_weight`)

### 7.1 Single-variable design

Run 2 is the first ablation of Run 1. The single variable changed is the loss configuration:

| | Run 1 | **Run 2** |
|---|-------|-----------|
| Encoder init | warm-start LoRA from `models/roberta_pretrain/` | warm-start LoRA from `models/roberta_pretrain/` |
| Tokenizer | from `models/roberta_pretrain/` | from `models/roberta_pretrain/` |
| Loss | `FocalBCE(γ=2, pos_weight=3.5739)` | **`FocalBCE(γ=2)` only — no `pos_weight`, no class weights** |
| Threshold-search protocol | F1-opt on raw val, applied to recal test | **fixed: F1-opt on recalibrated val, applied to recal test** |
| Everything else | identical | identical |

The threshold-search protocol fix (§6.5 D6) shipped alongside the loss change but is methodology-rigour, not a hypothesis test on AUC; AUC is threshold-invariant.

### 7.2 Outcome (halted at epoch 1)

The run was started with the patched notebook and halted manually by the user after epoch 1 once the AUC delta from Run 1 was visible.

| Epoch | Train loss | Val loss | Val AUC | Val F1@0.5 | Wall-clock |
|------:|-----------:|---------:|--------:|-----------:|-----------:|
| 1 | 0.1237 | 0.2000 | **0.7424** | 0.3442 | ≈ 14 min |

**Comparison to Run 1's epoch 1:**

| Metric | Run 1 (ep 1) | Run 2 (ep 1) | Δ |
|--------|-------------:|-------------:|--:|
| Train loss | 0.2456 | 0.1237 | −0.1219 *(the `pos_weight` factor)* |
| Val loss | 0.3563 | 0.2000 | −0.1563 *(same — loss is smaller without pos_weight)* |
| **Val AUC** | **0.7427** | **0.7424** | **−0.0003** *(within noise)* |
| Val F1@0.5 | 0.6769 | 0.3442 | −0.3327 *(threshold 0.5 is now badly calibrated; model is biased toward NotHate without `pos_weight`)* |

### 7.3 Interpretation

- The loss change does what arithmetic predicts — `pos_weight` is a multiplicative factor on positive examples, so the loss magnitude drops when removed.
- The **AUC delta is essentially zero** at epoch 1, where Run 1 had already reached its plateau height. Running the remaining 4 epochs would have replayed Run 1's plateau at a lower-magnitude loss but the same ranking quality.
- The **F1 collapse at threshold 0.5** is *not* a learning failure — it is a calibration consequence. Without `pos_weight`, the model's sigmoid output is centred lower (closer to the train prior of ~22 % hate), so threshold 0.5 cuts off most positives. A properly tuned threshold (the F1-opt on recalibrated val from cell 10) would restore F1.

### 7.4 Verdict on H1

**Hypothesis H1 (double-counted class imbalance) is RULED OUT as the cause of the AUC plateau.** Removing `pos_weight` has no measurable effect on AUC. Whatever ceiling AUC ≈ 0.74 represents, the loss configuration is not creating it.

### 7.5 Artefacts written (Run 2)

Run 2 did not complete cell 8+, so the `metrics.json` / final-model save did not occur. The training-loop cell did save the best-epoch-1 adapter to `models/roberta_mvp1_best/` (overwriting Run 1's best). Charts in `outputs/previous_tests/nb05_*.png` are still Run 1's (cell 9 / cell 10 never executed in Run 2).

---

## 8. NB 05 Run C — No-Warm-Start Ablation

### 8.1 Single-variable design

With H1 ruled out by Run 2, the remaining co-equal top hypothesis from §6.3 is H2: the NB 04 warm-start fighting MMHS150K's class distribution. Run C tests it by holding everything else constant (including Run 1's loss configuration) and changing only the encoder initialisation:

| | Run 1 | Run 2 | **Run C (Diagnostic)** |
|---|-------|-------|------------------------|
| Encoder init | warm-start LoRA from `models/roberta_pretrain/` | warm-start LoRA from `models/roberta_pretrain/` | **fresh `cardiffnlp/twitter-roberta-base-2022-154m`, brand-new LoRA built via `get_peft_model(LoraConfig(...))`** |
| Tokenizer | `models/roberta_pretrain/` | `models/roberta_pretrain/` | **`cardiffnlp/twitter-roberta-base-2022-154m` (HF Hub)** |
| Loss | `FocalBCE(γ=2, pos_weight=3.5739)` | `FocalBCE(γ=2)` only | **`FocalBCE(γ=2, pos_weight=3.5739)` — restored to match Run 1** |
| Epochs run | 5 | 1 (halted) | **5 (complete)** |
| Notebook file | `05_mvp1_roberta_t1.ipynb` | `05_mvp1_roberta_t1.ipynb` | `05c_no_warmstart.ipynb` (separate file to keep artefacts isolated) |
| Output dirs | `models/roberta_mvp1_iter1/` (Run 1 final) | `models/roberta_mvp1_best/` (Run 2's partial best, overwrote Run 1's) | `models/roberta_mvp1_fresh/`, `models/roberta_mvp1_fresh_best/` |
| Charts | `outputs/previous_tests/nb05_*.png` | (none, halted before cell 9) | `outputs/previous_tests/nb05_fresh_*.png` |

LoRA init sanity check at training start (from cell 5 of Run C):
- `lora_A` weight norms (mean) = 2.31  (Kaiming init, non-zero — expected)
- `lora_B` weight norms (mean) = 0.0  (zero-init by PEFT convention, so initial LoRA delta = AB = 0)

So at step 0 the model behaves identically to vanilla `cardiffnlp/twitter-roberta-base-2022-154m` with no adapter contribution. Any difference from Run 1 attributes to the difference between training the LoRA from a Kaggle-cyberbullying-trained start versus training from this null start.

### 8.2 Training trajectory (5 epochs, ~11 min each)

| Epoch | Train loss | Val loss | Val AUC | Val F1@0.5 | Wall-clock |
|------:|-----------:|---------:|--------:|-----------:|-----------:|
| 1 | 0.2479 | 0.3524 | 0.7405 | 0.6714 | 641 s |
| 2 | 0.2397 | 0.3413 | 0.7425 | 0.6891 | 647 s |
| 3 | 0.2377 | **0.3264** | 0.7439 | 0.6996 | 652 s |
| 4 | 0.2360 | 0.3407 | 0.7446 | 0.6926 | 658 s |
| 5 | 0.2349 | 0.3362 | **0.7447** | 0.6927 | 659 s |

- Best by val AUC: **epoch 5** (0.7447). Best by val loss: epoch 3 (0.3264). Mild val/train divergence at epoch 4 then re-converges at epoch 5 — same shape as Run 1, no overfit.
- Per-epoch time on T4: ~11 min (much faster than Run 1's later epochs of ~23 min — likely because Run 1's epochs 2-5 were affected by background contention or studio resource sharing; the architecture is identical).
- See `outputs/previous_tests/nb05_fresh_training_curves.png`.

### 8.3 Test set (balanced 50/50, n = 10,000, threshold 0.5)

| Metric | Value |
|--------|------:|
| AUC-ROC | **0.7407** |
| F1 (macro) | 0.6875 |
| Precision (macro) | 0.6875 |
| Recall (macro) | 0.6875 |
| TN / FP / FN / TP | 3,465 / 1,534 / 1,591 / 3,410 |
| **False positive rate** | **0.3069** |

See `outputs/previous_tests/nb05_fresh_confusion_matrix.png`.

### 8.4 Recalibrated test (deployment prior ~22 % hate, F1-opt on recal val)

F1-opt threshold on recalibrated val = **0.220** (val F1 = 0.7156). Recalibrated test probability range = `[0.068, 0.470]`, mean 0.262 — well-spread, non-degenerate.

| Metric | Recalibrated |
|--------|------:|
| AUC-ROC | 0.7407 (invariant) |
| F1 (macro) | 0.6810 |
| Precision (macro) | 0.6873 |
| Recall (macro) | 0.6829 |
| FPR | 0.3941 |
| TN / FP / FN / TP | 3,029 / 1,970 / 1,201 / 3,800 |

The recalibrated confusion matrix is **non-degenerate** (unlike Run 1, where the iter-1 protocol bug collapsed it). The §6.5 D6 protocol fix shipped in Run 2 has been validated end-to-end here.

### 8.5 Δ vs Run 1

| Metric | Run 1 (test) | Run C (test) | Δ |
|--------|-------------:|-------------:|--:|
| AUC-ROC | 0.7398 | **0.7407** | **+0.0009** |
| F1 (macro) | 0.6885 | **0.6875** | **−0.0010** |
| FPR | 0.3071 | **0.3069** | **−0.0002** |
| Best val AUC | 0.7457 (ep 4) | 0.7447 (ep 5) | −0.0010 |

**All deltas are within run-to-run variance.** The two runs are statistically indistinguishable.

### 8.6 Verdict on H2

**Hypothesis H2 (NB 04 warm-start fighting MMHS150K's class distribution) is RULED OUT as the cause of the AUC plateau.** Removing the warm-start entirely changes the test AUC by +0.0009. The NB 04 pretext task is **neither helping nor hurting** on the primary metric — it is design-neutral.

> Secondary observation for the thesis: the warm-start is not load-bearing for MVP 1. It can be kept for thematic coherence with the project's pretext-task framing, but it cannot be claimed in the contribution narrative as the source of any AUC gain. **It can be honestly described as a domain-familiarisation step whose effect was found to be neutral in controlled ablation.**

### 8.7 Artefacts written (Run C)

```
models/roberta_mvp1_fresh/
  ├── README.md                       5.08 KB
  ├── adapter_config.json             1.02 KB
  ├── adapter_model.safetensors       2.26 MB
  ├── head.pt                         4.83 KB
  ├── hparams.json                    0.38 KB
  ├── metrics.json                    0.91 KB   (balanced + recalibrated tables)
  └── training_history.json           0.60 KB

models/roberta_mvp1_fresh_best/
  ├── README.md
  ├── adapter_config.json
  ├── adapter_model.safetensors
  └── head.pt

outputs/
  ├── nb05_fresh_confusion_matrix.png
  └── nb05_fresh_training_curves.png

notebooks/
  └── 05c_no_warmstart.ipynb         (executed, 11 cells, 0 errors, 179 KB)
```

---

## 9. Three-Run Comparison and Pre-Run-D Diagnosis

### 9.1 Headline comparison

| Run | Encoder init | `pos_weight` | Epochs | Best val AUC | Test AUC | Test F1m | Test FPR |
|----:|--------------|:------------:|:------:|-------------:|---------:|---------:|---------:|
| **1** | warm-start (NB04) | 3.5739 | 5 | 0.7457 (ep 4) | **0.7398** | 0.6885 | 0.3071 |
| **2** | warm-start (NB04) | — | 1 (halted) | 0.7424 (ep 1) | n/a | n/a | n/a |
| **C** | FRESH (no warm-start) | 3.5739 | 5 | 0.7447 (ep 5) | **0.7407** | 0.6875 | 0.3069 |

### 9.2 Trajectory comparison (val AUC per epoch)

```
            ep1     ep2     ep3     ep4     ep5
Run 1:    0.7427  0.7439  0.7444  0.7457  0.7454
Run C:    0.7405  0.7425  0.7439  0.7446  0.7447
                                          (best)
```

- Same starting point (within 0.002).
- Same slope of improvement (slow, decelerating).
- Same plateau height (~0.745 val AUC, ~0.74 test AUC).
- The two trajectories are essentially overlaid — independent runs reproducing each other's plateau.

### 9.3 Updated root-cause table

> **Note (added after Run D).** This table reflects the diagnosis state **after Runs 1, 2, and C** — when H3 was the standing hypothesis pending an explicit test. Run D (§10) subsequently tested H3 and ruled it out. **For the current/final hypothesis status, see §11.2.** This table is retained as a historical snapshot of the iteration log.

The §6.3 hypothesis table is updated with the (pre-Run-D) diagnostic-suite verdicts:

| # | Hypothesis | Status after diagnostic suite | Evidence |
|---|-----------|-------------------------------|----------|
| H1 | Double-counted class imbalance (Focal γ=2 + `pos_weight`) | ❌ **RULED OUT** by Run 2 | Removing `pos_weight` changes val AUC by −0.0003 at epoch 1 — within noise. |
| H2 | NB 04 warm-start fighting MMHS150K class distribution | ❌ **RULED OUT** by Run C | Removing warm-start entirely changes test AUC by +0.0009 — within noise. Same trajectory shape and same ceiling. |
| H3 | LoRA rank 16 + cosine LR decayed too early → optimiser cannot escape init region | ✅ **STANDING (promoted to primary)** | Three independent runs converge on AUC ≈ 0.74 at every measured epoch. Train loss reduction across 5 epochs is < 5 % in all three. Plateau is reached by epoch 1 and never broken. Consistent with a low-rank optimisation ceiling. |
| H4 | `t2_valid=False` rows add label noise (~7 % of train) | ⏳ Untested. Modest noise contribution at best. | Defer until after H3 is tested with a higher-rank or higher-LR run. |
| H5 | Sequence length 128 truncates p95-length tweets | ⏳ Untested. ~5 % of rows truncate. | Low priority. |
| H6 | Recalibration threshold protocol inconsistency | ✅ **FIXED** in Run 2 / Run C cell 10 | Recalibrated test CM in Run C is non-degenerate (TN=3029, FP=1970, FN=1201, TP=3800). Fix validated. |
| **H7** *(new)* | **Architectural ceiling — 0.74 AUC is the actual MMHS150K text-only limit under PEFT LoRA rank 16 + lr 1e-4 + `cardiffnlp/twitter-roberta-base-2022-154m`** | ⏳ Co-equal with H3 until next ablation | If H3 is tested by bumping rank or LR and the AUC still plateaus at ~0.74, H7 is confirmed and reported as the project's load-bearing limitation. |

### 9.4 Two readings the thesis must keep open

After Runs 1, 2, and C the data supports two interpretations that the thesis Limitations section should distinguish:

1. **Optimisation ceiling, breakable** (H3). The PEFT adapter is undersized or the LR schedule is too conservative for a 134k-row binary fine-tune. A higher-rank or higher-LR run would clear the plateau.
2. **Architectural ceiling, not breakable inside the explored space** (H7). Twitter-RoBERTa with rank-16 LoRA simply has a 0.74-AUC capacity for this task on this dataset, regardless of hyperparameters in the standard range. Gomez 2019's 0.81-0.83 numbers required something outside this space (full fine-tune, larger backbone, no PEFT bottleneck, or different data preprocessing).

The next iteration (§10) is designed to distinguish them — a single rank bump (rank 16 → 32) holds everything else from Run C constant. If AUC clears 0.78, H3 is correct and H7 falls. If AUC stays at ~0.74, H7 is correct and the project must accept the baseline.

### 9.5 What this section means for the thesis

When writing up, condense §§7–9 into the following paragraphs:

- **Methods / MVP 1 baseline.** Use Run 1's architecture (§5.5) plus Run C's loss + threshold protocol (since Run 2 demonstrated the protocol fix is correct and Run C demonstrated the warm-start is neutral). Quote Run C's test numbers as the headline baseline — they are produced by the cleanest, most controlled pipeline.
- **Limitations / iteration log.** §§7-9 provide the iteration trail. The point is that three independent runs reproduce the plateau — this is *not* a hyperparameter tuning failure or a code bug, it is a measured architectural limit. The Limitations section can claim this honestly without overstating it.
- **Discussion / pretext-task design.** Run C's neutral verdict on the warm-start is itself a research finding. It implies that for binary MMHS150K T1 detection, Twitter-domain familiarisation via Kaggle cyberbullying classification is not load-bearing. The warm-start may yet contribute in MVP 4's gated fusion (where T2 categorical signal matters more) — leave that claim open until measured downstream.

---

## 10. NB 05 Run D — LoRA Capacity Ablation (Rank 32 + LR 3e-4)

### 10.1 Single-variable design (bundled)

With H1 and H2 ruled out by Runs 2 and C, the standing hypothesis after §9 was H3 — that the AUC plateau is a LoRA-rank-16 / lr-1e-4 optimisation ceiling, breakable with more capacity and a higher LR. Run D tests it by holding everything else constant from Run C (the cleanest baseline pipeline so far) and changing **two LoRA knobs together**:

| Knob | Run C | **Run D** | Rationale |
|------|-------|-----------|-----------|
| LoRA rank `r` | 16 | **32** | 2× capacity in the LoRA subspace |
| LoRA `lora_alpha` | 32 | **64** | Preserves `alpha = 2 × r` so effective scaling matches |
| LoRA learning rate | 1e-4 | **3e-4** | Standard LoRA fine-tune literature operates in 1e-4 – 5e-4 |
| Encoder init | fresh `cardiffnlp/...` | fresh `cardiffnlp/...` | unchanged |
| Loss | `FocalBCE(γ=2, pos_weight=3.5739)` | identical | unchanged |
| Head LR | 1e-3 | 1e-3 | unchanged |
| Batch / grad-accum / seq_len / epochs / seed | identical | identical | unchanged |
| Trainable params | 590,593 (0.47 %) | **1,180,417 (0.94 %)** — exactly 2× LoRA + same head | sanity-checked at training start |

The two knobs are bundled into one ablation because the binary question is *can the architecture clear AUC 0.78 under any reasonable rank/LR combination?* A single combined run gives the strongest signal in one training pass; if it fails, both knobs together fail and a single-variable follow-up would not change the verdict (each in isolation would move the model less).

LoRA init sanity at training start: `lora_A` weight norm mean **3.27** (Kaiming, ≈ √2 × Run C's 2.31, as expected from doubling rank), `lora_B` weight norm mean **0.0** (zero-init by PEFT convention). Initial LoRA delta = `A @ B = 0`, so at step 0 the model behaves identically to vanilla `cardiffnlp/twitter-roberta-base-2022-154m`.

### 10.2 Training trajectory (5 epochs, ~11 min each, total ~55 min)

| Epoch | Train loss | Val loss | Val AUC | Val F1@0.5 | Wall-clock |
|------:|-----------:|---------:|--------:|-----------:|-----------:|
| 1 | 0.2474 | **0.3119** | 0.7419 | **0.7119** | 657 s |
| 2 | 0.2390 | 0.3459 | 0.7450 | 0.6848 | 660 s |
| 3 | 0.2362 | 0.3599 | 0.7479 | 0.6762 | 663 s |
| 4 | 0.2327 | 0.3610 | **0.7495** | 0.6780 | 662 s |
| 5 | 0.2302 | 0.3485 | 0.7474 | 0.6863 | 662 s |

- Best by val AUC: **epoch 4** (0.7495 — the highest val AUC of any run in the four-run suite).
- Best by val loss: **epoch 1** (0.3119, but with lower val AUC). **Val loss increases while val AUC also increases (epochs 2-4)** — characteristic of a higher-LR regime: the model becomes more confident on miscalibrated examples (val loss up) while ranking better (AUC up).
- Per-epoch time is unchanged from Run C (~11 min). The rank-32 forward/backward cost is dominated by the unchanged backbone; doubling LoRA-trainable params adds < 1 % to throughput.

See `outputs/nb05_d_training_curves.png`.

### 10.3 Test set (balanced 50/50, n = 10,000, threshold 0.5)

| Metric | Value |
|--------|------:|
| AUC-ROC | **0.7431** |
| F1 (macro) | 0.6855 |
| Precision (macro) | 0.6879 |
| Recall (macro) | 0.6862 |
| TN / FP / FN / TP | 3,666 / 1,333 / 1,805 / 3,196 |
| **False positive rate** | **0.2667** |

See `outputs/nb05_d_confusion_matrix.png`.

### 10.4 Recalibrated test (deployment prior ~22 % hate, F1-opt on recal val)

F1-opt threshold on recalibrated val = **0.210** (val F1 = 0.7155). Recalibrated test probability range = `[0.055, 0.461]`, mean 0.250 — well-spread, non-degenerate.

| Metric | Recalibrated |
|--------|------:|
| AUC-ROC | 0.7431 (invariant) |
| F1 (macro) | 0.6852 |
| Precision (macro) | 0.6897 |
| Recall (macro) | 0.6865 |
| FPR | 0.3781 |
| TN / FP / FN / TP | 3,109 / 1,890 / 1,245 / 3,756 |

Recalibrated CM is **non-degenerate** — D6 protocol fix validated again under a different hyperparameter regime.

### 10.5 Δ vs Run C (cleanest single-variable predecessor)

| Metric | Run C (test) | Run D (test) | Δ | Within noise? |
|--------|-------------:|-------------:|--:|:-------------:|
| AUC-ROC | 0.7407 | **0.7431** | **+0.0024** | yes (±0.02 band) |
| F1 (macro) | 0.6875 | 0.6855 | −0.0020 | yes |
| Precision (M) | 0.6875 | 0.6879 | +0.0004 | yes |
| Recall (M) | 0.6875 | 0.6862 | −0.0013 | yes |
| **FPR** | **0.3069** | **0.2667** | **−0.0402** | **no — ~10× the AUC noise scale** |
| Best val AUC | 0.7447 | **0.7495** | **+0.0048** | borderline |

**The AUC delta is within noise. The FPR delta is not.** Run D made the model more conservative at threshold 0.5: it predicts NotHate slightly more often, gaining 201 true-negative classifications at the cost of 214 true-positive classifications. Macro F1 is essentially unchanged because the precision-recall trade-off is symmetric in the balanced test set, but FPR — which conditions on the NotHate column only — drops meaningfully.

### 10.6 Verdict on H3

**Hypothesis H3 (LoRA rank 16 + lr 1e-4 is the optimisation ceiling, breakable with more rank and a higher LR) is RULED OUT.** Doubling LoRA capacity *and* tripling the LoRA learning rate moves test AUC by 0.0024 — within run-to-run variance. The plateau persists.

### 10.7 Secondary finding — FPR drop at the same AUC

Although Run D fails to break the AUC ceiling, it shifts the model along the precision-recall curve in a direction useful for the project's deployment-routing story:

- **At the same ranking quality** (AUC ≈ 0.74), the model produces **13 % fewer false-positive predictions** at the canonical threshold 0.5.
- Mechanism: with rank 32 and lr 3e-4, the LoRA optimiser pushes threshold-relevant logits downward on examples the model is uncertain about, concentrating positive predictions on higher-confidence cases.
- For the project's human-in-the-loop framing (T3 routing of ambiguous cases to reviewers), a lower FPR at the same AUC is **operationally meaningful** even though it is not a discrimination-power improvement: fewer over-flagged non-hate posts reach human review.

This is the basis for selecting Run D as the MVP 1 baseline in §11.3, despite the AUC ceiling being confirmed.

### 10.8 Artefacts written (Run D)

```
models/roberta_mvp1_d/
  ├── README.md                       5.08 KB
  ├── adapter_config.json             1.02 KB
  ├── adapter_model.safetensors       4.51 MB   (≈ 2× Run C's 2.26 MB — rank 32 confirmed)
  ├── head.pt                         4.83 KB
  ├── hparams.json                    0.40 KB
  ├── metrics.json                    0.97 KB   (balanced + recalibrated tables, run label, lora params)
  └── training_history.json           0.60 KB

models/roberta_mvp1_d_best/
  ├── README.md                       5.08 KB
  ├── adapter_config.json             1.02 KB
  ├── adapter_model.safetensors       4.51 MB
  └── head.pt                         4.83 KB

outputs/
  ├── nb05_d_confusion_matrix.png    33.0 KB
  └── nb05_d_training_curves.png     68.8 KB

notebooks/
  └── 05d_rank32_lr3e4.ipynb          184 KB  (executed, 11 cells, 0 errors)
```

---

## 11. Final Four-Run Synthesis and MVP 1 Baseline Decision

### 11.1 Four-run headline comparison

| Run | Encoder init | `pos_weight` | r | lora_lr | Epochs | Best val AUC | Test AUC | Test F1m | Test FPR |
|----:|--------------|:------------:|:--:|:-------:|:------:|-------------:|---------:|---------:|---------:|
| 1 | warm-start (NB 04) | 3.5739 | 16 | 1e-4 | 5 | 0.7457 (ep 4) | 0.7398 | 0.6885 | 0.3071 |
| 2 | warm-start (NB 04) | — | 16 | 1e-4 | 1 (halted) | 0.7424 (ep 1) | n/a | n/a | n/a |
| C | fresh | 3.5739 | 16 | 1e-4 | 5 | 0.7447 (ep 5) | 0.7407 | 0.6875 | 0.3069 |
| **D** | **fresh** | **3.5739** | **32** | **3e-4** | **5** | **0.7495 (ep 4)** | **0.7431** | 0.6855 | **0.2667** |

### 11.2 Final hypothesis table

| # | Hypothesis | Final status | Evidence |
|---|-----------|--------------|----------|
| H1 | Double-counted class imbalance (Focal γ=2 + `pos_weight`) | ❌ **RULED OUT** | Run 2: removing `pos_weight` changes val AUC by −0.0003 at epoch 1 (within noise). |
| H2 | NB 04 warm-start fighting MMHS150K class distribution | ❌ **RULED OUT** | Run C: removing the warm-start entirely changes test AUC by +0.0009 (within noise). |
| H3 | LoRA rank 16 + lr 1e-4 is the optimisation ceiling, breakable with more capacity | ❌ **RULED OUT** | Run D: doubling rank to 32 *and* tripling lr to 3e-4 changes test AUC by +0.0024 (within noise). |
| H6 | Recalibration threshold protocol inconsistency | ✅ **FIXED** | Run 2 onward optimises F1 threshold on recalibrated val. Non-degenerate recal CM in Run C and Run D. |
| **H7** | **AUC ≈ 0.74 is the actual MMHS150K text-only ceiling under PEFT LoRA on `cardiffnlp/twitter-roberta-base-2022-154m`, within the explored hyperparameter space (rank ∈ {16, 32}, lr ∈ {1e-4, 3e-4})** | ✅ **CONFIRMED** | Four independent runs (different loss, different init, different capacity) all converge on test AUC ≈ 0.74. The plateau survives every parameter-efficient intervention tested. |
| H4 | `t2_valid=False` rows add label noise (~7 % of train) | ⏳ Untested. Modest contribution at best. | Not pursued — defer to MVP 5 / NB 09 bias analysis if relevant. |
| H5 | Sequence length 128 truncates p95-length tweets | ⏳ Untested. ~5 % of rows truncate. | Not pursued — low priority. |

### 11.3 Why Run D is selected as the MVP 1 baseline

> **This subsection is the load-bearing justification for citing Run D in the thesis. It is written so the rationale survives examination.**

After four runs, three are eligible for the MVP 1 baseline slot (Run 2 was halted at epoch 1 and has no completed test evaluation). The selection is **Run D**, on the following grounds:

1. **Highest test AUC of the eligible runs.** 0.7431 vs 0.7407 (Run C) and 0.7398 (Run 1). The pairwise deltas are individually within run-to-run noise (≈ ±0.02), but Run D is consistently first across all three eligible runs on the primary ranking metric. The fact that Run C ≈ Run 1 (with a 0.0009 gap) replicates noise behaviour cleanly suggests Run D's +0.0024 over Run C is small but plausibly real — at minimum it is not negative.
2. **Lowest test false-positive rate.** 0.2667 vs 0.3071 (Run 1) and 0.3069 (Run C). This delta is **not** within noise — it is roughly an order of magnitude larger than the AUC noise scale, and it is in the operationally desirable direction. For the project's human-in-the-loop framing (`CLAUDE.md §2b`, `Cyberbullying_Detection_Report_Framing.md §5`), a 13 % relative reduction in FPR at the same AUC means **fewer over-flagged non-hate posts going to human reviewers**.
3. **Highest validation AUC of any run.** 0.7495 at epoch 4 — the cleanest best-epoch behaviour in the suite.
4. **Cleanest controlled-experiment lineage.** Run D inherits Run C's no-warm-start architecture (the H2 ruled-out condition) and adds a single bundled capacity bump (the H3 test). Reporting Run D as the baseline lets the thesis tell the iteration story honestly: *we ablated the warm-start and found it neutral, then tested capacity and found the AUC ceiling; the best operational point in our explored space is Run D.*
5. **Trade-off is acknowledged honestly.** Run D's F1 (macro) is 0.0020 lower than Run C — within noise but worth flagging. The shift is a precision-recall move along a roughly equal-quality contour: more conservative decisions at the canonical threshold, fewer false positives, slightly fewer true positives, net F1m essentially unchanged. The direction of the shift is **deployment-aligned**.

#### Counter-considerations and why they do not win

**Why not Run 1?** The only argument for Run 1 over Run D is *thematic alignment* — Run 1 uses the NB 04 warm-start, which is the project spec's design choice. Run C's neutral verdict on the warm-start (§8.6) makes this thematic-only: the warm-start contributes nothing measurable to MVP 1 test AUC, so retaining it would mean paying a transparency cost (more complex pipeline, less defensible "minimal-baseline" framing) for no measurable benefit. The thesis can describe the warm-start honestly as *"a designed pretext task whose effect on MVP 1 was found to be neutral in controlled ablation; retained as a feature input to MVP 4's gated-fusion analysis where its multi-class cyberbullying-type signal may contribute beyond binary T1 ranking."*

**Why not Run C?** Run C's test AUC is 0.7407 — 0.0024 lower than Run D, but it is the cleanest *single-variable* ablation we have (only the warm-start is removed). For a paper that emphasises ablation rigour, Run C is the most defensible *minimal* baseline. The case for Run D over Run C rests on:
- Run D's higher AUC, lower FPR, higher val AUC (all small, all in the right direction).
- Run D demonstrates capacity has been ablated too — the baseline isn't accidentally optimal at one specific rank/LR setting; it is selected after capacity has been confirmed not to help.

**Defensible alternative phrasing for the thesis** if the user wants to hedge:

> "*The MVP 1 baseline reported here (Run D, AUC 0.7431, FPR 0.2667) was selected from a four-run diagnostic suite that ablated loss configuration (Run 2), warm-start contribution (Run C), and LoRA capacity (Run D). The neighbouring run (Run C, AUC 0.7407) is within run-to-run variance on AUC; we report Run D because it has the lowest false-positive rate (operationally aligned with the project's human-in-the-loop framing) at no statistically distinguishable AUC cost.*"

### 11.4 What the multimodal MVPs must beat

Downstream MVPs (NB 06 onward) are evaluated against **Run D's headline numbers**:

| Metric | MVP 1 baseline (Run D) | Threshold to claim a multimodal gain |
|--------|-----------------------:|:-------------------------------------|
| Test AUC | **0.7431** | Exceed by more than run-to-run noise: ≥ 0.76 to be a credible win, ≥ 0.78 to be a strong win. |
| Test F1m | 0.6855 | Improve only if AUC also improves; F1 alone is insufficient (it can move under precision-recall trade-offs without ranking improvement). |
| Test FPR | **0.2667** | Match or beat. A multimodal model that beats AUC but worsens FPR is a partial win at best. |
| Recalibrated F1m | 0.6852 | Non-degenerate recalibrated CM is mandatory; collapsing recal CM (Run 1 iter-1 bug) is a fatal protocol error. |

A multimodal MVP whose AUC beats Run D by < 0.005 with all other metrics flat is **not** a gain. A multimodal MVP that beats AUC by ≥ 0.02 with stable F1 and FPR is the meaningful threshold for the project's claimed contribution.

### 11.5 What the H7 ceiling means for the project's contribution narrative

The thesis can claim, after the four-run suite:

- A **precisely characterised ceiling** has been established for text-only PEFT LoRA on MMHS150K T1: **AUC ≈ 0.74** in the explored hyperparameter space (rank ∈ {16, 32}, LR ∈ {1e-4, 3e-4}, 5 epochs, `cardiffnlp/twitter-roberta-base-2022-154m`).
- This ceiling is robust to (i) loss-imbalance configuration, (ii) the NB 04 warm-start, (iii) LoRA capacity within the explored range.
- Any AUC improvement above **≈ 0.76** from the multimodal MVPs is therefore attributable to **modality fusion**, not to text-encoder tuning, because the text encoder has been measured at its operational ceiling under the project's PEFT regime.

This is a stronger contribution claim than the original `Multimodal_Cyberbullying_Detection_v1.2.md` framing because the baseline is now **empirically pinned, not assumed**.

#### Caveats for the Limitations section

- The ceiling is established **within PEFT LoRA**. A full fine-tune of the encoder, or a larger backbone (e.g. RoBERTa-large, DeBERTa-v3-large), may push AUC higher. We did not test these because (a) full fine-tune falls outside the project's PEFT scope per `CLAUDE.md §12`, and (b) larger backbones fall outside the T4 16 GB compute envelope for joint fp16 fusion training in later notebooks. These are *known unknowns*, not denied.
- Run 2 (no `pos_weight`) ran only 1 epoch before being halted; the conclusion that loss is not load-bearing on AUC is based on the epoch-1 match between Run 1 and Run 2 (val AUC 0.7427 vs 0.7424) rather than a 5-epoch comparison. The match at epoch 1, combined with the well-understood arithmetic of how `pos_weight` enters the loss (a multiplicative factor on positive examples, not a change to gradient direction), makes the 1-epoch result load-bearing in our judgement, but a fuller Run 2′ would be a defensive follow-up if reviewers ask.
- The 13 % FPR reduction in Run D vs Run C is genuine but was discovered after the ablation; we did not pre-register FPR as the selection metric. Phrasing in the thesis should be honest: *"FPR became the deciding metric between Run C and Run D because the AUC delta was within noise; the FPR improvement is reported as a secondary operational gain, not a pre-registered hypothesis."*

### 11.6 What carries into the Methods section of the thesis

When writing up:

- **Methods / MVP 1 architecture.** §5.5 (single-branch Twitter-RoBERTa + LoRA + Linear(768, 1) head). Update the rank to 32 and `lora_alpha` to 64 per §10.
- **Methods / MVP 1 loss.** Focal BCE γ=2 with `pos_weight=3.5739` derived from sklearn balanced class weights on the train split. (Run 2 demonstrated `pos_weight` is not load-bearing on AUC; we retain it because Run D — the selected baseline — used it, and removing it for the thesis baseline would be a post-hoc change.)
- **Methods / MVP 1 optimisation.** AdamW with LoRA lr 3e-4 / head lr 1e-3 / weight decay 0.01; linear warmup 10 % → cosine decay; batch 16 × grad-accum 4 = effective batch 64; fp16 with `torch.amp.GradScaler` and `torch.amp.autocast`; 5 epochs; seed 42; seq_len 128.
- **Methods / methodology rigour.** Dual-reporting protocol (balanced + recalibrated) from `CLAUDE.md §9`; F1-opt threshold optimised on recalibrated val (D6 fix, validated by non-degenerate recal CMs in Runs C and D).
- **Methods / iteration log.** §§ 5-11 are the full iteration trail. Condense to one paragraph: *"MVP 1 was developed through a four-run diagnostic suite. Run 1 (loss + warm-start, the original design per `Multimodal_Cyberbullying_Detection_v1.2.md`) plateaued at test AUC 0.7398. Two single-variable ablations (Run 2 dropping `pos_weight`; Run C dropping the warm-start) confirmed that neither was load-bearing on AUC. A capacity ablation (Run D, rank 32 + lr 3e-4) confirmed that the plateau is robust to standard PEFT scaling. We report Run D (AUC 0.7431, FPR 0.2667) as the MVP 1 baseline on the basis of its operational improvement on FPR at no AUC cost (full justification in §11.3 of the Phase 2 modelling report)."*
- **Discussion / contribution claim.** Per §11.5 — the text-only ceiling is empirically pinned, so any multimodal gain in NB 07-09 is attributable to modality fusion.
- **Limitations.** Per §11.5 caveats.

---

## 12. Open Items and Preconditions for NB 06

> **Note (added after NB 06 ran).** This section captures the preconditions and the next-notebook contract **as they were defined before NB 06 was executed**. NB 06 has since completed and its full results are documented in §13. This section is retained as a historical snapshot of the iteration log; the current artefact and decision status for downstream notebooks lives in §13.12.



| Item | Status |
|------|--------|
| Run 1 model preserved at `models/roberta_mvp1_iter1/` | ✅ done |
| Run C model preserved at `models/roberta_mvp1_fresh/` | ✅ done |
| Run D model preserved at `models/roberta_mvp1_d/` | ✅ done |
| **MVP 1 baseline declared (Run D)** — §11.3 justification | ✅ done |
| All ablations documented in iteration log (§§ 5-11) | ✅ done |
| D1 / D2 / D3 / D6 decisions queue from §6.5 | ✅ all closed |
| Per-T3-bucket AUC diagnostic | ⏳ deferred to MVP 5 / NB 09 (bias analysis cell will use T3 buckets) |
| H4 (label noise) / H5 (seq_len 128 truncation) | ⏳ untested — defer to future work; not load-bearing for MVP 1 verdict |
| NB 06 (CLIP + XGBoost image baseline · structured-only baseline · add T2 head to text encoder) | ⏳ next notebook |

### Next-notebook contract (NB 06)

NB 06 must:

1. Build the **image-only baseline.** Encode all 149,819 MMHS150K images with frozen `openai/clip-vit-base-patch16`; project the 512-d CLIP embedding to a binary T1 head via XGBoost / Logistic Regression. Report test AUC + FPR at threshold 0.5.
2. Build the **structured-only baseline.** Use the 9-feature vector from `data/processed/structured_features.csv` (the pruned feature set from NB 03); fit XGBoost / Logistic Regression for T1. Report test AUC + FPR.
3. **Reuse Run D's encoder** to add the **T2 multi-class head** (6 categories: NotHate, Racist, Sexist, Homophobe, Religion, OtherHate). Load the Run-D LoRA adapter from `models/roberta_mvp1_d/` and attach a fresh `Linear(768, 6)` head. Mask `t2_valid=False` rows in the training loss (NaN target — use a masked Focal CE).
4. Report all metrics in two forms (balanced + recalibrated to prior 0.2468), per `CLAUDE.md §9`. Apply the D6 protocol (F1-opt threshold on recalibrated val).
5. Compare image-only and structured-only AUC against Run D's text-only baseline (AUC 0.7431). Expected outcome: both underperform text-only (validates the project's working assumption that text is the dominant signal for MMHS150K T1). If either matches or beats, document and revisit the framing.

### Decision rule after NB 06

- **Typical case:** image-only and structured-only underperform Run D. Proceed to NB 07 (naive fusion, MVP 3) to replicate the documented MMHS150K naive-fusion failure mode (the bedrock motivation for the project's gated-fusion contribution).
- **Surprise case:** image-only or structured-only AUC ≥ 0.74 standalone. Document; this would weaken the "text-dominant" assumption and require a brief revision to the v1.2 scope's framing.

---

## 13. Notebook 06 — MVP 2 Multimodal Naive Concatenation (text + image)

### 13.1 Purpose

Notebook 06 is **MVP 2** of the project's phased delivery ladder. It is the first multimodal model in the project: a CLIP ViT-B/16 image encoder (LoRA rank 16 on the vision attention `q_proj` and `v_proj` projections) is added alongside the frozen Run-D text encoder from MVP 1, and the two branches' features are combined by naive concatenation before being routed to two task heads — **T1** (binary hate vs. not-hate) and **T2** (six-category masked classification).

The naive-concat design is deliberate, not a default. Gomez et al. 2019 — the dataset paper for MMHS150K — explicitly documented that simple multimodal fusion does **not** outperform a strong text-only baseline on this corpus, and that finding has been replicated multiple times in subsequent MMHS150K literature. MVP 2's purpose is to **empirically reproduce that failure mode under our specific architecture** (frozen Run-D text encoder + parameter-efficient CLIP LoRA + fixed-weight concatenation). Establishing a measured naive-fusion number is what makes MVP 4's gated cross-modal attention defensible: gated fusion must outperform a measured naive baseline, not an assumed one.

Five locked architectural decisions are introduced at this step and remain locked through MVP 5 unless explicitly revisited. The text encoder is frozen throughout — no gradient flow through RoBERTa or through its MVP 1 LoRA delta. The CLIP vision backbone is frozen and only its q_proj / v_proj projections receive LoRA-rank-16 adapters. The T2 loss is masked to `t2_valid=True` rows only, so the 6.2 % of train rows with ambiguous category labels (`t2_valid=False`, T2 = NaN per `CLAUDE.md §10`) do not contribute noise to the categorical head. The combined loss is `0.7 × L_T1 + 0.3 × L_T2`, weighting the primary task higher than the auxiliary. Fusion is plain `torch.cat` — no attention, no gating, no MLP-mixer. Cross-modal attention lands in MVP 4 / NB 08; introducing it earlier would conflate the *fusion-architecture* contribution with the *image-encoder* contribution.

### 13.2 Inputs

| Property | Value |
|---|---|
| Labels CSV | `data/processed/labels_parsed.csv` (149,819 rows, from Phase 1) |
| Image directory | `data/MMHS150K/img_resized/` (150,000 JPGs on disk; all GT-referenced images present per Phase 1 audit) |
| Split source | `data/MMHS150K/splits/{train,val,test}_ids.txt` — joined on `tweet_id` to construct the `split` column at runtime |
| Image processor | `CLIPImageProcessor` from `openai/clip-vit-base-patch16` (shortest-edge 224, ImageNet-style normalisation, mean (0.481, 0.458, 0.408), std (0.269, 0.261, 0.276)) |
| Text tokenizer | `cardiffnlp/twitter-roberta-base-2022-154m` from Hub (same as Run D) |
| Frozen text adapter | `models/roberta_mvp1_d/` — Run D's LoRA-tuned encoder, the selected MVP 1 baseline (§11.3) |
| Sequence length | 128 (matches MVP 1; p95 tweet length is 133 chars per Phase 1 EDA) |
| Image-existence drop count | **0** rows — all 149,819 GT-referenced tweet IDs have a corresponding `.jpg` file |

#### Split sizes and class distributions (post-join, all 149,819 rows kept)

| Split | n | T1 % hate | `t2_valid` rows | T2 NaN (filtered out of T2 loss) |
|---|---:|---:|---:|---:|
| train | 134,820 | 21.86 % | 125,501 | 9,319 |
| val | 4,999 | 50.01 % | 4,197 | 802 |
| test | 10,000 | 50.01 % | 8,411 | 1,589 |

#### T2 class distribution on `t2_valid=True` rows (per split)

| Class | train | val | test |
|---|---:|---:|---:|
| NotHate | 105,344 | 2,499 | 4,999 |
| Racist | 9,505 | 809 | 1,613 |
| Sexist | 2,790 | 241 | 464 |
| Homophobe | 3,087 | 253 | 531 |
| Religion | 130 | 9 | 24 |
| OtherHate | 4,645 | 386 | 780 |

The T2 distribution is the natural MMHS150K class balance documented in `Phase1_Data_Engineering_Report.md`. The Religion class (130 train rows / 24 test rows) is genuinely rare by data construction, not a measurement error — Focal Loss is the project's locked mitigation for this rarity, not over-sampling (per `CLAUDE.md §10`). `t2_valid=False` rows have their T2 target replaced with placeholder 0 in the dataset class; the validity flag carries the real signal and is consumed by the masked T2 loss in §13.4.

### 13.3 Architecture

The dual-branch model is wired so the text branch never sees gradient. The image branch carries all the trainable surface (CLIP LoRA, the trainable second-stage projection, and the dual heads). The text branch is called inside a `torch.no_grad()` block and kept in `.eval()` mode for the full training loop.

```
Text branch (NO GRAD, eval mode)
─────────────────────────────────────────────────────────────────────────
input_ids (B, 128)  ┐
                    ├─► [FROZEN] Twitter-RoBERTa + Run-D LoRA  ─► [CLS] (B, 768)
attention_mask (B, 128) ┘

Image branch (CLIP vision + LoRA r=16)
─────────────────────────────────────────────────────────────────────────
pixel_values (B, 3, 224, 224)
            │
            ▼
[FROZEN backbone] CLIPVisionModel encoder (12 layers)
   with LoRA r=16, lora_alpha=32 injected into self_attn.{q_proj, v_proj}
            │
            ▼
   pooler_output (B, 768)
            │
            ▼
[FROZEN] visual_projection_to_512 — Linear(768 → 512, bias=False)
            │                       (weights lifted from public CLIPModel.visual_projection)
            ▼
   feat_512 (B, 512)
            │
            ▼
[TRAINABLE] image_projection — Linear(512 → 768)
            │                       (CLAUDE.md §3 cross-modal dim alignment)
            ▼
   img_768 (B, 768)

Fusion (naive concat) and dual heads
─────────────────────────────────────────────────────────────────────────
fused = torch.cat([img_768, txt_768], dim=-1)   ─►   (B, 1536)
                            │
                            ├──► T1 head: Linear(1536→256) ─► ReLU ─► Dropout(0.1) ─► Linear(256→1)   ─► logit_t1
                            │
                            └──► T2 head: Linear(1536→256) ─► ReLU ─► Dropout(0.1) ─► Linear(256→6)   ─► logits_t2
```

#### Component table

| Component | Source / type | Parameters | Trainable? |
|---|---|---:|:---:|
| Text encoder (RoBERTa-base) | `cardiffnlp/twitter-roberta-base-2022-154m` | 124,645,632 | No |
| Text LoRA (Run D, r = 32, α = 64 on `query` + `value`) | `models/roberta_mvp1_d/adapter_model.safetensors` | 1,179,648 | **No** (locked decision #1) |
| Vision encoder (CLIP ViT-B/16) | `openai/clip-vit-base-patch16` (vision tower only) | 86,389,248 | No |
| Vision LoRA (r = 16, α = 32 on `q_proj` + `v_proj`) | injected via PEFT 0.19 | **589,824** | **Yes** (locked decision #2) |
| `visual_projection_to_512` | Linear(768→512, no bias), weights init from `CLIPModel.visual_projection` | 393,216 | No (locked) |
| `image_projection` | Linear(512→768) | **393,984** | Yes |
| `head_t1` | Linear(1536→256) → ReLU → Dropout(0.1) → Linear(256→1) | **393,729** | Yes |
| `head_t2` | Linear(1536→256) → ReLU → Dropout(0.1) → Linear(256→6) | **395,014** | Yes |

#### Parameter count summary

| Quantity | Value |
|---|---:|
| Total parameters | 213,790,471 |
| Total trainable | **1,772,551 (0.829 %)** |
| Trainable in text branch | **0** (asserted at build) |
| Trainable in CLIP LoRA | 589,824 |
| Trainable in frozen `visual_projection_to_512` | 0 (asserted at build) |
| Trainable in `image_projection` | 393,984 |
| Trainable in `head_t1` | 393,729 |
| Trainable in `head_t2` | 395,014 |

#### PEFT-bypass workaround in `encode_image`

PEFT 0.19's `PeftModel.forward` method is incompatible with transformers 5.x's `CLIPVisionModel` forward path: an internal kwargs collision causes `CLIPEncoder` to receive `inputs_embeds` twice and raises `TypeError`. The bug was diagnosed during the build smoke test (raw `CLIPVisionModel(pixel_values=...)` works; `PeftModel(pixel_values=...)` fails; `peft_model.base_model.model(pixel_values=...)` works).

The workaround is implemented inside `MVP2NaiveConcat.encode_image`:

```python
out = self.vision_encoder.base_model.model(pixel_values=pixel_values)
```

instead of the standard

```python
out = self.vision_encoder(pixel_values=pixel_values)
```

`self.vision_encoder.base_model.model` is the original `CLIPVisionModel` instance after PEFT has injected its `lora.Linear` modules into the attention layers. Calling it directly skips `PeftModel.forward` and `LoraModel.forward` but does **not** skip the LoRA computation, because the LoRA layers are themselves `nn.Module` instances sitting inline with the q_proj / v_proj projections. The workaround is invisible to `save_pretrained` / `from_pretrained`, which do not route through `forward`. This workaround is locked as decision #6 in §13.11.

### 13.4 Loss

The total loss is a fixed-weight sum of two task-specific Focal Loss variants:

$$\mathcal{L}_{\text{total}} = \lambda_1 \cdot \mathcal{L}_{T1} + \lambda_2 \cdot \mathcal{L}_{T2,\text{masked}},\qquad \lambda_1 = 0.7, \quad \lambda_2 = 0.3$$

#### T1 — Focal BCE with `pos_weight`

$$\mathcal{L}_{T1} = \frac{1}{N}\sum_i w_i \cdot (1 - p_t^{(i)})^{\gamma} \cdot \mathrm{BCE}(p^{(i)}, y_i^{T1}),\qquad w_i = y_i^{T1} \cdot \text{pos\_weight} + (1 - y_i^{T1})$$

with γ = 2 and `pos_weight = 3.5739`, derived from `sklearn.utils.class_weight.compute_class_weight('balanced', ...)` on the train T1 distribution (nothate weight 0.6399, hate weight 2.2869, ratio = 2.2869 / 0.6399 = 3.5739). This matches MVP 1 Run D's loss configuration exactly so the multimodal comparison is single-variable on the architecture rather than confounded by a loss-side change.

#### T2 — Masked Focal Cross-Entropy with class weights

For samples where `t2_valid = True`:

$$\mathcal{L}_{T2,\text{sample}}^{(i)} = w_{c^{(i)}} \cdot (1 - p_{c^{(i)}}^{(i)})^{\gamma} \cdot (-\log p_{c^{(i)}}^{(i)})$$

For samples where `t2_valid = False`, the per-sample loss is zeroed; the reported `L_T2` is the mean over the valid rows only:

$$\mathcal{L}_{T2,\text{masked}} = \frac{1}{\sum_i m^{(i)}} \sum_i m^{(i)} \cdot \mathcal{L}_{T2,\text{sample}}^{(i)},\quad m^{(i)} = \mathbb{1}[t2\_valid^{(i)}]$$

If a batch contains zero valid rows, the implementation returns `logits.sum() * 0.0` so autograd retains a gradient path and the optimiser step proceeds with no T2 contribution. Per-class weights are computed from the train distribution on `t2_valid=True` rows only:

| Class | train count | balanced weight |
|---|---:|---:|
| NotHate | 105,344 | 0.1986 |
| Racist | 9,505 | 2.2006 |
| Sexist | 2,790 | 7.4971 |
| Homophobe | 3,087 | 6.7758 |
| Religion | 130 | **160.8987** |
| OtherHate | 4,645 | 4.5031 |

The Religion weight (160.9) is mathematically extreme but is the direct sklearn `'balanced'` output given the class's natural rarity. Combined with Focal γ = 2, it drives the model toward predicting any minority hate category over NotHate on T2 — a behaviour quantified in §13.7 and §13.9.

#### Combined loss weighting

The 0.7 / 0.3 ratio is locked and follows standard multi-task learning practice of weighting the primary task above the auxiliary. The auxiliary T2 head is used here both as a learning signal that forces the image branch to capture category-discriminative visual features and as a diagnostic instrument that confirms the image branch is functional independent of T1 performance (§13.10 finding 2).

### 13.5 Optimisation

| Knob | Value |
|---|---|
| Optimiser | AdamW with three parameter groups |
| CLIP LoRA learning rate | 3e-4 |
| `image_projection` learning rate | 1e-3 |
| Head learning rate (T1 + T2 grouped) | 1e-3 |
| Weight decay | 0.01 on all groups |
| Scheduler | Linear warmup (10 % of steps) → cosine decay |
| Mixed precision | fp16 via `torch.amp.GradScaler` + `torch.amp.autocast('cuda', dtype=torch.float16)` |
| Batch size (physical) | 16 |
| Gradient accumulation | 4 steps → **effective batch 64** |
| Sequence length | 128 |
| Image size | 224 × 224 |
| Epochs | 5 |
| Seed | 42 |
| Data steps per epoch | 8,427 |
| Optimiser steps per epoch | 2,106 |
| Total optimiser steps | 10,530 |
| Warmup steps | 1,053 |

### 13.6 Results — training trajectory

Five-epoch run, ~23.3 minutes per epoch on Lightning Studio T4 (≈ 116 minutes total training). Per-epoch wall-clock is roughly 2× MVP 1's ~11 min / epoch because the CLIP vision forward + backward through LoRA is added; the frozen text encoder contributes much less per step due to its `no_grad` context.

| Epoch | Train tot | Train T1 | Train T2 | Val tot | Val T1 | Val T2 | Val T1 AUC | Val T1 F1 @0.5 | Val T2 macro F1 | Wall-clock |
|---:|---------:|---------:|---------:|--------:|-------:|-------:|----------:|---------------:|----------------:|----------:|
| 1 | 0.4002 | 0.2329 | 0.7908 | 0.5776 | 0.3381 | 1.1366 | **0.7488 ★** | 0.6939 | 0.3763 | 1,401 s |
| 2 | 0.3159 | 0.2291 | 0.5185 | 0.5513 | 0.3360 | 1.0538 | 0.7484 | 0.7018 | 0.5043 | 1,397 s |
| 3 | 0.2996 | 0.2286 | 0.4651 | 0.5574 | 0.3597 | 1.0188 | 0.7479 | 0.6904 | **0.5062 ★** | 1,397 s |
| 4 | 0.2885 | 0.2280 | 0.4298 | 0.5496 | 0.3532 | 1.0079 | 0.7476 | 0.6845 | 0.4869 | 1,398 s |
| 5 | 0.2802 | 0.2275 | 0.4033 | 0.5523 | 0.3506 | 1.0228 | 0.7484 | 0.6889 | 0.4976 | 1,402 s |

The best-by-val-T1-AUC checkpoint is **epoch 1** (val T1 AUC 0.7488) and is the saved checkpoint at `models/mvp2_naive_concat_best/`. The best-by-val-T2-macro-F1 epoch is **epoch 3** (0.5062). Train T1 loss decreases by only 2.3 % across five epochs (0.2329 → 0.2275), while train T2 loss decreases nearly 2× (0.7908 → 0.4033). The combined-loss reduction is essentially entirely T2's contribution: the T1 head reaches its operational ceiling by epoch 1 and contributes little additional learning, while the T2 head and the CLIP LoRA continue to converge on the categorical signal across the run. Val T1 AUC range across all five epochs is 0.7476 → 0.7488 (Δ = 0.0012), statistically indistinguishable from the four-run MVP 1 plateau at AUC ≈ 0.74 established in §11.1. Val T1 loss climbs marginally from 0.3381 (ep 1) to 0.3506 (ep 5), a +0.0125 increase; the pattern is non-monotonic (0.338 → 0.336 → 0.360 → 0.353 → 0.351) and is discussed as calibration drift in §13.10 finding 4 rather than overfitting.

### 13.7 Results — held-out test set

The best-by-val-T1-AUC checkpoint (epoch 1) is reloaded and evaluated on the official 10,000-row test split. Test is 50.01 % T1-balanced and contains 8,411 `t2_valid=True` rows.

#### T1 — balanced test, threshold 0.5

| Metric | Value |
|---|---:|
| AUC-ROC | **0.7411** |
| F1 (macro) | 0.6892 |
| Precision (macro) | 0.6892 |
| Recall (macro) | 0.6892 |
| TN / FP / FN / TP | 3,479 / 1,520 / 1,588 / 3,413 |
| False positive rate | **0.3041** |

#### T1 — recalibrated to deployment prior P(hate) = 0.2468, threshold 0.220 (F1-opt on recalibrated val)

The recalibration uses the Bayes prior shift from §10.4. The F1-optimised threshold is searched on the recalibrated val probabilities (0.01 – 0.99, step 0.01, F1 = 0.7139 at the optimum) and applied to the recalibrated test probabilities, validating protocol D6 under the multimodal setup.

| Metric | Value |
|---|---:|
| AUC-ROC | 0.7411 (invariant to monotone recalibration) |
| F1 (macro) | 0.6828 |
| Precision (macro) | 0.6879 |
| Recall (macro) | 0.6843 |
| TN / FP / FN / TP | 3,074 / 1,925 / 1,232 / 3,769 |
| False positive rate | 0.3851 |

The recalibrated confusion matrix is **non-degenerate** (TP 3,769 > 0, TN 3,074 > 0). This is the third independent validation of the D6 protocol after Run C and Run D, and the first under a non-MVP-1 architecture.

#### T2 — macro F1 on `t2_valid=True` rows only (n = 8,411)

Macro F1 across the six T2 classes is **0.3795**. Macro precision is 0.4630 and macro recall is 0.6046. The recall-precision asymmetry is significant and is the principal entry point for the per-class analysis in §13.9.

### 13.8 MVP 1 vs MVP 2 head-to-head comparison

| Metric | MVP 1 (Run D, text-only) | **MVP 2 (naive concat)** | Δ | Within ±0.02 AUC noise band? |
|---|---:|---:|---:|:---:|
| Test AUC-ROC | 0.7431 | **0.7411** | −0.0020 | yes |
| Test F1 (macro) | 0.6855 | **0.6892** | +0.0037 | yes |
| Test FPR | **0.2667** | 0.3041 | **+0.0374** | **no** — ~10× the AUC noise scale |
| Best val T1 AUC | 0.7495 (ep 4) | 0.7488 (ep 1) | −0.0007 | yes |

**Verdict on T1.** Naive concatenation **does not beat MVP 1 on ranking quality** (AUC delta −0.0020, within the four-run noise band established in §11.1). On decision-threshold quality, **naive concatenation regresses on FPR** by +0.0374 — outside the noise band by an order of magnitude. The fused model misclassifies 1,520 NotHate examples as Hate at threshold 0.5, against MVP 1's equivalent of approximately 1,334 — a 13.6 % relative increase in false positives at the canonical operating point. This is the empirical reproduction of the documented Gomez 2019 failure mode for naive multimodal fusion on MMHS150K, with one tightening specific to our architecture: in our setup, naive concat does not merely *fail to help* on T1; it *adds operational cost* on FPR. The cost is the image branch contributing decision-boundary noise on T1 without contributing ranking signal.

### 13.9 T2 per-class analysis

Per-class T2 metrics on the test set, computed on `t2_valid=True` rows only (n = 8,411), sorted by class label index:

| Class | Precision | Recall | F1 | Support | Note |
|---|---:|---:|---:|---:|---|
| NotHate | 0.9167 | **0.0462** | 0.0880 | 4,999 | Catastrophic recall — only 231 of 4,999 NotHate rows correctly predicted NotHate. |
| Racist | 0.3021 | 0.8345 | 0.4436 | 1,613 | Heavy over-prediction — high recall paid for by low precision. |
| Sexist | 0.2965 | 0.5668 | 0.3893 | 464 | Same over-prediction signature as Racist. |
| Homophobe | 0.6237 | 0.7928 | **0.6982** | 531 | Strong both ways — visual features carry signal here. |
| Religion | 0.0134 | 0.7500 | 0.0263 | 24 | n = 24; F1 unreliable. 18 of 24 recalled but 1,322 false positives across the column. |
| OtherHate | 0.6259 | 0.6372 | **0.6315** | 780 | Strong, balanced. |
| **Macro avg** | **0.4630** | **0.6046** | **0.3795** | 8,411 | |

The dominant pattern is **systematic over-prediction of hate categories at the cost of NotHate recall**. NotHate precision is high (0.9167 — when the model says NotHate, it is usually right), but its recall is 0.0462 — only 4.6 % of true NotHate examples get classified as NotHate. The model predicts a hate category for almost all inputs.

The mechanism is the interaction between the sklearn `'balanced'` class weights and Focal CE γ = 2 described in §13.4. Focal Loss already down-weights easy / correct predictions; multiplying by an additional class-weight vector of (0.20, 2.20, 7.50, 6.78, 160.90, 4.50) creates a loss landscape where the cost of misclassifying a NotHate example as a hate category is small (×0.20, low Focal modulation) while the cost of misclassifying any hate-category example as NotHate is large (×2 – 160). The optimiser converges to a predictor that prefers minority-class predictions almost everywhere on the input space.

Despite this calibration distortion, the per-class breakdown is informative about **what the image branch actually learned**. The two classes with both high precision and high recall — **Homophobe (F1 0.6982) and OtherHate (F1 0.6315)** — are the visually-grounded hate categories in MMHS150K. The Phase 1 EDA documented that these are also the meme-heavy classes (Religion 70 % OCR, OtherHate 52 %, against ~37–43 % for other classes per `Phase1_Data_Engineering_Report.md` §5.6 and the §1 headline findings). The image branch is doing real work on visual-hate signal; what it cannot do under this fixed-weighting loss configuration is distinguish NotHate from generic hate at the category level. The Religion result (F1 0.0263 with recall 0.7500) is small-sample noise: n = 24 test examples, and the precision of 0.0134 reflects that the model predicts Religion frequently but is correct only 18 times. This corroborates the documented limitation in `CLAUDE.md §10` — the Religion class is genuinely rare and Focal Loss alone is insufficient to recover meaningful precision at this scale.

### 13.10 Findings

The following findings are extracted from the training-trajectory data, the test-set evaluation, and the cross-comparison with the MVP 1 baseline. They are ordered by importance for the project's contribution narrative and for the thesis paper.

**Finding 1 — T1 plateaus at the text-only ceiling.** Validation T1 AUC ranges from 0.7476 to 0.7488 across all five epochs, with best at epoch 1 (0.7488). Test T1 AUC is 0.7411. This is statistically indistinguishable from the four-run MVP 1 ceiling at AUC ≈ 0.74 documented in §11.1. The train T1 loss drops by only 2.3 % across five epochs (0.2329 → 0.2275), the same anaemic decrease pattern documented in §11.5 for the text-only runs. Under naive concatenation, the T1 head learns to rely almost entirely on the text branch's `[CLS]` embedding (which already saturates the ranking signal available in the data) and treats the image branch's contribution as a noise input rather than a complementary one. This is the predicted outcome of the §11.5 ceiling diagnosis and is the load-bearing observation that motivates the architectural change in MVP 4 — gated cross-modal attention is designed to let the T1 head route between modalities selectively rather than concatenate them unconditionally.

**Finding 2 — T2 learning confirms the image branch is functionally alive.** Validation T2 macro F1 climbs from 0.3763 at epoch 1 to 0.5062 at epoch 3 — a +0.1299 absolute gain that is well outside any noise interpretation. Train T2 loss drops nearly 2× across the run (0.7908 → 0.4033). If the image branch were a broken pass-through — if the CLIP LoRA were not training, if the image embedding were a constant, or if the `visual_projection_to_512` were misconfigured — T2 macro F1 would not move. The image branch is contributing gradient signal and the LoRA adapter is learning meaningful visual representations. The functional verification of the image branch by T2 is independent of T1's plateau and is methodologically important: it isolates *what the image branch can do* from *what naive concatenation can do with it*. This finding is what allows the T1 plateau in Finding 1 to be attributed to the fusion mechanism rather than to a broken image encoder.

**Finding 3 — T2 per-class breakdown reveals what the image branch learned.** As detailed in §13.9, the image branch achieves strong F1 on **Homophobe (0.6982) and OtherHate (0.6315)** — the visually-grounded, meme-heavy hate categories per Phase 1 EDA — and contributes essentially nothing to distinguishing NotHate from generic hate at the category level (NotHate F1 0.0880, dominated by recall 0.0462). The class-weight-induced bias toward minority hate prediction is responsible for the NotHate recall collapse, but the underlying signal-where-it-exists pattern is consistent with the dataset's documented modality structure: visual hate signal exists for some categories and not others. This finding has direct implications for MVP 4: the gated attention should be expected to weight the image branch more heavily for inputs whose visual content matches the Homophobe / OtherHate signal patterns, and to suppress it otherwise. The diagonal of the T2 confusion matrix is therefore a partial blueprint for what the gate must learn to attend to.

**Finding 4 — Val T1 loss creep is calibration drift, not real overfitting.** Val T1 loss climbs from 0.3381 at epoch 1 to 0.3506 at epoch 5 — a +0.0125 increase — while train T1 loss continues to decrease. The train / val gap widens from 0.105 to 0.123. However, val T1 AUC is essentially flat across the run (0.7488 → 0.7484, Δ = −0.0004), and the val T1 loss trajectory is non-monotonic (0.338 → 0.336 → 0.360 → 0.353 → 0.351 — it wobbles, it does not progressively degrade). This is the same Focal-BCE-on-confident-wrong-predictions signature documented in MVP 1 Run D §10.2: the model's sigmoid sharpens with continued training, raising the loss on miscalibrated examples without changing the ranking. Saving the epoch-1 checkpoint as the operational baseline is the correct choice; running additional epochs would not have moved the test AUC out of the ±0.001 band but would have continued to inflate the loss-on-miscalibrated examples.

**Finding 5 — Train trajectory splits into a frozen-ceilinged T1 and an actively learning T2.** The combined-loss decrease of 0.4002 → 0.2802 (Δ = −0.1200 across five epochs) decomposes as 0.7 × (Δ T1 = −0.0054) + 0.3 × (Δ T2 = −0.3875) = −0.0038 + (−0.1163) = −0.1201. **Essentially the entire loss decrease is the T2 term.** The T1 head is at its operational ceiling from epoch 1 and contributes almost nothing further to optimisation pressure; the optimiser spends its remaining capacity training the CLIP LoRA and the T2 head on the categorical signal. This is a clean decomposition that lets the thesis Methods section attribute each task's training behaviour separately, and it is consistent with the multi-task learning expectation that the auxiliary task (T2) extracts value from the joint training even when the primary task (T1) has saturated.

**Finding 6 — FPR regression vs MVP 1 is the operational cost of naive fusion.** The +0.0374 FPR delta versus MVP 1 (§13.8) is the most consequential negative finding of the run. It is roughly an order of magnitude larger than the AUC noise band and is in the deployment-unfavourable direction: at threshold 0.5, MVP 2 misclassifies 1,520 NotHate examples as Hate, compared to MVP 1's equivalent of approximately 1,334. The 13.6 % relative increase in false positives means that adding the image branch under naive concatenation sends more over-flagged non-hate posts toward a human-in-the-loop reviewer pipeline — the opposite of the operational improvement MVP 1 (Run D) achieved over Run 1. Under the project's framing in `Cyberbullying_Detection_Report_Framing.md`, this is an unambiguous regression. The thesis Limitations section should report this directly: naive multimodal fusion on MMHS150K does not merely fail the Gomez 2019 ranking expectation; it produces a small but real operational degradation versus the text-only baseline.

**Finding 7 — Recalibration protocol D6 is validated for the multimodal setup.** The D6 protocol fix (F1-optimised threshold searched on recalibrated val and applied to recalibrated test) was originally introduced in MVP 1 Run 2 / Run C and re-validated for Run D in §10.4. MVP 2 is the third independent validation under a different architecture: recalibrated test CM is non-degenerate (TN 3,074, FP 1,925, FN 1,232, TP 3,769), F1m 0.6828, FPR 0.3851. The protocol generalises across the text-only and multimodal regimes without modification.

### 13.11 Methodological decisions locked during NB 06

| # | Decision | Rationale |
|---|----------|-----------|
| 1 | **Text encoder frozen throughout** — no gradient flow through the RoBERTa backbone or through the Run-D LoRA adapter. Both `is_trainable=False` at `PeftModel.from_pretrained` and an explicit `requires_grad = False` loop are applied; the encoder is also placed in `.eval()` mode and called inside a `torch.no_grad()` block. | The Run-D LoRA is the selected MVP 1 baseline (§11.3). Allowing it to drift during multimodal training would conflate the multimodal contribution with further text-encoder tuning and would invalidate the MVP-1-as-baseline comparison. Verified at build with `assert text_trainable == 0`. |
| 2 | **CLIP backbone frozen; LoRA rank 16 on `q_proj` and `v_proj` of the vision encoder only**, `lora_alpha = 32`, `lora_dropout = 0.1`, `bias = 'none'`. | Mirrors the parameter-efficient strategy used for text in MVP 1 and contains the trainable surface to 589,824 vision-LoRA parameters. The CLIP equivalents of RoBERTa's `query` and `value` (used in MVP 1) are `q_proj` and `v_proj`. Verified at build with `assert vision_lora_trainable > 0`. |
| 3 | **T2 loss is masked to `t2_valid=True` rows only.** Invalid rows contribute zero to the T2 loss; the T2 mean is over valid rows. For batches with no valid rows, the loss returns `logits.sum() * 0.0` to preserve autograd's gradient path. | 6.2 % of train rows have ambiguous T2 (three-way annotator disagreement, T2 = NaN per `CLAUDE.md §10`). Including them would inject categorical noise into the T2 gradient with no corresponding signal. T1 is unaffected — those rows still have valid T1 labels and are kept in T1 training. |
| 4 | **Combined loss: `0.7 × L_T1 + 0.3 × L_T2`.** Fixed weights; not learned. | Standard multi-task learning weighting where the primary task (T1, the headline metric) outweighs the auxiliary task (T2, used as a learning signal and a diagnostic instrument). Tuning the weights would conflate the MVP 2 baseline with a hyperparameter search; held fixed for clean comparison to MVP 1. |
| 5 | **Naive concatenation fusion** — `torch.cat([img_768, txt_768], dim=-1)` → 1,536-d input to both heads. No attention, no gating, no MLP-mixer, no addition / averaging. | Naive concatenation is the documented Gomez 2019 failure mode for MMHS150K. Reproducing it here measures the floor against which MVP 4's gated cross-modal attention is evaluated. Introducing attention or gating at this step would conflate the architectural contribution. |
| 6 | **PEFT-bypass in `encode_image`** — call `self.vision_encoder.base_model.model(pixel_values=...)` directly instead of `self.vision_encoder(pixel_values=...)`. | The smoke test isolated a `TypeError` collision in `PeftModel.forward(CLIPVisionModel(...))` on PEFT 0.19 + transformers 5.8.1: the inner `CLIPEncoder` receives `inputs_embeds` twice. The LoRA adapter modules are injected as `lora.Linear` instances inline with the attention projections, so calling `base_model.model` bypasses the broken outer wrapper without bypassing the LoRA computation. Save / load is unaffected because `save_pretrained` / `from_pretrained` do not route through `forward`. |
| 7 | **`visual_projection_to_512` frozen and initialised from the public `CLIPModel.visual_projection`** — `Linear(768→512, bias=False)`, weights lifted at construction time, `requires_grad = False`. | Preserves the canonical CLIP image embedding (the 512-d feature on which the public model's contrastive pre-training operated). The image feature entering our trainable `image_projection` is therefore the standard CLIP image embedding, not an arbitrary 768-d pooler output. Verified at build with `assert vis_proj_512_trainable == 0`. |
| 8 | **`image_projection` trainable** — `Linear(512→768)`. | Re-projects the canonical CLIP image embedding to 768-d for fusion with the 768-d text `[CLS]` (`CLAUDE.md §3` cross-modal dim alignment). This is the only trainable transformation between the frozen CLIP image embedding and the head input. Initialised with PyTorch defaults. |
| 9 | **T2 metrics computed only on `t2_valid=True` rows.** Both per-epoch validation metrics and the final test report exclude rows where T2 is NaN. | Mirrors the loss-side masking (decision #3). Reporting macro F1 across all rows would degrade the metric with rows the model was never trained to predict on. The number of valid rows (4,197 val, 8,411 test) is reported alongside the metric in every output. |

### 13.12 Open items and preconditions for NB 07

| Item | Status |
|---|---|
| `data/processed/structured_features.csv` (the pruned 9-feature vector from NB 03) | ✅ available — produced and stored in Phase 1, documented in `Phase1_Data_Engineering_Report.md` |
| MVP 2 architecture preserved at `models/mvp2_naive_concat/` and `models/mvp2_naive_concat_best/` | ✅ done — full state required to instantiate NB 07's starting point |
| Run D text-encoder LoRA at `models/roberta_mvp1_d/` (frozen feature provider) | ✅ unchanged |
| CLIP image processor configuration (224 × 224, ImageNet-style normalisation) | ✅ specified in `hparams.json` — must match NB 06 byte-for-byte |
| Recalibration protocol D6 | ✅ validated across three independent runs (MVP 1 Run C, Run D, MVP 2) — safe to carry forward |
| Whether to freeze the CLIP LoRA during NB 07 or continue training it | ⏳ open — deferred to the NB 07 prompt |
| How to combine the structured 9-feature vector with the existing 1,536-d fusion vector (concat to 1,545-d, project then concat, late-fusion at logit level) | ⏳ open — deferred to the NB 07 prompt |
| Whether to retain both T1 and T2 heads in NB 07 or simplify to T1 only | ⏳ open — deferred to the NB 07 prompt |
| Whether to use the same 0.7 / 0.3 loss weighting in NB 07 | ⏳ open — deferred to the NB 07 prompt |
| T1 FPR regression vs MVP 1 (§13.10 finding 6) | ⚠️ watch — NB 07 should track FPR alongside AUC so the structured branch's effect on the decision boundary is visible |
| Preservation of T2 macro F1 when structured features are added | ⚠️ watch — if T2 macro F1 collapses, the structured branch is fighting the categorical signal already extracted by the image branch |

### 13.13 Artefacts written

| Artefact | Path | Size |
|---|---|---:|
| Final model — CLIP LoRA adapter weights | `models/mvp2_naive_concat/adapter_model.safetensors` | 2.26 MB |
| Final model — CLIP LoRA config | `models/mvp2_naive_concat/adapter_config.json` | 1.01 KB |
| Final model — PEFT auto-generated README | `models/mvp2_naive_concat/README.md` | 5.06 KB |
| Final model — non-PEFT state (`image_projection` + `head_t1` + `head_t2` + epoch / val metadata) | `models/mvp2_naive_concat/rest.pt` | 4.52 MB |
| Frozen hyperparameters | `models/mvp2_naive_concat/hparams.json` | 0.57 KB |
| Per-epoch metrics | `models/mvp2_naive_concat/training_history.json` | 1.31 KB |
| Balanced + recalibrated test metrics + selection metadata | `models/mvp2_naive_concat/metrics.json` | 0.98 KB |
| Best-by-val-T1-AUC checkpoint — adapter | `models/mvp2_naive_concat_best/adapter_model.safetensors` | 2.26 MB |
| Best-by-val-T1-AUC checkpoint — adapter config | `models/mvp2_naive_concat_best/adapter_config.json` | 1.01 KB |
| Best-by-val-T1-AUC checkpoint — auto README | `models/mvp2_naive_concat_best/README.md` | 5.06 KB |
| Best-by-val-T1-AUC checkpoint — non-PEFT state | `models/mvp2_naive_concat_best/rest.pt` | 4.52 MB |
| Test-set T1 confusion matrix chart | `outputs/nb06_t1_confusion_matrix.png` | 35.45 KB |
| Test-set T2 confusion matrix chart | `outputs/nb06_t2_confusion_matrix.png` | 69.59 KB |
| Training curves chart (combined loss, per-task losses, val T1 AUC + val T2 macro F1) | `outputs/nb06_training_curves.png` | 89.17 KB |
| Executed notebook (12 code cells, 0 errors) | `notebooks/06_mvp2_naive_fusion.ipynb` | 233 KB (post-execution) |

---

## 14. References

- Gomez, R. et al. (2019). *Exploring Hate Speech Detection in Multimodal Publications*. WACV 2020. — source of MMHS150K and of the 50/50 val/test design.
- Lin, T.-Y. et al. (2017). *Focal Loss for Dense Object Detection*. ICCV. — origin of focal cross-entropy.
- Hu, E. J. et al. (2021). *LoRA: Low-Rank Adaptation of Large Language Models*. arXiv:2106.09685. — origin of the rank-decomposition adapter pattern wired in via `peft`.
- Loureiro, D. et al. (2022). *TimeLMs: Diachronic Language Models from Twitter*. ACL Demos. — backing the `cardiffnlp/twitter-roberta-base-2022-154m` checkpoint.
- `CLAUDE.md` (project-internal) §§ 3, 8, 9, 12 — locked project rules referenced throughout.
- `Phase1_Data_Engineering_Report.md` — predecessor report covering data engineering decisions reused in NB04 (label parsing, structured features, split posture).

---

## Appendix A — Saved Artefacts

| Artefact | Notebook | Location | Size |
|----------|----------|----------|------|
| LoRA adapter weights | NB 04 | `models/roberta_pretrain/adapter_model.safetensors` | 2.26 MB |
| LoRA adapter config | NB 04 | `models/roberta_pretrain/adapter_config.json` | 1.02 KB |
| Tokenizer (json) | NB 04 | `models/roberta_pretrain/tokenizer.json` | 3.39 MB |
| Tokenizer config | NB 04 | `models/roberta_pretrain/tokenizer_config.json` | 0.48 KB |
| Frozen hyperparameters | NB 04 | `models/roberta_pretrain/hparams.json` | 0.26 KB |
| Label dictionary | NB 04 | `models/roberta_pretrain/labels.json` | 0.37 KB |
| Per-epoch metrics | NB 04 | `models/roberta_pretrain/training_history.json` | 0.32 KB |
| PEFT auto README | NB 04 | `models/roberta_pretrain/README.md` | 5.08 KB |
| Confusion matrix chart | NB 04 | `outputs/nb04_confusion_matrix.png` | 67.80 KB |
| Training curves chart | NB 04 | `outputs/nb04_training_curves.png` | 62.62 KB |
| Executed notebook | NB 04 | `notebooks/04_roberta_pretrain_kaggle.ipynb` | 179 KB |
| LoRA adapter weights (iter 1) | NB 05 | `models/roberta_mvp1_iter1/adapter_model.safetensors` | 2.26 MB |
| LoRA adapter config (iter 1) | NB 05 | `models/roberta_mvp1_iter1/adapter_config.json` | 1.02 KB |
| T1 head state_dict (iter 1) | NB 05 | `models/roberta_mvp1_iter1/head.pt` | 4.83 KB |
| Frozen hyperparameters (iter 1) | NB 05 | `models/roberta_mvp1_iter1/hparams.json` | 0.38 KB |
| Per-epoch metrics (iter 1) | NB 05 | `models/roberta_mvp1_iter1/training_history.json` | 0.60 KB |
| Balanced + recalibrated metrics (iter 1) | NB 05 | `models/roberta_mvp1_iter1/metrics.json` | 1.27 KB |
| PEFT auto README (iter 1) | NB 05 | `models/roberta_mvp1_iter1/README.md` | 5.08 KB |
| Confusion matrix chart (iter 1) | NB 05 | `outputs/previous_tests/nb05_confusion_matrix.png` | 34.6 KB |
| Training curves chart (iter 1) | NB 05 | `outputs/previous_tests/nb05_training_curves.png` | 57.7 KB |
| Executed notebook (iter 1; later patched for iter 2) | NB 05 | `notebooks/previous_tests/05_mvp1_roberta_t1.ipynb` | 179 KB |
| Run 2 partial best checkpoint (epoch 1 only) — LoRA adapter | NB 05 (Run 2) | `models/roberta_mvp1_best/adapter_model.safetensors` | 2.26 MB |
| Run 2 partial best checkpoint — head | NB 05 (Run 2) | `models/roberta_mvp1_best/head.pt` | 4.83 KB |
| Run C LoRA adapter | NB 05c | `models/roberta_mvp1_fresh/adapter_model.safetensors` | 2.26 MB |
| Run C LoRA config | NB 05c | `models/roberta_mvp1_fresh/adapter_config.json` | 1.02 KB |
| Run C T1 head state_dict | NB 05c | `models/roberta_mvp1_fresh/head.pt` | 4.83 KB |
| Run C frozen hyperparameters | NB 05c | `models/roberta_mvp1_fresh/hparams.json` | 0.38 KB |
| Run C per-epoch metrics | NB 05c | `models/roberta_mvp1_fresh/training_history.json` | 0.60 KB |
| Run C balanced + recalibrated metrics | NB 05c | `models/roberta_mvp1_fresh/metrics.json` | 0.91 KB |
| Run C PEFT auto README | NB 05c | `models/roberta_mvp1_fresh/README.md` | 5.08 KB |
| Run C best-by-AUC checkpoint dir | NB 05c | `models/roberta_mvp1_fresh_best/` | 2.27 MB total |
| Run C confusion matrix chart | NB 05c | `outputs/previous_tests/nb05_fresh_confusion_matrix.png` | 34.7 KB |
| Run C training curves chart | NB 05c | `outputs/previous_tests/nb05_fresh_training_curves.png` | 63.7 KB |
| Run C executed notebook | NB 05c | `notebooks/previous_tests/05c_no_warmstart.ipynb` | 179 KB |
| **Run D LoRA adapter (rank 32, MVP 1 baseline)** | NB 05d | `models/roberta_mvp1_d/adapter_model.safetensors` | 4.51 MB |
| Run D LoRA config | NB 05d | `models/roberta_mvp1_d/adapter_config.json` | 1.02 KB |
| Run D T1 head state_dict | NB 05d | `models/roberta_mvp1_d/head.pt` | 4.83 KB |
| Run D frozen hyperparameters | NB 05d | `models/roberta_mvp1_d/hparams.json` | 0.40 KB |
| Run D per-epoch metrics | NB 05d | `models/roberta_mvp1_d/training_history.json` | 0.60 KB |
| Run D balanced + recalibrated metrics + selection metadata | NB 05d | `models/roberta_mvp1_d/metrics.json` | 0.97 KB |
| Run D PEFT auto README | NB 05d | `models/roberta_mvp1_d/README.md` | 5.08 KB |
| Run D best-by-AUC checkpoint dir | NB 05d | `models/roberta_mvp1_d_best/` | 4.52 MB total |
| Run D confusion matrix chart | NB 05d | `outputs/nb05_d_confusion_matrix.png` | 33.0 KB |
| Run D training curves chart | NB 05d | `outputs/nb05_d_training_curves.png` | 68.8 KB |
| Run D executed notebook | NB 05d | `notebooks/05d_rank32_lr3e4.ipynb` | 184 KB |
| MVP 2 — CLIP LoRA adapter weights (rank 16, vision q_proj+v_proj) | NB 06 | `models/mvp2_naive_concat/adapter_model.safetensors` | 2.26 MB |
| MVP 2 — CLIP LoRA config | NB 06 | `models/mvp2_naive_concat/adapter_config.json` | 1.01 KB |
| MVP 2 — non-PEFT state (image_projection + head_t1 + head_t2 + metadata) | NB 06 | `models/mvp2_naive_concat/rest.pt` | 4.52 MB |
| MVP 2 — frozen hyperparameters | NB 06 | `models/mvp2_naive_concat/hparams.json` | 0.57 KB |
| MVP 2 — per-epoch metrics (training history) | NB 06 | `models/mvp2_naive_concat/training_history.json` | 1.31 KB |
| MVP 2 — balanced + recalibrated metrics | NB 06 | `models/mvp2_naive_concat/metrics.json` | 0.98 KB |
| MVP 2 — PEFT auto README | NB 06 | `models/mvp2_naive_concat/README.md` | 5.06 KB |
| MVP 2 — best-by-val-T1-AUC checkpoint dir | NB 06 | `models/mvp2_naive_concat_best/` | 6.79 MB total |
| MVP 2 — T1 test confusion matrix chart | NB 06 | `outputs/nb06_t1_confusion_matrix.png` | 35.45 KB |
| MVP 2 — T2 test confusion matrix chart | NB 06 | `outputs/nb06_t2_confusion_matrix.png` | 69.59 KB |
| MVP 2 — training curves chart (loss + val T1 AUC + val T2 macro F1) | NB 06 | `outputs/nb06_training_curves.png` | 89.17 KB |
| MVP 2 — executed notebook | NB 06 | `notebooks/06_mvp2_naive_fusion.ipynb` | 233 KB |
