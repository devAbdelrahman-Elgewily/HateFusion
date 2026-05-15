# 💬 Multimodal Cyberbullying & Online Hate Speech Detection
### Final Year Project — Computer Science (AI Major)
### Full Project Scope & Technical Analysis  *(v1.1 — revised after technical critique)*

---

## Table of Contents

1. [Executive Summary](#1-executive-summary)
2. [Problem Statement](#2-problem-statement)
3. [Project Objectives](#3-project-objectives)
4. [Datasets](#4-datasets)
5. [Data Types & Modalities](#5-data-types--modalities)
6. [Target Features](#6-target-features)
7. [System Architecture](#7-system-architecture)
8. [Module Breakdown](#8-module-breakdown)
9. [AI & ML Concepts Used](#9-ai--ml-concepts-used)
10. [Input & Output Specification](#10-input--output-specification)
11. [Technology Stack](#11-technology-stack)
12. [Project Phases & Timeline](#12-project-phases--timeline)
13. [Evaluation Metrics](#13-evaluation-metrics)
14. [Ablation Study Plan](#14-ablation-study-plan)
15. [Explainability (XAI)](#15-explainability-xai)
16. [Deliverables](#16-deliverables)
17. [Risks & Mitigations](#17-risks--mitigations)
18. [Academic Contribution](#18-academic-contribution)
19. [Ethics & Responsible AI](#19-ethics--responsible-ai)

---

## 1. Executive Summary

This project builds an end-to-end multimodal AI system that detects cyberbullying and hate
speech in social media posts by fusing three fundamentally different data modalities:
**post images** (memes, photos, screenshots), **post text content** (tweet/caption text), and
**structured contextual features** (text-in-image OCR, hashtag patterns, URL presence,
post metadata).

The system addresses a genuine open research problem: existing single-modality hate speech
detectors miss attacks where the text alone seems neutral but the combination with an image
becomes harmful — and vice versa. Critically, the seminal paper that introduced the
MMHS150K dataset reported that current multimodal fusion models *failed to outperform*
text-only baselines. Recent 2024–2025 work has begun to close this gap with better attention
mechanisms, but it remains an active and unsolved problem.

The core contribution is a **gated cross-modal attention fusion layer** that learns when each
modality matters — recognising that some hate speech is text-driven, some image-driven, and
some only emerges from the *combination*. The system outputs four simultaneous predictions
(binary hate flag, hate category, attack severity, modality importance) with full
explainability, and is deployed as an interactive web dashboard.

---

## 2. Problem Statement

Cyberbullying and online hate speech cause measurable real-world harm. Studies link
sustained exposure to hate content with depression, anxiety, and suicidal ideation —
particularly in adolescents. Major platforms (X/Twitter, Instagram, Facebook) handle billions
of posts per day and rely heavily on AI moderation, yet existing systems have well-documented
failure modes.

**The fundamental gap:** most hate speech detectors operate on text alone. But modern hate
speech is increasingly multimodal:

- A neutral text like "look at this beauty" becomes harmful when paired with a degrading image
- A meme image alone may be ambiguous, but with overlaid text becomes targeted hate
- Coded language (dog whistles, emojis) gains meaning only with visual context
- Screenshots of conversations weaponise text quoted from elsewhere

Single-modality models miss all four scenarios. Yet — and this is the genuinely open research
question — the original MMHS150K paper demonstrated that naive multimodal fusion *also fails*
to outperform text-only models. The image and text signals fight each other rather than
complement each other.

**The central question:**

> Can a gated cross-modal attention architecture — that learns *when* each modality matters
> rather than always combining them equally — outperform both unimodal baselines and naive
> multimodal fusion on hate speech detection across multiple datasets?

---

## 3. Project Objectives

| # | Objective |
|---|-----------|
| O1 | Build a Vision Transformer (ViT-B/16) image encoder for post images and memes |
| O2 | Build a BERT/RoBERTa-based text encoder fine-tuned on hate speech corpora |
| O3 | Build a structured ML module for OCR text, hashtag/URL/mention features, and post metadata |
| O4 | Design a **gated cross-modal attention fusion layer** that learns modality importance dynamically |
| O5 | Implement multi-task learning across four simultaneous prediction heads |
| O6 | Conduct ablation studies comparing unimodal baselines, naive fusion, and gated fusion |
| O7 | Provide full explainability: GRAD-CAM attention maps, BERT token attribution, gate weights |
| O8 | Deploy an interactive moderation-assistant dashboard with confidence scores and reasoning |

---

## 4. Datasets

### 4.1 MMHS150K — Primary Dataset
- **URL:** https://gombru.github.io/2019/10/09/MMHS/
- **License:** Public, free for academic research
- **Size:** **150,000 tweets**, each with image + text + manual annotations
- **Source:** Real Twitter posts collected Sep 2018 – Feb 2019, sampled using 51 Hatebase terms
- **Annotation:** **3 human annotators per tweet** via Amazon Mechanical Turk
- **Labels:** 6-class classification per tweet:
  - No attacks to any community
  - Racist
  - Sexist
  - Homophobic
  - Religion-based attack
  - Attack to other community
- **Pre-extracted features included:**
  - Image text via OCR (Google Vision API output)
  - Tweet text (cleaned)
  - Multi-annotator labels (we can compute agreement scores)
- **Format:** JSON metadata + JPEG images (6 GB total)
- **Why it's the right dataset:** Largest manually annotated multimodal hate speech dataset.
  Real social media data, not synthetic. Multi-annotator labels enable severity/confidence
  derivation. Active benchmark in 2024–2025 papers.

### 4.2 Hateful Memes Dataset — Secondary Validation
- **URL:** https://hatefulmemeschallenge.com / https://ai.facebook.com/datasets/hateful-memes
- **License:** Free for research, requires acceptance of terms (instant)
- **Size:** 10,000 multimodal memes (text + image)
- **Source:** Created by Facebook AI Research specifically for hate speech research
- **Annotation:** Expert annotators with multi-stage validation
- **Labels:** Binary hateful/not-hateful + meme template metadata
- **Why it's included:** Cross-dataset validation. If our model trained on MMHS150K
  generalises to Hateful Memes (different distribution, different platform style), that
  proves robustness. Strong claim for the final report.
- **Special property:** Designed to be hard for unimodal models — text-only and image-only
  baselines deliberately perform poorly. Forces genuine multimodal reasoning.

### 4.3 Cyberbullying Detection Dataset (Kaggle) — Text-Only Pre-Training
- **URL:** https://www.kaggle.com/datasets/andrewmvd/cyberbullying-classification
- **License:** Open data
- **Size:** 47,000 tweets labeled by cyberbullying type (age, ethnicity, gender, religion)
- **Role:** Pre-train BERT branch on this corpus before fine-tuning on MMHS150K — gives the
  text encoder strong domain knowledge before multimodal training begins

### 4.4 Dataset Integration Strategy

The three datasets serve different roles, no fuzzy joins needed:

```
MMHS150K (primary training)
    ├── 150K tweets — train all 4 targets here
    ├── 80% train / 10% val / 10% test split
    └── Stratified by 6-class label

Hateful Memes (cross-domain test)
    └── Held-out evaluation only — never trained on
       Tests whether model generalises beyond Twitter

Cyberbullying Kaggle (text branch warm-start)
    └── Used only to pre-train BERT before MMHS150K
       Strengthens text encoder before multimodal fine-tuning
```

**Final unified training set:** 150,000 multimodal posts (MMHS150K) split 80/10/10.
**Final cross-domain test:** 10,000 Hateful Memes for generalisation claims.

---

## 5. Data Types & Modalities

| Modality | Source | Format | Notes |
|----------|--------|--------|-------|
| **Images** (unstructured) | MMHS150K image archive | JPEG, varied dimensions (resize to 224×224) | Memes, photos, screenshots, infographics |
| **Text** (unstructured) | Tweet text field in MMHS150K JSON | Raw string, typically 5–50 words | Real Twitter language — slang, emojis, hashtags |
| **Structured** (tabular) | Engineered from raw posts | ~15 numeric/categorical features | OCR text features, hashtag/URL counts, metadata |

---

## 6. Target Features

| # | Target | Type | Classes / Range | Label Source |
|---|--------|------|-----------------|-------------|
| T1 | **Hate Speech Flag** | Binary classification | Hate / Not Hate | Direct from MMHS150K — collapse 5 hate classes vs "no attack" |
| T2 | **Hate Category** | Multi-class classification | Racist / Sexist / Homophobic / Religion / Other / None | Direct from MMHS150K — 6 classes from majority vote |
| T3 | **Annotator Agreement Score** | Regression 0.0–1.0 | Continuous | **Direct from MMHS150K** — fraction of 3 annotators who agreed; proxy for severity/clarity |
| ~~T4~~ | ~~Modality Importance~~ | **REMOVED as supervised target** | — | Inferred post-hoc from learned gate weights during analysis, not used for training |

**Why T4 was removed:** Originally proposed as a supervised target derived from baseline model
confidence thresholds, T4 had a circular dependency problem — labels would have been derived
from model outputs themselves, creating confirmation bias and pseudo-ground-truth leakage. The
academically defensible approach is to *infer* modality reliance from the learned gate
distributions of the final model, post-hoc, as analytical output rather than training signal.

**Label derivation notes (for remaining targets):**

- **T1 and T2** are direct from MMHS150K's 3-annotator majority vote. This is real
  human-labeled ground truth — no fabrication.
- **T3** uses inter-annotator agreement. When all 3 annotators agree, agreement = 1.0.
  When 2 of 3 agree, agreement = 0.67. When all disagree, agreement = 0.33. This is a real
  signal — low agreement often correlates with subtle/borderline cases.
- **T3 is treated as auxiliary**, not a primary objective — see Phased Training below.

---

## 6b. Phased Training Strategy — Multi-Task Risk Mitigation

Training T1, T2, and T3 simultaneously from day one risks gradient conflicts. Instead, follow
a staged training plan:

| MVP Stage | Tasks Active | Deliverable |
|-----------|--------------|-------------|
| MVP 1 | T1 only (binary hate)                  | Working classifier, baseline metrics |
| MVP 2 | T1 + T2 (binary + 6-class category)    | Multi-class classification working |
| MVP 3 | T1 + T2 with naive fusion              | Replicating MMHS150K paper failure mode |
| MVP 4 | T1 + T2 with gated fusion              | Core contribution validated |
| MVP 5 | + T3 + ablations + bias analysis       | Complete project |

**Survivability principle:** if any MVP stage breaks, the project is still finishable at the
previous stage. MVP 1 alone is a valid (if minimal) submission.

---

## 7. System Architecture

```
╔════════════════════════════════════════════════════════════════════╗
║                          INPUT LAYER                               ║
║  ┌──────────────────┐  ┌───────────────────┐  ┌─────────────────┐ ║
║  │  Post Image      │  │  Post Text        │  │  Structured     │ ║
║  │  (224×224 RGB    │  │  (tweet content,  │  │  Features       │ ║
║  │   meme/photo/    │  │   typically 5-50  │  │  (~15 cols)     │ ║
║  │   screenshot)    │  │   words)          │  │                 │ ║
║  └────────┬─────────┘  └─────────┬─────────┘  └────────┬────────┘ ║
╚═══════════╪═════════════════════╪═════════════════════╪══════════╝
            │                     │                     │
            ▼                     ▼                     ▼
╔══════════════════╗  ╔════════════════════════╗  ╔═════════════════╗
║   BRANCH A       ║  ║   BRANCH B             ║  ║   BRANCH C      ║
║   ViT-B/16       ║  ║   RoBERTa Fine-Tuned   ║  ║   XGBoost +     ║
║                  ║  ║   on Cyberbullying     ║  ║   TabNet        ║
║ Pretrained:      ║  ║   Corpus               ║  ║                 ║
║ ImageNet-21K     ║  ║                        ║  ║ Features:       ║
║                  ║  ║ Stages:                ║  ║ - OCR text len  ║
║ Learns:          ║  ║ 1. Pre-train on        ║  ║ - hashtag count ║
║ - Visual         ║  ║    Cyberbully Kaggle   ║  ║ - URL present   ║
║   targeting cues ║  ║ 2. Fine-tune on        ║  ║ - mention count ║
║ - Symbol/icon    ║  ║    MMHS150K            ║  ║ - emoji count   ║
║   recognition    ║  ║                        ║  ║ - all_caps frac ║
║ - Meme template  ║  ║ Learns:                ║  ║ - text length   ║
║   recognition    ║  ║ - Coded language       ║  ║ - OCR present   ║
║ - Image text     ║  ║ - Slurs in context     ║  ║ - exclamations  ║
║   regions        ║  ║ - Sarcasm/irony        ║  ║ - sentiment     ║
║                  ║  ║ - Hashtag semantics    ║  ║   polarity      ║
║ Output: 768-d    ║  ║                        ║  ║                 ║
║ patch tokens +   ║  ║ Output: 768-d [CLS]    ║  ║ Output: 256-d   ║
║ [CLS] embed      ║  ║ + token-level embeds   ║  ║ embedding       ║
╚════════╤═════════╝  ╚═══════════╤════════════╝  ╚════════╤════════╝
         │                        │                        │
         └────────────────────────┼────────────────────────┘
                                  │
                  ╔═══════════════▼══════════════════╗
                  ║   GATED CROSS-MODAL ATTENTION    ║
                  ║                                  ║
                  ║  Step 1: Compute per-modality    ║
                  ║          gate weights g_A, g_B,  ║
                  ║          g_C ∈ [0, 1]            ║
                  ║                                  ║
                  ║  Step 2: Weighted cross-attention║
                  ║          between modalities      ║
                  ║                                  ║
                  ║  Step 3: Gates learn WHICH       ║
                  ║          modality matters per    ║
                  ║          sample (the novelty)    ║
                  ╚═══════════════╤══════════════════╝
                                  │
                  ╔═══════════════▼══════════════════╗
                  ║   SHARED REPRESENTATION          ║
                  ║   FC(1792 → 512) → ReLU          ║
                  ║   → BatchNorm → Dropout(0.3)     ║
                  ╚═══╤═══════════╤═══════╤══════════╝
                      │           │       │
                      ▼           ▼       ▼
          ┌──────────────┐ ┌─────────┐ ┌──────────────────┐
          │ HEAD T1      │ │ HEAD T2 │ │ HEAD T3 + T4     │
          │ Hate Binary  │ │ 6-class │ │ Agreement (MSE)  │
          │ (Focal BCE)  │ │ (Focal  │ │ + Modality       │
          │              │ │  CE)    │ │ Importance (CE)  │
          └──────────────┘ └─────────┘ └──────────────────┘

Joint Loss:
L = λ1·FocalBCE(T1) + λ2·FocalCE(T2) + λ3·MSE(T3) + λ4·CE(T4)
λ values: [1.2, 1.0, 0.6, 0.4] — tunable via grid search
```

---

## 8. Module Breakdown

### 8.1 Data Engineering Module

**Goal:** Build the unified labeled multimodal dataset for training

- Download MMHS150K (~6 GB) from official Google Drive / Mega mirror
- Download Hateful Memes dataset (held-out test set only)
- Download Cyberbullying Kaggle dataset (text branch pre-training)
- Parse MMHS150K JSON metadata → extract tweet text, image filename, 3 annotator labels
- Compute majority vote labels for T1 (binary hate) and T2 (6-class category)
- Compute annotator agreement scores for T3
- Resize all images to 224×224 RGB; apply ImageNet normalisation
- Engineer 15 structured features per post (OCR text length, hashtag count, URL presence,
  emoji count, all-caps fraction, sentiment polarity via VADER, etc.)
- Build train/val/test split: 80/10/10 stratified by T2 class
- For Hateful Memes: minimal preprocessing — use as held-out cross-domain test set only

**Output:** ~150K records with image paths, tweet text, structured features, and 4 labels

---

### 8.2 Branch A — Vision Transformer Module

**Goal:** Extract visual cues for hate speech: targeting symbols, meme templates, image-text
regions

- **Model:** ViT-B/16 pretrained on ImageNet-21K, fine-tuned on MMHS150K images
- **Why ViT over CNN:** Hate speech images are diverse (memes, photos, screenshots,
  infographics). ViT's global attention captures the *layout relationships* between subjects
  and overlaid text — critical for memes where the joke depends on placement
- **Input:** 3-channel RGB image, 224×224, ImageNet normalised
- **Fine-tuning:** All layers unfrozen, learning rate 1e-5, linear warmup over 2 epochs
- **Augmentation:** Random horizontal flip, mild colour jitter. **No vertical flip or rotation
  — text orientation matters for meme legibility**
- **Output:** 768-d [CLS] token embedding + 196 patch tokens (used for cross-modal attention)
- **Auxiliary task during fine-tuning:** Predict T1 (binary hate) directly from ViT head before
  fusion training to give the backbone domain awareness

---

### 8.3 Branch B — RoBERTa Text Module

**Goal:** Decode hate language including slang, slurs, sarcasm, and coded references

- **Model:** `roberta-base` (120M parameters) — better than BERT-base on social media text per
  benchmark studies
- **Training pipeline:**
  1. **Pre-train** on Cyberbullying Kaggle dataset (47K tweets, 4-class) for 3 epochs.
     Gives the encoder strong familiarity with hate speech language patterns.
  2. **Fine-tune** on MMHS150K with auxiliary T1 prediction head for 5 epochs
- **Tokenisation:** Standard RoBERTa BPE tokeniser. Max sequence length 128 tokens
  (tweets are short — longer sequences waste compute)
- **Special handling:** Preserve hashtags, mentions, URLs as tokens. Do NOT remove them — they
  carry signal (e.g., trending hate hashtags are predictive)
- **Output:** 768-d [CLS] token + token-level hidden states (for token-level attention in XAI)

---

### 8.4 Branch C — Structured Feature Module *(simplified)*

**Goal:** Capture metadata and surface-level text patterns missed by deep models. Kept
intentionally lightweight — 15 features cannot compete with RoBERTa's 768-d embeddings on
their own, so this branch is treated as auxiliary enhancement, not a core branch.

**Full feature list (15 features):**

| Feature | Type | Why it matters |
|---------|------|----------------|
| Tweet text length | numeric | Very short or very long tweets have different hate patterns |
| Hashtag count | numeric | Hate often weaponises hashtags |
| Mention count | numeric | Targeted attacks have @mentions |
| URL count | binary | Links to extremist sources |
| Emoji count | numeric | Coded emoji use |
| All-caps fraction | numeric | Aggressive style marker |
| Exclamation count | numeric | Aggressive punctuation |
| Question marks | numeric | Rhetorical aggression |
| OCR text length | numeric | Text-heavy memes vs photos |
| OCR text present | binary | Has overlaid text or not |
| VADER neg sentiment | numeric | Pre-computed negative sentiment |
| VADER neu sentiment | numeric | Neutrality score |
| Repeated chars (heyyyy) | numeric | Casual / aggressive style |
| Profanity count (lexicon) | numeric | Surface-level vulgarity |
| Hate keyword count | numeric | From Hatebase lexicon match |

- **Models for ablation baseline:** XGBoost on the structured features alone (CPU, no GPU
  needed). This serves the structured-only ablation cell.
- **In the fusion architecture:** features are *not* embedded via TabNet. Standardised and
  passed as a 15-d vector directly into the fusion layer. This decision was made to:
  - Avoid TabNet adding a second source of training instability
  - Save 1–2 weeks of engineering time
  - Reflect the honest expectation that 15 features can't justify a deep tabular model
- **Preprocessing:** Standardise numeric features; no embedding for binaries

---

### 8.5 Gated Cross-Modal Attention Fusion Layer — The Novel Contribution

**Refined positioning:** This project does not invent gated fusion. The contribution is
*applying adaptive modality gating specifically to the MMHS150K failure mode and analysing
modality reliance.* This is the safe, defensible academic framing.

The original MMHS150K paper documented that naive fusion *hurt* performance. Our gated variant
addresses that specific failure by learning *when* each modality matters per sample.

**Mechanism:**

```
# Branch outputs
A_cls  = ViT [CLS] embedding         (768-d)
A_pat  = ViT patch tokens            (196 × 768)
B_cls  = RoBERTa [CLS] embedding     (768-d)
B_tok  = RoBERTa token embeddings    (128 × 768)
C_emb  = Structured features (raw)   (15-d, lightweight)

# Step 1: Compute per-modality gate weights
gate_input = concat([A_cls, B_cls, C_emb])    → (1551-d)
gate_logits = FC(1551 → 3)                     → (3,)
g_A, g_B, g_C = softmax(gate_logits)            three weights summing to 1

# Step 2: Modality-weighted cross-attention (image patches attend to text)
Q = Linear(A_pat)                               → (196, 512)
K = Linear(B_tok)                               → (128, 512)
V = Linear(B_tok)                               → (128, 512)

cross_attn_AB = softmax(Q·K^T / √512) · V       → (196, 512)
visual_text   = mean_pool(cross_attn_AB) · g_A  gated by image relevance

# Step 3: Final gated fused representation
fused = concat([
    g_A · A_cls,             # image contribution, gated
    g_B · B_cls,             # text contribution, gated
    g_C · C_emb,             # structured contribution, gated
    visual_text              # cross-modal interaction
])                                              → (1567-d)

shared_repr = FC(1567 → 512) → ReLU → BN → Dropout(0.3)
```

### 8.5b — Gate Collapse Mitigation (Critical Diagnostic)

**The biggest hidden risk in this architecture:** the gate may collapse to "90% text, 5%
image, 5% structured" for every sample, making the multimodal system effectively a text
classifier with decorative inputs.

**Required diagnostics in every training run:**

1. **Log gate entropy per epoch** — `H(gates) = -Σ g·log(g)`. If average entropy drops below
   0.5 (out of theoretical max ~1.1 for 3 modalities), gates have collapsed
2. **Plot gate weight distributions per class** — does the model use the image branch for
   image-heavy classes (memes, infographics) more than for plain text screenshots?
3. **Per-class gate statistics report** — required output of every training run

**Entropy regularisation term in the loss:**

```
L_total = L_task − λ_entropy · H(gates_batch_mean)

with λ_entropy = 0.05
```

This explicitly rewards the model for using more than one modality. Without this term, gate
collapse is the default outcome on hate speech datasets where text dominates.

**Sanity check during analysis:** if final gate distribution shows g_A < 0.15 across all
samples, the multimodal claim is not supported and the result must be reported honestly.

### 8.5c — The Honest Hypothesis

The expected, *defensible* outcome is:

- Gates will lean text-heavy on average (text is the strongest signal in hate datasets)
- But on specific subsets (memes with overlaid text, ambiguous photos with sharp captions)
  the image gate should rise meaningfully
- The contribution is documenting **when** the image modality contributes, not claiming
  it dominates everywhere

This framing protects the project from "your fusion is just a text classifier" attacks at
defense.

---

### 8.6 Multi-Task Output Heads *(simplified after T4 removal)*

```
shared_repr (512-d) feeds three heads (with phased activation):

T1 Hate Binary:        FC(512 → 128 → 1)  + Sigmoid    → [0, 1]
                        Loss: Focal BCE (γ=2) — class imbalance
                        Active from MVP 1

T2 Hate Category:      FC(512 → 128 → 6)  + Softmax    → 6-class probs
                        Loss: Focal CE (γ=2) — "no attack" dominates
                        Active from MVP 2

T3 Annotator Agreement: FC(512 → 64 → 1)  + Sigmoid    → [0, 1]
                        Loss: MSE — measures certainty/severity
                        Active from MVP 5 (auxiliary, optional)

Joint Loss (MVP 4 — primary architecture):
L_total = 1.2·FocalBCE(T1) + 1.0·FocalCE(T2) − 0.05·H(gates)

Joint Loss (MVP 5 — full system with T3):
L_total = 1.2·FocalBCE(T1) + 1.0·FocalCE(T2) + 0.4·MSE(T3) − 0.05·H(gates)

Note: H(gates) = entropy regularisation, prevents gate collapse (see 8.5b)
```

---

## 9. AI & ML Concepts Used

| Concept | Where Applied |
|---------|---------------|
| **Computer Vision** | ViT-B/16 on post images |
| **Vision Transformer** | Self-attention over image patches |
| **Transfer Learning** | ImageNet ViT → fine-tune on MMHS150K |
| **NLP — Transformers** | RoBERTa fine-tuned for hate speech |
| **Two-stage Pre-training** | Cyberbullying Kaggle → MMHS150K |
| **Classical ML** | XGBoost on structured features baseline |
| **Tabular Deep Learning** | TabNet for differentiable structured embedding |
| **Gated Cross-Modal Attention** | Novel architectural contribution |
| **Multi-Task Learning** | 4 simultaneous prediction heads |
| **Focal Loss** | Class imbalance handling for T1 and T2 |
| **OCR Feature Engineering** | Image-text region extraction |
| **Sentiment Lexicons** | VADER scores as structured features |
| **Cross-Domain Evaluation** | MMHS150K → Hateful Memes generalisation test |
| **Explainable AI (XAI)** | GRAD-CAM, BERT attention, gate weight visualisation |

---

## 10. Input & Output Specification

### 10.1 System Input

**Mode A — Single Post Analysis**
| Field | Type | How Provided |
|-------|------|-------------|
| Post Image | JPEG upload | User uploads image |
| Post Text | Free text | User enters tweet/caption |
| Auto-extracted | OCR + features | Computed by system |

**Mode B — Batch Moderation Queue**
- User uploads CSV with image paths + text fields
- System processes batch, returns sorted list by hate confidence
- Sortable by hate flag, severity, or category

### 10.2 System Output

```
╔════════════════════════════════════════════════════════════════════╗
║  💬 CONTENT MODERATION REPORT                                      ║
║  Post ID: 8472361                                                  ║
║  Submitted: 2026-05-07 14:23 UTC                                   ║
╠════════════════════════════════════════════════════════════════════╣
║                                                                    ║
║  🚩 HATE SPEECH DETECTED   (confidence: 89%)                       ║
║                                                                    ║
║  📋 CATEGORY:  Sexist content  (84% conf)                          ║
║      Runners-up: Other (8%) | Religious (5%) | Racist (3%)        ║
║                                                                    ║
║  ⚖️  ANNOTATOR AGREEMENT:  0.91   (high — clear case)              ║
║      Predicted clarity matches typical 3-of-3 annotator consensus  ║
║                                                                    ║
║  🔍 MODALITY IMPORTANCE:  Multimodal-required                      ║
║      Text alone: ambiguous (52%)                                   ║
║      Image alone: ambiguous (61%)                                  ║
║      Combined: clear (89%) — both modalities needed                ║
║                                                                    ║
╠══════════════════ EXPLAINABILITY ═════════════════════════════════╣
║                                                                    ║
║  🎛️  Gate Weights Learned:                                         ║
║      Image:  g_A = 0.42                                            ║
║      Text:   g_B = 0.51                                            ║
║      Struct: g_C = 0.07                                            ║
║      → Both image and text equally weighted (multimodal case)      ║
║                                                                    ║
║  🖼️  Visual Evidence (ViT GRAD-CAM):                                ║
║      Highest attention on: lower-right caption region              ║
║      Secondary attention: subject's face                           ║
║                                                                    ║
║  📋 Text Evidence (BERT Token Attribution):                        ║
║      Highest tokens: 'belongs', 'kitchen' (combined contextual)    ║
║      Hashtag #womenoughtto contributed +0.31 to hate logit        ║
║                                                                    ║
║  📊 Structured Signals (SHAP):                                     ║
║      + OCR text overlay present (+0.18)                            ║
║      + 3 hashtags including derogatory tag (+0.15)                 ║
║      - No URLs or external links (-0.04)                           ║
║                                                                    ║
║  💡 Decision Reasoning:                                            ║
║      Text alone is borderline (kitchen reference is contextual).   ║
║      Image alone is a benign photo. Combination reveals clear      ║
║      sexist meme pattern. High agreement among synthetic           ║
║      annotators (0.91) suggests this is NOT a borderline case.     ║
║                                                                    ║
╠══════════════════ SIMILAR FLAGGED POSTS ═══════════════════════════╣
║  Top-3 most similar by joint embedding:                           ║
║  [1] Post #4128 — Sexist meme  (cosine: 0.87)                     ║
║  [2] Post #9183 — Sexist meme  (cosine: 0.82)                     ║
║  [3] Post #3641 — Sexist meme  (cosine: 0.79)                     ║
╚════════════════════════════════════════════════════════════════════╝
```

---

## 11. Technology Stack

| Layer | Tool |
|-------|------|
| **Language** | Python 3.11 |
| **Deep Learning** | PyTorch 2.x |
| **Computer Vision** | timm (ViT-B/16), torchvision |
| **NLP** | HuggingFace Transformers (roberta-base) |
| **Tabular ML** | XGBoost, PyTorch TabNet |
| **OCR (already in dataset)** | Pre-extracted via Google Vision |
| **Sentiment** | VADER (vaderSentiment) |
| **XAI** | pytorch-grad-cam, captum (BERT attribution), SHAP |
| **Embedding similarity** | FAISS for retrieval-based XAI |
| **Experiment tracking** | Weights & Biases |
| **Dashboard** | Streamlit + Plotly |
| **Deployment** | Docker, HuggingFace Spaces |
| **Cloud GPU (optional)** | Kaggle Notebooks (free T4 16GB) |
| **Version control** | Git + GitHub |

---

## 12. Project Phases & Timeline *(revised — MVP ladder structure)*

### Phase 1 — Data Engineering (Weeks 1–2)
- [ ] Download MMHS150K (~6 GB) and verify integrity
- [ ] Download Hateful Memes (held-out test set)
- [ ] Download Cyberbullying Kaggle (text pre-training)
- [ ] Parse MMHS150K JSON: extract text, image paths, 3-annotator labels
- [ ] Compute T1, T2, T3 labels from majority votes and agreement
- [ ] Engineer 15 structured features per post
- [ ] Build stratified 80/10/10 split
- [ ] **Implement identity-term masking augmentation (20% replacement)**
- [ ] Sanity check: class distribution, image accessibility, no nulls
- [ ] **Deliverable:** Unified processed dataset + EDA report

### Phase 2 — MVP 1: Text-Only Baseline (Week 3)
- [ ] Pre-train RoBERTa on Cyberbullying Kaggle dataset
- [ ] Fine-tune RoBERTa on MMHS150K with T1 only
- [ ] **Deliverable:** Working binary hate classifier — minimum viable submission

### Phase 3 — MVP 2: Image Branch + T2 (Weeks 4–5)
- [ ] Fine-tune ViT-B/16 on MMHS150K with T1
- [ ] Train XGBoost on structured features (T1 baseline)
- [ ] Add T2 (6-class hate category) to RoBERTa
- [ ] **Record per-modality baseline metrics**
- [ ] **Add OCR-text-only ablation** (uses pre-extracted OCR as input to RoBERTa)
- [ ] **Deliverable:** Three trained branches + baseline report

### Phase 4 — MVP 3: Naive Fusion (Week 6)
- [ ] Implement simple concatenation fusion
- [ ] Train multi-task heads (T1 + T2)
- [ ] Confirm this matches or underperforms text-only — replicating documented failure mode
- [ ] **Deliverable:** Naive fusion baseline + replicated failure documentation

### Phase 5 — MVP 4: Gated Fusion *(Core Contribution)* (Weeks 7–9)
- [ ] Implement gated cross-modal attention layer
- [ ] **Implement entropy regularisation in loss**
- [ ] **Implement gate weight diagnostics + logging**
- [ ] Train end-to-end with frozen branches initially
- [ ] Unfreeze and joint fine-tune
- [ ] **Run parameter-matched baseline** (RoBERTa-large with similar param count)
- [ ] **Deliverable:** Trained gated fusion model + parameter-matched comparison

### Phase 6 — MVP 5: Full System + Bias Analysis (Weeks 10–12)
- [ ] Add T3 (annotator agreement) as auxiliary task
- [ ] Run 8-cell ablation table (see Section 14)
- [ ] **Identity term counterfactual testing (100 samples)**
- [ ] **Subgroup performance analysis (per hate category)**
- [ ] Cross-domain stress-test on Hateful Memes
- [ ] Statistical significance tests (McNemar's, bootstrap CIs)
- [ ] **Failure mode analysis: 20+ documented examples**
- [ ] **Deliverable:** Full evaluation report including bias analysis

### Phase 7 — XAI & Dashboard (Weeks 13–14)
- [ ] GRAD-CAM for ViT branch
- [ ] BERT token-level attribution via captum
- [ ] SHAP for structured features
- [ ] **Gate weight visualisation per sample**
- [ ] FAISS similar-post retrieval panel
- [ ] **Failure mode browser in dashboard**
- [ ] Build Streamlit dashboard
- [ ] Dockerise and deploy
- [ ] **Deliverable:** Live interactive dashboard

### Phase 8 — Documentation & Defense (Weeks 15–16)
- [ ] Write technical report / thesis chapters
- [ ] Final presentation slides
- [ ] 5-minute demo video
- [ ] Prepare defense Q&A with planned responses to likely attacks
- [ ] **Deliverable:** Final report + slides + video + GitHub repo

---

## 13. Evaluation Metrics

### T1 — Hate Binary Classification
| Metric | Why |
|--------|-----|
| AUC-ROC | Standard for binary classification under imbalance |
| F1 (macro) | Balances precision/recall across both classes |
| Precision @ 0.95 Recall | Operational: how many false positives at high recall? |
| Precision @ 0.85 Precision | Operational: how many cases caught at acceptable false positive rate? |

### T2 — 6-Class Hate Category
| Metric | Why |
|--------|-----|
| Macro F1 | Per-category performance, averaged equally |
| Confusion Matrix | Reveals which categories are confused (sexist vs other?) |
| Per-class Recall | Critical for rare categories like "religion-based" |

### T3 — Annotator Agreement Regression
| Metric | Why |
|--------|-----|
| RMSE | Standard regression metric |
| Pearson correlation | Does predicted agreement track actual agreement? |
| Calibration plot | Are confidence levels well-calibrated? |

### T4 — Modality Importance
| Metric | Why |
|--------|-----|
| Macro F1 (3-class) | Balanced across text/image/multimodal classes |
| Multimodal-class recall | Most important — these are the cases naive fusion fails on |

### Cross-Domain Generalisation (Hateful Memes)
| Metric | Why |
|--------|-----|
| AUC-ROC delta | Performance drop when moving to a different distribution |
| Per-meme-type accuracy | Does it generalise to different hate patterns? |

---

## 14. Ablation Study Plan *(expanded)*

| Configuration | Active Branches | T1 AUC | T2 Macro F1 | Hateful Memes AUC |
|---------------|-----------------|--------|------------|------------------|
| Image only (A) | ViT | ? | ? | ? |
| Text only (B) | RoBERTa | ? | ? | ? |
| **OCR-text-only** | **RoBERTa on OCR text only** | **?** | **?** | **?** |
| Structured only (C) | XGBoost | ? | ? | ? |
| A + B naive | concat | ? | ? | ? |
| A + B + C naive | concat | ? | ? | ? |
| **A + B + C gated** | **full (core contribution)** | **?** | **?** | **?** |
| **Param-matched RoBERTa-large** | **text-only, similar params to fusion** | **?** | **?** | **?** |

**Why these added cells matter:**

- **OCR-text-only baseline:** memes are often primarily textual. If RoBERTa on OCR text alone
  matches the full multimodal system, then ViT's contribution is questionable. This must be
  documented honestly.
- **Parameter-matched baseline:** addresses likely defense attack — "is your gain from
  architecture or just more parameters?" Run RoBERTa-large (355M) which has comparable
  parameter count to ViT-base + RoBERTa-base fusion (≈210M). If gated fusion still wins, the
  contribution is real.

**Expected findings (hypothesised honestly):**

- **Text-only (B)** strong baseline — replicates MMHS150K paper finding
- **Image-only (A)** weak alone — most posts need text context
- **OCR-text-only** may be surprisingly strong on meme-heavy data
- **Naive fusion (concat)** roughly matches or slightly underperforms text-only —
  replicating the documented problem
- **Gated fusion** should outperform on multimodal-required samples specifically
- **Param-matched RoBERTa-large** may close the gap — if so, contribution becomes "comparable
  performance with built-in interpretability via gates" rather than "raw accuracy gain"

The ablation must show that gated fusion specifically helps on samples where neither single
modality is sufficient. **If gated fusion does not beat either text-only or
parameter-matched RoBERTa, the result is reported honestly as a negative finding** with
analysis of why.

---

## 15. Explainability (XAI) *(softened claims)*

**Wording principle (academically defensible):** attention and gate weights provide
*interpretable signals correlated with model focus*, not causal proof of model reasoning.
We avoid the strong claim "this explains why the model decided" — that is academically
contested (Jain & Wallace 2019; Wiegreffe & Pinter 2019).

### 15.1 GRAD-CAM Image Attention (Branch A)
- Class activation maps over input image
- Shows image regions correlated with prediction confidence
- Useful for moderator review, not causal reasoning

### 15.2 BERT Token Attribution (Branch B)
- Per-token importance via Captum integrated gradients
- Highlights words/hashtags correlated with the prediction logit
- Shows surface signals the model relied on

### 15.3 SHAP Values (Branch C)
- Per-feature contribution for structured branch
- Force plots and global summary plots
- More robust theoretical grounding than attention maps

### 15.4 Gate Weight Visualisation
- Per-sample gate weights (g_A, g_B, g_C) displayed in dashboard
- Provides *interpretable modality reliance signal* — not causal proof
- Honest framing: "the model assigned X% weight to this modality"

### 15.5 Similar-Post Retrieval
- FAISS cosine similarity over fused embeddings
- Top-3 similar training-set posts with their labels
- Implicit validation by similarity

### 15.6 Failure Mode Browser *(new — required deliverable)*
- Dashboard panel showing 20+ documented failure cases
- Categorised by failure type (sarcasm misread, OCR error, ambiguous coded language,
  political satire, identity-term over-flagging)
- Builds defense credibility through honest disclosure of model limitations

---

## 15b. Error Analysis Strategy *(new section — required for defense credibility)*

Every model has failure modes. Documenting them is required for academic credibility.

**Categories of expected failure (to be empirically verified):**

| Failure Category | What we'll measure | Documentation requirement |
|------------------|---------------------|---------------------------|
| **Sarcasm/irony** | Manual review of 50 confidently-wrong predictions | 5+ examples with analysis |
| **Coded language / dog whistles** | Identify slur-substitution patterns | 5+ examples |
| **OCR errors propagating** | Cross-reference image OCR errors with prediction errors | Quantitative: % errors caused by bad OCR |
| **Ambiguous humour** | Cases with annotator agreement <0.5 | Distribution analysis |
| **Political satire** | Hand-curated subset evaluation | Per-subset metrics |
| **Identity-term over-flagging** | Counterfactual swap test (see Bias Analysis) | Quantitative flip rate |

**Outputs:**
- Failure mode browser in dashboard with searchable categorisation
- Dedicated failure analysis chapter in technical report (5–10 pages)
- 20+ documented examples with explanations

---

## 16. Deliverables

| # | Deliverable | Format |
|---|-------------|--------|
| D1 | Unified multimodal MMHS150K dataset | Processed CSV + image folder |
| D2 | Data engineering pipeline | Python scripts |
| D3 | Trained ViT-B/16 branch | PyTorch checkpoint |
| D4 | Trained RoBERTa branch | HuggingFace model dir |
| D5 | Trained XGBoost + TabNet | Pickle + PyTorch checkpoint |
| D6 | Trained gated fusion model | PyTorch checkpoint |
| D7 | Ablation study results | CSV + matplotlib charts |
| D8 | Cross-domain evaluation report | PDF + plots |
| D9 | XAI module | Python module |
| D10 | Interactive moderation dashboard | Streamlit app (Dockerised) |
| D11 | Technical report | PDF (60–80 pages) |
| D12 | Demo video | MP4 (5 minutes) |
| D13 | GitHub repository | Public repo with README + notebooks |

---

## 17. Risks & Mitigations

| Risk | Likelihood | Impact | Mitigation |
|------|------------|--------|------------|
| MMHS150K download link broken | Low | High | Multiple mirrors (CVC, Google Drive, Mega) confirmed in 2024 papers |
| Class imbalance (no-attack dominates) | High | Medium | Focal Loss, class-weighted sampling, stratified splits |
| Naive fusion fails (matches text-only) | High | Low | This is *expected* — we need this to motivate gated fusion |
| Gated fusion doesn't beat text-only | Medium | High | Document carefully which subsets it helps on; T4 modality breakdown is the safety net |
| Hateful Memes generalisation fails | Medium | Medium | Acceptable — different distribution. Document as limitation |
| Dataset contains offensive content | Certain | N/A | Document content warnings, ethics statement, never display unfiltered hate in dashboard |
| RTX 3060 OOM on joint training | Medium | Medium | Train branch-by-branch first; use fp16; Kaggle T4 fallback |
| RoBERTa pre-training overfits to Kaggle distribution | Low | Low | Use small learning rate during MMHS150K fine-tune; early stopping |

---

## 18. Academic Contribution *(repositioned for safer defense)*

**Refined contribution framing:** This project does not invent gated multimodal fusion —
gated fusion architectures exist broadly in the literature. The contribution is *applying
adaptive modality gating specifically to the MMHS150K failure mode and analysing modality
reliance patterns.* This wording is academically defensible.

1. **Empirical Investigation of Gated Fusion on a Documented Failure Mode:** The original
   MMHS150K paper (Gomez et al. 2019) explicitly reported that multimodal fusion failed to
   improve over text-only baselines and called for future research. This project provides
   that empirical investigation using gated fusion, with full diagnostics for gate collapse
   and entropy regularisation.

2. **Modality Reliance Analysis as Interpretability Signal:** Rather than predicting modality
   importance as a supervised target (which is academically problematic), we analyse the
   learned gate distributions post-hoc. This produces *correlational interpretability*
   (not causal) about when each modality contributes.

3. **Cross-Distribution Stress Testing Framework:** Evaluating MMHS150K-trained models on
   Hateful Memes provides honest stress-testing — *not* a robustness proof. The wording
   matters academically. We document where the model generalises and where it doesn't.

4. **Annotator Agreement as Auxiliary Confidence Signal:** T3 (predicting agreement score)
   provides uncertainty-aware deployment information — the model can flag its own borderline
   cases for human review.

5. **Practical Bias Mitigation in Hate Speech Detection:** Identity-term masking augmentation
   and counterfactual testing applied to a known biased dataset, with subgroup performance
   reporting. Not a fairness solution — a documented mitigation.

**What this project deliberately does NOT claim:**
- Does not claim to solve hate speech detection
- Does not claim gate weights are causally interpretable
- Does not claim robustness to adversarial inputs
- Does not claim deployment readiness
- Does not claim to invent new fusion mechanisms

These honest exclusions strengthen rather than weaken the project — they show academic
maturity.

---

## 19. Ethics & Responsible AI

This project deals with offensive content. We commit to the following:

- **Content warnings** in every deliverable — README, report, presentation, dashboard
- **No display of raw hate content** in public-facing dashboard examples — use blurred or
  synthetic illustrative samples
- **No deployment claims** — this is an academic research artifact, not a production
  moderation system
- **Acknowledged limitations** — documented in failure mode analysis section of the report
- **Bias awareness** — MMHS150K is English Twitter only. Will not generalise to other
  languages, platforms, or cultural contexts. This is stated explicitly
- **Responsible disclosure** — if the system has biases (e.g., higher false positive rate on
  AAVE), this is documented prominently, not buried
- **No personal data** — only public posts already de-identified by dataset authors

---

## 19b. Bias Analysis Strategy *(new section — practical, undergraduate-realistic)*

**Scope clarification:** This is not a fairness research project. We do not attempt to
*solve* dataset bias — that is PhD-thesis scope. We implement four practical mitigations
that improve model results, demonstrate academic awareness, and prevent embarrassing
failures during defense.

### 19b.1 — Identity Term Masking Augmentation *(training-time mitigation)*

During training, randomly replace identity terms with a `[IDENTITY]` token 20% of the time.
Forces the model to learn from context rather than memorising identity words.

**Implementation:** Add to data loader as a transform, runs on every training batch.

```python
identity_terms = load_terms('identity_lexicon.txt')  # ~200 terms
def mask_identity_terms(text, p=0.20):
    if random.random() < p:
        for term in identity_terms:
            text = text.replace(term, '[IDENTITY]', 1)
    return text
```

**Cost:** ~30 lines of code, no compute overhead.
**Expected effect:** 5–10% reduction in identity-term over-reliance, measured by counterfactual flip rate.

### 19b.2 — Counterfactual Identity Swap Testing *(evaluation-time diagnostic)*

Take 100 confidently-predicted hate examples. Programmatically swap identity terms (women ↔
men, Muslim ↔ Christian, gay ↔ straight). Measure how often the prediction flips.

**Pass criterion:** flip rate < 15%. If flip rate > 25%, the model is learning identity words,
not hate patterns. Document honestly.

**Implementation:** Single evaluation script, runs once on test set.

### 19b.3 — Subgroup Performance Reporting *(transparency requirement)*

Report metrics broken down by hate category:

| Category | Precision | Recall | F1 | False Positive Rate |
|----------|-----------|--------|----|---------------------|
| Sexist | ? | ? | ? | ? |
| Racist | ? | ? | ? | ? |
| Homophobic | ? | ? | ? | ? |
| Religion-based | ? | ? | ? | ? |
| Other | ? | ? | ? | ? |
| No attack | ? | ? | ? | ? |

If a category has F1 below 50% while others are above 80%, this is reported prominently —
not buried.

### 19b.4 — Class-Balanced Sampling Within Hate Categories *(training mitigation)*

MMHS150K has known overrepresentation of explicit slurs in certain hate categories. Use
weighted sampling so each *hate category* gets equal exposure during training, not just
hate vs not-hate balance.

**Implementation:** PyTorch `WeightedRandomSampler` with weights inversely proportional to
class frequency.

### 19b.5 — Documented Limitations Section in Final Report

Required content:
- 5+ false positive examples documented with analysis
- Identity-term flip rate reported numerically
- Per-category metrics in a table
- Honest discussion of what the model cannot do
- Cite Gomez et al. 2019 on documented dataset biases
- Acknowledge English Twitter limitation explicitly

**This section alone is what gets you through ethics-focused defense questions.** Going
further than this is PhD scope.

---

## 20. Defense Preparation *(new section)*

Anticipated attacks during project defense and prepared responses:

### Attack 1: "How is this different from existing multimodal transformers?"
**Response:** Reference literature comparison table. Position contribution as
*application-specific empirical investigation* of the documented MMHS150K failure mode,
not architectural invention. Show that we're addressing a *specific* known failure with
appropriate diagnostics (gate collapse, entropy regularisation).

### Attack 2: "Are gate weights truly interpretable?"
**Response:** "We claim correlational interpretability, not causal. Gate weights show what
the model focused on, not why it decided. We acknowledge this is an active academic debate
(Jain & Wallace 2019, Wiegreffe & Pinter 2019) and frame our claims accordingly."

### Attack 3: "Your modality labels are synthetic."
**Response:** "T4 was removed from the supervised training pipeline specifically because
of this concern. Modality reliance is now inferred post-hoc from learned gate distributions,
which is a defensible analytical approach rather than a circular supervision target."

### Attack 4: "How do you know performance gain isn't from increased parameter count?"
**Response:** Reference parameter-matched baseline (RoBERTa-large with comparable parameter
count). Show that gated fusion either (a) outperforms the parameter-matched baseline, or
(b) matches it with built-in interpretability — both are defensible outcomes.

### Attack 5: "How do you handle dataset bias?"
**Response:** Reference Bias Analysis section — identity-term masking, counterfactual testing,
subgroup reporting. Explicitly acknowledge this is mitigation, not a fairness solution.

### Attack 6: "Why didn't you just use a better text model?"
**Response:** Reference OCR-text-only ablation and parameter-matched RoBERTa-large baseline.
If those win, we report so honestly. The contribution is then *interpretability through
gating*, not raw accuracy.

---

*Document version: 1.1 | Revised after technical critique | Prepared for Final Year CS Project (AI Major)*

## Changelog v1.0 → v1.1
- **Removed T4** as supervised target (circular dependency with model outputs)
- **Added MVP ladder** for staged training with survivability
- **Added gate collapse diagnostics** + entropy regularisation
- **Lightened structured branch** (removed TabNet, kept simple concat)
- **Added bias analysis section** (identity masking, counterfactual testing, subgroup metrics)
- **Added error analysis section** with required failure mode browser
- **Added defense preparation section** with anticipated attacks and responses
- **Added parameter-matched baseline** to ablation study
- **Added OCR-text-only baseline** to ablation study
- **Softened XAI wording** to correlational not causal interpretability
- **Repositioned novelty claim** as application-specific, not architectural invention
- **Reframed cross-domain** as stress-test, not robustness proof
