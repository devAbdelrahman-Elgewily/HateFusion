# HateFusion 🔍

**Multimodal hate speech detection on MMHS150K using CLIP image features, Twitter-RoBERTa,
and gated cross-modal attention fusion.**

> Final year Computer Science (AI Major) undergraduate project.
> Addresses the documented failure mode where naive multimodal fusion underperforms
> text-only baselines on the MMHS150K benchmark (Gomez et al., 2019).

---

## Overview

Modern hate speech increasingly combines seemingly innocent text with hostile images —
attack patterns that text-only moderation systems miss. HateFusion addresses this by
fusing three modalities through a **gated cross-modal attention layer** that learns
*when* each modality matters, rather than combining them equally.

The project targets the specific open research problem documented in the MMHS150K paper:
naive multimodal fusion fails to outperform text-only baselines. Our gated fusion
architecture directly addresses this failure mode with entropy regularization and full
gate collapse diagnostics.

---

## Architecture

```
┌─────────────────┐   ┌─────────────────────┐   ┌─────────────────┐
│  Branch A       │   │  Branch B            │   │  Branch C       │
│  CLIP ViT-B/16  │   │  Twitter-RoBERTa     │   │  Structured     │
│  + LoRA (r=16)  │   │  + LoRA (r=16)       │   │  Features (9-d) │
│  → 512-d embed  │   │  → 768-d embed       │   │                 │
└────────┬────────┘   └──────────┬───────────┘   └────────┬────────┘
         │                       │                        │
         └───────────────────────┼────────────────────────┘
                                 │
                  ┌──────────────▼──────────────┐
                  │  Gated Cross-Modal Attention │
                  │  g_A, g_B, g_C ∈ [0,1]      │
                  │  + Entropy Regularization    │
                  └──────────────┬──────────────┘
                                 │
                  ┌──────────────▼──────────────┐
                  │  Shared Representation       │
                  │  FC(2057 → 512) → ReLU → BN │
                  └───┬──────────┬──────────┬───┘
                      │          │          │
                   T1: Hate   T2: Type   T3: Agreement
                   Binary     6-class    Regression
```

**Key design principle:** The gating mechanism learns to weight each modality
per sample — amplifying image signals on meme-style posts, down-weighting noisy
image branches on text-dominant hate content.

---

## Datasets

| Dataset | Role | Size | License |
|---------|------|------|---------|
| [MMHS150K](https://gombru.github.io/2019/10/09/MMHS/) | Primary training | 150K tweets, ~6 GB | Academic research |
| [Hateful Memes](https://hatefulmemeschallenge.com) | Cross-domain test only | 10K memes, ~3 GB | Facebook AI research terms |
| [Cyberbullying Tweets](https://www.kaggle.com/datasets/andrewmvd/cyberbullying-classification) | RoBERTa warm-start | 47K tweets, 50 MB | Open data |

> ⚠️ Datasets are **not** included in this repository. See [Dataset Setup](#dataset-setup)
> for download instructions. Dataset usage is subject to original dataset licenses —
> this codebase's MIT license does not grant rights to the underlying data.

---

## Prediction Targets

| Target | Task | Description |
|--------|------|-------------|
| **T1** | Binary classification | Hate speech / Not hate speech |
| **T2** | 6-class classification | Racist / Sexist / Homophobic / Religion / Other / None |
| **T3** | Regression [0, 1] | Annotator agreement score — routes borderline cases to human review |

T3 enables **human-in-the-loop deployment**: high-agreement predictions are
auto-moderated; low-agreement cases are flagged for human reviewer.

---

## Results

> 🚧 **Training in progress.** Results will be updated as each MVP is completed.

### Ablation Study

| Configuration | T1 AUC | T2 Macro F1 | Hateful Memes AUC |
|---------------|--------|-------------|-------------------|
| Text-only (Twitter-RoBERTa) | — | — | — |
| Image-only (CLIP) | — | — | — |
| OCR-text-only | — | — | — |
| Structured-only (XGBoost) | — | — | — |
| Naive concat fusion | — | — | — |
| **Gated fusion (ours)** | — | — | — |
| Param-matched RoBERTa-large | — | — | — |

### Modality Reliance Analysis

> Per-sample interaction pattern distribution — updated after MVP 4.

| Pattern | Count | % of Test Set |
|---------|-------|--------------|
| Convergent Correct | — | — |
| Text Saved | — | — |
| Image Saved | — | — |
| Emergent Multimodal | — | — |
| Fusion Failure | — | — |

---

## Project Structure

```
HateFusion/
├── data/
│   ├── mmhs150k/              ← MMHS150K dataset (not tracked by git)
│   ├── hateful_memes/         ← Hateful Memes dataset (not tracked by git)
│   └── cyberbullying/         ← Cyberbullying Kaggle CSV (not tracked by git)
├── notebooks/
│   ├── 01_data_loading.ipynb          ✅ Complete
│   ├── 02_eda.ipynb                   ✅ Complete
│   ├── 03_structured_features.ipynb   ✅ Complete
│   ├── 04_roberta_pretrain_kaggle.ipynb
│   ├── 05_mvp1_roberta_t1.ipynb
│   ├── 06_mvp2_clip_xgb.ipynb
│   ├── 07_mvp3_naive_fusion.ipynb
│   ├── 08_mvp4_gated_fusion.ipynb
│   ├── 09_mvp5_t3_ablations_bias.ipynb
│   ├── 10_per_sample_modality_analysis.ipynb
│   └── 11_xai_dashboard_prep.ipynb
├── data/processed/
│   ├── labels_parsed.csv              ✅ 149,819 rows, T1/T2/T3 labels
│   └── structured_features.csv        ✅ 9-feature vector, 149,819 rows
├── models/                    ← checkpoints saved here (not tracked by git)
├── outputs/                   ← charts, ablation tables, logs
├── app/
│   └── dashboard.py           ← Streamlit moderation dashboard
├── Reports/
│   ├── Multimodal_Cyberbullying_Detection_v1.2.md
│   └── Cyberbullying_Detection_Report_Framing.md
├── CLAUDE.md                  ← Claude Code session directives
├── requirements.txt
└── README.md
```

---

## Dataset Setup

### 1. MMHS150K
```bash
# Download from official page (choose any mirror):
# https://gombru.github.io/2019/10/09/MMHS/
# Direct: https://datasets.cvc.uab.es/MMHS150K/MMHS150K.zip

# Extract to:
data/mmhs150k/
├── img_resized/
├── img_txt/
├── splits/
├── MMHS150K_GT.json
└── hatespeech_keywords.txt
```

### 2. Hateful Memes
```bash
# Accept terms and download from:
# https://hatefulmemeschallenge.com
# Or Kaggle mirror: https://www.kaggle.com/datasets/parthplc/facebook-hateful-meme-dataset

# Extract to:
data/hateful_memes/
├── img/
├── train.jsonl
├── dev.jsonl
└── test.jsonl
```

### 3. Cyberbullying Tweets
```bash
# Download from Kaggle (free account required):
# https://www.kaggle.com/datasets/andrewmvd/cyberbullying-classification

# Place CSV at:
data/cyberbullying/cyberbullying_tweets.csv
```

---

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/HateFusion.git
cd HateFusion

# Create conda environment
conda create -n hateful_project python=3.11
conda activate hateful_project

# Install dependencies
pip install -r requirements.txt
```

### Requirements
```
torch>=2.0.0
transformers>=4.35.0
peft>=0.7.0
open-clip-torch>=2.20.0
datasets>=2.14.0
scikit-learn>=1.3.0
xgboost>=2.0.0
vaderSentiment>=3.3.2
better-profanity>=0.7.0
streamlit>=1.28.0
plotly>=5.17.0
grad-cam>=1.4.8
captum>=0.6.0
shap>=0.43.0
faiss-cpu>=1.7.4
wandb>=0.16.0
pandas>=2.0.0
numpy>=1.24.0
Pillow>=10.0.0
tqdm>=4.65.0
```

---

## Usage

### Run the notebooks
```bash
conda activate hateful_project
jupyter lab
```
Open notebooks in order (01 → 11). Each notebook is self-contained
with a summary cell documenting findings and downstream decisions.

### Run the moderation dashboard
```bash
streamlit run app/dashboard.py
```

---

## Reproducibility

All experiments use fixed seeds:
```python
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
torch.cuda.manual_seed_all(42)
```

Official MMHS150K splits are used as-is (train 134,820 / val 4,999 / test 10,000)
for direct comparability with Gomez et al. 2019. Val and test are
deliberately 50/50 hate-balanced by dataset design.

---

## Bias & Ethics

This project deals with offensive content. Important disclosures:

- **No deployment claims** — academic research artifact only
- **English Twitter only** — does not generalize to other languages or platforms
- **Inherits MMHS150K biases** — known keyword sampling bias and annotation inconsistency
  documented in Davidson et al. (2019) and Sap et al. (2019)
- **Bias mitigations implemented:** identity-term masking (20% during training),
  counterfactual swap testing, subgroup performance reporting
- **Human-in-the-loop design** — T3 routes borderline cases to human reviewers;
  system is not intended for fully automated moderation

---

## Citation

If you use this code in your research:

```bibtex
@misc{hateFusion2025,
  title   = {HateFusion: Multimodal Hate Speech Detection via
             Gated Cross-Modal Attention on MMHS150K},
  author  = {[Your Name]},
  year    = {2025},
  url     = {https://github.com/yourusername/HateFusion}
}
```

### Key References

```bibtex
@inproceedings{gomez2020exploring,
  title     = {Exploring Hate Speech Detection in Multimodal Publications},
  author    = {Gomez, Raul and Gibert, Jaume and Gomez, Lluis and Karatzas, Dimosthenis},
  booktitle = {WACV},
  year      = {2020}
}

@article{kiela2020hateful,
  title   = {The Hateful Memes Challenge: Detecting Hate Speech in Multimodal Memes},
  author  = {Kiela, Douwe and others},
  journal = {NeurIPS},
  year    = {2020}
}
```

---

## License

This project is licensed under the **MIT License** — see [LICENSE](LICENSE) for details.

Dataset usage is subject to the original dataset licenses. This codebase's MIT license
does not grant rights to the underlying datasets.

---

## Project Status

| Phase | Status | Notes |
|-------|--------|-------|
| Data Engineering | ✅ Complete | 149,819 labeled rows, 9-feature structured branch |
| EDA | ✅ Complete | 6 charts, key findings documented |
| Structured Features | ✅ Complete | Correlation-pruned to 9 features |
| MVP 1 — Text Baseline | 🔄 In progress | Twitter-RoBERTa on T1 |
| MVP 2 — Image + T2 | ⏳ Pending | CLIP + XGBoost baselines |
| MVP 3 — Naive Fusion | ⏳ Pending | Replicating documented failure mode |
| MVP 4 — Gated Fusion | ⏳ Pending | Core contribution |
| MVP 5 — Full System | ⏳ Pending | T3 + ablations + bias analysis |
| Per-Sample Analysis | ⏳ Pending | Modality reliance categorisation |
| Dashboard | ⏳ Pending | Streamlit moderation app |
