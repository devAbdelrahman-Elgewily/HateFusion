# Data

This folder holds the three datasets used by the project. The two heavy image
datasets are **not** committed to git (see `.gitignore`) — download them
separately as described below. The smaller text/tabular files **are** committed.

## What is committed

| Path | Size | Source |
|---|---|---|
| `cyberbullying_tweets.csv` | 7 MB | Cyberbullying Classification (J. Wang, Kaggle) — used for RoBERTa warm-start (Notebook 04) |
| `processed/labels_parsed.csv` | 29 MB | Output of Notebook 01 (T1/T2/T3 derivation from MMHS150K_GT.json) |
| `processed/structured_features.csv` | 7 MB | Output of Notebook 03 (15 engineered features pruned to 9) |

## What you need to download

### MMHS150K (~6 GB)

Primary dataset. Original release: Gomez et al., *"Exploring Hate Speech
Detection in Multimodal Publications"*, WACV 2020.

- Download: https://gombru.github.io/2019/10/09/MMHS150K/
- Expected layout after extraction:
  ```
  data/MMHS150K/
    MMHS150K_GT.json           # 49 MB, annotations
    img_resized/               # ~150K .jpg files
    img_txt/                   # ~150K .json files (pre-extracted OCR)
    splits/                    # train_ids.txt, val_ids.txt, test_ids.txt
    hatespeech_keywords.txt
  ```
- Splits used: official train (134,823) / val (5,000) / test (10,000). Do not re-split.

### Hateful Memes Challenge (~3.4 GB)

Used only as the cross-domain test set. Never trained on.

- Download: https://hatefulmemeschallenge.com/ (requires accepting Meta's
  research data agreement) or the HuggingFace mirror
  `neuralcatcher/hateful_memes`.
- Expected layout:
  ```
  data/The Hateful Memes Challenge/
    img/                       # ~10K .png files
    train.jsonl
    dev.jsonl
    test.jsonl
  ```

### Cyberbullying tweets

Already in this repo. Original source for reference:
https://www.kaggle.com/datasets/andrewmvd/cyberbullying-classification

## Reproducing `processed/`

The two CSVs in `processed/` are deterministic outputs of Notebooks 01 and 03
given the MMHS150K data above. If you re-run those notebooks with the same
seeds (42), you should get identical files. They are committed so reviewers
can skip the ~10 min Phase 1 pipeline and jump straight to Notebook 04+.
