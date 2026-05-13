# 📑 Phase 1 — Data Engineering Report
### Multimodal Cyberbullying & Online Hate Speech Detection
### Final Year Project — Computer Science (AI Major)

---

## Document Information

| Field | Value |
|-------|-------|
| **Document type** | Phase 1 deliverable + report-writing source material |
| **Phase** | 1 — Data Engineering (Weeks 1–2 of project plan) |
| **Status** | Complete |
| **Date** | 2026-05-07 |
| **Companion documents** | `Multimodal_Cyberbullying_Detection_v1.1.md` (technical scope), `Cyberbullying_Detection_Report_Framing.md` (significance/defence) |
| **Notebooks delivered** | `01_data_loading.ipynb`, `02_eda.ipynb`, `03_structured_features.ipynb` |
| **Persisted artefacts** | `data/processed/labels_parsed.csv`, `data/processed/structured_features.csv`, six EDA charts in `outputs/` |

---

## Table of Contents

1. [Executive Summary](#1-executive-summary)
2. [Environment & Reproducibility](#2-environment--reproducibility)
3. [Dataset Inventory](#3-dataset-inventory)
4. [Notebook 01 — Data Loading & Label Derivation](#4-notebook-01--data-loading--label-derivation)
5. [Notebook 02 — Exploratory Data Analysis](#5-notebook-02--exploratory-data-analysis)
6. [Notebook 03 — Structured Feature Engineering](#6-notebook-03--structured-feature-engineering)
7. [Methodological Decisions Locked During Phase 1](#7-methodological-decisions-locked-during-phase-1)
8. [Open Items & Phase 2 Preconditions](#8-open-items--phase-2-preconditions)
9. [References](#9-references)
10. [Appendix A — Saved Artefacts](#appendix-a--saved-artefacts)
11. [Appendix B — Project File Tree (post-Phase 1)](#appendix-b--project-file-tree-post-phase-1)

---

## 1. Executive Summary

Phase 1 of the multimodal cyberbullying detection project — data engineering — is complete. Three notebooks were delivered, a unified label DataFrame and a 9-feature structured vector were persisted to disk, and an exploratory data analysis surfaced eight findings that drive the design of Phase 2 onwards.

### Phase 1 deliverables

| # | Deliverable | Output |
|---|-------------|--------|
| 1 | Parsed and cleaned label DataFrame for MMHS150K | `data/processed/labels_parsed.csv` (28.6 MB, 149,819 × 10) |
| 2 | Engineered structured feature vector (post correlation gate) | `data/processed/structured_features.csv` (6.6 MB, 149,819 × 10) |
| 3 | EDA charts (six PNGs) | `outputs/eda_*.png` |
| 4 | Eight EDA findings with actionable downstream implications | Section 5 of this report |
| 5 | Locked methodological decisions captured in `CLAUDE.md` | Section 7 of this report |

### Headline findings

1. **All 149,819 GT-referenced images are present on disk** — no row drop required for image-branch training. Plus 181 orphan `.jpg` files with no GT entry (ignored).
2. **The MMHS150K val and test splits are intentionally 50/50 hate-balanced** (Gomez et al. 2019 design) while train reflects the natural ~22% hate rate. This is undocumented in our spec but methodologically load-bearing — metric reporting must distinguish balanced-test results from recalibrated deployment estimates.
3. **The T3 deployment-routing premise is empirically confirmed.** NotHate is unambiguous (mean T3 = 0.838, median = 1.0); every hate class has a collapsed interquartile range at T3 = 0.667 (Q25 = Q75 = 0.667), with Sexist the most ambiguous (mean = 0.705). T3 is a defensible signal for human-in-the-loop case routing.
4. **22 rows had T3 > 1.0 (mathematically impossible)** because their `labels` array contained 4 or 5 entries instead of the documented 3. A wider audit revealed 74 such rows in total. Investigation showed they are predominantly AAVE / in-group reclamation cases that received extended annotator review and were ultimately judged non-hate. Generalised T3 formula (`max_count / len(labels)`) was adopted; 4 single-annotator rows were dropped.
5. **The structured branch is dominated by VADER sentiment** (`vader_neg` corr = +0.17, `vader_neu` corr = −0.13). All other surviving features have |corr| < 0.05. Six of 15 features failed the |corr| ≥ 0.02 gate and were dropped, including `url_present` (zero variance — every Twitter post has a `t.co` URL wrapper) and structural noise like `n_questions` and `text_len_chars`.
6. **`hate_keyword_count` correlation is structurally confounded** — every tweet in MMHS150K contains at least one Hatebase keyword by construction (the keywords were the dataset's seed terms). The feature is near-constant. Kept for completeness; flagged for the Limitations section of the final report.
7. **Religion (163 rows, 0.3%) is genuinely rare**, not a measurement error. Per locked decision, no over-sampling or synthetic augmentation. Focal Loss + intra-hate-class weighted sampling will be the only mitigations. Low-recall expectation documented as a known limitation.
8. **39% of MMHS150K images contain OCR-detectable text**, with strong skew toward the meme-driven classes — Religion 62%, OtherHate 52%, vs ~37–41% for the other classes. The OCR-only ablation cell in MVP 2 is genuinely competitive on the meme-heavy classes; for the rest it will track text-only baseline closely.

---

## 2. Environment & Reproducibility

### 2.1 Hardware / OS
- **OS:** Windows 11 Pro 10.0.26200
- **Shell:** PowerShell (primary), Bash (secondary, via Git)
- **GPU:** NVIDIA RTX 3060 Laptop, 6 GB VRAM (Phase 2+ training target)
- **Storage:** Project directory on D: drive

### 2.2 Conda environment
- **Name:** `cyberbully_project`
- **Location:** `D:\Anaconda\envs\cyberbully_project`
- **Python:** 3.11.15

### 2.3 Phase 1 package versions

| Package | Version | Used in |
|---------|---------|---------|
| `pandas` | 3.0.2 | All notebooks |
| `numpy` | 2.4.3 | All notebooks |
| `scikit-learn` | 1.8.0 | (reserved for later, installed up front) |
| `matplotlib` | 3.10.9 | Notebook 02 charts |
| `seaborn` | 0.13.2 | Notebook 02 styling |
| `jupyterlab` | (latest at install date) | Notebook execution |
| `ipykernel` | (latest at install date) | Kernel registration |
| `Pillow` (PIL) | 12.2.0 | Notebook 02 image dimension audit |
| `tqdm` | 4.67.3 | Progress bars |
| `vaderSentiment` | 3.3.2 | Notebook 03 — sentiment scoring |
| `better-profanity` | 0.7.0 | Notebook 03 — profanity lexicon |

> **Versioning note:** pandas 3.0 and numpy 2.x are bleeding-edge releases at project start. Both passed Phase 1 without issue. Pin the environment before MVP 1 to avoid downstream regression risk.

### 2.4 Reproducibility settings

Every notebook seeds the same way:

```python
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)            # added from Phase 2 onwards
torch.cuda.manual_seed_all(42)   # added from Phase 2 onwards
```

### 2.5 Notebook execution
- Kernel display name: `Python (cyberbully_project)`
- All notebooks executed end-to-end via `jupyter nbconvert --execute` to guarantee that committed outputs match committed source.

---

## 3. Dataset Inventory

Three datasets are staged on disk under `D:\Cyberbullying Detection\data\`. None were downloaded during Phase 1 — all were already present.

### 3.1 MMHS150K (primary training corpus)

| Property | Value |
|----------|-------|
| **Source** | Gomez et al. 2019 (WACV 2020) — *Exploring Hate Speech Detection in Multimodal Publications* |
| **URL** | https://gombru.github.io/2019/10/09/MMHS/ |
| **Collection method** | Real Twitter posts (Sep 2018 – Feb 2019), keyword-seeded via 51 Hatebase terms (the on-disk file contains 86 terms — 51 single-word hate slurs plus 35 multi-word phrases; both the single-word and multi-phrase variants were used during sampling) |
| **Annotation** | 3-of-3 Amazon Mechanical Turk annotators per tweet (with 70 exceptions: 30 with 2 annotators, 37 with 4, 3 with 5, 4 with 1) |
| **GT entries** | 149,823 in the JSON dictionary |
| **Image files** | 150,000 `.jpg` files in `img_resized/`, of which 149,819 are GT-referenced and 181 are orphans |
| **OCR text** | Pre-extracted via Google Vision API into `img_txt/{tweet_id}.json`, 149,819 files |
| **Official splits** | `train_ids.txt` (134,823) / `val_ids.txt` (5,000) / `test_ids.txt` (10,000); val/test 50/50 hate-balanced by design |
| **Image dimensions** | All images pre-resized to shortest side ≥ 500 px |
| **Disk footprint** | 6.0 GB extracted (`MMHS150K.zip` 6.4 GB kept as backup) |

### 3.2 The Hateful Memes Challenge (cross-domain test)

| Property | Value |
|----------|-------|
| **Source** | Facebook AI Research (Kiela et al. 2020) |
| **Format** | JSONL files: `train.jsonl` (8,499), `dev.jsonl` (499), `test.jsonl` (999) — total 9,997 |
| **Schema** | `{id, img, label, text}` |
| **Role in this project** | **Held-out cross-domain evaluation only.** Never trained on. Used to assess generalisation in MVP 5. |

### 3.3 Cyberbullying Kaggle (text-branch warm-start)

| Property | Value |
|----------|-------|
| **Source** | Kaggle: `andrewmvd/cyberbullying-classification` |
| **Format** | Single CSV |
| **Rows** | 47,985 |
| **Schema** | `{tweet_text, cyberbullying_type}` (text-only, no images) |
| **Role in this project** | RoBERTa pre-training (3 epochs) before MMHS150K fine-tune. Phase 2 / MVP 1 deliverable. |

---

## 4. Notebook 01 — Data Loading & Label Derivation

### 4.1 Goal
Parse `MMHS150K_GT.json` into a unified DataFrame, derive the supervised targets `T1`, `T2`, `T3`, handle data anomalies, and persist the result to disk for downstream notebooks.

### 4.2 Source data structure

`MMHS150K_GT.json` is a top-level dict keyed by `tweet_id` (string). Each entry has:

```json
{
  "tweet_url":  "https://twitter.com/user/status/...",
  "img_url":    "http://pbs.twimg.com/...",
  "tweet_text": "<raw tweet text including hashtags, mentions, URLs, emojis>",
  "labels":     [<int>, <int>, <int>],         // nominally 3 entries, see anomalies
  "labels_str": ["<class_name>", "<class_name>", "<class_name>"]
}
```

**Label encoding:** 0 = NotHate, 1 = Racist, 2 = Sexist, 3 = Homophobe, 4 = Religion, 5 = OtherHate.

### 4.3 Parse strategy

The JSON is loaded once (~270 MB on disk → ~600 MB in Python objects), flattened to a list of records, materialised as a pandas DataFrame, then the raw dict and intermediate list are explicitly `del`-ed to free memory.

The DataFrame after flattening has columns `[tweet_id, tweet_text, img_filename, labels, labels_str]` with shape `(149,823, 5)` and zero nulls in any column.

### 4.4 Label derivation

#### 4.4.1 T1 — binary hate flag (Focal BCE target)

> T1 = 1 if the majority-vote class is non-zero, else 0.

#### 4.4.2 T2 — 6-class hate category (Focal CE target)

> T2 = the most frequent class in the `labels` array.

**Tie-break rule** (3-way disagreement, all classes distinct): pick the lowest-index hate class present (1–5), or 0 if no hate class is in the tie.

Examples:
- `labels = [4, 1, 3]` → counts `{4:1, 1:1, 3:1}`, top set `{1, 3, 4}`, hate-only sorted `[1, 3, 4]` → T2 = 1 (Racist)
- `labels = [0, 3, 5]` → counts `{0:1, 3:1, 5:1}`, top set `{0, 3, 5}`, hate-only sorted `[3, 5]` → T2 = 3 (Homophobe)

This rule is **deterministic** and documented; it preserves a pseudo-label for ambiguous cases at the cost of inflating hate-class counts. For T2 training, ambiguous rows are masked to NaN (see Section 4.5).

#### 4.4.3 T3 — annotator agreement score (MSE target, auxiliary)

> T3 = (max vote count) / (number of annotators).

**Originally:** `T3 = max_count / 3`, hard-coded to 3 annotators.
**After anomaly resolution (Section 4.5):** `T3 = max_count / len(labels)`, generalised so T3 is bounded in [0, 1] regardless of annotator count.

Standard 3-annotator bucket distribution after generalisation:
- Unanimous (3/3 agree) → T3 = 1.0
- Majority of 2 (2-of-3) → T3 ≈ 0.667
- 3-way disagreement → T3 ≈ 0.333

### 4.5 Anomaly handling — the discovery and resolution

The first-pass T3 distribution surfaced **22 rows with T3 > 1.0**, which is mathematically impossible under the documented 3-annotator schema. This triggered a wider audit.

#### 4.5.1 Length distribution of `labels` arrays across the full dataset

```
len(labels)         count
       1                4    ← single annotator (no agreement signal)
       2               30    ← T3 = 0.5 or 1.0 under generalised formula
       3          149,749    ← documented schema
       4               37    ← T3 ∈ {0.25, 0.5, 0.75, 1.0}
       5                3    ← T3 ∈ {0.2, 0.4, 0.6, 0.8, 1.0}
─────────────────────────
   total          149,823
```

**74 rows total** (0.05% of the dataset) had non-3 annotator counts. Only 22 of those produced T3 > 1.0 under the original `max_count / 3` formula (the others silently mixed into the existing 0.333/0.667/1.0 buckets).

#### 4.5.2 Qualitative inspection of the 22 anomalies

All 22 are `T1 = 0` (non-hate) and `T2 = 0` (NotHate). The pattern is striking: **predominantly AAVE / in-group reclamation language and casual hip-hop vernacular**, with all 4 or 5 annotators ultimately agreeing they are non-hate. Examples of the surface vocabulary include the n-word in casual usage, "twat", "retarded", "hillbilly".

**Inferred curation pattern** (treated as a hypothesis to verify, then verified empirically in EDA Q8): extra annotators were brought in by the dataset authors to adjudicate ambiguous AAVE cases that initially produced annotator disagreement; the additional review confirmed those tweets were not hate. The 22 anomalies are therefore *valuable hard non-hate training examples* — exactly the cases a fairness-aware model needs to see.

#### 4.5.3 Resolution (locked decisions)

Applied in `Cell 6` of `01_data_loading.ipynb`:

1. **Drop the 4 rows with `len(labels) == 1`** — single-annotator rows have no agreement signal possible.
2. **Generalise T3** to `max_count / len(labels)`, bounding it in [0, 1] for all annotator counts.
3. **Add `n_annotators` column** = `len(labels)`, so downstream sessions can identify extended-review rows.
4. **Set `T2 = NaN` where there is no majority** (`max_count == 1`, equivalent to T3 = 0.333 in the standard 3-annotator case). Use the nullable `Int8` dtype to preserve integer semantics with NaN support.
5. **Add `t2_valid` boolean column** = True iff T2 is not NaN.
6. **Use masked CE during T2 training** to ignore NaN rows, while keeping T1 and T3 valid for those same rows.

#### 4.5.4 Verification (final values from the executed Cell 6)

- Rows: 149,823 → 149,819 (4 dropped)
- T3 range: [0.333, 1.000] — bounded ✓
- T2 NaN count: 11,710 (matches `t2_valid=False` count exactly)
- All other columns clean: 0 nulls in `tweet_id`, `tweet_text`, `img_filename`, `labels`, `labels_str`, `T1`, `T3`, `n_annotators`, `t2_valid`

### 4.6 Final distributions (after Cell 6)

#### 4.6.1 T1 — binary hate
| T1 | count | fraction |
|----|------:|---------:|
| 0 (no_hate) | 112,841 | 75.32% |
| 1 (hate)    |  36,978 | 24.68% |

#### 4.6.2 T2 — 6-class category (NaN-aware)
| T2 | count | fraction (of total) |
|----|------:|--------------------:|
| NotHate    | 112,842 | 75.32% |
| Racist     |  11,927 |  7.96% |
| Sexist     |   3,495 |  2.33% |
| Homophobe  |   3,871 |  2.58% |
| Religion   |     163 |  0.11% |
| OtherHate  |   5,811 |  3.88% |
| **NaN**    |  11,710 |  7.82% |

> **Why hate-class counts dropped after T2 masking:** the original tie-break rule deterministically labelled 3-way ambiguous cases with the lowest-index hate class, inflating the apparent hate-class counts. Masking those rows as NaN reveals that the original distributions over-stated several hate categories by 30–50%. OtherHate (index 5) is unchanged because under the lowest-index-hate tie-break it can only "win" a tie with NotHate — never against another hate class — so it never won a 3-way tie. Religion (index 4) lost 232 rows for the same structural reason.

#### 4.6.3 T3 — annotator agreement
| T3 (rounded) | count |
|-------------:|------:|
| 0.333 | 11,699 |
| 0.500 |     16 |
| 0.667 | 76,068 |
| 0.750 |     13 |
| 0.800 |      2 |
| 1.000 | 62,021 |
| **mean** | **0.7786** |
| **median** | **0.6667** |

#### 4.6.4 n_annotators
| n_annotators | count |
|-------------:|------:|
| 2 |        30 |
| 3 |   149,749 |
| 4 |        37 |
| 5 |         3 |

### 4.7 Output schema — `data/processed/labels_parsed.csv`

| Column | Type | Description |
|--------|------|-------------|
| `tweet_id` | str | Primary key (string, retains big-int precision) |
| `tweet_text` | str | Raw tweet text including hashtags, mentions, URLs, emojis |
| `img_filename` | str | `{tweet_id}.jpg` — file inside `data/MMHS150K/img_resized/` |
| `labels` | list[int] | Raw annotator vote array, 1–5 entries |
| `labels_str` | list[str] | Human-readable class names matching `labels` |
| `T1` | int8 | Binary hate flag {0, 1} |
| `T2` | nullable Int8 | 6-class category {0..5}, NaN for ambiguous rows |
| `T3` | float32 | Agreement score in [0, 1] |
| `n_annotators` | int8 | Length of `labels` array (2..5) |
| `t2_valid` | bool | True iff T2 is not NaN |

**File:** 28.6 MB CSV. List columns are stored as Python repr strings; recover with `pd.read_csv(..., converters={"labels": ast.literal_eval, "labels_str": ast.literal_eval})`.

---

## 5. Notebook 02 — Exploratory Data Analysis

### 5.1 Approach

Eight specific questions were defined up front, each tied to a downstream design decision. Generic plotting was deliberately avoided in favour of decision-driving analysis.

### 5.2 Q1 — Image accessibility audit

**Method:** Single `Path.iterdir()` scan of `img_resized/` builds a set of existing filenames; membership-test against `df["img_filename"]`. O(N) once vs. O(N) syscalls.

**Result:**
- 150,000 `.jpg` files found.
- 149,819 / 149,819 GT-referenced files present — **0 missing**.
- 181 orphan `.jpg` files in `img_resized/` with no GT entry.

**Implication:** No row drop needed before MVP 2 image-branch training. The 181 orphans are ignored.

### 5.3 Q2 — Tweet length distribution

**Method:** `df["tweet_text"].str.len()` for chars, `df["tweet_text"].str.split().str.len()` for words.

**Result:**

| Statistic | Chars | Words |
|-----------|------:|------:|
| 50th pct | 82 | 11 |
| 90th pct | 126 | 19 |
| 95th pct | 133 | 21 |
| 99th pct | 139 | 24 |
| max | 164 | 38 |

**100% of tweets are ≤ 128 words.**

**Implication:** RoBERTa `max_seq_len=128` is adequate (probably overkill — could plausibly drop to 64 after BPE tokenisation). Decision deferred to MVP 1 — re-verify after BPE encoding (BPE typically expands by 1.3–1.6× word count for English social text).

### 5.4 Q3 — Annotator agreement (T3) by T2 class

**Method:** Group by T2, compute mean/median/Q25/Q75 of T3 per class.

**Result:**

| T2 class | mean T3 | median T3 | Q25 | Q75 | n |
|----------|--------:|----------:|----:|----:|--:|
| NotHate    | 0.838 | 1.000 | 0.667 | 1.000 | 112,842 |
| Racist     | 0.714 | 0.667 | 0.667 | 0.667 |  11,927 |
| Sexist     | **0.705** | 0.667 | 0.667 | 0.667 |   3,495 |
| Homophobe  | 0.743 | 0.667 | 0.667 | 0.667 |   3,871 |
| Religion   | 0.714 | 0.667 | 0.667 | 0.667 |     163 |
| OtherHate  | 0.728 | 0.667 | 0.667 | 0.667 |   5,811 |

**Striking pattern:** the interquartile range of every hate class collapses onto T3 = 0.667. NotHate is unambiguous (median = 1.0); every hate class has at least 50% of its rows at the 2-of-3-agreement point.

**Implication:** **T3 is the right signal for human-in-the-loop deployment routing** — the framing-document deployment story (Section 5 of `Cyberbullying_Detection_Report_Framing.md`) is empirically supported. Sexist is the strongest candidate for human review (lowest agreement). For the technical report's *Significance* and *Deployment* sections, this finding is load-bearing.

### 5.5 Q4 — Text features by T1 (hashtags, mentions, URLs, emoji)

**Method:** Vectorised `.str.count(regex)` on the **full dataset (n = 149,819)**, grouped by T1, ratio of hate to no-hate means.

**Result:**

| Feature | no_hate mean | hate mean | hate / no_hate ratio |
|---------|-------------:|----------:|---------------------:|
| `n_urls`     | 1.066 | 1.063 | **0.997** (noise) |
| `n_hashtags` | 0.191 | 0.241 | **1.262 (+26%)** |
| `n_mentions` | 0.520 | 0.565 | **1.087 (+9%)** |
| `n_emoji`    | 0.600 | 0.427 | **0.712 (−29% inverse)** |

**Implication:**
- **Hashtags +26%** and **emoji −29%** are real signals. Mention count's +9% is a meaningful but weaker signal than the original 20K-sample suggested (+14%) — magnitudes shifted on the full dataset.
- **URLs are noise as a count** because every Twitter post embeds the image as a `t.co` URL. Use as a binary `url_present` instead. (See Q4 discussion in Section 6.4 — the binary still failed the gate due to zero variance for the same reason.)

### 5.6 Q5 — OCR text presence and length by T2 class

**Method:** `ThreadPoolExecutor(max_workers=16)` reads all 149,819 OCR JSON files (`img_txt/{tweet_id}.json`) into memory, computes `ocr_present` (binary) and `ocr_len` (chars), groups by T2.

**Result:**

| T2 class | OCR present rate | mean OCR length (chars) | n |
|----------|-----------------:|------------------------:|--:|
| NotHate    | 38.8% | 30.1 | 112,842 |
| Racist     | 38.2% | 29.4 |  11,927 |
| Sexist     | 41.3% | 31.1 |   3,495 |
| Homophobe  | 36.8% | 26.9 |   3,871 |
| **Religion** | **62.0%** | **62.6** | **163** |
| **OtherHate** | **52.0%** | **55.5** | **5,811** |
| **overall** | **39.5%** | **31.4** | **149,819** |

**Implication:**
- **Religion and OtherHate are clearly meme-driven** (high OCR presence and long OCR text). On these classes, the **OCR-only ablation cell in MVP 2 will be genuinely competitive**.
- For Racist / Sexist / Homophobe / NotHate, OCR coverage is roughly equal (~37–41%) and OCR-only will likely track text-only baseline closely — informative but not differentiating.
- The Religion 62% rate replaces an earlier 70% estimate from a noisy 10-row sample; the full-class result on n = 163 is now defensible.

### 5.7 Q6 — Image dimensions

**Method:** Sample 1,000 random images, open each via PIL, record `(w, h)`. Sampling is appropriate here because image reads are slow and the property of interest (preprocessing compatibility) is dataset-level.

**Result:**

| Statistic | width (px) | height (px) | aspect (w/h) | min_side (px) |
|-----------|-----------:|------------:|-------------:|--------------:|
| 5th pct | 500 | 500 | 0.54 | 500 |
| 50th pct | 527 | 500 | 1.07 | 500 |
| 95th pct | 953 | 933 | 1.91 | 500 |
| max | 2,358 | 1,734 | 4.7 | 500 |

- **All sampled images have `min_side ≥ 500`** (the dataset is pre-resized to that constraint).
- 0% require upsampling for a 224×224 ViT input.

**Implication:** Standard `timm` preprocessing (`shortest-side-resize` → `center-crop` → `224×224`) is appropriate. No special handling needed.

### 5.8 Q7 — Official split balance (load-bearing methodological finding)

**Method:** Read each split file (`train_ids.txt`, `val_ids.txt`, `test_ids.txt`), attach a `split` column, group by split.

**Result:**

| Split | rows | T1 hate rate | mean T3 |
|-------|-----:|-------------:|--------:|
| train | 134,820 | **21.86%** (natural) | 0.785 |
| val   |   4,999 | **50.01%** (engineered) | 0.717 |
| test  |  10,000 | **50.01%** (engineered) | 0.717 |

T2 distribution is also re-balanced in val and test (NotHate ≈ 59%) compared to train (NotHate ≈ 84%).

**Implication (this is the most important methodological finding of EDA):**
- **Val/test were intentionally constructed to be 50/50 hate-balanced** by Gomez et al. 2019. Train is left at the natural Twitter-sampled distribution.
- AUC-ROC and other balanced-class metrics on val/test reflect performance on a balanced sample, **not real-world deployment conditions**.
- **Reporting rule:** every metric must be reported in two forms — (a) raw on the balanced val/test, and (b) recalibrated via Bayes-shift to ~22% hate (the train distribution, which approximates real-world Twitter rates).
- **Class-weighting in training loss must use the train distribution (~22% hate)**, never val/test.
- Val/test must not be re-balanced or re-split — preserving the Gomez 2019 design is required for direct comparability with prior work on this benchmark.

### 5.9 Q8 — Extended-annotator pattern (hypothesis verification)

**Method:** Filter to rows with `n_annotators ≠ 3`, compute hate rate and mean T3 vs the full dataset.

**Result:**
- 70 extended-annotator rows in total (0.047% of the dataset).
- Hate rate: **21.4%** (vs 24.7% overall) — slightly lower.
- Mean T3: **0.834** (vs 0.779 overall) — meaningfully higher.
- T2: 55 NotHate, 3 Racist, 1 Homophobe, 11 still NaN (3-way disagreement persisted even with 4–5 annotators on those 11 rows).

**Implication:** The hypothesis from Section 4.5.2 is empirically confirmed — **extra annotators were brought in to adjudicate originally-disputed cases that turned out to be non-hate** (mostly AAVE / in-group reclamation language). This pattern is documented in `Cell 5` of `01_data_loading.ipynb` and again in this report; the bias-analysis section of the final report should reference it.

### 5.10 Saved chart artefacts

| Chart file | Question | Content |
|------------|----------|---------|
| `outputs/eda_tweet_length.png` | Q2 | Char- and word-length histograms with `max_seq_len=128` reference line |
| `outputs/eda_t3_by_class.png` | Q3 | Stacked bucket plot + mean T3 per T2 class |
| `outputs/eda_text_features_by_T1.png` | Q4 | Mean count by T1 for URLs / hashtags / mentions / emoji |
| `outputs/eda_ocr_by_class.png` | Q5 | OCR presence rate and mean OCR length by T2 class |
| `outputs/eda_image_dimensions.png` | Q6 | Width × height scatter + aspect-ratio histogram |
| `outputs/eda_split_balance.png` | Q7 | Hate rate per split + T2 distribution per split |

### 5.11 Cross-cutting findings

- The dataset is **honestly labelled** but **structurally non-trivial**: keyword-seeded sampling, 50/50 engineered splits, and a non-uniform annotator schema all impose constraints on how results must be interpreted and reported.
- The deployment story (T3-based human routing) is empirically defensible.
- The bias story (AAVE under-flagged due to extended annotator review on contested cases) is empirically visible in the data and needs to be honestly documented.

---

## 6. Notebook 03 — Structured Feature Engineering

### 6.1 Goal
Engineer the 15-dimensional structured feature vector that will feed Branch C of the multimodal architecture, then prune any feature that fails a minimum-correlation threshold against T1.

### 6.2 The 15-feature spec (per technical scope §8.4)

| # | Feature | Type | Source |
|---|---------|------|--------|
| 1 | `text_len_chars` | numeric | tweet_text length in chars |
| 2 | `n_hashtags` | numeric | `#\w+` count |
| 3 | `n_mentions` | numeric | `@\w+` count |
| 4 | `url_present` | binary | tweet contains `https?://...` |
| 5 | `n_emoji` | numeric | matches against 12 Unicode emoji ranges (incl. flags) |
| 6 | `allcaps_frac` | numeric | uppercase letters / all alphabetic letters |
| 7 | `n_exclamations` | numeric | count of `!` |
| 8 | `n_questions` | numeric | count of `?` |
| 9 | `repeated_chars` | numeric | runs of 3+ same letter (e.g. "heyyyy") |
| 10 | `profanity_count` | numeric | better-profanity bundled lexicon (916 words), word-boundary regex |
| 11 | `hate_keyword_count` | numeric | `hatespeech_keywords.txt` (86 terms — the dataset's seed lexicon), word-boundary regex |
| 12 | `vader_neg` | numeric | VADER negative sentiment score |
| 13 | `vader_neu` | numeric | VADER neutral sentiment score |
| 14 | `ocr_len` | numeric | length of OCR text from `img_txt/{id}.json` |
| 15 | `ocr_present` | binary | OCR text non-empty |

### 6.3 Implementation notes

- **OCR text is bulk-loaded** for all 149,819 rows via `ThreadPoolExecutor(max_workers=16)` (~30–60 sec).
- **VADER scoring** is also threaded for consistency with the OCR loader, though VADER is pure Python and the GIL limits realistic speedup; sequential and threaded runtimes are roughly comparable.
- **Emoji regex** covers 12 Unicode ranges including regional indicators (flags), emoticons, supplemental pictographs, and dingbats — a complete pattern, not the simplified two-range version used in initial drafts.
- **Profanity counting via `better-profanity`** required a workaround: the package's `CENSOR_WORDSET` exposes `VaryingString` instances (for leetspeak fuzzy matching) which `re.escape` cannot process. The bundled `profanity_wordlist.txt` is read directly to obtain plain string lexicon (916 words).
- **Hate-keyword counting** uses the dataset's own `hatespeech_keywords.txt` (86 terms — the seed lexicon used during MMHS150K collection).

### 6.4 Correlation gate methodology

After all 15 features are engineered, Pearson correlation with T1 (binary 0/1) is computed for each. Features with `|corr(feature, T1)| < 0.02` are dropped — the threshold is pragmatic: features below this magnitude add noise without lift to a model that has access to RoBERTa's 768-d embedding for the same input.

### 6.5 Correlation audit (full dataset, n = 149,819)

Sorted by `|corr|` descending:

| Feature | corr_T1 | abs_corr | keep? |
|---------|--------:|---------:|:-----:|
| `vader_neg` | +0.1714 | 0.1714 | ✅ |
| `vader_neu` | −0.1326 | 0.1326 | ✅ |
| `n_emoji` | −0.0478 | 0.0478 | ✅ |
| `hate_keyword_count` | +0.0328 | 0.0328 | ✅ (with confound — see 6.6) |
| `ocr_present` | +0.0277 | 0.0277 | ✅ |
| `profanity_count` | +0.0275 | 0.0275 | ✅ |
| `n_hashtags` | +0.0272 | 0.0272 | ✅ |
| `ocr_len` | +0.0271 | 0.0271 | ✅ |
| `n_mentions` | +0.0234 | 0.0234 | ✅ |
| ── threshold |corr| ≥ 0.02 ── | | | |
| `repeated_chars` | −0.0178 | 0.0178 | ❌ |
| `allcaps_frac` | +0.0131 | 0.0131 | ❌ |
| `text_len_chars` | +0.0095 | 0.0095 | ❌ |
| `n_exclamations` | +0.0024 | 0.0024 | ❌ |
| `n_questions` | +0.0014 | 0.0014 | ❌ |
| `url_present` | NaN | NaN | ❌ (zero variance — see 6.6) |

**Kept (9):** `vader_neg`, `vader_neu`, `n_emoji`, `hate_keyword_count`, `ocr_present`, `profanity_count`, `n_hashtags`, `ocr_len`, `n_mentions`

**Dropped (6):** `repeated_chars`, `allcaps_frac`, `text_len_chars`, `n_exclamations`, `n_questions`, `url_present`

**Top-line interpretation:**
- VADER sentiment dominates the structured branch. Only two features have `|corr| > 0.05`; the remaining seven survivors hover at 0.023–0.048.
- The structured branch is intentionally lightweight (no TabNet) — and the audit confirms there is little independent signal beyond what RoBERTa would learn anyway. This vindicates the v1.1 spec's decision to keep Branch C as a 15-d (now 9-d) raw concatenation.

### 6.6 Documented confounds

#### 6.6.1 `hate_keyword_count` — confounded by dataset construction
The 86 hate keywords are the same Hatebase terms Gomez et al. 2019 used to **seed** MMHS150K. Every tweet in the dataset matches at least one term by construction (mean 0.98, std 0.27 — near-constant). The +0.033 correlation with T1 is structurally inflated by the sampling pipeline, not by an independent semantic signal.
**Decision:** kept for completeness rather than dropped — over-pruning a dataset-level artefact alongside genuine noise risks too-aggressive feature reduction. **Flagged for the report's *Limitations* section.**

#### 6.6.2 `url_present` — zero variance
Every MMHS150K tweet contains a Twitter `t.co` URL wrapper (Twitter automatically embeds image links this way). The feature is therefore effectively constant across the dataset, producing a NaN correlation with T1 (zero-variance denominator). This is a property of Twitter's link-handling, not a property of hate speech.
**Decision:** dropped from the structured features. The feature **is preserved in the engineering code** for future use on non-Twitter datasets where it may carry signal.

### 6.7 Output schema — `data/processed/structured_features.csv`

| Column | dtype | range |
|--------|-------|-------|
| `tweet_id` | str | 149,819 unique values |
| `vader_neg` | float32 | [0.000, 0.960] |
| `vader_neu` | float32 | [0.040, 1.000] |
| `n_emoji` | int16 | [0, 62] |
| `hate_keyword_count` | int16 | [0, 10] |
| `ocr_present` | int8 | {0, 1} |
| `profanity_count` | int16 | [0, 11] |
| `n_hashtags` | int16 | [0, 14] |
| `ocr_len` | int32 | [0, 3128] (clipping recommended at ~99th pct = 391 before model input) |
| `n_mentions` | int16 | [0, 9] |

**File:** 6.6 MB CSV. Shape `(149,819, 10)`. Zero nulls in any column. Joinable to `labels_parsed.csv` on `tweet_id`.

### 6.8 Per-feature summary statistics

```
                       count    mean     std   min    25%    50%     75%      max
vader_neg           149819.0   0.175   0.188  0.00  0.000  0.147   0.299     0.96
vader_neu           149819.0   0.728   0.207  0.04  0.575  0.719   1.000     1.00
n_emoji             149819.0   0.557   1.561  0.00  0.000  0.000   0.000    62.00
hate_keyword_count  149819.0   0.981   0.267  0.00  1.000  1.000   1.000    10.00
ocr_present         149819.0   0.395   0.489  0.00  0.000  0.000   1.000     1.00
profanity_count     149819.0   1.215   0.816  0.00  1.000  1.000   1.000    11.00
n_hashtags          149819.0   0.203   0.792  0.00  0.000  0.000   0.000    14.00
ocr_len             149819.0  31.437  84.341  0.00  0.000  0.000  21.000  3128.00
n_mentions          149819.0   0.531   0.831  0.00  0.000  0.000   1.000     9.00
```

**Pre-model preprocessing recommendations** (to be implemented in MVP 2 dataloader):
- **Standardise** the float and count features (z-score normalisation per training-set statistics).
- **Clip `ocr_len` at the 99th percentile (~391 chars)** before standardisation to prevent the long tail (max 3,128 chars from a screenshot of an article) from dominating the standardised vector.
- Binary features (`ocr_present`) are already in [0, 1] and should not be standardised.

---

## 7. Methodological Decisions Locked During Phase 1

Captured in `CLAUDE.md` (project-root file). Each is "locked" — re-litigation requires explicit user reopening.

| # | Decision | Source / rationale |
|---|----------|-------------------|
| 1 | **T4 removed as a supervised target** | Pre-Phase-1 (technical scope v1.1). Circular dependency: labels would have been derived from model outputs. |
| 2 | **Structured branch is lightweight** (15 raw features, no TabNet, concatenated raw into fusion) | Pre-Phase-1 (v1.1 §8.4). Empirically supported by Phase 1 §6.5 audit — most features have negligible correlation. |
| 3 | **Phased MVP ladder** (T1 → T1+T2 → naive fusion → gated fusion → +T3 + bias) | Pre-Phase-1 (v1.1 §6b). |
| 4 | **Tie-break for T2** = lowest-index hate class on 3-way tie | Phase 1 §4.4.2. Documented and deterministic. |
| 5 | **T2 = NaN where `max_count == 1`** (no-majority); add `t2_valid` boolean; mask T2 loss accordingly | Phase 1 §4.5.3 (this report). |
| 6 | **Drop rows with `len(labels) == 1`** (4 rows, no agreement signal) | Phase 1 §4.5.3. |
| 7 | **Generalise T3 = max_count / len(labels)** so T3 ∈ [0, 1] for any annotator count | Phase 1 §4.5.3. |
| 8 | **Use the official MMHS150K splits as-is** (134,823 / 5,000 / 10,000) | EDA Q7. Locked by Gomez 2019 design. Never re-balanced. |
| 9 | **Val/test are 50/50 hate-balanced; train is naturally ~22% hate** — class weighting must use the train distribution; metrics must be reported balanced and recalibrated | EDA Q7. |
| 10 | **Religion class (~0.3%) is genuinely rare, not a measurement error** — no over-sampling, augmentation, or synthesis. Focal Loss + intra-hate-class weighted sampling are the only mitigations | EDA + technical scope. Documented as a known limitation. |
| 11 | **Pre-extracted OCR text is used directly from `img_txt/{tweet_id}.json`** | Pre-Phase-1. Confirmed in §3.1, §5.6. |
| 12 | **Hashtags, mentions, URLs preserved as RoBERTa tokens** | Pre-Phase-1. Empirically supported (hashtag/mention counts carry T1 signal — EDA Q4). |
| 13 | **Augmentation:** horizontal flip + mild colour jitter only; no vertical flip or rotation (text orientation matters in memes) | Pre-Phase-1. |
| 14 | **RoBERTa pre-train sequence:** Cyberbullying Kaggle (3 epochs) → MMHS150K (5 epochs) | Pre-Phase-1. |
| 15 | **XAI claims are correlational, not causal.** Cite Jain & Wallace 2019, Wiegreffe & Pinter 2019. | Pre-Phase-1. |
| 16 | **Novelty framing:** "application of gated fusion to the documented MMHS150K failure mode," not "we invented gated fusion." | Pre-Phase-1. |

---

## 8. Open Items & Phase 2 Preconditions

### 8.1 Open items deferred from Phase 1

- **`max_seq_len=128` decision:** EDA Q2 confirms 100% of tweets are ≤ 128 words; could plausibly drop to 64 after BPE tokenisation. Re-verify after RoBERTa BPE encoding in MVP 1 (Notebook 06). Tentatively keep 128 for now.
- **`ocr_len` clipping at 99th percentile (~391 chars):** to be applied in the MVP 2 dataloader before standardising the structured feature vector.
- **Profanity feature semantics:** `better-profanity`'s leetspeak fuzzy matching is currently disabled (we use the raw bundled wordlist for plain regex). Consider re-evaluating in Phase 2 if the structured branch shows weakness.

### 8.2 Phase 2 preconditions (all satisfied)

- ✅ `data/processed/labels_parsed.csv` keyed on `tweet_id`, joinable to anything else.
- ✅ `data/processed/structured_features.csv` keyed on `tweet_id`, joinable to labels.
- ✅ Image accessibility audit complete; no row drop required.
- ✅ Official splits documented; per-split T1/T2 distributions known.
- ✅ Methodological decisions locked in `CLAUDE.md` (project-root file).
- ✅ Six EDA charts saved in `outputs/`.

### 8.3 Next notebook (immediate next step)

**`04_train_val_test_split.ipynb`** — load the official `splits/*_ids.txt` files, attach a `split` column to the merged label + feature DataFrame, persist as a single split-aware artefact, re-verify per-split T1/T2 distributions match Phase 1 EDA Q7.

After 04: `05_roberta_pretrain_kaggle.ipynb` (RoBERTa warm-start on Cyberbullying Kaggle CSV) → `06_mvp1_roberta_t1.ipynb` (the MVP 1 deliverable: working binary hate classifier).

---

## 9. References

### 9.1 Datasets
- **Gomez, R., Gibert, J., Gomez, L., & Karatzas, D. (2019).** Exploring Hate Speech Detection in Multimodal Publications. *WACV 2020*. — MMHS150K paper. Dataset: https://gombru.github.io/2019/10/09/MMHS/
- **Kiela, D., Firooz, H., Mohan, A., et al. (2020).** The Hateful Memes Challenge: Detecting Hate Speech in Multimodal Memes. *NeurIPS 2020*. — Hateful Memes dataset.
- **Andrew M. (2020).** Cyberbullying Classification — Kaggle dataset. URL: https://www.kaggle.com/datasets/andrewmvd/cyberbullying-classification

### 9.2 Methods used in Phase 1
- **Hutto, C.J., & Gilbert, E. (2014).** VADER: A Parsimonious Rule-based Model for Sentiment Analysis of Social Media Text. *ICWSM 2014*. — `vaderSentiment` package.
- **Better Profanity package** (Pham, A.) — bundled English profanity wordlist, version 0.7.0.

### 9.3 Companion project documents
- `Multimodal_Cyberbullying_Detection_v1.1.md` — full technical scope (project root: `Reports/`).
- `Cyberbullying_Detection_Report_Framing.md` — significance, deployment philosophy, defence preparation.
- `CLAUDE.md` — operational rules and locked methodological decisions (project root).

---

## Appendix A — Saved Artefacts

### A.1 Data artefacts (under `data/processed/`)

| File | Size | Shape | Notes |
|------|-----:|------:|-------|
| `labels_parsed.csv` | 28.6 MB | (149,819, 10) | Phase-1 final labels |
| `structured_features.csv` | 6.6 MB | (149,819, 10) | tweet_id + 9 surviving features |

### A.2 Chart artefacts (under `outputs/`)

| File | Source | Resolution |
|------|--------|-----------:|
| `eda_tweet_length.png` | Notebook 02 Q2 | 120 dpi |
| `eda_t3_by_class.png` | Notebook 02 Q3 | 120 dpi |
| `eda_text_features_by_T1.png` | Notebook 02 Q4 | 120 dpi |
| `eda_ocr_by_class.png` | Notebook 02 Q5 | 120 dpi |
| `eda_image_dimensions.png` | Notebook 02 Q6 | 120 dpi |
| `eda_split_balance.png` | Notebook 02 Q7 | 120 dpi |

### A.3 Notebook artefacts (under `notebooks/`)

| Notebook | Cells | Status |
|----------|------:|--------|
| `01_data_loading.ipynb` | 6 | Executed end-to-end |
| `02_eda.ipynb` | 11 | Executed end-to-end |
| `03_structured_features.ipynb` | 8 | Executed end-to-end |

---

## Appendix B — Project File Tree (post-Phase 1)

```
D:\Cyberbullying Detection\
├── CLAUDE.md                                    # operational rules + locked decisions
├── data\
│   ├── MMHS150K.zip                             # source archive (kept as backup)
│   ├── MMHS150K\                                # extracted dataset
│   │   ├── MMHS150K_GT.json                     # ground truth dict (149,823 entries)
│   │   ├── MMHS150K_readme.txt
│   │   ├── hatespeech_keywords.txt              # 86 Hatebase seed terms
│   │   ├── img_resized\                         # 150,000 .jpg files
│   │   ├── img_txt\                             # 149,819 OCR JSONs (Google Vision)
│   │   └── splits\
│   │       ├── train_ids.txt                    # 134,823
│   │       ├── val_ids.txt                      #   5,000
│   │       └── test_ids.txt                     #  10,000
│   ├── The Hateful Memes Challenge\             # held-out cross-domain test
│   │   ├── train.jsonl                          # 8,499
│   │   ├── dev.jsonl                            #   499
│   │   ├── test.jsonl                           #   999
│   │   ├── img\
│   │   ├── README.md
│   │   └── LICENSE.txt
│   ├── cyberbullying_tweets.csv                 # Kaggle text-only, 47,985 rows
│   └── processed\                               # Phase 1 outputs
│       ├── labels_parsed.csv                    # (149,819 × 10), 28.6 MB
│       └── structured_features.csv              # (149,819 × 10), 6.6 MB
├── notebooks\
│   ├── 01_data_loading.ipynb                    # Phase 1 ✓
│   ├── 02_eda.ipynb                             # Phase 1 ✓
│   └── 03_structured_features.ipynb             # Phase 1 ✓
├── models\                                      # (empty — populated from MVP 1)
├── outputs\                                     # six EDA chart PNGs
└── Reports\
    ├── Multimodal_Cyberbullying_Detection_v1.1.md
    ├── Cyberbullying_Detection_Report_Framing.md
    └── Phase1_Data_Engineering_Report.md        # this file
```

---

*Document version 1.0 — Phase 1 Data Engineering Report — prepared for use as report-writing source material and as a phase-handover snapshot for downstream sessions.*
