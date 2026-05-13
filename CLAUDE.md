# CLAUDE.md — Multimodal Cyberbullying Detection

## 1. Project Status
Phase 1 (data engineering) complete — notebooks 01, 02, 03 all done. Next task: Notebook 05 (MVP 1 — text-only Twitter-RoBERTa baseline on T1). Pre-train notebook 04 must run first on Cyberbullying Kaggle.

## 2. What This Project Is
Multimodal hate-speech classifier on MMHS150K. Three branches — CLIP ViT-B/16 (`openai/clip-vit-base-patch16`, image), Twitter-RoBERTa (`cardiffnlp/twitter-roberta-base-2022-154m`, text), 9-d structured vector (pruned from 15 engineered features by |corr|<0.02 threshold) — fused via gated cross-modal attention. Targets the documented MMHS150K failure mode where naive fusion underperforms text-only baselines.

## 2b. Critical Findings (data-driven, must inform downstream decisions)
- **T3 deployment-routing premise CONFIRMED (EDA Q3).** Mean T3 by T2 class: NotHate 0.838 (median 1.0); all hate classes 0.705–0.743 (median 0.667, Q25=Q75=0.667). Every hate class has ambiguous majority-vote cases — Sexist is the worst (0.705). T3 is the right signal to route borderline hate cases to human reviewers in the deployment story.
- **Val/test are 50/50 hate-balanced; train is naturally ~22% hate (EDA Q7).** Affects how metrics are interpreted and how training loss is class-weighted. See Section 9 rules.
- **All 149,819 GT-referenced images are present on disk (EDA Q1).** Plus 181 orphan .jpg files in `img_resized/` that have no GT entry — ignore them.
- **Image dimensions: every image has min_side ≥ 500 (EDA Q6).** No upsampling needed for 224×224. Use timm's standard shortest-side-resize + center-crop.
- **39% of MMHS150K images contain OCR text (EDA Q5).** Religion (70%) and OtherHate (52%) are meme-driven; other hate classes 36–43%. OCR-only ablation is genuinely competitive on text-heavy classes.
- **Hashtag count (+20%), mention count (+14%), emoji count (-27%) carry T1 signal (EDA Q4).** URL count is noise — use as binary instead.
- **Extended-annotator hypothesis CONFIRMED (EDA Q8).** 70 rows have n=2/4/5 annotators; they skew NotHate (21% vs 25% overall) with higher T3 (0.83 vs 0.78). Pattern: extra annotators were used to adjudicate originally-disputed cases that turned out non-hate.

## 3. Key Decisions Already Made (do not re-debate)
- T4 removed as supervised target (circular dependency). Only T1, T2, T3 exist.
- Structured branch is 9 features concatenated into fusion (pruned from 15 by |corr|<0.02 threshold in notebook 03). Kept: `vader_neg`, `vader_neu`, `n_emoji`, `hate_keyword_count`, `ocr_present`, `profanity_count`, `n_hashtags`, `ocr_len`, `n_mentions`. No TabNet.
- Phased MVP ladder: T1 → T1+T2 → naive fusion → gated fusion → +T3 + bias analysis.
- Augmentation: horizontal flip + mild colour jitter only. No vertical flip, no rotation.
- OCR text is pre-extracted in `data/MMHS150K/img_txt/`. Do not run OCR.
- Hashtags, mentions, URLs are preserved as RoBERTa tokens. Do not strip.
- Cross-domain test = Hateful Memes. Never trained on.
- Splits: use the official MMHS150K `splits/` directory (train 134,823 / val 5,000 / test 10,000). Not stratified, but matches Gomez et al. 2019 for comparability.
- RoBERTa pre-train: Cyberbullying Kaggle (3 epochs) → MMHS150K fine-tune (5 epochs).
- XAI claims are correlational, not causal. Cite Jain & Wallace 2019, Wiegreffe & Pinter 2019.
- Novelty framing: "application of gated fusion to MMHS150K failure mode" — not "we invented gated fusion."
- LoRA adapters (rank 16) on both encoders via PEFT library — never full fine-tune.
- CLIP image embedding is 512-d; project to 768-d via `Linear(512→768)` before fusion (cross-attention dim compatibility).
- MoE-style fusion is documented as future stretch goal only — do NOT implement during MVP 1–5 ladder.

## 4. Targets
- **T1 (binary hate, Focal BCE γ=2):** not started. Active from MVP 1.
- **T2 (6-class category, Focal CE γ=2):** not started. Active from MVP 2.
- **T3 (annotator agreement 0–1, MSE):** auxiliary, not primary. Active from MVP 5 only.

## 5. File Locations
- Project root: `D:\Cyberbullying Detection`
- Data: `data/MMHS150K/` (`MMHS150K_GT.json`, `img_resized/`, `img_txt/`, `splits/`), `data/The Hateful Memes Challenge/`, `data/cyberbullying_tweets.csv`
- Notebooks: `notebooks/`
- Model checkpoints: `models/<branch_or_mvp_name>/`
- Charts, EDA outputs, ablation tables, logs: `outputs/`
- Specs: `Reports/Multimodal_Cyberbullying_Detection_v1.2.md` (scope), `Reports/Cyberbullying_Detection_Report_Framing.md` (significance/defence)

## 6. Notebook Order
1. [x] `01_data_loading.ipynb` — parse GT JSON, derive T1/T2/T3
2. [x] `02_eda.ipynb` — class distribution, agreement distribution, image accessibility audit
3. [x] `03_structured_features.ipynb` — engineer 15 features, prune to 9 by |corr|<0.02
4. [ ] `04_roberta_pretrain_kaggle.ipynb` — Cyberbullying Kaggle warm-start
5. [ ] `05_mvp1_roberta_t1.ipynb` — MVP 1 deliverable
6. [ ] `06_mvp2_clip_xgb.ipynb` — CLIP + XGBoost baselines, add T2 head
7. [ ] `07_mvp3_naive_fusion.ipynb` — replicate documented failure mode
8. [ ] `08_mvp4_gated_fusion.ipynb` — core contribution
9. [ ] `09_mvp5_t3_ablations_bias.ipynb` — full system + bias analysis
10. [ ] `10_per_sample_modality_analysis.ipynb` — categorise each test sample (Convergent Correct / Text Saved / Image Saved / Emergent Multimodal / Fusion Failure); runs after MVP 5
11. [ ] `11_xai_dashboard_prep.ipynb` — GRAD-CAM, captum, SHAP, gate viz, FAISS

## 7. Environment
- OS: Windows 11 Pro. Shell: PowerShell.
- Python 3.11 in conda env `cyberbully_project`.
- Activate: `conda activate cyberbully_project`
- Launch Jupyter: `jupyter lab`
- Launch dashboard: `streamlit run app/dashboard.py`
- Always use the `cyberbully_project` kernel in notebooks.

## 8. Hardware Rules
- GPU: RTX 3060 Laptop, **6 GB VRAM**.
- Never train CLIP and Twitter-RoBERTa simultaneously without fp16 — will OOM.
- CLIP ViT-B/16 fine-tune (LoRA rank 16): batch size 16 with fp16, 8 without.
- Twitter-RoBERTa fine-tune (LoRA rank 16): batch size 16 with fp16 at seq_len 128.
- Joint fusion (MVP 4): freeze branches first, then unfreeze with fp16 + gradient accumulation. If still OOM, move to Kaggle T4 (16 GB).
- Gradient accumulation steps = 4 (physical batch 16 → effective batch 64). Required for MVP 4 fusion training.
- Always wrap training loops in `torch.cuda.amp.autocast()` + `GradScaler` for fp16.
- Always call `torch.cuda.empty_cache()` between branch-training cells.
- Set `num_workers=2`, `pin_memory=True` in DataLoaders. Higher workers will thrash on 6 GB VRAM + Windows.

## 9. Dataset Rules
- Primary key in MMHS150K: `tweet_id` (string). Image = `img_resized/{tweet_id}.jpg`. OCR = `img_txt/{tweet_id}.json`.
- Some images may be missing or corrupt — drop with logged count, never silently skip.
- Hateful Memes uses its own `id` field. Not joined to MMHS150K.
- Cyberbullying Kaggle is text-only. No image join.
- After any merge or filter, print `df.shape` and `df.isna().sum()`.
- Never load all 150K images into memory. Use a Dataset that reads from disk per `__getitem__`.
- **MMHS150K val/test are intentionally 50/50 hate-balanced** (Gomez 2019 design). Train is at the natural ~22% hate rate.
- Never re-balance the splits. Never replace them with a custom stratified split.
- When reporting metrics, always include BOTH: (a) balanced-test metrics (raw val/test numbers), and (b) recalibrated-distribution metrics (Bayes-shift to ~22% hate so numbers reflect deployment conditions).
- Class weighting in training loss must use the **train distribution (~22% hate)**, NOT val/test (50% hate).

## 10. MMHS150K-Specific Rules
- Each tweet has (nominally) 3 annotator labels in `labels` (0=NotHate, 1=Racist, 2=Sexist, 3=Homophobe, 4=Religion, 5=OtherHate). 74 tweets have non-3 annotator counts (30 with n=2, 37 with n=4, 3 with n=5, 4 with n=1). 4 dropped (n=1, no agreement signal); rest kept with T3 generalized to `max_count/len(labels)`. See `n_annotators` column in `labels_parsed.csv`.
- T1 = `1` if majority label != 0, else `0`.
- T2 = majority-vote class. On 2-vs-1 majority, take the majority class. On 3-way disagreement (e.g. labels `[1,2,4]`), set `T2=NaN` and `t2_valid=False`. The original lowest-index tie-break rule applied during initial parsing but was overridden by the NaN masking decision in Cell 6 of notebook 01.
- T3 = (max vote count) / `len(labels)`.
  - For n=3: 1.0 / 0.667 / 0.333
  - For n=4: 1.0 / 0.75 / 0.5
  - For n=5: 1.0 / 0.8 / 0.6 / 0.4
  - Computed in notebook 01 Cell 6 after generalization.
- **T2 ambiguity rule:** `T2 = NaN` where no single hate category has majority support (i.e. `max_count < 1/2` of total annotators). For n=3 this means `T3 = 0.333`. For n=4: `max_count <= 2` (`T3 <= 0.5`). For n=5: `max_count <= 2` (`T3 <= 0.4`). The `t2_valid` boolean column in `labels_parsed.csv` encodes this definitively — use that, not the T3 threshold. T1 and T3 stay valid for these rows. Use masked CE during T2 training to ignore NaN rows.
- **Religion class (~0.3%) is NOT a bug.** It reflects real Twitter distribution. Do not over-sample, augment, or otherwise "fix" it. Focal Loss + class-weighted sampling within hate categories is the only mitigation. Document the low-recall expectation as a known limitation.
- **Official splits are kept as-is.** Train 134,823 / val 5,000 / test 10,000. The val:test ratio (1:2) is by Gomez 2019 design — smaller val, larger test for more reliable test metrics. Do not "rebalance" it.
- Pre-extracted OCR text lives in `img_txt/{tweet_id}.json`. Use it directly.
- `tweet_url` is for traceability only. Never use as a feature.

## 11. Python Rules
- KeyError on a column name usually means a rename happened before the column list was defined — re-check cell order.
- Always assign back: `df = df.rename(...)`. Do not rely on `inplace=True`.
- Always specify `dtype` when reading large CSVs to avoid int/object surprises.
- Never use `df.iterrows()` over 150K rows — use vectorised ops.
- Wrap OCR JSON parsing in try/except and log failures. Do not crash the full pass.
- Always seed: `random.seed(42)`, `np.random.seed(42)`, `torch.manual_seed(42)`, `torch.cuda.manual_seed_all(42)`.

## 12. Model Rules
- Phased training is mandatory. Do not jump to fusion before unimodal baselines exist.
- Focal Loss (γ=2) for T1 and T2. No vanilla cross-entropy on these.
- Entropy regularisation in MVP 4: `L_total = L_task − 0.05 · H(gates_batch_mean)`. Not optional.
- Required diagnostics every training run: per-epoch gate entropy, per-class gate statistics, gate weight histograms saved to `outputs/`.
- If average gate entropy < 0.5 by epoch 3, gates have collapsed — stop and inspect.
- If `g_image < 0.15` across all samples in final eval, report honestly: multimodal claim is not supported.
- Identity-term masking: replace identity terms with `[IDENTITY]` token at 20% probability during training. Mandatory in MVP 5.
- Parameter-matched ablation: train RoBERTa-large (355M) text-only baseline. Required ablation cell.
- OCR-text-only ablation: feed pre-extracted OCR into Twitter-RoBERTa. Required ablation cell.
- Loss weights: λ1=1.2, λ2=1.0, λ3=0.4. Tune via small grid search at MVP 5.
- Two-stage LR: linear warmup (10% of steps) + cosine decay. Peak LRs differ per component — LoRA adapters 1e-4; gate / fusion / heads 1e-3.

## 13. How To Work — General Rules
**Communication:**
- No "great question" or filler affirmations.
- No preamble before code.
- No end-of-response summary of what was just done.
- Direct. Factual. No hedging. Use commands, not "should consider."

**Code behaviour:**
- Before writing code, state what it does and how it will be verified.
- After writing code, run it.
- Never claim code works without running it first.
- One instruction at a time. No dumped checklists.
- One notebook cell at a time. Never dump whole notebooks.

**Data hygiene per cell:**
- Each cell that transforms data must print `df.shape` and `df.isna().sum()` after.
- Each cell that produces a chart must save it via `plt.savefig('outputs/<descriptive_name>.png', dpi=120, bbox_inches='tight')`.

**Files:**
- Edit existing files in preference to creating new ones.
- Never create READMEs, summary docs, or tutorials unless explicitly asked.

**Errors:**
- Identify root cause, not symptom.
- KeyError on column name → check cell order, rename happened earlier.
- CUDA OOM → reduce batch size first, then fp16, then gradient accumulation, then move to Kaggle T4.

## 14. What Not To Do
- Never auto-commit to git. Commit only when explicitly asked.
- Never run `pip install` or `conda install` without asking. Env is already configured.
- Never download datasets. All three are on disk.
- Never run OCR on MMHS150K images. Pre-extracted in `img_txt/`.
- Never strip hashtags, mentions, or URLs from RoBERTa input.
- Never add TabNet to the structured branch.
- Never re-introduce T4 as a supervised target.
- Never apply vertical flip or rotation to image augmentation.
- Never train ViT + RoBERTa jointly without fp16 and frozen branches first.
- Never claim gate weights are causal explanations. They are correlational interpretability.
- Never claim the project solves bias. It acknowledges and partially mitigates.
- Never claim the system is deployment-ready. It is an academic research artefact.
- Never display raw hate content in the public dashboard. Use blurred or synthetic samples.
- Never skip the gate-entropy diagnostic in MVP 4 training runs.
- Never over-sample, augment, or synthesise the Religion class (~0.3%). It is real distribution, not measurement error.
- Never re-split or "rebalance" the official MMHS150K splits. Train 134,823 / val 5,000 / test 10,000 is by Gomez 2019 design.
- Never full fine-tune CLIP or Twitter-RoBERTa. Use PEFT LoRA adapters (rank 16) only.
- Never implement MoE-style fusion during MVP 1–5. Documented as future stretch goal only.
