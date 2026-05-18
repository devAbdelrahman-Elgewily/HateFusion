# Identity Lexicon Build Report (v1.0)
**Source:** HateXplain (Mathew et al. 2021, AAAI). 20,148 posts × 3 annotators with target-community annotations. Downloaded 2026-05-17 from `https://github.com/hate-alert/HateXplain`.
**Output:** `data/processed/identity_lexicon.json` (346,361 bytes).
## Methodology
1. **Parse HateXplain `dataset.json`.** Total posts: 20,148.
2. **Determine the per-post target community.** For each post, take the majority vote (≥ 2 of 3 annotators) of the `target` field across annotators. Posts with majority `None` or `Other`, or with no clear majority, are skipped. This filter retains posts where the annotator consensus identifies a specific targeted identity community.
   - Posts with clear community target: **11,814** (of 20,148, 58.6 %).
   - Skipped: `majority_None`=6,588, `no_majority`=1,095, `majority_Other`=651.
3. **Token aggregation per community.** From the cleaned `post_tokens` list of each retained post, accumulate per-community token frequency Counters. Apply the following filters before counting: lowercase normalisation; drop tokens of length < 3; drop English stopwords (hardcoded list of ~150 items); drop tokens starting with `@` (user mentions), `#`, `http`/`www`; drop tokens that are pure punctuation, pure numeric, or non-leading-alphanumeric.
   - Total tokens in targeted posts (after filters): **137,892**.
   - Unique tokens in targeted posts: **19,942**.
4. **PMI scoring per (community, token).** For each token *t* and community *C*, compute add-eps-smoothed pointwise mutual information against the contrast distribution of all other targeted posts:

   ```
   PMI(t, C) = log( (count(t,C) + eps) / N_C    /    (count(t,!C) + eps) / N_!C )
   ```

   Admit token *t* into the lexicon for community *C* if `count(t,C) >= 5` AND `PMI(t,C) >= 1.0`.
   - **F_min = 5.** Selected to require minimum statistical support before any community claim; chosen empirically — F_min = 3 admitted many spurious low-frequency tokens; F_min = 10 dropped real identity terms for smaller communities (e.g. Disability, Christian).
   - **PMI ≥ 1.0.** A token at this threshold is at least *e ≈ 2.72×* more frequent in the community-targeting subset than in the contrast (non-community-targeting-but-still-targeted) subset. Documented as the working point for this lexicon.
   - **Smoothing constant ε = 1e-6** in probability space. Required because some tokens have zero contrast frequency (would otherwise log(0)). Small enough not to dominate scores at the F_min ≥ 5 working point.
5. **Aggregate into single lexicon.** Tokens are de-duplicated; a single token can identify multiple communities (recorded as a list per entry). Entries also retain the per-community HateXplain frequency and per-community PMI score for downstream debugging.
6. **MMHS150K coverage audit.** For every entry, record its raw frequency in the MMHS150K training-split tweet text (the cleaned `tweet_text` column of `labels_parsed.csv` filtered to the official train_ids.txt). Tokenisation uses whitespace + stripped punctuation + lowercase to match the simple lexical-overlap intended use case in MVP4-IW.

## Token-count summary by community
| Community | HateXplain posts | Lexicon tokens identifying this community |
|---|---:|---:|
| African | 3,014 | 237 |
| Islam | 1,881 | 253 |
| Homosexual | 1,650 | 126 |
| Jewish | 1,605 | 185 |
| Women | 1,179 | 116 |
| Refugee | 813 | 166 |
| Arab | 546 | 47 |
| Caucasian | 390 | 38 |
| Asian | 335 | 43 |
| Hispanic | 247 | 30 |
| Men | 53 | 8 |
| Disability | 39 | 5 |
| Christian | 32 | 3 |
| Hindu | 14 | 2 |
| Indian | 9 | 1 |
| Economic | 5 | 0 |
| Indigenous | 1 | 0 |
| Buddhism | 1 | 0 |
| **Total unique tokens** | — | **1,177** |

## Top 20 tokens by HateXplain frequency (within targeted posts)
| Rank | Token | Communities | HateXplain freq | MMHS150K train freq | Max PMI |
|---:|---|---|---:|---:|---:|
| 1 | `white` | Caucasian | 1,923 | 2,985 | 2.03 |
| 2 | `nigger` | African | 1,892 | 4,536 | 3.04 |
| 3 | `jews` | Jewish | 1,034 | 15 | 2.92 |
| 4 | `kike` | Jewish | 980 | 6 | 3.74 |
| 5 | `women` | Men, Women | 924 | 308 | 2.19 |
| 6 | `people` | Caucasian, Disability | 831 | 1,284 | 1.12 |
| 7 | `muslim` | Islam | 740 | 76 | 3.14 |
| 8 | `niggers` | African, Arab | 737 | 35 | 2.11 |
| 9 | `bitch` | Women | 572 | 3,047 | 2.65 |
| 10 | `faggot` | Homosexual | 554 | 5,212 | 2.98 |
| 11 | `black` | African | 546 | 1,333 | 2.29 |
| 12 | `shit` | Disability | 544 | 4,114 | 1.61 |
| 13 | `gay` | Homosexual | 536 | 693 | 3.52 |
| 14 | `fucking` | Disability | 530 | 2,945 | 2.25 |
| 15 | `ghetto` | African | 519 | 42 | 2.40 |
| 16 | `immigrants` | Refugee | 496 | 16 | 4.16 |
| 17 | `muslims` | Hindu, Islam | 397 | 23 | 2.48 |
| 18 | `hate` | Caucasian | 392 | 737 | 1.04 |
| 19 | `islam` | Islam | 365 | 146 | 3.29 |
| 20 | `illegal` | Hispanic, Refugee | 364 | 77 | 3.83 |

## MMHS150K coverage audit
- **Total unique HateXplain identity tokens (N_lex):** 1,177
- **Of those, appearing at least once in MMHS150K train split (N_overlap):** 1,057
- **Coverage fraction (N_overlap / N_lex):** 0.8980
- **Mean occurrences per overlapping lex token in MMHS train (134,820 tweets):** 224.17
- **Tokens with zero MMHS occurrences:** 120

## Limitations
- **English-only.** HateXplain is monolingual English; the lexicon does not transfer to other languages. MMHS150K is multilingual in practice (predominantly English), so non-English tweets are inherently outside the lexicon's scope.
- **No in-group reclamation handling.** AAVE in-group reclaimed slurs (e.g. specific reclaimed terms used by African-American speakers in non-hate contexts) appear in the lexicon because HateXplain conflates surface form with target. Downstream MVP4-IW must treat lexicon hits as a *signal* to be weighted by other context, not as a binary in-group / out-group decision.
- **Static and time-bounded.** The lexicon reflects HateXplain's 2020 Twitter snapshot. Identity-term reclamation and slur evolution are continuous processes; periodic re-extraction would be required for production deployment.
- **PMI thresholds are heuristic.** F_min = 5 and PMI = 1.0 are working points chosen to balance recall (covering real identity terms) against precision (excluding spurious co-occurrence). Different downstream applications may require re-tuning.
- **No MMHS150K-derived terms.** Per the project's non-circularity rule, the lexicon is sourced exclusively from HateXplain. MMHS150K is used only for the coverage audit; no token enters the lexicon by appearing in MMHS150K.
- **Coverage is partial.** A non-trivial fraction of the lexicon does not appear in MMHS150K train; this is recorded in the audit table above. Downstream MVP4-IW must report the operational coverage on its evaluation splits, not assume the lexicon applies to every sample.
- **Per-token community membership is a set, not a probability.** A token can identify multiple communities; the lexicon does not encode which community is most likely at inference time. Downstream resolution requires additional context (e.g. co-occurring tokens) and is out of scope for this artefact.

---

## Appendix A — Hatebase Seed Overlap Audit

### A.1 Purpose

This appendix audits the overlap between the **MMHS150K Hatebase seed-keyword list** (the 86-keyword Hatebase subset used to filter Twitter Streaming API content into the MMHS150K candidate pool, per Gomez et al. 2019 §3.1) and the HateXplain-derived identity lexicon documented in §1–§6 of this report. The purpose is to detect a circular dependency: if the HateXplain lexicon largely re-derives the same tokens that seeded the MMHS150K data collection, then any downstream IW-attention model that conditions on the lexicon is partly measuring distributional artefacts of the seeding process rather than genuinely transferred identity-term signal from an independent source.

### A.2 Hatebase seed list

**File:** `data/MMHS150K/hatespeech_keywords.txt` (952 bytes).

| Quantity | Value |
|---|---:|
| Total raw entries (lines) | 86 |
| Single-word entries | 57 |
| Multi-word entries | 29 |
| Total unique tokens (single + extracted from multi-word, lowercased) | **85** |

Multi-word phrases (e.g. `"white power"`, `"go back to your country"`) are split into their constituent tokens before matching, so a multi-word phrase contributes each of its words to the candidate vocabulary. Both lists are lowercased and stripped before the set-membership comparison.

### A.3 Overlap statistics

| Quantity | Value |
|---|---:|
| Hatebase token vocabulary (denominator) | 85 |
| HateXplain lexicon tokens | 1,177 |
| **Overlapping tokens** | **35** |
| **Overlap fraction (overlap / Hatebase vocab)** | **0.4118 (41.2 %)** |
| Interpretation category | **MODERATE** |


### A.4 Overlapping tokens — full list with HateXplain detail

The full set of overlapping tokens, grouped by their HateXplain community label(s). Per-token columns: HateXplain global frequency (within targeted posts), MMHS150K train frequency, max PMI score across community assignments, and the source Hatebase phrase(s) that contributed the token.

| Token | HateXplain community/ies | HX freq | MMHS train freq | max PMI | Source Hatebase phrase(s) |
|---|---|---:|---:|---:|---|
| `nigger` | African | 1,892 | 4,536 | 3.04 | `border nigger`, `white nigger`, `nigger`, `house nigger`, `white nigger`, `white nigger` |
| `nigga` | African | 153 | 73,825 | 1.64 | `nigga` |
| `coon` | African | 40 | 38 | 2.27 | `bamboo coon` |
| `monkey` | African | 32 | 57 | 2.98 | `surrender monkey`, `surrender monkey` |
| `moon` | African | 14 | 74 | 1.32 | `moon cricket` |
| `park` | African | 12 | 160 | 1.37 | `trailer park trash`, `trailer park trash` |
| `raghead` | Arab | 60 | 17 | 3.81 | `raghead` |
| `arab` | Arab | 35 | 45 | 2.87 | `arab terror` |
| `camel` | Arab | 29 | 9 | 2.83 | `camel fucker`, `camel fucker`, `camel fucker` |
| `asian` | Asian | 61 | 259 | 3.76 | `asian drive` |
| `chinaman` | Asian | 36 | 77 | 5.42 | `chinaman`, `chinaman` |
| `white` | Caucasian | 1,923 | 2,985 | 2.03 | `white nigger`, `white trash`, `white trash`, `white nigger`, `white nigger` |
| `trash` | Caucasian | 148 | 2,531 | 2.69 | `trailer park trash`, `trailer trash`, `white trash`, `white trash`, `trailer trash`, `trailer park trash` |
| `trailer` | Caucasian | 35 | 519 | 3.48 | `trailer park trash`, `trailer trash`, `trailer trash`, `trailer park trash` |
| `redneck` | Caucasian | 28 | 2,800 | 4.95 | `redneck` |
| `hillbilly` | Caucasian | 13 | 1,925 | 4.63 | `hillbilly` |
| `wigger` | Caucasian | 8 | 101 | 3.94 | `wigger`, `wigger` |
| `retarded` | Disability, Men | 283 | 5,055 | 3.90 | `retarded` |
| `retard` | Disability | 82 | 3,314 | 4.34 | `retard` |
| `spic` | Hispanic | 121 | 246 | 4.80 | `spic` |
| `wetback` | Hispanic | 58 | 70 | 4.78 | `wetback` |
| `border` | Hispanic, Refugee | 41 | 260 | 2.69 | `border jumper`, `border nigger` |
| `faggot` | Homosexual | 554 | 5,212 | 2.98 | `faggot` |
| `dyke` | Homosexual | 174 | 6,166 | 3.43 | `dyke`, `dyke` |
| `pussy` | Homosexual, Women | 70 | 897 | 1.33 | `sideways pussy` |
| `card` | Homosexual | 20 | 1,858 | 1.36 | `victim card`, `race card` |
| `islam` | Islam | 365 | 146 | 3.29 | `islam terrorism` |
| `muzzie` | Islam | 157 | 62 | 3.36 | `muzzie` |
| `terrorism` | Islam | 37 | 92 | 1.71 | `islam terrorism` |
| `terror` | Islam | 36 | 41 | 2.50 | `arab terror` |
| `banislam` | Islam | 13 | 86 | 4.03 | `banislam` |
| `conspiracy` | Jewish | 14 | 890 | 1.52 | `conspiracy theorist` |
| `cunt` | Women | 96 | 13,058 | 1.45 | `cunt` |
| `sjw` | Women | 34 | 1,681 | 1.21 | `sjw` |
| `feminazi` | Women | 22 | 206 | 4.24 | `feminazi` |

### A.5 Interpretation

**Overlap fraction = 0.4118 (41.2 %), category = MODERATE.** overlap is moderate. This is the expected range for any hate-speech lexicon since core slurs appear universally across datasets, but the overlap should be documented as a methodological caveat in the thesis Limitations section. The 89.8 % MMHS150K coverage reported in the lexicon's body audit is partly attributable to overlap with the seeding keywords (which by construction appear at elevated frequency in MMHS150K), but the majority of the lexicon still derives from HateXplain content that is independent of the MMHS150K seed list.
