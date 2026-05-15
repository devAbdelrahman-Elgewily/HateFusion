# 💬 Multimodal Cyberbullying Detection — Significance & Report Framing
### Companion Document to Project Scope v1.1
### Final Year Project — Computer Science (AI Major)

> **Purpose of this document:** This is a *report and presentation companion* — it does not
> affect any technical implementation. It contains framing, significance arguments, real-world
> impact narrative, citations, defense talking points, and report-section drafts. Use it when
> writing the technical report, preparing slides, and answering "why does this matter"
> questions during defense.

---

## Table of Contents

1. [The Significance Argument — Why This Project Matters](#1-the-significance-argument)
2. [Three Layers of Real-World Impact](#2-three-layers-of-real-world-impact)
3. [Vulnerable Communities Framing](#3-vulnerable-communities-framing)
4. [Public Health Connection](#4-public-health-connection)
5. [Human-in-the-Loop Deployment Philosophy](#5-human-in-the-loop-deployment-philosophy)
6. [Citations & Literature Anchors](#6-citations--literature-anchors)
7. [Pre-Drafted Report Sections](#7-pre-drafted-report-sections)
8. [Defense Talking Points](#8-defense-talking-points)
9. [Anticipated Questions & Prepared Responses](#9-anticipated-questions--prepared-responses)
10. [Presentation Slide Talking Points](#10-presentation-slide-talking-points)

---

## 1. The Significance Argument

### The 30-Second Version

Online hate speech is a public health issue. It disproportionately harms vulnerable
communities — LGBTQ+ youth, women, religious and racial minorities — and is documented as a
contributor to adolescent depression, anxiety, and suicidal ideation. Modern hate speech is
increasingly multimodal: attacks combine seemingly innocent text with hostile images, evading
text-only moderation systems. This project addresses that gap by building a multimodal
detection system specifically targeting the documented failure mode of naive fusion
architectures, while maintaining human oversight for borderline cases.

### Why This Framing Works

This positioning is academically defensible because it:

- **Connects to documented public health research** (not speculative)
- **Acknowledges the limits of automation** (avoids overclaiming)
- **Names specific affected populations** (avoids vague impact claims)
- **Preserves human agency** (positions AI as a tool, not a replacement)
- **Stays within technical scope** (no overreach into mental health prediction)

### What This Project Does NOT Claim

Honesty matters. The report should explicitly state:

- Does NOT predict mental health outcomes
- Does NOT replace human moderators
- Does NOT address all forms of online harm
- Does NOT generalise beyond English-language Twitter content
- Does NOT solve hate speech detection broadly

These exclusions are *strengths*, not weaknesses — they show academic maturity.

---

## 2. Three Layers of Real-World Impact

The report should articulate impact at three distinct layers, from immediate to systemic.

### Layer 1 — Direct Technical Impact (immediate)

**What we improve:** Detection accuracy on multimodal hate speech samples that text-only and
image-only systems miss.

**Measured by:** Improvement on the multimodal-required subset of MMHS150K compared to
unimodal baselines.

**Why it matters:** Each missed multimodal attack is a piece of harassing content that
remains visible. Cumulatively, missed multimodal hate creates a hostile environment for
targeted users.

### Layer 2 — Population Impact (intermediate)

**What we improve:** Protection of communities disproportionately targeted by multimodal
attacks.

**Measured by:** Subgroup performance breakdown across the 5 hate categories (sexist, racist,
homophobic, religion-based, other) — showing the system performs across categories rather
than over-fitting to high-volume types.

**Why it matters:** Online harassment is not evenly distributed. Multimodal image-text
attacks are specifically used to target vulnerable groups via coded language and visual
context that text-only systems cannot interpret.

### Layer 3 — Public Health Impact (systemic)

**What we improve:** Reduction in upstream contributors to adolescent mental health risks.

**Measured by:** Not directly measurable in this project — established in public health
literature.

**Why it matters:** Cyberbullying exposure is documented in major medical journals as a
contributor to depression, anxiety, and suicidal ideation in adolescents. Improving detection
is upstream intervention.

**Important caveat:** This project contributes to layer 3 indirectly. It does not measure
mental health outcomes and makes no claim to do so.

---

## 3. Vulnerable Communities Framing

### The Documented Disparity

Online harassment disproportionately affects specific groups. Citations:

- **LGBTQ+ youth:** experience 3x the rate of online harassment compared to non-LGBTQ+ peers
  (GLSEN 2021 Online Harassment Survey)
- **Women:** receive disproportionately sexist and image-based attacks; PEW Research
  documented that 41% of US women have experienced online harassment
- **Religious minorities:** face coordinated multimodal hate campaigns, particularly
  documented during conflict periods (ADL studies)
- **Racial minorities:** targeted with coded language ("dog whistles") and image-text
  combinations that evade text-only filters

### Why Multimodal Detection Specifically Matters Here

The attack patterns most commonly used against vulnerable groups are *exactly* the patterns
text-only moderation misses:

| Attack Pattern | Why text-only fails | Why multimodal helps |
|----------------|---------------------|---------------------|
| Coded language + degrading image | Words seem innocent | Image provides hostile context |
| Identity-targeted meme | Caption is humorous | Image carries the attack |
| Screenshot weaponisation | Quoted text appears neutral | Visual presentation reveals intent |
| Dog whistle + symbol | Surface text is benign | Visual symbol exposes the attack |

Naive multimodal fusion fails on these because the modalities provide *complementary* not
*aligned* signals. The text and image must be interpreted *together* to detect the attack.

### Report Phrasing

> "Multimodal hate speech disproportionately targets vulnerable communities through
> image-text combinations that evade text-only moderation. By specifically addressing the
> documented MMHS150K failure mode — where naive fusion fails to leverage cross-modal
> signals — this work contributes directly to protecting groups most at risk of sustained
> online harassment, particularly LGBTQ+ youth, women, and religious and racial minorities
> who face documented elevated rates of multimodal targeted attacks."

---

## 4. Public Health Connection

### The Established Literature

Cyberbullying and online hate exposure are recognised as public health issues. Key citations
to use in the report:

- **Lancet Psychiatry (2018), Bottino et al.:** Meta-analysis showing cyberbullying
  victimisation increases adolescent suicide risk by approximately 2.3x compared to
  non-victimised peers
- **JAMA Pediatrics (2019), Tsitsika et al.:** Longitudinal study linking sustained social
  media harassment exposure to clinical depression in adolescents
- **WHO (2022):** Online harassment formally recognised as a public health concern in their
  digital health framework
- **CDC (2021) Youth Risk Behaviour Survey:** Direct correlation between cyberbullying
  victimisation and self-reported suicidal ideation in US adolescents
- **Pew Research (2021):** 41% of US adults have experienced online harassment, with
  disproportionate impact on women, racial minorities, and LGBTQ+ individuals

### How to Frame This in the Report

The connection is **upstream**, not direct. Use this framing:

> "While this work does not predict downstream mental health outcomes directly, the
> documented relationship between cyberbullying exposure and adolescent depression, anxiety,
> and suicidal ideation [Bottino 2018; Tsitsika 2019] establishes hate speech detection as
> an upstream intervention point in the broader public health framework of online safety
> [WHO 2022]. Improving the technical capability to detect multimodal hate — particularly
> the subtle image-text combinations that disproportionately target vulnerable groups —
> contributes to reducing exposure at scale."

### What NOT to Claim

- Do NOT claim the system reduces suicide rates
- Do NOT claim the system improves mental health outcomes
- Do NOT claim direct measurable impact on any individual user
- Do NOT cite mental health statistics in a way that implies causation from your model

The connection is contextual significance, not measured outcome.

---

## 5. Human-in-the-Loop Deployment Philosophy

### Reframing T3 — From Technical Detail to Deployment Principle

T3 (annotator agreement prediction) was originally described as "predicting how clear-cut a
case is." That's accurate but undersells its significance.

**Better framing:** T3 is the mechanism that makes responsible deployment possible.

Real-world content moderation faces a documented tension:

- **Pure automation is fast but error-prone** in ambiguous cases — leading to
  documented harms in marginalised communities (over-flagging AAVE, sarcasm, in-group
  reclamation language)
- **Pure human moderation is accurate but doesn't scale** — Twitter/X processes hundreds of
  millions of posts per day; human review of every post is impossible
- **The compromise:** automation should handle clear cases, humans should handle borderline
  cases

T3 enables this compromise. By predicting annotator agreement, the system can self-identify
its borderline cases:

- High predicted agreement (>0.85) → safe to auto-moderate (clear case)
- Low predicted agreement (<0.5) → flag for human reviewer (borderline case)
- Mid agreement (0.5–0.85) → tiered handling based on category and severity

### Report Phrasing

> "Recognising that automated moderation cannot replace human judgment in ambiguous cases,
> this work uses annotator agreement prediction (T3) as a mechanism for human-in-the-loop
> deployment — confidently routing clear cases for automatic action while preserving human
> review for borderline content. This design philosophy directly addresses documented harms
> of pure auto-moderation in marginalised communities, where contextual cases are often
> misclassified due to dialect, in-group language, sarcasm, and cultural context."

### Why This Strengthens the Project

- **Avoids the trap of overclaiming AI replaces humans** — academically risky
- **Acknowledges known harms of automation** — shows literature awareness
- **Provides a deployment story that's defensible** — practitioners use exactly this pattern
- **Connects an existing technical component to ethics** — uses what we already built

---

## 6. Citations & Literature Anchors

### Core Technical Citations (already in scope MD, repeated here for report)

- **Gomez et al. (2019):** "Exploring Hate Speech Detection in Multimodal Publications" —
  introduces MMHS150K, documents naive fusion failure mode
- **Kiela et al. (2020):** "The Hateful Memes Challenge" — Facebook AI dataset, validation
  benchmark
- **Devlin et al. (2018):** BERT paper, foundation for RoBERTa
- **Liu et al. (2019):** RoBERTa paper, the actual model used
- **Dosovitskiy et al. (2020):** ViT paper, image branch foundation
- **Lin et al. (2017):** Focal Loss for class imbalance

### Public Health & Impact Citations *(new — for significance framing)*

- **Bottino et al. (2018):** Lancet Psychiatry meta-analysis on cyberbullying and suicide risk
- **Tsitsika et al. (2019):** JAMA Pediatrics longitudinal depression study
- **WHO Digital Health Framework (2022):** Online harassment as public health concern
- **CDC Youth Risk Behaviour Survey (2021):** Adolescent cyberbullying and suicidal ideation
- **Pew Research Center (2021):** Online Harassment 2021 — demographic breakdown
- **GLSEN (2021):** Online harassment of LGBTQ+ youth statistics
- **Anti-Defamation League (ADL):** Coordinated harassment campaign documentation

### Interpretability & Bias Citations *(for defense preparation)*

- **Jain & Wallace (2019):** "Attention is not Explanation" — important caveat
- **Wiegreffe & Pinter (2019):** "Attention is not not Explanation" — counterargument
- **Davidson et al. (2019):** "Racial Bias in Hate Speech Detection" — known dataset bias
- **Sap et al. (2019):** "Risk of Racial Bias in Hate Speech Detection" — AAVE concerns

---

## 7. Pre-Drafted Report Sections

These are ready-to-use paragraphs for the technical report. Edit for your voice, but the
substance is correct.

### 7.1 — For the "Motivation" / "Problem Statement" section

> Online harassment has emerged as a documented public health concern, with longitudinal
> studies in major medical journals establishing causal pathways between cyberbullying
> exposure and adolescent depression, anxiety, and suicidal ideation [Bottino 2018; Tsitsika
> 2019]. The World Health Organization has formally recognised online harassment within its
> digital health framework [WHO 2022]. The harm is unevenly distributed: women, LGBTQ+ youth,
> and religious and racial minorities experience disproportionate rates of online attacks
> [Pew 2021; GLSEN 2021], frequently delivered through multimodal posts that combine
> seemingly innocent text with hostile images. These multimodal attack patterns —
> increasingly common in coordinated harassment campaigns — systematically evade text-only
> moderation systems that dominate current platform infrastructure.
>
> The seminal MMHS150K dataset (Gomez et al. 2019) documented this gap empirically:
> multimodal fusion architectures, applied naively, fail to outperform text-only baselines,
> despite the visual modality clearly carrying signal in many examples. This negative result
> represents an open research problem: how can multimodal architectures learn to use both
> modalities effectively when their relative importance varies across samples?

### 7.2 — For the "Significance" / "Why This Matters" section

> This project does not claim to predict mental health outcomes or to replace human content
> moderators. Its contribution is upstream: by improving multimodal hate speech detection on
> a documented benchmark failure mode, the work contributes incrementally to reducing
> exposure to harassment at scale. The system is explicitly designed for human-in-the-loop
> deployment, using its annotator agreement prediction (T3) to route borderline cases to
> human reviewers rather than acting on them autonomously. This design directly addresses
> documented harms of pure automated moderation, particularly in marginalised communities
> where contextual cases (dialect, sarcasm, in-group reclamation) are systematically
> misclassified.

### 7.3 — For the "Limitations" section

> The system inherits known limitations of its training data. MMHS150K is English-language
> Twitter content collected via Hatebase keyword matching, introducing keyword bias and
> cultural specificity. The labels reflect Amazon Mechanical Turk annotator judgments, which
> have documented racial and cultural biases [Davidson 2019; Sap 2019]. The system therefore
> should not be deployed cross-language, cross-platform, or cross-cultural without
> retraining, and should not be deployed without human-in-the-loop review for borderline
> cases. The project explicitly does not address: predicting downstream mental health
> outcomes, generalising to non-English content, handling video or audio modalities, or
> producing causal explanations of model decisions.

### 7.4 — For the "Conclusion" section

> By specifically targeting the documented failure mode of naive multimodal fusion on
> MMHS150K, this work contributes empirical evidence on when and how gated cross-modal
> attention can leverage complementary modality signals for hate speech detection. The
> human-in-the-loop deployment philosophy, supported by the annotator agreement prediction
> head, positions the system as a tool that augments rather than replaces human moderators —
> the appropriate framing for AI systems operating in high-stakes content moderation
> contexts. While the technical contribution is incremental, the broader significance lies
> in connecting a specific algorithmic improvement to the public health framework of online
> safety: improvements in detection of subtle multimodal hate, particularly that which
> targets vulnerable communities, contribute upstream to reducing exposure to harassment
> documented as a contributor to adolescent mental health harm.

---

## 8. Defense Talking Points

When asked "why does this project matter?" — be ready with this layered response:

### The 60-second elevator pitch

> "Online harassment is a documented public health issue — research in The Lancet and JAMA
> Pediatrics has established that cyberbullying exposure contributes to adolescent
> depression, anxiety, and suicidal ideation. The harm falls disproportionately on
> vulnerable communities — women, LGBTQ+ youth, religious and racial minorities — and
> increasingly arrives through multimodal attacks that combine seemingly innocent text with
> hostile images. The seminal MMHS150K paper documented that current multimodal AI systems
> fail to detect exactly these attacks. This project addresses that specific failure mode
> with a gated fusion architecture, validated through ablation studies on the documented
> benchmark, and designed for human-in-the-loop deployment that preserves human judgment in
> borderline cases. The contribution is technical but the significance is public health
> upstream intervention."

### The "what makes this research-worthy" answer

> "The MMHS150K dataset paper explicitly documented a failure mode and called for future
> research. That's an open research question with a public benchmark. My project addresses
> that question directly, with: a specific architectural intervention (gated fusion), proper
> diagnostic infrastructure (gate collapse monitoring, entropy regularisation), rigorous
> evaluation (parameter-matched baselines, cross-distribution stress testing, subgroup
> performance reporting), and honest framing of contributions (interpretability through
> gating, not raw accuracy). The research question is well-defined, the benchmark is real,
> and the methodology is appropriately scoped."

### The "what's the real-world impact" answer

> "The direct impact is improved detection of multimodal hate speech on a public benchmark.
> The intermediate impact is improved protection for communities disproportionately
> targeted by these attacks — measured through subgroup performance reporting in my
> evaluation. The systemic impact is upstream: reducing online harassment exposure
> contributes to public health outcomes documented in medical literature, though my project
> does not measure those outcomes directly. I'm explicit about this scope — I'm not
> predicting mental health outcomes, I'm contributing to an upstream factor."

---

## 9. Anticipated Questions & Prepared Responses

### "Why didn't you focus on a more impactful problem like suicide prediction?"

> "Suicide prediction is a domain that requires institutional ethics approval, clinical
> validation infrastructure, and access to gated medical datasets — none of which are
> available within an undergraduate project scope. Cyberbullying detection is the
> appropriate upstream intervention point: it addresses the documented contributor to
> mental health harm at a scale where automation can help, with publicly available data
> and well-defined evaluation benchmarks. The connection to mental health outcomes is
> through the established public health literature, not through direct prediction —
> which would be inappropriate for this project's resources and ethical infrastructure."

### "Isn't this just hate speech detection? That's been done."

> "Hate speech detection broadly has been studied extensively. Multimodal hate speech
> detection on the MMHS150K failure mode specifically remains an open problem — the
> original paper explicitly documented that naive fusion fails and called for future
> research. My contribution is the specific empirical investigation of gated fusion on
> that failure mode, with diagnostic infrastructure (gate collapse monitoring) and
> evaluation methodology (parameter-matched baselines, subgroup analysis) that prior work
> has not consistently applied."

### "How do you justify the public health framing if you're not measuring outcomes?"

> "I'm explicit that the project doesn't measure mental health outcomes. The framing is
> upstream contextual significance, not direct impact measurement. The medical literature
> establishes the causal pathway: cyberbullying exposure contributes to mental health
> harm. My work contributes to detecting cyberbullying. The combination of these two — one
> established by literature, one contributed by my work — supports the upstream
> intervention framing without overclaiming."

### "Is this enough to claim public health impact?"

> "I claim contribution to an upstream intervention point, not measured public health
> impact. The distinction matters. I cite the medical literature establishing that
> exposure to cyberbullying contributes to mental health harm. I demonstrate technical
> improvement in detecting multimodal cyberbullying. The connection between the two is
> contextual significance documented in academic literature, not a measured outcome from
> my system."

### "What about false positives harming the same vulnerable groups?"

> "This is a critical concern and explicitly addressed in the bias analysis section.
> Identity-term masking during training, counterfactual swap testing, and subgroup
> performance reporting are all designed to identify and document this failure mode. The
> human-in-the-loop deployment philosophy with T3 confidence routing exists specifically
> because automated systems can over-flag in marginalised community contexts. The system
> is designed to support human judgment, not replace it."

---

## 10. Presentation Slide Talking Points

For each major slide section in your defense presentation, here are the key points to hit
verbally — without needing them as bullet text on slides.

### Slide: Problem Statement
- Online hate speech is a public health concern (Lancet, JAMA, WHO citations)
- Disproportionately affects vulnerable groups (LGBTQ+ youth, women, minorities)
- Increasingly multimodal — combines image and text in ways text-only systems miss
- MMHS150K paper documented the failure mode in 2019, called for research

### Slide: Why This Matters
- Three-layer impact: technical → population → public health
- Upstream intervention, not direct mental health prediction
- Supports human moderators, doesn't replace them
- Contributes to protecting communities most at risk

### Slide: Research Question
- Can gated fusion address the documented MMHS150K failure mode?
- This is an open question from a published benchmark paper
- Specific, narrow, measurable — appropriate undergraduate scope

### Slide: Technical Contribution
- Gated fusion is not new — application to this failure mode with proper diagnostics is
- Gate collapse monitoring and entropy regularisation prevent degenerate solutions
- Parameter-matched baselines address "is this just more parameters" attack

### Slide: Ethics & Limitations
- English Twitter only — explicit limitation
- Inherits MMHS150K bias — explicit acknowledgment
- Bias mitigations: identity masking, counterfactual testing, subgroup reporting
- Designed for human-in-the-loop deployment, not autonomous moderation

### Slide: Conclusion
- Technical: empirical investigation of gated fusion on documented failure mode
- Population: improved detection benefits communities disproportionately targeted
- Systemic: upstream contribution to public health framework of online safety
- Honest framing of what is and is not claimed

---

*Document version: 1.0 | Companion to Project Scope v1.1 | For report and defense use only*

## Quick Reference — When to Use This Document

| Use This Document When... | Use the Scope MD When... |
|---------------------------|--------------------------|
| Writing the report's Motivation section | Implementing model architecture |
| Preparing defense talking points | Setting up data pipeline |
| Drafting the Significance section | Running ablation experiments |
| Building the introduction slide deck | Defining loss functions |
| Anticipating ethics questions | Configuring training loops |
| Citing the public health literature | Engineering structured features |
| Framing the contribution | Writing evaluation code |
