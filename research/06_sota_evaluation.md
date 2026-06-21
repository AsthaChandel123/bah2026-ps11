# BAH 2026 PS-11 — SOTA Landscape, Exact Evaluation Methodology, and Winning Strategy

**Problem:** Cross-Modal Satellite Image Retrieval Using Multi-Sensor Remote Sensing Data
(optical RGB / multispectral / SAR; same-modal + cross-modal; rank top-5 / top-10; report retrieval time).

**Scored metrics (from `idea.md`):**
1. F1-score@5 (same-modal)
2. F1-score@10 (same-modal)
3. F1-score@5 (cross-modal)
4. F1-score@10 (cross-modal)
5. Average retrieval time per query
> "Cross-modal retrieval may be given additional importance because it is more challenging." → **spend most effort on cross-modal F1**, and keep latency low for the 5th metric.

This document covers: (1) the competitive/SOTA landscape, (2) the exact F1@K evaluation methodology + worked example + evaluation-matrix design, (3) a ranked list of winning moves with metric impact and latency cost, and (4) robust cross-checking via multiple datasets/backbones.

---

## PART 0 — TL;DR for the team

- **F1@K we will use:** for a query *q*, `P@K = r_K/K`, `R@K = r_K/min(K, R_q)`, `F1@K = 2·P@K·R@K/(P@K+R@K)`, where `r_K` = # relevant in top-K and `R_q` = total relevant for *q* in the gallery. Average F1@K over all queries (per matrix cell), then macro-average cells. **Use `min(K, R_q)` in the recall denominator** so a class with few members is not unfairly capped — but report both variants because graders may use raw `R_q`. (See Part 2.)
- **What wins today:** For **image↔text** RS retrieval, **CLIP-style foundation models fine-tuned on RS** dominate (GeoRSCLIP > RemoteCLIP > pre-CLIP dual-encoders; newest iEBAKER/PIR-CLIP push further). For **SAR↔optical image↔image**, the winning recipe is a **shared-space multimodal foundation backbone** (CROMA-style radar-optical contrastive MAE) **+ a text/semantic anchor** (CLOSP-style) to bridge the heterogeneity gap, with **FAISS** for speed and **deep hashing** when latency must be tiny.
- **Top 8 winning moves (ranked):** (1) shared multimodal foundation backbone, (2) per-modality whitening + L2-norm to close the modality gap, (3) projection head trained with symmetric InfoNCE + ArcFace, (4) a text/class **anchor space** for cross-modal alignment, (5) single FAISS index over all modalities + ANN for latency, (6) k-reciprocal re-rank of top candidates, (7) calibrate K vs relevant-set / class-balanced gallery, (8) multi-backbone ensemble (concat+whiten) + TTA. Details + costs in Part 3.
- **Where to focus the cross-modal weight:** the **alignment objective and the anchor** (moves 2–4) and a **single unified index** (move 5). Cross-modal F1 is bottlenecked by the modality gap, not by the index.

---

## PART 1 — Competitive / SOTA Landscape

### 1A. Image↔Text RS retrieval (RSITR) — methods transfer directly to our embedding/alignment design

These methods learn a shared image–text space; the **image encoders, alignment losses, and re-ranking tricks transfer directly** to our image↔image cross-modal task. Benchmarks: **RSITMD** (4,743 imgs, 23,715 captions, 5 captions/img; subset of RSICD, 256×256) and **RSICD** (10,921 imgs, 30 scene classes, 5 captions/img, 224×224). Metric = **Recall@1/5/10** and **mean Recall (mR)** = mean of the 6 R@K values (i2t + t2i).

**Verified leaderboard (from iEBAKER comparison table, arXiv 2504.05644):**

| Method | Year / type | RSITMD mR | RSICD mR | Notes |
|---|---|---|---|---|
| AMFMN | 2022, dual-branch + MVSA self-attn, dynamic-margin triplet | 29.32 | 16.42 | Classic baseline; visual feats guide text |
| GaLR | 2022, global+local fusion | 31.41 | 18.96 | Multi-level dynamic fusion, removes redundancy |
| KAMCL | 2023, knowledge-aided momentum contrast | 36.14 | 26.10 | Momentum/MoCo-style + knowledge |
| PE-RSITR | 2023, **foundation-model** (CLIP) + parameter-efficient tuning | 44.47 | 31.12 | First strong CLIP-transfer baseline |
| RemoteCLIP | 2023, CLIP fine-tuned, 12× enlarged RS data | 49.38 | 35.26 | ViT-B/32, ViT-L/14, RN50; +9.14 mR over prior SOTA (RSITMD), +8.92 (RSICD) per abstract |
| GeoRSCLIP | 2024, CLIP trained on **RS5M (5M pairs)** | 51.81 | 38.87 | +3–6% on retrieval over baselines |
| EBAKER | 2025, eliminate-before-align + keyword reasoning | — / 40.70 (RSICD) | 40.70 | CLIP-based |
| **iEBAKER-Split** | 2025, improved EBAKER | **53.55** | **45.07** | Current best in this table |

Other notable RSITR methods (same family, also competitive): **SWAN** (context-sensitive integration to cut semantic ambiguity), **HVSA** (handles image variance vs text similarity / hard-sample difficulty), **PIR / PIR-CLIP** (prior-instruction representation: Spatial-PAE + Temporal-PAE, ResNet-on-AID instruction encoder; PIR-CLIP reports +7.3%/+9.4% over open-domain SOTA on RSICD/RSITMD), **SSJDN** (+2.82 mR over GaLR), **MSSA**, **PriorCLIP/PR-CLIP**.

**What wins and why (image↔text):**
- **Foundation-model transfer dominates** the pre-CLIP dual-encoders by a wide margin (PE-RSITR 44.47 vs AMFMN 29.32 on RSITMD). The jump from AMFMN→GeoRSCLIP is ~+22 mR on RSITMD.
- Among CLIP variants, **more + cleaner RS pretraining data wins** (RemoteCLIP's 12× data; GeoRSCLIP's RS5M 5M pairs).
- Incremental winners (iEBAKER, PIR) add **data cleaning / "eliminate-before-align"** (filtering pseudo-matched pairs) and **keyword/prior reasoning** on top of CLIP.

### 1B. Image↔Image cross-modal & cross-source RS retrieval (SAR↔optical, optical↔MS) — the actual task

**(i) Shared-space multimodal foundation backbones (the backbone we should build on):**
- **CROMA** (NeurIPS 2023, `2311.00566`): **Contrastive Radar-Optical Masked Autoencoder**. Separately encodes masked Sentinel-1 (SAR) and Sentinel-2 (optical) aligned in space/time, does **cross-modal contrastive learning** (→ aligned unimodal SAR & optical embeddings), and a multimodal encoder **fuses** them (→ joint embedding). Novel 2D-ALiBi/X-ALiBi position biasing enables test-time extrapolation to larger images. **Backbones:** CROMA-B (ViT-B), CROMA-L (ViT-L). **Result:** joint multimodal beats optical-only by **+1.2% on BigEarthNet** and **+4.9% (B)/+3.5% (L) on DFC2020**; CROMA-B beats SatMAE-B on fMoW-Sentinel by **+3.3%**; avg **+1.8% finetune / +2.4% linear-probe** over prior SoTA. **Why relevant:** gives us *aligned SAR + optical + joint* embeddings out-of-the-box → directly seeds a shared retrieval space.
- Other multimodal RS foundation models for SAR/optical/MS/DEM shared spaces: **DOFA** (wavelength-conditioned, any-sensor), **SkySense**, **MMEarth**, **Prithvi-EO-2.0** (4.2M HLS multispectral time-series, IBM/NASA), **SatMAE**, **GFM/GeoPile**, **CSMoE** (soft mixture-of-experts). These are the candidate backbones for our shared embedding space.

**(ii) Text/semantic anchor to bridge SAR↔optical (a winning trick):**
- **CLOSP** (`2507.10403`, "Contrastive Language Optical SAR Pretraining"): three encoders (text, S1-SAR, S2-optical) → **shared 384-D space**; **text acts as the anchor** linking each image modality *independently* (no direct SAR-optical pairing needed). Trained on **CrisisLandMark** (647k+ S1/S2 pairs). **Text→image retrieval (Table 6): nDCG@1000 = 57.76% (GeoCLOSP)** vs best baseline **SkyCLIP-T 37.88%** — nearly **+20 absolute nDCG**. Per-modality (Table 7): on **SAR (S1)**, CLOSP nDCG@1000 = **55.65% vs BiCLIP 37.35% (+18)**; on optical (S2) it ties the best (54.80% vs 55.72%). **Key finding:** semantic signatures learned from multispectral (vegetation/water) **transfer to SAR** via the text anchor *without* direct SAR-optical alignment. **Takeaway for us:** if SAR↔optical pairs are scarce, **anchor both to a shared label/text space** to get cross-modal retrieval "for free."

**(iii) Sensor-agnostic image↔image retrieval (closest to our exact task):**
- **CSMAE** (`2401.07782`, "Cross-Sensor Masked Autoencoders for sensor-agnostic image retrieval"): adapts MAE to handle **uni-modal (S2→S2, S1→S1)** and **cross-modal (S1→S2, S2→S1)** CBIR on **BigEarthNet-MM**. First systematic study of MAEs for sensor-agnostic CBIR; builds a common space so a query from one sensor retrieves from another. (Exact mAP tables not extractable from PDF here; the framework design — masked reconstruction + cross-sensor alignment — is the transferable contribution.)
- **DUCH** (`2201.08125`, ICASSP 2022): **Deep Unsupervised Contrastive Hashing** for cross-modal RS retrieval — feature module + hashing module with contrastive (similarity), adversarial (cross-modal consistency), and binarization objectives; **unsupervised, no labels**, scalable binary codes. Good template for the **hashing/latency** branch.
- Earlier two-stream FCNs map optical & SAR to a **common feature space** for comparison (heterogeneity-gap framing); **CMIR-NET** (deep cross-modal retrieval in RS).

**(iv) SAR↔optical matching/registration (descriptor lessons, not ranking metrics):**
- **PromptMID** (`2502.18104`): modality-invariant descriptors from **diffusion models + vision foundation models + land-use text prompts**. On WHU-OPT-SAR (seen): **SR 100%, NCM 452, RMSE 1.870**; generalizes to unseen SEN1-2 (SR 99.0), A (100), B (96.0, vs ReDFeat 83.5). Beats 12 baselines (SIFT, RIFT, LoFTR, RoMa, ReDFeat…). **Lesson:** semantic/text-conditioned, foundation-model descriptors are the most *modality-robust* — reinforces moves (1) and (4).

**(v) Same-modal (optical→optical) CBIR baselines — sets the ceiling for our same-modal F1:**
- **Deep hashing on UC-Merced (mAP @ 64 bits):** DPSH 81.7% → DHNN ~97.1% → DHCNN 98.02% → MiLan 99.1% → **DHPL 99.21%** → **HPSH 99.7%**. (DHPL across bits: 98.53/98.83/99.01/99.21 @ 16/32/48/64.) **Implication:** within-modality optical retrieval is essentially solved (>99% mAP); our **same-modal F1@5/@10 should be very high** — the differentiator is cross-modal.
- **Triplet/metric-learning** (MHCLN, triplet deep metric nets) and **CNN-feature CBIR** on UC-Merced/AID/PatternNet/NWPU-RESISC45 are the classical baselines; modern foundation backbones beat them.
- **Standard benchmark archives:** UC-Merced (21 cls ×100), AID (30 cls, 10k), PatternNet (38 cls ×800), NWPU-RESISC45, WHU-RS19, RSSCN7; **BigEarthNet-MM** (590,326 S1+S2 pairs, 19-class multi-label) and **SEN12MS** (180,662 S1/S2/MODIS triplets, 13 bands), **SEN1-2** (282,384 co-registered S1/S2 patches) for multi-modal.

### 1C. Net "what currently wins and why"
1. **Foundation-model embeddings beat task-specific CNNs** (huge mR gaps in 1A; backbone quality dominates).
2. **For cross-modal, a shared multimodal backbone (CROMA) + a semantic/text anchor (CLOSP) is the winning structure** — it directly attacks the heterogeneity gap and lets you fill missing-modality gaps via the anchor.
3. **Latency is won by FAISS ANN and/or deep hashing**; accuracy ceiling for same-modal optical is ~99% mAP (deep hashing), so **the score is decided by cross-modal F1 + retrieval time**.

---

## PART 2 — Exact Evaluation Methodology (F1@K)

### 2A. Definitions

For a single query *q* and cutoff *K* ∈ {5, 10}:
- `retrieved_K` = the top-K ranked gallery items (by similarity).
- `rel(q)` = the **relevant set** for *q* in the gallery (definition below). `R_q = |rel(q)|`.
- `r_K` = `|retrieved_K ∩ rel(q)|` = number of relevant items among the top-K.

**Precision@K** = `r_K / K` (fraction of the K returned that are relevant).
**Recall@K** = `r_K / R_q` (fraction of all relevant items captured in top-K).
**F1@K** = harmonic mean = `2 · P@K · R@K / (P@K + R@K)` = `2·r_K / (K + R_q)`.

> The closed form **F1@K = 2·r_K/(K + R_q)** is exact and the cleanest way to compute it. (It follows by substituting P and R.)

**Aggregation:** compute F1@K per query, then **average over all queries** in a given evaluation cell. Then average across cells per the matrix in 2D. (This is "macro over queries"; it matches how `mR`/Recall@K are averaged in RSITR.)

### 2B. The recall-denominator subtlety (critical — affects the score)

The naive `R@K = r_K/R_q` has a structural problem at fixed K:
- If a class has **many** gallery members (`R_q ≫ K`), then even a perfect top-K gives `R@K = K/R_q` → recall is **capped low** → F1 capped low.
  - e.g., `R_q = 100`, `K = 5`: max recall = 0.05, so **max F1@5 = 2·5/(5+100) = 0.095** even with a *perfect* ranking. The metric punishes large classes.
- If a class has **few** members (`R_q < K`), you cannot fill K slots with relevants, so **precision** drops: with `R_q = 2`, `K = 5`, perfect ranking gives `P@5 = 2/5`, `R@5 = 1`, **F1@5 = 2·2/(5+2) = 0.571**.

**Consequence:** F1@K is **dominated by the relevant-set sizes** `R_q`, not only by ranking quality. Two systems with identical rankings score differently if the gallery class balance differs.

**Two conventions (report both; pick the fair one as primary):**
- **(A) Raw recall:** `R@K = r_K/R_q`. This is the literal reading and what an automated grader most likely implements. Use it as the **reported/primary** number to match graders.
- **(B) Capped recall:** `R@K = r_K/min(K, R_q)`. This removes the "large-class cap," so a perfect top-K → recall 1 → F1 reflects ranking quality. This is the **fair** number for internal model selection.

**Recommendation:** optimize and select models on (B), but always report (A) too. If the grader's exact formula is unknown, **assume (A)** for the leaderboard and **engineer the gallery so `R_q ≈ K`** (see move 7) — that makes (A) and (B) nearly coincide and maximizes the achievable F1@5/@10.

### 2C. Definition of the "relevant set" `rel(q)`

`idea.md` says relevance is judged by "**semantic class, geographic correspondence, or predefined relevance labels**." Three operational definitions (support all; choose per data availability):
1. **Same semantic label (primary, most likely grader rule):** gallery item is relevant iff it shares the query's land-cover/land-use/scene class.
   - Single-label data (UC-Merced/AID/PatternNet/NWPU scenes): exact class match.
   - **Multi-label data (BigEarthNet-MM, 19 classes):** define relevant via a **label-overlap threshold** — e.g., Jaccard(query labels, gallery labels) ≥ τ, or "shares ≥1 label," or **weighted/graded relevance** (used by CLOSP's nDCG). State τ explicitly.
2. **Geographic correspondence:** for *paired* datasets (SEN1-2, SEN12MS, BigEarthNet-MM), the co-located patch in another modality is the (or a) ground-truth match — this is the **cleanest cross-modal definition** and ideal for SAR↔optical where the paired tile is unambiguously "the answer."
3. **Predefined relevance labels:** if the grader ships an explicit `query_id → [relevant gallery_ids]` map, use it verbatim.

> **Best practice:** use **geographic pairs for cross-modal** ground truth where available (unambiguous), and **semantic class** for same-modal and for cross-modal when pairs are absent. Document the rule per matrix cell.

### 2D. Worked example (end-to-end)

Same-modal optical→optical query, class "airport", gallery has **`R_q = 8`** other "airport" images. System returns top-10; the ranked relevance pattern is:
`[1,1,1,0,1,1,0,1,0,1]` (1 = relevant). Then `r_5 = 4`, `r_10 = 7`.

- **@5:** `P@5 = 4/5 = 0.800`. Raw `R@5 = 4/8 = 0.500` → **F1@5 = 2·0.8·0.5/(0.8+0.5) = 0.615** (= 2·4/(5+8)). Capped `R@5 = 4/min(5,8)=4/5=0.8` → F1@5 = 0.800.
- **@10:** `P@10 = 7/10 = 0.700`. Raw `R@10 = 7/8 = 0.875` → **F1@10 = 2·0.7·0.875/(1.575) = 0.778** (= 2·7/(10+8)). Capped `R@10 = 7/min(10,8)=7/8=0.875` → F1@10 = 0.778 (identical here since R_q≤10).

Cross-modal example, optical→SAR, **`R_q = 1`** (single co-located SAR tile = the only relevant), top-5 = `[0,1,0,0,0]`: `r_5=1`. `P@5=1/5=0.2`, raw `R@5=1/1=1.0` → **F1@5 = 2·1/(5+1)=0.333** (capped identical). This shows why **paired (R_q small) cross-modal caps precision**: even a perfect rank-1 hit gives F1@5 = 0.333. → **calibrate K / gallery so R_q is sensible** (move 7), or, when reporting paired retrieval, also report Recall@K / Top-K accuracy alongside F1.

### 2E. Evaluation-matrix design (query modality × gallery modality)

Build separate **query/gallery splits** and evaluate every cell. With modalities {OPT, MS, SAR}:

| Query ↓ \ Gallery → | OPT | MS | SAR |
|---|---|---|---|
| **OPT** | same-modal | cross-modal | cross-modal |
| **MS** | cross-modal | same-modal | cross-modal |
| **SAR** | cross-modal | cross-modal | same-modal |

- **Same-modal score** = average of the diagonal cells' F1@K (OPT→OPT, MS→MS, SAR→SAR).
- **Cross-modal score** = average of the off-diagonal cells' F1@K. `idea.md` highlights **OPT↔SAR** and **OPT↔MS** specifically — ensure those four directional cells are present and weighted.
- **Mixed-gallery cell (recommended extra):** also evaluate **query=any modality, gallery = ALL modalities mixed** (one unified index). This matches the stated outcome ("gallery may contain images from the same modality or different modalities") and is what a single FAISS index naturally serves.

**Split construction rules:**
- Disjoint query vs gallery items per cell; a query image is **never in its own gallery**.
- For paired data, put one modality of a tile in the query set and other modalities in the gallery so geographic ground truth is exploitable.
- **Class-balanced gallery** (≈ equal members per class) so F1@K isn't dominated by a few huge classes (Part 2B). Fix the gallery size and report it.
- **Average retrieval time** = wall-clock per query for {embed query if needed} + {ANN search + re-rank}. Report (a) **search-only** and (b) **end-to-end incl. encoder**; specify hardware, batch=1, warm cache, and whether the query embedding is precomputed. Report median and p95, not just mean.

---

## PART 3 — Winning Strategy: Ranked Moves (with metric impact & latency cost)

Legend — **Impact:** ★ low → ★★★★ high. **Latency cost:** train-time (no query cost) / query-time add.

### Move 1 — Shared multimodal foundation backbone for a common space ★★★★ (esp. cross-modal)
Use a backbone that already aligns SAR+optical(+MS): **CROMA** (radar-optical contrastive MAE; gives aligned unimodal + joint embeddings), or DOFA/SkySense/Prithvi/SatMAE for MS. Extract per-modality embeddings; this is the single biggest lever (Part 1A shows backbone choice swings mR by 20+ points).
- **Metric impact:** all four F1 cells, **largest on cross-modal** (closes the bulk of the modality gap before any head training).
- **Latency:** train/precompute embeddings offline; **0 added query cost** (embeddings cached). Choose ViT-B over ViT-L if encoder latency matters for unseen queries.
- **Cross-modal focus:** YES — this is where the cross-modal weight is won.

### Move 2 — Per-modality whitening + L2-normalization to close the modality gap ★★★ (cross-modal)
CLIP-style spaces have a documented **modality gap** (modalities occupy disjoint cones). Fit a **per-modality whitening transform (PCA-whiten or ZCA)** on gallery embeddings, then L2-normalize, so SAR / optical / MS distributions are centered and isotropic → cross-modal cosine similarities become comparable. (Mitigating the gap is an active CLIP research line — AlignCLIP etc.)
- **Metric impact:** **cross-modal F1@5/@10 ↑** (same-modal largely unchanged); cheap, near-free win.
- **Latency:** offline fit; query-time = one matrix-multiply (~µs). Negligible.
- **Cross-modal focus:** YES.

### Move 3 — Train a projection head with symmetric InfoNCE + ArcFace ★★★★ (both, esp. cross-modal)
Freeze (or lightly tune) the backbone; train a small projection MLP per modality into a shared D-dim space using **symmetric InfoNCE** (CLIP-style, pull cross-modal positives together) **plus an ArcFace / additive-angular-margin classification head** on land-cover class (intra-class compactness + inter-class margin on the hypersphere). InfoNCE handles cross-modal pairing; ArcFace sharpens class clusters → **directly lifts F1@K** because relevance is class-based. Combo of contrastive + angular-margin is established for retrieval.
- **Metric impact:** **all four F1 cells**; ArcFace especially helps **F1@K** since same-class items cluster tightly (good precision@K) and InfoNCE aligns modalities (cross-modal recall).
- **Latency:** offline; **0 added query cost**.
- **Cross-modal focus:** YES — make the InfoNCE term cross-modal-weighted (sample more OPT↔SAR / OPT↔MS positive pairs).

### Move 4 — Semantic/text anchor space for cross-modal alignment (fills missing modalities) ★★★★ (cross-modal)
Adopt the **CLOSP recipe**: anchor every modality to a **shared text/label space** (class-name prompts or captions), training each image encoder against the anchor rather than requiring SAR↔optical pairs. CLOSP gets **+18–20 nDCG on SAR retrieval** this way and **transfers MS semantics to SAR** with no direct pairs. Use GeoRSCLIP/RemoteCLIP text encoder as the anchor.
- **Metric impact:** **cross-modal F1 ↑↑**, and crucially enables retrieval when a modality has **few/no pairs** (Goal 4 — fill gaps via broad data/anchor).
- **Latency:** offline training; **0 added query cost** (text encoder only needed if doing text queries).
- **Cross-modal focus:** YES — top lever when paired data is scarce.

### Move 5 — ONE FAISS index over all modalities + fast ANN ★★★ (latency + cross-modal usability)
After moves 1–4, all images live in **one aligned space** → build a **single FAISS index over the entire mixed-modality gallery**. Use **HNSW** for best recall/latency at moderate scale, **IVF-Flat/IVF-PQ** for large/RAM-constrained galleries (PQ → ~8 bytes/vector, top-K in µs on GPU). This is what serves the cross-modal/mixed-gallery query directly.
- **Metric impact:** **retrieval-time metric ↓↓** (ms→sub-ms); cross-modal *capability* (one query hits all modalities). Tune `efSearch`/`nprobe` so ANN recall ≥ ~0.98 to avoid hurting F1.
- **Latency:** **this is the latency win.** HNSW query ≈ sub-ms to few-ms at million scale; IVF-PQ µs on GPU.
- **Cross-modal focus:** YES (unified index = native cross-modal serving).

### Move 6 — Cheap k-reciprocal re-ranking of the top candidates ★★★ (F1, esp. @10)
Retrieve top-N (e.g., N=50–100) via FAISS, then **re-rank with k-reciprocal encoding + Jaccard distance** (Zhong et al., CVPR'17). It promotes mutual-NN matches and is **unsupervised, no labels** — reliably lifts mAP/Recall in re-ID and CBIR.
- **Metric impact:** **F1@5 and F1@10 ↑** (better ordering of relevants into top-K), typically the best accuracy-per-millisecond add after the backbone.
- **Latency:** small **query-time add** (re-rank only N candidates, not the whole gallery) — a few ms; bounded by N. Keep N small to protect the time metric.
- **Cross-modal focus:** apply within each cell; especially helps cross-modal where first-pass ranking is noisier. Optionally use **query expansion (average QE)**: average the query with its top-few neighbors' descriptors before re-search — cheap mAP boost.

### Move 7 — Calibrate K vs relevant-set size; class-balanced gallery ★★★ (all F1)
Because F1@K is bounded by `R_q` (Part 2B), **engineer the gallery so `R_q` is near K** (≈5–10 members per class for same-modal; for paired cross-modal where R_q=1, additionally report Top-K accuracy/Recall and consider augmenting positives). Use a **class-balanced gallery** so no huge class caps recall. If the grader fixes the gallery, instead **tune which K to emphasize** and ensure the ranking front-loads relevants.
- **Metric impact:** can raise **achievable F1@5/@10** substantially (the ceiling itself moves) — see Part 2B math.
- **Latency:** none.
- **Cross-modal focus:** neutral, but ensures cross-modal F1 isn't artificially floored.

### Move 8 — Multi-backbone ensemble (concat + whiten) + TTA ★★★ (robustness, all F1)
Concatenate embeddings from **multiple complementary backbones** (e.g., CROMA + GeoRSCLIP-image + a SatMAE/DOFA), **whiten** the concatenation, L2-norm → a more robust descriptor that cross-verifies across backbones (Goal 4: agreement/ensembling). Add **test-time augmentation** (multi-crop/flip; average descriptors — average QE-style) for a small consistent mAP lift.
- **Metric impact:** **all four F1 cells ↑** modestly + robustness across datasets/sensors; reduces variance.
- **Latency:** **query-time add** (run >1 encoder + larger index dim) — the **most expensive** move per query; mitigate by PCA-reducing the concatenation and/or precomputing gallery embeddings. Drop ensemble at inference if the time metric is tight (keep it only for offline gallery quality).
- **Cross-modal focus:** YES for robustness — agreement across backbones is most valuable on the hard cross-modal cells.

### Lower-priority / supporting moves
- **Deep hashing branch (DUCH/DHPL-style):** if the latency metric is weighted hard or gallery is huge, output **binary codes** for Hamming-distance search (UC-Merced optical mAP ~99% with 64-bit) — trades a little accuracy for big speed. Use as an **optional fast path**; keep float re-rank for the top candidates.
- **GeM pooling** instead of average/CLS pooling on CNN/ViT features — small retrieval gain.
- **Hard-negative mining / dynamic-margin triplet** (AMFMN-style) within InfoNCE batches — helps the high intra-class similarity of RS scenes.
- **Modality-specific preprocessing:** SAR speckle filtering + dB scaling + per-band normalization; MS band selection / spectral normalization; resize to backbone's expected input. Bad SAR normalization is a common silent F1 killer.

### Where to spend the cross-modal weight (explicit)
1. **Alignment objective + anchor (moves 2, 3, 4)** — closes the modality gap; this is 60–70% of the cross-modal score.
2. **Shared backbone (move 1)** — provides the already-aligned starting space.
3. **Unified FAISS index (move 5)** — makes cross-modal serving native and fast.
4. **Re-rank (move 6) on cross-modal cells** — cleans noisy first-pass cross-modal rankings.
Same-modal optical is near-saturated (≈99% mAP achievable), so **do not over-invest there**; spend the marginal effort on OPT↔SAR and OPT↔MS.

---

## PART 4 — Robust Cross-Checking (Goal 4)

- **Multiple datasets:** validate on **BigEarthNet-MM** (S1+S2, 19-label, geographic pairs → clean cross-modal GT), **SEN12MS / SEN1-2** (paired S1/S2), and single-modal scene archives (**UC-Merced, AID, PatternNet, NWPU-RESISC45**) for same-modal sanity. Cross-dataset evaluation guards against overfitting one archive's class distribution.
- **Multiple backbones (ensemble/agreement):** combine CROMA (radar-optical), GeoRSCLIP/RemoteCLIP (semantic), SatMAE/DOFA (MS/any-sensor). **Agreement** between independently-trained embeddings is a strong relevance signal; disagreement flags hard/ambiguous queries. Concatenate+whiten (move 8) or rank-fuse (e.g., Reciprocal Rank Fusion) their result lists.
- **Fill missing-modality gaps with broad data + anchor:** when a modality lacks pairs, use the **text/label anchor (move 4)** so its embeddings still land in the shared space (CLOSP transfers MS→SAR semantics with zero SAR-optical pairs). Pretrain/fine-tune on the **largest** available multi-modal corpora (RS5M for semantics; BigEarthNet/SEN12MS for SAR-optical) to cover modalities and seasons (the problem explicitly mentions different seasons/acquisition conditions).
- **Cross-verify the metric:** compute F1@K with **both** recall conventions (raw and capped, Part 2B) and **both** relevance definitions (semantic-class and geographic-pair) and confirm rankings of candidate systems are stable across them before trusting a single number.

---

## Sources

- Cross-modal RS retrieval / SAR-optical common space (survey + 2024-25): https://arxiv.org/html/2507.10403v1 (CLOSP) · https://arxiv.org/pdf/2204.09868 · https://www.sciencedirect.com/science/article/pii/S092427162500245X · https://link.springer.com/chapter/10.1007/978-3-030-00776-8_36
- RSITR methods & benchmark numbers: https://arxiv.org/html/2504.05644v1 (iEBAKER table — AMFMN/GaLR/KAMCL/PE-RSITR/RemoteCLIP/GeoRSCLIP/EBAKER mR) · https://github.com/jaychempan/Awesome-RSITR · https://www.mdpi.com/2072-4292/17/24/3995 (2025 review) · https://arxiv.org/html/2405.10160v1 (PIR) · https://github.com/xiaoyuan1996/GaLR · https://arxiv.org/pdf/2405.03373 (KAMCL) · https://github.com/ZhanYang-nwpu/PE-RSITR
- RemoteCLIP / RS5M / GeoRSCLIP: https://arxiv.org/abs/2306.11029 · https://arxiv.org/html/2306.11300 · https://github.com/om-ai-lab/RS5M
- CROMA (radar-optical shared space): https://arxiv.org/abs/2311.00566 · https://github.com/antofuller/CROMA · https://liner.com/review/croma-remote-sensing-representations-with-contrastive-radaroptical-masked-autoencoders
- Sensor-agnostic / cross-modal image↔image retrieval & hashing: https://arxiv.org/pdf/2401.07782 (CSMAE) · https://arxiv.org/pdf/2201.08125 (DUCH) · https://arxiv.org/pdf/2204.08707 · https://www.sciencedirect.com/science/article/abs/pii/S0167865520300453 (CMIR-NET)
- SAR↔optical matching descriptors: https://arxiv.org/pdf/2502.18104 (PromptMID)
- Multi-modal datasets: https://arxiv.org/abs/2105.07921 (BigEarthNet-MM) · https://bigearth.net/ · https://isprs-annals.copernicus.org/articles/IV-2-W7/153/2019/ (SEN12MS) · https://arxiv.org/pdf/2103.08259 (QXS-SAROPT)
- Same-modal CBIR baselines & deep hashing mAP: https://mdpi.com/2072-4292/13/15/2924/htm (DHPL proxy loss) · https://www.mdpi.com/2072-4292/12/17/2789 (HPSH) · https://arxiv.org/pdf/1904.01258 (MHCLN metric+hash) · https://arxiv.org/pdf/1706.03424 (PatternNet benchmark) · https://arxiv.org/pdf/1902.05818 (triplet metric learning)
- F1@K / IR metrics: https://en.wikipedia.org/wiki/F-score · https://apxml.com/courses/basics-model-evaluation-metrics/chapter-2-metrics-for-classification/f1-score-metric
- Modality gap (CLIP): https://arxiv.org/abs/2406.17639 (AlignCLIP / Mitigate the Gap)
- ArcFace: https://arxiv.org/abs/1801.07698
- k-reciprocal re-ranking: https://arxiv.org/abs/1701.08398
- TTA / query expansion for retrieval: https://arxiv.org/pdf/2002.01642 · https://arxiv.org/pdf/1811.00202 (GeM)
- FAISS / ANN (HNSW, IVF, PQ) latency vs recall: https://pyimagesearch.com/2026/02/16/vector-search-with-faiss-approximate-nearest-neighbor-ann-explained/ · https://www.pingcap.com/article/approximate-nearest-neighbor-ann-search-explained-ivf-vs-hnsw-vs-pq/
- Multimodal RS foundation models (DOFA/CROMA/SatMAE/Prithvi/CSMoE): https://arxiv.org/html/2503.22081v1 · https://arxiv.org/pdf/2509.14104
