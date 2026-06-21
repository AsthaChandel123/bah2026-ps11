# Cross-Modal Alignment for Multi-Sensor Satellite Image Retrieval

**BAH 2026 — Problem Statement 11: Cross-Modal Satellite Image Retrieval Using Multi-Sensor Remote Sensing Data**

Research focus: methods to align optical (RGB), multispectral (MS), and SAR modalities into a **shared embedding space** so that semantically similar scenes are nearest neighbors regardless of sensor — enabling optical↔SAR and optical↔MS cross-modal retrieval. This is the highest-weighted part of the challenge (F1@5/F1@10 cross-modal), so this survey is deliberately rigorous.

---

## 0. Problem framing and what "alignment" must achieve

The retrieval system computes a compact descriptor `z = f(x)` for every query and gallery image, then ranks the gallery by cosine similarity (or L2) to the query. For **same-modal** retrieval this is standard metric learning. For **cross-modal** retrieval the killer requirement is that `f_optical(scene)` and `f_SAR(same/similar scene)` land at (almost) the same point. Two failure modes dominate:

1. **Semantic misalignment** — embeddings encode sensor-specific texture (SAR speckle, optical color) instead of land-cover semantics, so a SAR forest and an optical forest are far apart.
2. **Modality gap** — even when *semantically* aligned, contrastive encoders place each modality in its own narrow cone of the hypersphere; SAR embeddings cluster near SAR embeddings and optical near optical, with a constant offset between the two clusters (Section 13). Cross-modal cosine similarities are then systematically depressed and intra-modal distractors out-rank true cross-modal matches.

A strong solution must address **both**. Below are 12+ distinct architectural/loss patterns, then a dedicated modality-gap section, then a concrete recommended pipeline and a zero-shot fallback.

**Evaluation reminder (drives the recommendation):** metric is F1@5 / F1@10 for same- and cross-modal, plus retrieval time per query. Relevance is by semantic class / geo-correspondence. So we optimize for *ranking neighbors of the same semantic class across modalities*, with a fast (ideally frozen-backbone + FAISS) inference path.

---

## 1. CLIP-style symmetric InfoNCE contrastive alignment between modality pairs

**Mechanism.** Treat co-registered (optical, SAR) patches of the same location as positive pairs; all other pairs in the batch are negatives. Two encoders map each modality to L2-normalized embeddings; a symmetric InfoNCE (NT-Xent) loss pulls matched pairs together and pushes mismatched apart. This is the bilateral generalization of CLIP from (image, text) to (sensorA, sensorB).

**Loss.** For a batch of `N` aligned pairs, with optical embeddings `u_i`, SAR embeddings `v_i` (both unit-norm), temperature `τ`:

```
L_o→s = -(1/N) Σ_i log[ exp(u_i·v_i / τ) / Σ_j exp(u_i·v_j / τ) ]
L_s→o = -(1/N) Σ_i log[ exp(v_i·u_i / τ) / Σ_j exp(v_i·u_j / τ) ]
L_clip = ½ (L_o→s + L_s→o)
```

`τ` is usually a learnable scalar (init ≈ 0.07, i.e. logit scale ≈ 14–100). The symmetric form is what CLIP, RemoteCLIP, SARCLIP, and PR-CLIP optimize.

**Architecture.** Modality-specific encoders (CNN or ViT), each with a small linear/MLP projection head to a shared D-dim space (256–512). No weight sharing required; the alignment lives entirely in the loss. Optionally initialize each encoder from a unimodal pretrained checkpoint.

**Pros for retrieval.** Directly optimizes the cosine-similarity geometry used at query time; produces a single shared space queryable with one FAISS index; strong zero-shot transfer; scales with batch size (more negatives → sharper space). The de-facto baseline for cross-modal retrieval.

**Cons.** Needs **co-registered positive pairs** (e.g. SEN12MS, BigEarthNet-MM); large batches matter (more negatives); **induces a modality gap** by construction (Section 13) which must be corrected post-hoc; speckle/seasonal noise creates false negatives among in-batch pairs.

**Data needs.** Paired/co-registered (no class labels strictly required — geo-correspondence supplies positives). Unpaired data cannot be used directly.

**Compute.** Moderate; benefits from large batch (gradient cache / memory bank if GPU-limited). Inference is cheap (one forward pass + dot product).

**Reference.** Radford et al., *Learning Transferable Visual Models From Natural Language Supervision* (CLIP), arXiv:2103.00020. RS adaptations: RemoteCLIP (arXiv:2306.11029); SARCLIP (arXiv:2510.22665); PR-CLIP (Remote Sensing 17(13):2117).

---

## 2. Modality-specific encoders + a SHARED projection / embedding head (late alignment)

**Mechanism.** Keep separate per-modality backbones (so each can model speckle vs. color natively) but force their *final* layers into a **single shared projection head** `g(·)` whose weights are identical across modalities. Alignment happens "late": each backbone outputs a feature, then the shared head maps all modalities through the same function, so the output geometry is forced to be common. Trained with InfoNCE and/or a metric/classification loss on top.

**Loss.** Any of: symmetric InfoNCE (Section 1), supervised contrastive (if labels), or cross-entropy over shared class prototypes through the shared head. The shared head + a classification or contrastive objective is what makes the spaces commensurable.

**Architecture.** `z = g(f_modality(x))`, where `f_optical`, `f_SAR`, `f_MS` differ but `g` is shared. Often `g` is a 2–3 layer MLP with the *last* linear layer shared (or the whole head shared, with optional per-modality BatchNorm to absorb scale differences).

**Pros for retrieval.** Cheap and robust; partial weight sharing biases the network toward a common space and tends to **reduce (not eliminate) the modality gap** vs. fully separate heads; easy to bolt onto frozen foundation backbones (this is the head used in the recommended pipeline, Section 14).

**Cons.** Shared head alone does not guarantee semantic alignment — still needs a cross-modal loss; if backbones are too modality-specific the shared head may be a bottleneck.

**Data needs.** Paired (for contrastive) or class-labeled (for shared-prototype CE). 

**Compute.** Low — especially if backbones are frozen and only the head trains.

**Reference.** This is a standard pattern in cross-modal hashing/retrieval; see deep cross-modal hashing surveys and the head design in CSMAE/CLIP variants. Architecturally identical to the "projection head" in SimCLR/CLIP applied per-modality with sharing.

---

## 3. Single shared backbone with modality conditioning (modality tokens / FiLM / hypernetwork — DOFA-style dynamic weights)

**Mechanism.** Use **one** transformer backbone for all sensors; inject modality information at the *input* so the same weights process any sensor. Variants:
- **Modality tokens / learned channel embeddings** added to patch tokens.
- **FiLM** (feature-wise linear modulation): per-modality scale/shift conditions intermediate activations.
- **Hypernetwork-generated patch embedding (DOFA)**: a wavelength-conditioned hypernetwork *generates the patch-embedding convolution weights* from each channel's central wavelength, so a single ViT handles any number/type of spectral bands (SAR VV/VH, S2 12-band, hyperspectral 224-band).

**DOFA mechanism (concrete).** Central wavelengths `λ ∈ R^C` are sine-cosine encoded, passed through FC + a small transformer with learnable query tokens to emit dynamic conv weights `M_w` and biases `M_b`; patch embedding is `Conv(X, reshape(M_w), M_b)`. A **shared ViT** then processes all modalities identically. Pretraining = masked image modeling + cosine feature distillation from an ImageNet teacher (RGB proxy for >3-band inputs):

```
L = (1/N) Σ_i ||X_i - X'_i||²  -  cos( FP(F_s_i), F_t_i )
```

**Architecture.** Shared backbone; modality only changes the input adapter (tokens/FiLM/hypernet). Embeddings for different sensors are *already in the same space* because they pass through identical weights.

**Pros for retrieval.** A single model and **single embedding space by construction** — minimal modality gap because there is one encoder, not two cones; handles *unseen* sensors (DOFA generalized to Landsat-8 never seen in pretraining); ideal when you want one backbone for optical+MS+SAR. Public ViT-B/ViT-L weights (HuggingFace `XShadow/DOFA`, GitHub `zhu-xlab/DOFA`).

**Cons.** Conditioning must be expressive enough to separate sensors when needed; pretraining a hypernetwork is non-trivial (but you can use released weights); a shared backbone can underfit a very SAR-specific texture compared to a dedicated SAR encoder.

**Data needs.** Pretraining used 5 sensors (S1, S2, Gaofen-2, NAIP, EnMAP), ~8M samples; *for the challenge you use the released weights* and only fine-tune a head, so your data need is small (paired or labeled).

**Compute.** Pretraining is large-scale; **using** DOFA is cheap (frozen backbone + head). This is the top recommended backbone (Section 14).

**Reference.** Xiong et al., *Neural Plasticity-Inspired Multimodal Foundation Model for Earth Observation* (DOFA / "One for All"), arXiv:2403.15356. FiLM: Perez et al., AAAI 2018. Hypernetworks: Ha et al., ICLR 2017.

---

## 4. Cross-modal masked modeling / cross-reconstruction (CROMA, CSMAE — one modality predicts the other)

**Mechanism.** Self-supervised MAE-style learning where masked patches are reconstructed using context from *both* the same modality (uni-modal) and the *other* modality (cross-modal). Forcing reconstruction across sensors compels a shared latent that carries semantics transferable between modalities. Two flagship RS models:

### 4a. CROMA (Contrastive Radar-Optical Masked Autoencoders)
Three encoders: a unimodal **SAR (S1) ViT**, a unimodal **optical (S2) ViT**, and a **fusion encoder** producing joint encodings; a lightweight decoder predicts masked patches. **Two objectives combined**: (i) cross-modal **InfoNCE** between the pooled SAR and optical representations (Section 1 form), and (ii) **MAE reconstruction** of masked multispectral+SAR patches via the fusion encoder. Introduces X-ALiBi / 2D-ALiBi attention biases enabling extrapolation to images up to 17.6× larger at test time. Produces **three usable embeddings: radar-only, optical-only, joint** — radar-only and optical-only live in the contrastively-aligned shared space, perfect for cross-modal retrieval. ViT-B and ViT-L; pretrained on SSL4EO-S12. Public weights.

### 4b. CSMAE (Cross-Sensor Masked Autoencoders) — *purpose-built for this exact task*
Explicitly designed for **sensor-agnostic content-based image retrieval** in RS. A CSMAE = multi-sensor encoder + cross-sensor encoder + multi-sensor decoder (all ViT). Reconstructs masked patches from (a) unmasked patches of the same sensor (uni-modal) and (b) unmasked patches of the *paired other-sensor* image (cross-modal):

```
L_UMR = (1/|M|) Σ_{n∈M} || d(Z^j_U, z^j_m) - p^j_n ||²        (uni-modal)
L_CMR = (1/|M|) Σ_{n∈M} || d(Z^k_U, z^k_m) - p^j_n ||²        (cross-modal, k≠j)
L_CSMAE = Σ_{j∈{1,2}} ( L_UMR(x^j) + L_CMR(x^j) )
```

Plus two **latent-alignment** terms to close the gap directly:
- **Inter-modal Discrepancy Elimination** `L_MDE = -(1/|B|) Σ_i log(1 + exp(S(c1_i, c2_i)))` — pulls paired sensor embeddings to small angular distance.
- **Mutual-Information Maximization** `L_MIM` — NT-Xent that pulls same-location pairs together, pushes others apart.

Four variants by encoder/decoder sharing: **CECD / CESD / SECD / SESD** (Common vs Sensor-specific Encoder, Common vs Sensor-specific Decoder). Retrieval uses **global average pooling** of patch tokens (beats [CLS]) + kNN. **Best for cross-sensor retrieval: SECD + L_MIM** (sensor-specific encoders, shallow cross-sensor layers), giving balanced ~71% F1@10 on both S1→S2 and S2→S1 on BigEarthNet-MM; without inter-modal losses cross-modal drops to ~62–67%.

**Architecture.** Separate or shared ViT encoders + a cross-sensor fusion encoder + decoder; alignment from cross-reconstruction *and* explicit latent losses.

**Pros for retrieval.** CSMAE is the closest published analogue to PS-11 and reports F1@10 directly; cross-reconstruction yields semantically deep shared features; the explicit MDE/MIM terms **shrink the modality gap**; works without class labels.

**Cons.** Heavier to pretrain than a contrastive-only model; needs co-registered pairs; CROMA's contrastive head still leaves a residual gap that benefits from post-hoc centering.

**Data needs.** Paired/co-registered, no labels needed (SSL4EO-S12 ~251k or BigEarthNet-MM ~270k–590k pairs).

**Compute.** Pretraining is GPU-heavy; using released CROMA weights or training CSMAE on a subset is feasible. Inference cheap.

**Reference.** Fuller et al., *CROMA*, NeurIPS 2023, arXiv:2311.00566 (code + weights public). Hackel/Clasen et al., *Exploring Masked Autoencoders for Sensor-Agnostic Image Retrieval in Remote Sensing* (CSMAE), IEEE TGRS, arXiv:2401.07782, GitHub `jakhac/CSMAE`.

---

## 5. Decoupling common + unique representations (DeCUR) — retrieve on the COMMON part

**Mechanism.** Split each modality's embedding into a **common** block (`K_c` dims, aligned across modalities) and a **unique** block (`K_u` dims, decorrelated across modalities). Uses Barlow-Twins-style **redundancy reduction** on the cross-correlation matrix between modalities: drive the common block's cross-correlation toward the **identity** (align) and the unique block's cross-correlation toward **zero** (decorrelate). Plus intra-modal Barlow terms per modality.

**Loss.** With normalized cross-correlation `C_ij = Σ_b z^A_{b,i} z^B_{b,j} / (||z^A_{:,i}|| ||z^B_{:,j}||)`:

```
L_com = Σ_i (1 - C^c_ii)²  + λ_c Σ_i Σ_{j≠i} (C^c_ij)²      (common → identity)
L_uni = Σ_i (C^u_ii)²       + λ_u Σ_i Σ_{j≠i} (C^u_ij)²      (unique → zero)
L_M1, L_M2 = intra-modal Barlow terms (full dims)
L = L_com + L_uni + L_M1 + L_M2
```

**Architecture.** Modality-specific encoders `E1,E2` + projectors `P1,P2`; embedding partitioned into common/unique. For SAR–optical they use `K_c` ≈ 87.5% of dims common.

**Retrieve on the common part.** At query time use **only the common block** for cross-modal matching — it is explicitly aligned, so the modality gap on that sub-space is minimized; keep the unique block for same-modal retrieval if helpful. This is a clean, principled way to separate "what's shared" (retrieval-relevant across sensors) from "what's sensor-specific" (noise for cross-modal).

**Pros for retrieval.** Directly produces a cross-modal-ready sub-embedding; no negative sampling / large batches needed (Barlow Twins is negative-free); robust to missing modality; consistent gains on radar-optical, RGB-DEM, RGB-depth.

**Cons.** Choosing `K_c/K_u` split is a hyperparameter; Barlow needs large embedding dim and batch for stable covariance; alignment is dimension-wise (decorrelation) not instance-wise, so very fine semantic ranking may still want a metric-loss finetune.

**Data needs.** Paired/co-registered, **no labels**.

**Compute.** Moderate pretraining; **public weights available** so you can use as a frozen backbone.

**Reference.** Wang et al., *DeCUR: Decoupling Common and Unique Representations for Multimodal Self-Supervised Learning*, ECCV 2024, arXiv:2309.05300, code public.

---

## 6. Knowledge distillation: strong optical teacher → SAR student into the same space

**Mechanism.** Train (or take) a strong optical encoder as a **frozen teacher**; train a SAR **student** to mimic the teacher's embedding for the *same location*. The student inherits the teacher's semantic space, so SAR embeddings become directly comparable to optical embeddings — alignment without a symmetric contrastive game.

**Loss.** Feature distillation on co-registered pairs (teacher `t`, student `s`):

```
L_distill = (1/N) Σ_i || s(x^SAR_i) - sg[ t(x^opt_i) ] ||²        (or 1 - cos)
```

optionally + relational/KD terms (match pairwise similarity matrices) and a small contrastive term. `sg` = stop-gradient (teacher frozen). Cross-source RS retrieval works (Xiong et al.) ensemble source-shared + source-specific teacher classifiers and distill into a student.

**Architecture.** Teacher = pretrained optical backbone (frozen); student = SAR backbone (trainable); shared output dim. The optical embedding *defines* the space; SAR is mapped into it.

**Pros for retrieval.** The optical space (often higher quality, easier semantics) becomes the anchor; **no modality gap by construction** for SAR↔optical because SAR is *forced onto* optical embeddings, not into its own cone; asymmetric so no large-batch negative requirement; great when one modality has a much stronger pretrained model.

**Cons.** SAR can only ever approximate optical semantics it cannot observe (e.g. color); error ceiling set by teacher; for 3-way (optical/MS/SAR) you distill each non-anchor modality separately; if optical teacher itself is biased, student inherits it.

**Data needs.** Paired/co-registered. No labels needed (teacher provides soft targets). 

**Compute.** Cheap — only the student trains; teacher inference can be cached.

**Reference.** Liu et al., *Applications of Knowledge Distillation in Remote Sensing: A Survey*, arXiv:2409.12111. ERVD ViT distillation for RS retrieval, arXiv:2412.18136. Cross-Source Image Retrieval via ensemble + KD (Xiong et al., 2021). General cross-modal distillation: Gupta et al., CVPR 2016.

---

## 7. Modality translation (SAR→optical via GAN/diffusion, then match in optical space)

**Mechanism.** Translate the SAR query into a *synthetic optical* image with a generative model (CycleGAN, pix2pix, or conditional diffusion), embed the fake-optical with a single optical encoder, and match against real optical galleries. Alignment is sidestepped: everything is compared in *one* modality's space.

**Architecture.** Generator `G: SAR → optical` (+ discriminator for GAN, or a denoising UNet/diffusion for the modern version) → frozen optical encoder → cosine search.

**Pros for retrieval.** Reuses a single strong optical retrieval model; interpretable (you can see the translated image); diffusion variants give high-fidelity, fewer artifacts; useful as a *re-ranking* or visualization aid.

**Cons (significant for retrieval).** Translation is **lossy and hallucination-prone** — GANs suffer mode collapse / unstable training under the large SAR↔optical domain gap; diffusion is **slow at inference** (many denoising steps), which directly hurts the "retrieval time per query" metric; errors in `G` propagate into wrong neighbors; needs paired data to train `G` well; does not give a true joint space (MS↔SAR would need more translators). Generally a **fallback / augmentation**, not the primary alignment engine for a latency-sensitive retrieval system.

**Data needs.** Paired SAR-optical to train the translator (pix2pix) or unpaired (CycleGAN, Schrödinger bridge). 

**Compute.** Training and (for diffusion) inference are expensive; GAN inference is fast but quality-limited.

**Reference.** *Generative models for SAR–optical image translation: a systematic review*, ISPRS (2025), ScienceDirect S1569843225006569. Diffusion: Bai et al. (brain-inspired diffusion SAR→optical, PMC10861657); Adversarial Consistency Distillation, arXiv:2407.06095. CycleGAN: Zhu et al., ICCV 2017. SOMA-1M alignment dataset, arXiv:2602.05480.

---

## 8. Deep CCA / Deep Canonical Correlation Analysis (two-view alignment)

**Mechanism.** Learn nonlinear transforms of two views (optical, SAR) whose outputs are **maximally linearly correlated**. CCA finds projections maximizing canonical correlation; DCCA replaces the linear projections with neural nets and optimizes the sum of top-k singular values of the cross-covariance of the two views' outputs. DGCCA extends to >2 views (optical+MS+SAR).

**Loss (DCCA).** Maximize total correlation `corr(f1(X1), f2(X2))`, computed as the trace norm of the whitened cross-covariance:

```
maximize  Tr( (Σ11^{-1/2} Σ12 Σ22^{-1/2})ᵀ (Σ11^{-1/2} Σ12 Σ22^{-1/2}) )^{1/2}
```

where `Σ12` is the cross-covariance of the two networks' outputs, `Σ11,Σ22` the per-view covariances (with regularization). Solved with full/large-batch gradients (needs covariance estimates).

**Architecture.** Two networks → shared-dim outputs aligned by correlation; at test time concatenate or sum the canonical components for cross-modal matching.

**Pros for retrieval.** Statistically grounded; negative-free; produces a genuinely correlated (hence comparable) space; good when paired data is moderate; can fuse >2 modalities (DGCCA).

**Cons.** Covariance estimation needs **large batches** and is numerically finicky (matrix inverse-sqrt, regularization); correlation ≠ semantic discriminability, so often paired with a downstream metric/classifier; less popular than contrastive at scale; historically used for image-text, fewer RS-SAR results.

**Data needs.** Paired/co-registered. Labels optional.

**Compute.** Moderate but large-batch; SVD per step.

**Reference.** Andrew et al., *Deep Canonical Correlation Analysis*, ICML 2013. DGCCA: Benton et al., 2017. Cross-modal retrieval with DCCA: Yan & Mikolajczyk (CVPR 2015) for image-text; DCCA-PHS (Neurocomputing 2017).

---

## 9. Optimal-transport / Wasserstein feature alignment between modality distributions

**Mechanism.** Rather than aligning *instances*, align the *distributions* of optical vs SAR embeddings by minimizing an Optimal-Transport (Wasserstein) cost. Solve for a soft transport plan `P` matching SAR samples to optical samples (Sinkhorn-regularized), and minimize the transport cost — pulling the two modality point-clouds onto each other. Can be added as a regularizer to a contrastive backbone.

**Loss.** Entropic-regularized OT between batches of optical features `{u_i}` and SAR features `{v_j}`:

```
W_ε = min_{P∈Π(a,b)}  Σ_ij P_ij C_ij  -  ε H(P),   C_ij = ||u_i - v_j||²  (or 1 - cos)
```

solved by Sinkhorn iterations; the resulting `W_ε` is back-propagated. Variants: Gromov-Wasserstein (for spaces without direct correspondence), Bures-Wasserstein over per-modality Gaussians/GMMs (DecAlign), Wasserstein-barycenter consistency for multimodal contrastive (OT-CMA).

**Architecture.** Any two encoders; OT term added to the training objective. Optionally cluster each modality into prototypes and do multi-marginal OT over prototypes (cheaper, more stable).

**Pros for retrieval.** Aligns global geometry → **directly attacks the modality gap** (distribution-level, not just per-pair); works even with *weakly* paired or unpaired data (GW needs no correspondence); complements InfoNCE (handles the part contrastive misses).

**Cons.** Sinkhorn is O(N²) per batch (cost matrix), adds compute; sensitive to `ε` and batch composition; pure OT can align distributions while permuting semantics (a forest mapped to a field) unless combined with a correspondence/semantic term; more of a *regularizer* than a standalone retrieval objective.

**Data needs.** Unpaired (GW) up to paired; class-balanced batches help.

**Compute.** Moderate-high (Sinkhorn loop); prototype-level OT is much cheaper.

**Reference.** Courty et al., *Optimal Transport for Domain Adaptation*, TPAMI 2017. OTAdapt, arXiv:2205.10738. *Enhancing Multimodal Contrastive Learning via OT-Based Consistent Modality Alignment*, Springer 2024. DecAlign (Bures-Wasserstein over modality GMMs).

---

## 10. Prototype / cluster / anchor alignment (shared class prototypes; SwAV-style)

**Mechanism.** Define a shared set of learnable **prototypes** (cluster centers) used by *all* modalities. Each image is assigned a soft cluster code; cross-modal alignment is enforced by **swapped prediction** — predict the SAR view's cluster code from the optical view's embedding and vice-versa (SwAV across modalities). Because both modalities map to the *same* prototype bank, semantically similar scenes share codes → shared space. With labels, the prototypes are simply class centroids and you align via cross-entropy to shared centroids (cross-modal center loss).

**Loss (SwAV-style swapped prediction).** With shared prototypes `C`, codes `q` from Sinkhorn, predictions `p = softmax(z·C / τ)`:

```
L = - Σ [ q^SAR · log p^opt  +  q^opt · log p^SAR ]
```

(codes computed by Sinkhorn-Knopp to avoid collapse). Supervised variant: cross-modal center loss pulling each sample to its class centroid shared across modalities.

**Architecture.** Per-modality encoders + a **single shared prototype/centroid matrix**; no pairwise negatives needed (online clustering).

**Pros for retrieval.** Scalable (no pairwise comparisons / huge batches); shared prototypes create a **modality-agnostic semantic skeleton** ideal for class-based F1 retrieval; works with class labels (very natural for the challenge's class-based relevance) or unsupervised; cluster structure aids fast approximate search.

**Cons.** Number of prototypes/clusters is a hyperparameter; soft-assignment can blur fine intra-class distinctions; collapse risk without Sinkhorn balancing; unsupervised version doesn't guarantee clusters = the evaluation's semantic classes.

**Data needs.** Unpaired + clustering (SwAV) **or** class-labeled (center/prototype loss); paired helps but not required.

**Compute.** Low-moderate (online clustering is efficient).

**Reference.** Caron et al., *SwAV: Unsupervised Learning of Visual Features by Contrasting Cluster Assignments*, NeurIPS 2020, arXiv:2006.09882. Cross-modal center loss: Jing et al., CVPR 2021 (cross-modal retrieval). PCL/prototypical contrastive: Li et al., ICLR 2021.

---

## 11. Adapter / LoRA / projection fine-tuning of a frozen multimodal foundation model (parameter-efficient)

**Mechanism.** Take a **frozen** multimodal RS foundation backbone (DOFA / CROMA / DeCUR / Galileo) and adapt it with a *tiny* number of trainable parameters: bottleneck **Adapters** inserted in transformer blocks, **LoRA** low-rank updates `W + BA` on attention/MLP weights, or simply a trainable **projection head** on top. Train these with a cross-modal alignment loss (Section 1/12) while the backbone stays fixed.

**Loss.** Symmetric InfoNCE + metric loss on the adapter/head outputs (backbone frozen):
```
W_eff = W_frozen + (α/r) B A          (LoRA; A∈R^{r×d}, B∈R^{d×r}, r≪d)
L = L_clip(z) + λ L_triplet(z)        z = head(backbone_LoRA(x))
```

**Architecture.** Frozen foundation encoder → LoRA/adapter (optional) → small per-modality projection head → shared space.

**Pros for retrieval.** **Best practicality/compute trade-off** — leverages billions of pretraining samples, trains in hours on one GPU, tiny checkpoints; the foundation backbone already encodes RS semantics so the head only has to fix alignment + close the gap; fast inference (one forward pass). Directly fits the challenge's "limited compute, fast inference, public weights" constraints.

**Cons.** Bounded by the frozen backbone's quality; if the backbone wasn't trained on your sensor/wavelength, may underperform (mitigated by DOFA's wavelength conditioning); LoRA adds slight inference cost unless merged (it can be merged into `W` so zero overhead).

**Data needs.** Small paired set for the head/adapters (or labels for a supervised metric loss). Far less data than pretraining.

**Compute.** **Lowest of all training methods.** Recommended path (Section 14).

**Reference.** Hu et al., *LoRA*, ICLR 2022. Houlsby et al., *Adapter*, ICML 2019. RS PEFT: *Fine-tune Smarter, Not Harder: PEFT for Geospatial Foundation Models*, arXiv:2504.17397; PeftCD, arXiv:2509.09572.

---

## 12. Triplet / metric alignment with cross-modal positive/negative mining

**Mechanism.** Classic deep metric learning generalized to be cross-modal. Build triplets `(anchor, positive, negative)` where anchor and positive are **different modalities of the same semantic class/location** and the negative is a different class (possibly any modality). A margin ranking loss makes cross-modal positives closer than negatives. Mining strategy (semi-hard → hard, curriculum) is critical.

**Loss (cross-modal triplet).**
```
L = Σ max( 0, m + d(a, p) - d(a, n) ),   a∈modalityX, p∈modalityY (same class), n∈ different class
```
with `d` = cosine/L2 distance, margin `m`. Add an **intra-modal consistency** term so the space is also well-ordered *within* each modality (improves same-modal F1 and stabilizes cross-modal). Often combined with InfoNCE (which is essentially a soft, all-negatives version of this).

**Architecture.** Per-modality encoders + shared head; sampler that draws cross-modal triplets; optional cross-modal **center loss** to anchor each class.

**Pros for retrieval.** Optimizes the *ranking* objective the metric (F1@k) rewards; mining hard cross-modal negatives sharpens the boundary between true matches and same-class-wrong-modality distractors; works with **class labels** (the challenge supplies semantic-class relevance) without needing dense pixel co-registration; complements InfoNCE as a fine-ranking term.

**Cons.** Hardest-negative mining can destabilize training (the hardest negatives are often label-noise) — use semi-hard/curriculum; triplet sampling is O(combinatorial), needs careful batching; pure triplet converges slower than InfoNCE on large data.

**Data needs.** **Class-labeled** (or geo-paired) — does not require co-registered pixel pairs, only class/location correspondence; very compatible with PS-11's relevance definition.

**Compute.** Moderate; sampling overhead. Inference cheap.

**Reference.** Schroff et al., *FaceNet* (triplet + semi-hard mining), CVPR 2015. *Intramodal consistency in triplet-based cross-modal learning*, Machine Learning (Springer), 2024 (>9% mAP from semi-hard→hard curriculum). Cross-modal center/triplet loss surveys. SAR-optical hard-negative mining via GAN, Remote Sensing 10(10):1552.

---

## 13. The MODALITY GAP — diagnosis and remedies (critical for cross-modal F1)

**What it is.** In CLIP-like spaces, embeddings of each modality occupy a **narrow cone**, and the two cones are **separated by a near-constant offset**: SAR embeddings cluster with SAR, optical with optical, and the *mean* of each modality sits at "arm's length" from the other on the hypersphere. Even semantically matched cross-modal pairs then have lower cosine similarity than mismatched *same-modality* pairs — so a naive shared-index retrieval returns mostly same-modality neighbors and cross-modal F1 collapses.

**Why it arises (Liang et al., "Mind the Gap", NeurIPS 2022).**
1. **Cone effect / nonlinear activations** — each layer shrinks pairwise angles, so *random-init* networks already emit embeddings in a tight cone (measured avg cosine 0.56–0.99). Two separate encoders → two *different* cones at initialization, before any training.
2. **Different random inits** per encoder → spatially distinct cones that training does not merge.
3. **Contrastive loss + low temperature preserves the gap.** Their embedding-shift experiment shows the *default* gap is actually a **global minimum** of the contrastive loss — shifting the clusters together *increases* the loss when `τ` is small (CLIP's `τ≈0.01`). With high `τ` (≥0.1) the repulsion vanishes and closing the gap becomes optimal. So **temperature literally controls gap size.**

The follow-up *"It's Not a Modality Gap: the Contrastive Gap"* (arXiv:2405.18570) and *"Decipher the Modality Gap"* (arXiv:2510.03268) argue the separation is driven by the low-dim/uniformity dynamics of contrastive optimization, and that closing it can *help* cross-modal alignment for retrieval (distinct from Liang et al.'s task-dependent stance).

### Remedies (most useful first for retrieval)

**(1) Per-modality mean-centering (GR-CLIP) — cheapest, highest ROI.** Compute the mean embedding of each modality on a representative set and subtract it before similarity:
```
e'^SAR_i = e^SAR_i - μ_SAR ,   e'^opt_i = e^opt_i - μ_opt ,   (μ = E[e] per modality)
```
then L2-renormalize and use cosine. This *removes the constant offset between cones*, directly lifting cross-modal cosine similarities. GR-CLIP reports up to **+26 NDCG@10** over baseline CLIP for mixed-modality search and beats a SoTA model with **75× less compute** — a post-hoc, training-free fix. For a mixed-modality gallery, center each gallery item by *its own* modality mean. **This is the single best gap remedy for PS-11.**

**(2) Whitening / PCA / ZCA per modality (isotropy).** Beyond centering, decorrelate and normalize variance: `e' = Σ^{-1/2}(e - μ)`. Whitening makes each modality's cloud isotropic, so cosine similarity becomes a reliable semantic measure and residual anisotropy that depresses cross-modal scores is removed. Optionally remove the top principal component(s) (the dominant "modality/frequency" direction often encodes sensor identity, not semantics). Established for sentence embeddings (BERT-whitening) and shown to improve cross-modal/cross-lingual retrieval; apply with a *shared* whitening fit on the union, or per-modality then a common rotation. Watch overfitting the whitening matrix on small data (shrink/regularize `Σ`).

**(3) Temperature tuning / scheduling.** Since `τ` controls the gap, treat it as a deliberate knob: a *moderately higher* `τ` (or a learnable `τ` with a ceiling, or a `τ` schedule warm→cool) reduces the repulsion that manufactures the gap, trading a little same-modal sharpness for much better cross-modal alignment. Tune on cross-modal F1 directly.

**Supporting remedies.**
- **Modality-balanced batches.** Ensure each batch has equal optical/MS/SAR and many cross-modal positives, so the loss can't trivially separate by modality; balanced sampling reduces the gap and improves cross-modal negatives' usefulness.
- **Explicit gap/alignment losses in training.** CSMAE's `L_MDE` (angular pull on paired sensors) and `L_MIM`, an L2 "pull cross-modal pairs to coincide" term, or an OT distribution-matching term (Section 9) shrink the gap *during* training rather than post-hoc.
- **Shared encoder / shared head (Sections 2–3).** One backbone (DOFA) or a shared projection head produces a single cone → minimal gap by construction; the strongest *architectural* prevention.
- **L2-normalize + cosine** is standard; combine with centering (cosine after centering ≈ correlation, which is gap-robust).

**Top-3 to recommend (see summary):** (i) per-modality mean-centering (GR-CLIP), (ii) per-modality whitening/PCA isotropization (+ top-PC removal), (iii) temperature tuning + modality-balanced batches.

**Reference.** Liang et al., *Mind the Gap*, NeurIPS 2022, arXiv:2203.02053 (code `Weixin-Liang/Modality-Gap`). *Closing the Modality Gap for Mixed Modality Search* (GR-CLIP), arXiv:2507.19054. *It's Not a Modality Gap: the Contrastive Gap*, arXiv:2405.18570. BERT-whitening: Su et al., 2021. *On Isotropy of Multimodal Embeddings*, Information 14(7):392.

---

## 14. RECOMMENDED PRACTICAL PIPELINE (best for PS-11 constraints)

**Constraints:** limited compute, must be fast at inference, leverage public pretrained weights, maximize cross-modal F1@5/@10 while keeping same-modal strong and retrieval time low.

### Architecture

```
            ┌─────────────────────────────────────────────────────────┐
 optical ─► │ FROZEN multimodal RS foundation backbone                 │ ─► f_opt
   MS    ─► │   (DOFA ViT-B preferred: wavelength-conditioned, single  │ ─► f_ms
  SAR    ─► │    backbone handles S1+S2+RGB; CROMA or DeCUR as alt.)   │ ─► f_sar
            └─────────────────────────────────────────────────────────┘
                                   │  pooled feature (GAP over tokens)
                                   ▼
              per-modality light projection head g_m (2-layer MLP),
              LAST linear layer SHARED across modalities  (Section 2)
                                   │  z_m ∈ R^256, then per-modality
                                   ▼  mean-center + whiten + L2-norm  (Section 13)
                              SHARED EMBEDDING SPACE  →  FAISS index
```

- **Backbone (frozen):** **DOFA ViT-B** is the top pick — one wavelength-conditioned backbone natively ingests optical RGB, S2 multispectral, and S1 SAR, so cross-sensor embeddings start in *one* space (smallest intrinsic gap), and it generalizes to sensors unseen in pretraining. **CROMA** is the strongest pure SAR↔optical alternative (its radar-only/optical-only embeddings are already contrastively aligned). **DeCUR** is ideal if you want to retrieve on an explicitly aligned *common* sub-embedding. Use GAP over patch tokens (CSMAE shows GAP > [CLS] for retrieval).
- **Heads (trainable, tiny):** per-modality 2-layer MLP `g_m: 768→512→256`, with the **final linear layer weight-shared** across modalities (late alignment, Section 2) and per-modality BatchNorm. Optionally add **LoRA (r=8)** on the backbone's last few blocks if a little more capacity is needed (merge at inference for zero overhead). Total trainable params ≈ a few M → trains in hours on one GPU.
- **Gap closing (Section 13):** fit per-modality **mean `μ_m`** and **whitening `W_m = Σ_m^{-1/2}`** on the training embeddings; at inference, `z = normalize(W_m (g_m(f) - μ_m))`. Optionally drop the top-1 principal direction (sensor-identity).

### Loss (exact formulation)

Train heads (and optional LoRA) with **symmetric InfoNCE + cross-modal triplet + (optional) latent-pull**, on modality-balanced batches of co-registered (or same-class) pairs. Let `z^a_i, z^b_i` be L2-normalized head outputs for the two modalities of pair `i`, temperature `τ` (learnable, init 0.07, clamp ≥ 0.05), margin `m`:

```
Symmetric InfoNCE (over all modality pairs present in the batch, e.g. {opt-sar, opt-ms}):
  L_NCE = ½ Σ_{(a,b)} [ CE_row(S/τ) + CE_col(S/τ) ],   S_ij = z^a_i · z^b_j

Cross-modal triplet (hard-but-not-hardest / semi-hard mining):
  L_tri = Σ_i max(0, m + d(z^a_i, z^b_i⁺) - d(z^a_i, z^b_{n}⁻)),   d = 1 - cosine
          (positive = same location/class other-modality; negative = different class, any modality)

Optional explicit gap-pull (CSMAE-style, helps cross-modal F1):
  L_pull = (1/N) Σ_i || z^a_i - z^b_i ||²

Total:
  L = L_NCE  +  λ_tri · L_tri  +  λ_pull · L_pull        (λ_tri ≈ 1.0, λ_pull ≈ 0.1–0.5)
```

Add an **intra-modal** triplet/InfoNCE term (same-modality positives by class) so same-modal F1 stays high. Use a memory bank / gradient cache if GPU limits batch size (InfoNCE wants many negatives).

### Why this wins for PS-11
- Frozen foundation backbone → **public weights, low compute, fast inference** (one forward pass + FAISS), satisfies the latency metric.
- InfoNCE gives the global shared geometry; cross-modal **triplet** mining sharpens exactly the same-class-different-modality ranking that cross-modal F1@k rewards; `L_pull` and **mean-centering/whitening** crush the modality gap that otherwise tanks cross-modal F1.
- Shared final head + DOFA single-backbone → minimal *architectural* gap; per-modality centering/whitening removes the *residual* gap post-hoc.
- Class labels (if available) feed the triplet/center loss; if only geo-pairs exist, InfoNCE + `L_pull` still work.

### Inference / serving
Embed all gallery images once, store in a **FAISS** index (`IndexFlatIP` for exact cosine on centered+whitened vectors, or IVF/HNSW for speed). At query: embed → center+whiten by query's modality → search → return top-5/top-10. For a mixed-modality gallery, store everything in the one shared space; center each item by its own modality mean (GR-CLIP recipe). Report F1@5/@10 per scenario and per-query latency.

---

## 15. FALLBACK — zero-shot, NO training

If training is infeasible (time/compute), use foundation embeddings **directly** with only a cheap statistical calibration:

1. Pick a frozen multimodal backbone (**DOFA** for optical+MS+SAR in one model, or **CROMA** for SAR+optical). Embed every image (GAP over tokens).
2. On a small reference set, compute per-modality **mean `μ_m`** and **whitening `Σ_m^{-1/2}`** (regularize `Σ` with shrinkage on small data); optionally remove the top principal component.
3. At query/gallery time: `z = normalize( Σ_m^{-1/2} (e - μ_m) )` → cosine similarity / FAISS.

This needs **zero gradient steps** — just mean-centering + whitening (Section 13 remedies 1–2), which alone delivered up to +26 NDCG@10 for GR-CLIP. It is the safest baseline to submit first and to ablate against the trained pipeline. Same-modal retrieval will already be strong (foundation features are good intra-modal); centering+whitening is what makes the *cross-modal* numbers usable without training.

---

## 16. Method comparison table

| # | Method | Encoders | Loss core | Data need | Train cost | Cross-modal strength | Gap behavior |
|---|--------|----------|-----------|-----------|-----------|---------------------|--------------|
| 1 | CLIP InfoNCE | per-modality + heads | symmetric InfoNCE | paired | med (big batch) | high | **induces** gap |
| 2 | Shared head (late) | per-modality, shared head | InfoNCE/CE | paired/labeled | low | med-high | reduces gap |
| 3 | Shared backbone + cond. (DOFA) | one ViT + hypernet/FiLM | MIM+distill (pretrain) | use weights | low (frozen) | high | minimal gap |
| 4 | Cross-recon (CROMA/CSMAE) | uni + fusion ViT | InfoNCE + MAE (+MDE/MIM) | paired, no labels | high pretrain | very high (CSMAE F1≈71) | shrinks gap |
| 5 | DeCUR common/unique | per-modality + projectors | Barlow common=I/unique=0 | paired, no labels | med | high (common part) | aligns common dims |
| 6 | KD optical→SAR | teacher(frozen)+student | feature distill | paired, no labels | **low** | high (SAR↔opt) | no gap (forced) |
| 7 | SAR→optical translation | generator + 1 encoder | GAN/diffusion + L1 | paired/unpaired | high (slow infer) | med, lossy | single-modality space |
| 8 | Deep CCA | two nets | maximize canonical corr | paired | med (large batch) | med-high | correlated space |
| 9 | Optimal transport | any two | Sinkhorn-Wasserstein | weak/unpaired | med-high | med (as regularizer) | **distribution-level** close |
| 10 | Prototypes/SwAV | per-modality + shared protos | swapped-pred / center | labeled/unpaired | low-med | high (class-based) | shared skeleton |
| 11 | Adapter/LoRA on FM | frozen FM + LoRA/head | InfoNCE+triplet | small paired/labeled | **lowest** | high | inherit + post-fix |
| 12 | Cross-modal triplet | per-modality + head | margin ranking + mining | labeled/paired | med | high (ranking) | needs centering |

---

## 17. Key references (consolidated)

- CLIP — Radford et al. 2021, arXiv:2103.00020. RemoteCLIP arXiv:2306.11029; SARCLIP arXiv:2510.22665; PR-CLIP MDPI RS 17(13):2117.
- **CROMA** — Fuller et al., NeurIPS 2023, arXiv:2311.00566 (weights public).
- **CSMAE** (purpose-built sensor-agnostic CBIR) — arXiv:2401.07782, IEEE TGRS, GitHub `jakhac/CSMAE`.
- **DOFA** — Xiong et al., arXiv:2403.15356 (HF `XShadow/DOFA`, GitHub `zhu-xlab/DOFA`).
- **DeCUR** — Wang et al., ECCV 2024, arXiv:2309.05300.
- **Galileo** (highly-multimodal RS FM, global+local contrastive + masked) — Tseng et al., ICML 2025, arXiv:2502.09356.
- Modality gap — Liang et al. *Mind the Gap*, NeurIPS 2022, arXiv:2203.02053; **GR-CLIP / mean-centering** arXiv:2507.19054; Contrastive Gap arXiv:2405.18570; Decipher the Gap arXiv:2510.03268.
- Knowledge distillation in RS — survey arXiv:2409.12111; ERVD arXiv:2412.18136.
- SAR↔optical translation — ISPRS review S1569843225006569; CycleGAN ICCV 2017; diffusion arXiv:2407.06095.
- Deep CCA — Andrew et al. ICML 2013; DGCCA Benton et al. 2017.
- Optimal transport DA — Courty et al. TPAMI 2017; OTAdapt arXiv:2205.10738.
- SwAV — Caron et al., NeurIPS 2020, arXiv:2006.09882; cross-modal center loss CVPR 2021.
- LoRA — Hu et al. ICLR 2022; Adapter Houlsby et al. ICML 2019; RS PEFT arXiv:2504.17397.
- Triplet/metric — FaceNet CVPR 2015; intramodal-consistency cross-modal triplet, Springer ML 2024.
- Datasets — SEN12MS (arXiv:2104.00704); BigEarthNet-MM; SSL4EO-S12; SOMA-1M arXiv:2602.05480.
