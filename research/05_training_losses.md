# 05 — Training Losses, Mining, Sampling & Augmentation for Cross-Modal Satellite Retrieval

**Problem:** BAH 2026 PS-11 — Cross-modal & same-modal satellite image retrieval (optical RGB ↔ SAR ↔ multispectral). Metric: **F1@5 / F1@10** for both same-modal and cross-modal, plus average retrieval time. Cross-modal weighted more heavily.

**Goal of this document:** define the best *training recipe* (losses, mining, sampling, augmentation, optimization) that produces an embedding space where images of the same land-cover / land-use / scene class are close **irrespective of sensor modality**, maximizing rank-based F1@K.

**Key framing for loss choice (the single most important decision):** F1@K is **rank-based** and (for this benchmark) **class/semantic-relevance based** (relevance = same land-cover label or geographic correspondence). That means:
- Losses that produce **tight, well-separated semantic clusters** (angular-margin/proxy/classification losses) tend to win F1@K because relevance is class-defined.
- Losses that **directly optimize ranking/AP** (Smooth-AP, FastAP) are theoretically best-aligned with the metric.
- **Cross-modal alignment** (CLIP-style symmetric InfoNCE) is what makes optical↔SAR↔MS retrieval work at all — without it, modalities form separate cones and cross-modal F1 collapses.
- The winning recipe is therefore a **combination**: a cross-modal alignment term + a class-discriminative term (+ optionally a ranking term).

---

## 0. Background facts that drive every decision

### 0.1 Always L2-normalize embeddings
- Normalize the final descriptor to the unit hypersphere (‖z‖₂ = 1). With L2-normalized vectors, **cosine similarity ≡ (monotone in) Euclidean distance**, so the ranking is identical whether you use inner product or L2. This is what lets a single FAISS `IndexFlatIP` serve all queries.
- L2-normalization gives a "general improvement of recall over unnormalized variants" in instance retrieval and is mandatory for all angular/cosine-margin and InfoNCE losses (their math assumes points on the sphere).
- *Source: image-retrieval tricks benchmark (arXiv:1907.11854); CNN-image-retrieval (arXiv:1711.02512).*

### 0.2 Embedding dimension
- **128–512 are all viable.** Proxy-Anchor reports "stable performance ≥ 128 dimensions." Retrieval-model studies find native backbone dims (DINOv2 = 768, ViT-B) work well *without* reduction.
- **Recommendation for PS-11:** train/store **256-D** (sweet spot: enough capacity for 3 modalities + many classes, small enough for fast FAISS and low retrieval time). Use **512-D** if you have a large gallery and ample compute; **128-D** only if retrieval-time is the binding constraint. Larger dim ⇒ slightly higher F1 but slower search.

### 0.3 PCA-whitening as post-processing (powerful, cheap, training-free)
- Whitening CNN/foundation-model descriptors **down-weights co-occurring (bursty) dimensions** that dominate cosine similarity, and improves recall. Classic pipeline: extract features → **PCA-whiten (optionally reduce dim)** → **L2-normalize** → search.
- Euclidean distance between PCA-whitened features = **Mahalanobis distance** between raw descriptors (decorrelates the space).
- For cross-modal: fit the whitening matrix on a **balanced mix of all modalities** (or per-modality then a shared rotation) so no modality dominates the principal axes.
- *Source: "Negative Evidences and Co-occurrences… benefit of PCA and Whitening"; arXiv:1907.11854; arXiv:1711.02512.*

### 0.4 Temperature controls hard-negative emphasis (uniformity–tolerance dilemma)
- In InfoNCE/NT-Xent, **temperature τ controls the penalty strength on hard negatives.** **Small τ (0.05–0.07)** ⇒ concentrate gradient on the closest hard negatives, more *uniform* embedding, less tolerant of semantically-similar samples. **Large τ (0.2–0.5)** ⇒ penalty spread evenly, more tolerant of near-duplicates / same-class neighbors.
- This is the **"uniformity–tolerance dilemma"**: small τ separates classes well but can push apart legitimately-similar same-class items (bad when intra-class spread is large, as in seasonal/condition variation in RS).
- *Source: "Understanding the Behaviour of Contrastive Loss" (CVPR'21, arXiv:2012.09740).*

---

## 1. InfoNCE / NT-Xent (self-supervised contrastive)

**Formulation.** For a batch of 2N augmented views, with positive pair (i, j) and cosine similarity sim(·,·):

```
ℓ_{i,j} = − log [ exp(sim(z_i, z_j)/τ) / Σ_{k=1, k≠i}^{2N} exp(sim(z_i, z_k)/τ) ]
L_NTXent = (1/2N) Σ over all positive pairs ℓ
```
`sim(u,v) = uᵀv / (‖u‖‖v‖)`; τ = temperature.

**When it helps retrieval.** Pretraining / self-supervised stage when you have **no labels** but have natural positive pairs: (a) two augmentations of the same image (same-modal invariance), or (b) **two modalities of the same geo-location** (cross-modal — see §2). Builds a general, smooth embedding manifold before metric-supervised fine-tuning.

**How to mine pairs.** Positives = augmentation pairs and/or geo-paired SAR↔optical patches (e.g. BigEarthNet-MM, SEN1-2). Negatives = all other in-batch items. **Large batch (or memory bank, §10) is essential** because InfoNCE's only negatives are the other batch items.

**Hyperparameters.** τ default **0.07** (pytorch-metric-learning `NTXentLoss`); SimCLR used 0.1–0.5 depending on batch. For RS with high intra-class variation, **start τ ≈ 0.1–0.2** (more tolerant). Batch ≥ 256 (or use a queue).

**Pitfalls.** (1) Small batch ⇒ too few negatives ⇒ weak signal (fix with MoCo queue, §10). (2) Too-small τ over-separates same-class seasonal variants. (3) "False negatives": two different geo-locations of the **same land-cover class** are treated as negatives — this fights the F1@K objective. Mitigate by switching to **SupCon (§3)** once labels exist, or de-bias with class-aware masking.

---

## 2. Symmetric cross-modal InfoNCE (CLIP-style) — the backbone of cross-modal retrieval

**Formulation.** Two encoders f_A, f_B (e.g. optical, SAR), each L2-normalized, learnable temperature τ (stored as log-scale, clamped). For a batch of N **paired** items, logits `L = (Z_A Z_Bᵀ)/τ` (N×N), labels = identity (diagonal = positives):

```
L_clip = ½ [ CE(L, diag) + CE(Lᵀ, diag) ]
       = ½ [ (1/N)Σ_i −log softmax_row_i(L)[i]  +  (1/N)Σ_j −log softmax_col_j(L)[j] ]
```
The **symmetry** (image→text AND text→image, here modality-A→B AND B→A) is what forces a *shared* space rather than a one-directional projection.

**Extending to 3 modalities (optical, SAR, MS).** Two proven patterns:
1. **Anchor/bridge modality (CLOSP pattern).** Pick the richest modality (optical, or a *text* class-description) as the anchor; align SAR↔anchor and MS↔anchor with separate symmetric InfoNCE terms. **You do NOT need triple-aligned tuples** — only pairs to the anchor. CLOSP aligned Sentinel-1 (SAR) and Sentinel-2 (optical) *through text*, sidestepping the lack of temporally co-registered SAR/optical. Empirically this **improved SAR retrieval** (SAR is hard to interpret alone).
2. **All-pairs symmetric InfoNCE.** If you have co-registered triples, sum InfoNCE over all modality pairs {O-S, O-M, S-M}.

**Why it helps retrieval.** It is the *only* family here that explicitly closes the cross-modal gap; cross-modal F1@K is near-zero without it. SARCLIP reports 84.2% avg recall in cross-modal SAR retrieval; CLOSP/Mind-the-Modality-Gap show large cross-modal gains on BigEarthNet.

**Mining.** Positives = the **geo-paired** cross-modal patches (same lat/lon, same tile). In-batch negatives = all non-matching pairs. **Modality-balanced batch** (equal A and B) is important.

**Hyperparameters.** **Learnable τ initialized to 0.07** (= log(1/0.07)), clamp logit-scale ≤ ln(100)=4.6 as in CLIP to avoid blow-up. Batch 256–1024 (or queue). Cosine-annealed LR, warmup.

**Pitfalls.** (1) **Modality gap / cone effect**: encoders start as separate cones; low τ keeps them separated. Mitigate with **shared top layers / shared projection head**, modality-balanced sampling, and not making τ too small. (2) **Geo-pairing ≠ semantic identity**: same location can contain mixed land cover; multi-label relevance helps (BigEarthNet-MM is multi-label). (3) Only pulls *exact* pairs together — does not cluster all same-class items ⇒ **combine with SupCon/ArcFace** to also tighten same-class same-modal structure (which drives same-modal F1).

---

## 3. Supervised Contrastive (SupCon) — use land-cover labels

**Formulation.** For anchor i in a multiview batch, P(i) = all other samples with the same label, A(i) = all others:

```
L_SupCon = Σ_{i∈I} (−1/|P(i)|) Σ_{p∈P(i)} log [ exp(z_i·z_p/τ) / Σ_{a∈A(i)} exp(z_i·z_a/τ) ]
```

**When it helps retrieval.** This is **the natural fit for F1@K when relevance = class label.** Unlike InfoNCE (one positive), SupCon pulls **all same-class samples** together and pushes all other classes apart ⇒ tight class clusters ⇒ high precision in the top-K neighborhood. Directly raises **same-modal F1@K**.

**Cross-modal SupCon (key trick for PS-11).** Build the multiview batch so that **positives include same-class samples *from other modalities*** (anchor optical, positive = SAR with same land-cover label). This makes SupCon simultaneously a class-clustering *and* cross-modal-alignment loss — arguably the single best loss for this benchmark's relevance definition.

**Mining/sampling.** Requires a **P×K class-balanced sampler** (§ samplers) so each batch has several samples per class (and ideally each class represented in ≥2 modalities). Larger |P(i)| ⇒ better.

**Hyperparameters.** τ default **0.1** (pytorch-metric-learning `SupConLoss`); paper used **0.07–0.1**. Batch as large as possible.

**Pitfalls.** (1) **Multi-label RS data** (BigEarthNet has 19 multi-labels): "same class" is ambiguous. Use **multi-label SupCon** (positive weight ∝ label overlap / Jaccard), or define single dominant class. (2) Needs labels (not for the unlabeled pretraining stage). (3) Class imbalance ⇒ rare classes under-pulled; use class-balanced sampling.

---

## 4. Triplet loss + batch-hard / semi-hard mining (incl. cross-modal triplets)

**Formulation.** For triplet (anchor a, positive p, negative n), margin α:
```
L_triplet = (1/|T|) Σ [ ‖f(a) − f(p)‖² − ‖f(a) − f(n)‖² + α ]_+
```
FaceNet margin **α = 0.2** (for L2-normalized 128-D). pytorch-metric-learning `TripletMarginLoss` default margin **0.05**.

**Mining strategies (decisive for triplet performance).**
- **Batch-hard (BH):** within each P×K batch, for each anchor pick the **hardest positive** (farthest same-class) and **hardest negative** (closest diff-class). Strong, standard for re-ID/retrieval.
- **Semi-hard:** pick negatives that are farther than the positive but still inside the margin: `‖a−p‖² < ‖a−n‖² < ‖a−p‖² + α`. FaceNet found pure-hardest negatives cause **collapse to degenerate minima**; semi-hard is more stable early.
- Practical: **semi-hard for warmup → batch-hard later**, or use **batch-hard with a margin** and modality-balanced batches.

**Cross-modal triplets (core for PS-11).** Anchor = optical patch of class c; positive = **SAR (or MS) patch of class c**; negative = any-modality patch of class c'≠c. Also include **cross-modal hardest-negative mining**: for an optical anchor, the hardest negative is the *closest SAR/MS patch of a different class* — explicitly trains the boundary in the shared space that cross-modal F1 depends on.

**When it helps retrieval.** Triplet's **ranking formulation is directly suited to retrieval** (preserves relative similarities). Good complement when added on top of an alignment/classification loss; sharpens local rank order in the top-K.

**Pitfalls.** (1) **Sampling-sensitive & slow** (O(K³) candidate triplets); most triplets quickly become non-informative. (2) Hardest-negative collapse. (3) Needs P×K sampler. (4) Margin/embedding-scale coupling — pick α relative to whether embeddings are normalized.

---

## 5. ArcFace / CosFace / Sub-center ArcFace / AdaCos (angular-margin classification → strong retrieval)

These treat training as **classification over land-cover classes**, but the *normalized class weights become class prototypes* and the *features become retrieval embeddings*. They are the **strongest small-batch option** and produce highly discriminative, compact clusters → excellent F1@K. They also need **no pair/triplet mining**.

**Common setup.** L2-normalize both feature z and class weights W_j; logit = `s·cos(θ_j)` where `cos(θ_j)=W_jᵀz`. Scale s "re-inflates" the bounded cosine so softmax can saturate.

**ArcFace (additive angular margin).**
```
L = −log [ exp(s·cos(θ_{y}+m)) / ( exp(s·cos(θ_{y}+m)) + Σ_{j≠y} exp(s·cos θ_j) ) ]
```
Margin **m** added to the *angle* (geodesic) of the true class ⇒ constant linear angular margin. Recommended **s = 64, m = 0.5** (face); pytorch-metric-learning default `margin=28.6°(≈0.5 rad), scale=64`.

**CosFace (additive cosine margin).** `L = −log[ exp(s(cosθ_y − m)) / (… ) ]`. Margin subtracted from cosine; simpler, monotone. Defaults **s = 64, m = 0.35**. Bound: `0 ≤ m ≤ 1 − max(W_iᵀW_j)`.

**Combined margin (general form).** `s·(cos(m₁θ + m₂) − m₃)` unifies SphereFace(m₁), ArcFace(m₂), CosFace(m₃).

**Sub-center ArcFace.** Each class gets **K sub-centers**; a sample only needs to be near **its nearest sub-center**: `θ̃ = arccos(max_k W_{j,k}ᵀz)`. Robust to **intra-class variation and label noise** — *highly relevant for RS* where a "Forest" class spans many appearances, seasons, sensors. Default **K = 3 sub-centers** (s=64, m=0.5). This is a top pick for noisy/heterogeneous satellite classes.

**AdaCos (adaptive scale, no s/m tuning).** Drops manual scale; sets s dynamically from batch statistics:
```
s^(0) = √2 · log(N−1)                              (N = #classes)
s^(t) = log(B_avg^(t)) / cos(min(π/4, θ_med^(t))),  t>0
```
θ_med = median of true-class angles in batch (large ⇒ far from optimum ⇒ softer supervision; small ⇒ stricter). Removes the brittle s/m search. Excellent default when you don't want to tune.

**When they help retrieval.** Best when **relevance = class** and **batch is small** (they "work well even with smaller batch sizes," unlike pair losses). Produce maximally-separated angular clusters ⇒ strong F1@K, especially **same-modal**. For **cross-modal**, share the classifier head across all modalities (one set of class prototypes, fed by all encoders) → forces every modality of class c toward the same prototype = implicit cross-modal alignment.

**Pitfalls.** (1) Need a label set; the classifier head's LR must be **much higher** than the backbone LR (e.g. classifier LR ≈ 1.0 vs backbone 1e-6) — getting this wrong is "disastrous." (2) Large #classes ⇒ big weight matrix. (3) Plain ArcFace sensitive to label noise → prefer **Sub-center ArcFace** or **AdaCos** for messy RS labels. (4) m too large early ⇒ no convergence; warm up margin from 0.

---

## 6. Proxy-Anchor & Proxy-NCA++ (fast proxy-based metric learning)

**Idea.** Replace expensive sample-sample pairs with **learnable proxies** (one+ per class). Combines triplet-style data-to-data hardness with proxy speed/stability.

**Proxy-NCA / Proxy-NCA++.** Assign one proxy per class; positive = same-class proxy, negatives = other proxies. ++ adds **temperature scaling**, proxy-assignment improvements. Default `softmax_scale=1`.
```
L_ProxyNCA = − log [ exp(−d(z, p⁺)) / Σ_{p∈neg} exp(−d(z, p)) ]
```

**Proxy-Anchor (recommended proxy loss).** Each **proxy is the anchor**; it interacts with **all** batch samples, with LogSumExp weighting so **harder positives are pulled harder, harder negatives pushed harder** (data-to-data hardness, unlike Proxy-NCA's constant positive gradient):
```
L = (1/|P⁺|) Σ_{p∈P⁺} log(1 + Σ_{z∈X_p⁺} e^{−α(s(z,p)−δ)})
  + (1/|P|)  Σ_{p∈P}  log(1 + Σ_{z∈X_p⁻} e^{ α(s(z,p)+δ)})
```
P⁺ = proxies with positives in batch; s = cosine. **Defaults α = 32, δ = 0.1.** Converges fast, stable, low complexity O(MC).

**When it helps retrieval.** Excellent **speed/accuracy trade-off** and robust on smaller batches; strong F1@K with little tuning. For cross-modal, use **one shared proxy per land-cover class** fed by all modalities → cross-modal clustering for free.

**Pitfalls.** (1) Proxies assume class structure (need labels). (2) Stable ≥128-D embeddings, batch ≥ ~300 best on large datasets. (3) Single proxy per class can't model multimodal class appearance — consider multiple proxies (akin to sub-centers) for visually diverse RS classes.

---

## 7. Multi-Similarity (MS) loss with general pair weighting

**Formulation.** Two-stage **mine then weight**, using self-(P), positive-(S) and negative-(N) similarities. After hard-pair mining (keep positives harder than the hardest negative −ε, negatives easier than hardest positive +ε):
```
L_MS = (1/B) Σ_i [ (1/α) log(1 + Σ_{k∈P_i} e^{−α(S_{ik}−λ)})
                 + (1/β) log(1 + Σ_{k∈N_i} e^{ β(S_{ik}−λ)}) ]
```
**Defaults: α = 2, β = 50, λ (base) = 0.5–1.0, ε = 0.1.** (pytorch-metric-learning `MultiSimilarityLoss(alpha=2, beta=50, base=0.5)`.)

**When it helps retrieval.** A top **pair-based** loss for **large-batch** retrieval; weights informative pairs by all three similarity types ⇒ robust top-K ordering. Retrieval-model study recommends MS (with online miner) for **batch ≥ 256**.

**Mining.** Built-in MS-miner; pair with **P×K sampler**. Cross-modal: include cross-modal positive/negative pairs in the batch.

**Pitfalls.** (1) Wants large, well-sampled batches; weak with tiny batches (use ArcFace/CosFace there). (2) Several coupled hyperparameters; α/β/λ interact. (3) β=50 makes it aggressive on hard negatives — can over-fit noisy labels.

---

## 8. Circle loss (unified pair-similarity optimization)

**Formulation.** Re-weights each similarity by how far it is from its optimum (self-paced):
```
L = log[ 1 + Σ_{j=1}^{L} exp(γ·α_n^j (s_n^j − Δ_n)) · Σ_{i=1}^{K} exp(−γ·α_p^i (s_p^i − Δ_p)) ]
α_p^i = [O_p − s_p^i]_+ ,   α_n^j = [s_n^j − O_n]_+
O_p = 1+m, O_n = −m, Δ_p = 1−m, Δ_n = m   ⇒ circular boundary (s_n)² + (s_p−1)² = 2m²
```
**Defaults:** pair-wise labels **γ = 80, m = 0.4** (pytorch-metric-learning `CircleLoss(m=0.4, gamma=80)`); class-level labels **γ = 256, m = 0.25**.

**When it helps retrieval.** Unlike triplet/softmax (equal penalty on all scores), Circle **emphasizes the least-optimized scores** ⇒ more flexible optimization, definite convergence target. One formula handles **both class-level and pair-wise labels**, convenient for mixed supervision. Competitive with MS/ArcFace on re-ID/retrieval.

**Pitfalls.** γ large ⇒ sharp gradients, can be unstable; tune γ down if diverging. Needs in-batch positives/negatives (P×K sampler).

---

## 9. Ranking/AP-direct losses: NormSoftmax, Smooth-AP, FastAP (best metric-alignment with F1@K)

Because **F1@K is rank-based**, losses that *directly* optimize AP/ranking are the most metric-aligned and often give the best top-K behavior.

**NormSoftmax (normalized-softmax / "classification is a strong baseline").**
```
L = −log [ exp(W_{y}ᵀz / τ) / Σ_j exp(W_jᵀz / τ) ],  ‖z‖=‖W_j‖=1
```
Just a cosine softmax with temperature (no margin). Default **τ = 0.05** (`NormalizedSoftmaxLoss`). With high-dim embeddings it **outperforms triplet baselines** on retrieval and is dead-simple/stable — a great fast baseline classification loss. (ArcFace/CosFace = NormSoftmax + margin.)

**Smooth-AP (recommended ranking loss).** Replaces the non-differentiable indicator in AP with a sigmoid `G(x;τ)=1/(1+e^{−x/τ})` over pairwise score differences `D_ij = s_i − s_j`:
```
AP_q ≈ smoothed rank ratios using G(D_ij; τ);   L = (1/m) Σ_k (1 − AP_k)
```
**τ ≈ 0.01**; needs **|P| ≈ 4 positives per query** and **batch 256–384** (more in-batch positives & hard negatives). Closer to true AP and **simpler than FastAP/Blackbox-AP**; directly optimizes ranking (not a distance surrogate) ⇒ excellent for F1@K. *Source: Smooth-AP, ECCV'20 (arXiv:2007.12163).*

**FastAP.** Approximates AP via **distance quantization + soft histogram binning** (piecewise-linear soft assignment, cheap derivative). Default `num_bins=10`. Lower-variance, SGD-friendly; uses a special minibatch sampler. Slightly looser AP approximation than Smooth-AP. *Source: Cakir et al., "Deep Metric Learning to Rank," CVPR'19.*

**RS-specific note.** **OSAP-Loss** (TGRS) adapts AP optimization specifically for **remote-sensing image retrieval** ("involving samples after positive ones"), confirming AP-direct losses are effective in this domain.

**When they help.** When you can afford **medium-large batches with several positives per class** and want to optimize *exactly the ranking your F1@K measures*. Best as a **fine-tuning / final-stage loss** on top of a good initialization.

**Pitfalls.** (1) Need multiple positives per query in-batch (P×K sampler, K≥4). (2) τ/bin granularity matters. (3) Noisier gradients than classification losses early ⇒ pretrain with ArcFace/SupCon first, then add Smooth-AP.

---

## 10. Memory bank / MoCo-style queue (many negatives on small batches)

**Idea.** Decouple #negatives from batch size: maintain a **FIFO queue** of past encoded keys (e.g. 16k–65k) and a **momentum (EMA) key encoder** `θ_k ← m·θ_k + (1−m)·θ_q` (momentum m ≈ 0.999). Loss = InfoNCE with the queue as negatives:
```
L_MoCo = −log [ exp(q·k₊/τ) / ( exp(q·k₊/τ) + Σ_{k∈queue} exp(q·k/τ) ) ]
```

**When it helps retrieval.** When **GPU memory limits batch size** but InfoNCE/CLIP-style training needs many negatives. Hugely relevant for PS-11 if training on modest hardware: get SimCLR-scale negatives at small batch. **Cross-modal MoCo**: keep **separate queues per modality** (an optical query attends to a SAR/MS key queue) → many cross-modal negatives → stronger cross-modal boundaries.

**Hyperparameters.** Queue 16,384–65,536; momentum 0.999; τ 0.07–0.2. Enqueue current keys, dequeue oldest.

**Pitfalls.** (1) Queue staleness if encoder changes fast (momentum fixes this). (2) Extra memory for the queue + EMA encoder. (3) Class-level false negatives accumulate in the queue (same-class items as negatives) — consider class-aware filtering when labels exist.

---

## 11. Cross-cutting strategy choices

### 11.1 Samplers — **P×K (class-balanced) is mandatory** for pair/triplet/SupCon/MS/Circle/AP losses
- **P×K sampler:** pick **P classes**, then **K samples/class** ⇒ batch = P·K, guarantees positives & negatives exist in-batch. K=2–4 typical.
- **Modality-balanced + class-balanced (PS-11 specific):** sample **P classes × K samples, with the K split across modalities** (e.g. K=4 = 2 optical + 1 SAR + 1 MS of the same class/area) so every batch contains **same-class cross-modal positives** and **cross-modal negatives**. This single change makes SupCon/triplet/MS *cross-modal-aware* for free.
- For **classification losses** (ArcFace/NormSoftmax) use **1 sample/class** (just balance classes); they don't need in-batch positives.
- **Hard-class sampling:** instead of random P classes, sample the **most similar/confusable classes** together to mine harder negatives (improves discrimination of look-alike land covers, e.g. different crop types).
- *Sources: PyTorch-Metric-Learning samplers; Graph-Sampling DML (arXiv:2104.01546).*

### 11.2 Cross-modal hardest-negative mining
- For an **optical anchor**, mine the **closest different-class SAR/MS sample** as the negative (and symmetrically). This explicitly hardens the **inter-modal class boundary** that cross-modal F1 depends on. Use semi-hard early to avoid collapse.

### 11.3 Temperature / scale schedules
- **Learnable τ** (CLIP) is robust; **clamp** logit-scale ≤ ln(100). If fixed: anneal τ **from larger (0.2, tolerant) → smaller (0.07, sharper)** over training (warm tolerant, then tighten clusters).
- **Margin warmup** for ArcFace/CosFace/Circle: ramp m from 0 → target over the first few epochs to avoid early divergence.
- **AdaCos** removes scale tuning entirely (adaptive) — good default if unsure.

### 11.4 Combining losses (the winning pattern)
Multi-task heads sharing one backbone+embedding consistently beat single losses for retrieval. Proven combos:
- **Cross-modal alignment + class discrimination:** `λ₁·SymInfoNCE(cross-modal) + λ₂·SubCenterArcFace(shared head)`. Alignment closes the modality gap; ArcFace tightens semantic clusters (drives same-modal F1). **Primary recommendation.**
- **+ ranking polish:** add `λ₃·Smooth-AP` (or batch-hard triplet) in the **final stage** to directly optimize top-K order.
- Classic precedents: contrastive+softmax (face id-verification), triplet+softmax (face recognition) — auxiliary classification + metric loss is a standard, reliable recipe.
- **Weighting:** start equal (λ=1), then up-weight the cross-modal term (e.g. λ_align = 1.0–2.0) since **cross-modal F1 is weighted higher** in PS-11 scoring. Normalize each loss to similar scale before summing.

---

## 12. AUGMENTATION — modality-specific (critical for SAR/optical/MS robustness)

**Golden rule for paired training:** apply **the SAME geometric transform to all co-registered modalities of a pair** (shared random seed for flip/rot/crop) so the **pixel correspondence (co-registration) is preserved**. Apply **modality-specific photometric/radiometric** augmentations **independently** per modality. Sample random deformations on the fly per pair, then crop at the same position in both images.

### 12.1 SAR (Sentinel-1 VV/VH) — the most distinctive
- **Speckle is multiplicative, Gamma-distributed** with mean 1, variance 1/L (L = #looks). Augment by **multiplying** the image by a Gamma(L, 1/L) speckle field — **vary L ∈ {1,2,4,8}** to span clean→noisy. This is the single most important SAR augmentation for robustness.
- **Despeckle augmentation (both directions):** randomly apply **Lee / refined-Lee / Gamma-MAP** despeckling (or a learned despeckler) so the model sees both speckled and smoothed versions → speckle-invariant features.
- **Log / dB handling:** SAR intensity should be fed in **dB (10·log₁₀σ°)**; speckle becomes additive in log-domain (homomorphic). Augment with small **radiometric jitter** (additive dB offset / gain) to mimic calibration differences. Clip/standardize per channel (e.g. VV∈(−23,0) dB, VH∈(−28,−5) dB) then normalize to ~N(0,1) or [0,1].
- **Polarization (VV/VH) handling:** **random channel dropout** (train on VV-only, VH-only, and VV+VH) so the model is robust to which polarizations are available; optionally add the **VV/VH ratio** as a derived channel. Random small per-channel gain.
- **Geometric:** flips, rot90, random-resized-crop, **small rotations** (shared with the paired optical/MS crop).

### 12.2 Optical RGB
- **Color jitter** (brightness/contrast/saturation/hue), **random grayscale** (forces structure over color — but in cross-modal training grayscale also nudges optical toward SAR-like cues), **Gaussian blur**, **Cutout / Random-Erasing** (occlusion robustness, cloud-like masks), **RandAugment** (parameter-free policy; the "All-You-Need" retrieval study uses RandAugment).
- **RS-specific:** simulate **haze / thin cloud** (alpha-blend with white), **seasonal color shift**, sun-angle/illumination jitter.
- **Geometric:** flips, rot90 (RS imagery is rotation-agnostic — unlike natural images), random-resized-crop, small rotations.

### 12.3 Multispectral (Sentinel-2, 10–13 bands)
- **Band dropout / spectral tube masking:** randomly zero/mask a subset of bands (e.g. tube-mask along spectral dim, ratio up to ~75% for MAE-style, milder ~10–30% for contrastive) → robustness to missing/different band sets and band-order. Critical because gallery MS images may have different band availability.
- **Spectral jitter:** small independent per-band gain/offset (radiometric variation across acquisitions/sensors).
- **Band-order robustness:** occasionally permute / randomly select band subsets so the encoder isn't order-locked (supports cross-sensor MS).
- **Index augmentation:** append/jitter **NDVI = (NIR−Red)/(NIR+Red)** and similar indices (NDWI, etc.) as extra channels, or augment them — injects domain priors that align MS with optical semantics.
- **Do NOT blindly use color-suppressing aug (grayscale)** on MS — it destroys spectral signal that is the whole point of MS. Adapt ColorJitter to operate over all channels.
- **Geometric:** same as above, shared with paired modalities.

### 12.4 Geometric (all modalities) — preserve co-registration
- **Horizontal/vertical flips, rot90 (×4), random-resized-crop, small (±10–15°) rotations.** RS scenes have **no canonical orientation**, so full flip+rot90 is safe and very effective (8× effective data).
- **For paired training, use the identical geometric transform across the modalities of a pair** (shared seed) so positives stay pixel-aligned. Independent geometry would break the cross-modal correspondence signal.

### 12.5 Cross-modal mixup & modality dropout (force reliance on the shared space)
- **Modality dropout:** during training, randomly drop a modality from a multimodal sample/batch so the model must produce consistent embeddings from any subset → **modality-agnostic representation**, prevents over-reliance on the "easiest" modality (usually optical). Directly improves cross-modal robustness and handles missing modalities at query time.
- **Cross-modal mixup:** convex combinations across modalities (or mixup within the shared embedding space) to populate the inter-modal manifold and smooth the shared space (Multimodal-Mixup-Contrastive shows gains).

---

## 13. RECOMMENDED TRAINING RECIPE (ranked, concrete)

### Datasets to train/eval on
- **Primary: BigEarthNet-MM** — 590,326 **co-registered Sentinel-1 (SAR) + Sentinel-2 (MS/optical)** patch pairs with **19-class multi-label CORINE land cover**. Ideal for paired cross-modal training *and* class-supervised losses, and matches PS-11's relevance definition (semantic class + geo-correspondence). Supplement with **SEN1-2** (282k SAR-optical pairs) for more cross-modal pairs.

### Backbone & head
- **Per-modality encoders** (or one shared ViT with modality-specific patch-embed/stem): optical RGB, SAR (VV/VH), MS (10–13 band). Initialize from a **remote-sensing foundation model** where possible (Prithvi-EO/DINOv2-RS/SatCLIP) for a strong start.
- **Shared projection head → 256-D → L2-normalize.** One **shared classifier/proxy head** (class prototypes) consumed by all modalities (this is itself a cross-modal aligner).

### Primary loss combination (Stage B) — **ranked #1**
```
L = 1.5 · SymInfoNCE_crossmodal(learnable τ≈0.07)     # closes modality gap (weighted ↑: cross-modal scored higher)
  + 1.0 · SubCenterArcFace(s=64, m=0.5, K=3 subcenters, shared head)   # tight, noise-robust semantic clusters
  + 0.5 · BatchHardTriplet_crossmodal(margin≈0.1, semi-hard→hard)      # sharpen top-K rank order
```
- **Why this trio:** SymInfoNCE gives cross-modal alignment (without it cross-modal F1≈0); Sub-center ArcFace gives the **best small-batch, label-noise-robust class clustering** that F1@K rewards and adds cross-modal pull via the shared prototypes; batch-hard cross-modal triplet directly hardens the inter-modal boundary and polishes ranking. Sub-center handles RS classes' huge intra-class/seasonal/sensor variation.
- **Alternatives / swaps:**
  - Replace ArcFace term with **SupCon (cross-modal positives)** if you prefer pure contrastive and have large batches → also excellent, simpler conceptually.
  - **Large-batch (≥256) variant:** swap the triplet term for **Multi-Similarity (α=2,β=50,base=0.5)** or, for max metric-alignment, **Smooth-AP (τ=0.01, K≥4 positives)** in a final fine-tuning phase.
  - **Small-batch / limited GPU:** keep ArcFace as the main loss and add a **MoCo cross-modal queue (65k, momentum 0.999)** to supply InfoNCE negatives.

### Two-stage schedule
- **Stage A — alignment pretrain (frozen-ish backbone, train head + top blocks):** SymInfoNCE (cross-modal) + SupCon, large effective batch (use MoCo queue if needed), **τ warm 0.2→0.07**, **head LR ~1e-3**, backbone frozen or LR 1e-5. ~10–15 epochs. Builds the shared space cheaply.
- **Stage B — full fine-tune with the primary trio:** unfreeze all; **margin warmup** (ArcFace m: 0→0.5 over 3 epochs); add cross-modal batch-hard triplet. ~20–40 epochs (early-stop on val cross-modal F1@10, patience 3).
- **Stage C (optional polish):** add **Smooth-AP** (or FastAP) for a few epochs to directly optimize ranking.

### Sampler
- **Modality-balanced P×K sampler:** **P = 16–32 classes**, **K = 4** per class **split across modalities** (e.g. 2 optical + 1 SAR + 1 MS of the same area/class) ⇒ batch 64–128 (scale up with hardware). Guarantees in-batch cross-modal positives + cross-modal hard negatives. Optionally **hard-class sampling** (group confusable land covers). For the ArcFace term, balanced classes suffice.

### Optimizer / schedule
- **AdamW**, weight-decay 0.05 (ViT) / 1e-4 (CNN).
- **Differential LR:** **backbone 1e-5–3e-5** (full finetune) or **1e-6** if backbone strong; **head/classifier LR 1e-3–1.0** (ArcFace classifier likes a *much* higher LR — separate it!). Frozen-backbone+trained-head stage: head LR 1e-3.
- **Cosine-annealing** with **linear warmup** (3–5 epochs / ~5–10% of steps).
- **Batch:** as large as memory allows (≥256 effective preferred for pair/AP losses; MoCo queue if not). **Mixed precision (AMP).** **EMA of weights** for eval.
- **Epochs:** A ≈ 10–15, B ≈ 20–40, C ≈ 5; early-stop on **val cross-modal F1@10**.
- **Embedding dim 256**, L2-normalized.

### Post-processing & indexing (boosts F1@K, near-free)
- **PCA-whiten** (fit on modality-balanced sample) → **L2-normalize** → store.
- **FAISS** `IndexFlatIP` (cosine) for accuracy, or **IVF-PQ / HNSW** for speed at large gallery (meets "low retrieval time" criterion). Optionally **binary-hash** embeddings (sign function, 32× compression) — IBM/Prithvi showed binary codes give **near-identical mAP** with far faster search, directly helping the retrieval-time score.
- **Query-time:** L2-normalize the query embedding identically; same whitening matrix.

### Augmentation policy (per modality), summarized
| Modality | Photometric/radiometric (independent) | Geometric (shared across paired modalities) |
|---|---|---|
| **SAR** | multiplicative **Gamma speckle (L∈{1,2,4,8})**, random **Lee/refined-Lee despeckle**, **dB** input + radiometric (dB) jitter, **VV/VH dropout** + ratio channel | flips, rot90, RRC, ±10–15° rot |
| **Optical RGB** | color jitter, random grayscale, blur, **Cutout/Random-Erasing**, **RandAugment**, haze/cloud sim | flips, rot90, RRC, ±10–15° rot |
| **Multispectral** | **band dropout/spectral tube-mask (10–30%)**, **spectral jitter**, band-order/subset randomization, **NDVI/index** aug | flips, rot90, RRC, ±10–15° rot |
| **All (cross-modal)** | **modality dropout**, **cross-modal mixup** | identical seed per pair to preserve co-registration |

---

## 14. ZERO-TRAINING FALLBACK (no GPU-training; frozen foundation embeddings)

If training is infeasible or as a strong day-1 baseline:
1. **Pick a remote-sensing / general foundation model** per modality and extract frozen embeddings:
   - **Multispectral/optical:** **Prithvi-EO** (handles 6+ bands; reported **97.62% mAP@20** on BigEarthNet-43, training-free) or **DINOv2** (strong zero-shot retrieval) or **SatCLIP**. Use **CLS token** (ViT) or **mean-pooled patch tokens**; native dim (768 for Prithvi/DINOv2-base).
   - **SAR:** a SAR-pretrained encoder (SARMAE/SARCLIP/SSL4EO-S1); if unavailable, feed dB-scaled VV/VH through the same backbone.
2. **For cross-modal:** if using a CLIP-style RS model with a **shared text space** (CLOSP/SARCLIP/Mind-the-Modality-Gap), embed every modality into that **shared space** → cross-modal retrieval works zero-shot. Otherwise embed a small **text/class-name prompt** as the bridge.
3. **Post-process (this is the entire "training"):**
   - **PCA-whitening** (fit on a modality-balanced sample of the gallery) → optional **dim reduction to 256** → **L2-normalize**.
   - Optional **per-modality mean-centering** before whitening to **reduce the modality gap** (subtract each modality's mean embedding so optical/SAR/MS cones overlap) — cheap, effective modality-gap mitigation.
   - **Temperature/scale** is irrelevant for pure cosine retrieval (ranking is scale-invariant after L2-norm); only matters if you fuse scores or calibrate.
4. **Index with FAISS** `IndexFlatIP` (cosine); **binary-hash** for speed (near-free mAP per IBM/Prithvi). Cosine similarity → rank → top-5/top-10.
5. **Relevance/F1 scoring** uses the provided class/geo labels exactly as the metric defines.

**Expected:** strong same-modal F1 immediately (foundation models are excellent same-modal retrievers); cross-modal F1 depends on having a shared-space (CLIP-style RS) model or applying mean-centering + whitening to overlap modality cones. This fallback already satisfies the "low retrieval time" criterion (frozen features + FAISS/binary).

---

## 15. One-paragraph executive summary

Use a **two-stage, multi-loss recipe**: **(1)** align modalities with **symmetric CLIP-style InfoNCE** on geo-paired patches (learnable τ≈0.07, optionally via a text/optical *anchor* so you don't need triple-aligned data, plus a **MoCo cross-modal queue** for negatives if batches are small); **(2)** make clusters semantically tight with **Sub-center ArcFace** (s=64, m=0.5, K=3) on a **shared land-cover classifier head** fed by all modalities, and **sharpen top-K ranking** with a **cross-modal batch-hard triplet** (semi-hard→hard), optionally finishing with **Smooth-AP** to directly optimize the rank-based F1@K. Feed it with a **modality-balanced P×K sampler** (P=16–32 classes, K=4 split across optical/SAR/MS). Train with **AdamW + cosine + warmup**, **differential LR** (backbone 1e-5, ArcFace head up to ~1.0), **256-D L2-normalized** embeddings, **PCA-whitening** post-processing, and **FAISS (+ optional binary hashing)**. Augment **per modality**: SAR = multiplicative Gamma speckle (L∈{1,2,4,8}) + Lee/refined-Lee despeckle + dB-scaling + VV/VH dropout; optical = color jitter/grayscale/blur/Cutout/RandAugment; MS = band dropout/spectral-tube-mask + spectral jitter + NDVI-index aug; **all** = flips/rot90/RRC/small-rotations with a **shared seed across the paired modalities** to preserve co-registration, plus **modality dropout** and **cross-modal mixup** to force reliance on the shared space. Keep a **zero-training fallback**: frozen Prithvi/DINOv2 (and a SAR encoder / shared-text-space CLIP model) embeddings → per-modality mean-centering → **PCA-whitening + L2-normalization** → FAISS cosine search.

---

## Sources
- InfoNCE / NT-Xent & temperature: NT-Xent glossary & temperature-free InfoNCE (arXiv:2501.17683); "Understanding the Behaviour of Contrastive Loss" (CVPR'21, arXiv:2012.09740); "Dynamically Scaled Temperature" (arXiv:2308.01140).
- CLIP-style cross-modal RS: **CLOSP** (arXiv:2507.10403); **SARCLIP** (ScienceDirect S0924271625004058); **Mind the Modality Gap** (arXiv:2402.09816); modality-gap analysis "Mind the Gap" (arXiv:2203.02053).
- SupCon: Khosla et al. "Supervised Contrastive Learning" (arXiv:2004.11362); semi-supervised RS SupCon (arXiv:2112.06437); multi-label soft contrastive RS (arXiv:2405.20462).
- Triplet & mining: FaceNet/triplet (Wikipedia; Moindrot blog); "Sampling Matters in Deep Embedding Learning" (arXiv:1706.07567).
- ArcFace/CosFace/Sub-center/AdaCos: ArcFace (arXiv:1801.07698, CVPR'19); AdaCos (arXiv:1905.00292); Deep-Metric-Learning survey (hav4ik.github.io).
- Proxy-Anchor / Proxy-NCA++: Kim et al. (arXiv:2003.13911, CVPR'20).
- Multi-Similarity: Wang et al. (arXiv:1904.06627, CVPR'19).
- Circle loss: Sun et al. (arXiv:2002.10857).
- NormSoftmax / classification baseline: "Classification is a Strong Baseline for DML" (arXiv:1811.12649); NormSoftmax (OpenReview 4g7nCbpjNwd).
- Smooth-AP: Brown et al. (arXiv:2007.12163, ECCV'20). FastAP: Cakir et al. "Deep Metric Learning to Rank" (CVPR'19). OSAP-Loss for RS retrieval (TGRS).
- MoCo / memory bank: He et al. Momentum Contrast; enhanced negative sampling (arXiv:2501.16360).
- Default hyperparameters: **PyTorch-Metric-Learning** docs (kevinmusgrave.github.io/pytorch-metric-learning/losses, /samplers).
- Practical retrieval recipe: "All You Need to Know About Training Image Retrieval Models" (arXiv:2503.13045).
- PCA-whitening / L2-norm: image-retrieval tricks benchmark (arXiv:1907.11854); fine-tuning CNN retrieval (arXiv:1711.02512); "Negative Evidences and Co-occurrences…".
- Samplers (P×K, hard-class): PyTorch-Metric-Learning samplers; Graph-Sampling DML (arXiv:2104.01546).
- SAR augmentation / speckle / dB: deep SAR despeckling review (MDPI 11/13/1532); speckle Gamma/looks (arXiv:2307.06855; Sentinel-1 dB preprocessing, Google Earth Engine guide; OPERA RTC arXiv:2501.09129).
- MS augmentation / band dropout: spectral foundation model (arXiv:2503.01628); SSL-in-RS review (arXiv:2206.13188); rank-based geo-regularization MS contrastive (arXiv:2601.02289).
- Modality dropout / cross-modal mixup: Modality-Dropout topic (EmergentMind); Multimodal Mixup Contrastive (arXiv:2409.17777); improved modality dropout (MICCAI'25 paper 2038).
- Datasets: **BigEarthNet-MM** (arXiv:2105.07921); **SEN1-2** (arXiv:1807.01569); self-supervised cross-modal RS retrieval (arXiv:2202.11429).
- Foundation-model retrieval (zero-training): IBM/Prithvi RS retrieval (arXiv:2403.02059; github.com/IBM/remote-sensing-image-retrieval); Prithvi-EO-2.0 (arXiv:2412.02732).
