# 02 — Foundation Models & Pretrained Backbones for Cross-Modal Satellite Image Retrieval

**Problem Statement 11 (BAH 2026):** Cross-Modal Satellite Image Retrieval Using Multi-Sensor Remote Sensing Data.
**Downstream use:** extract a compact L2-normalized embedding per image → FAISS/ANN retrieval (cosine/inner-product), with optional contrastive fine-tuning of a lightweight projection head. Modalities in play: **Optical RGB**, **Multispectral (Sentinel-2, 12–13 bands)**, **SAR (Sentinel-1 VV/VH, 2 bands)**, optionally **hyperspectral / DEM**.

**Date compiled:** 2026-06-21. All HF ids / GitHub links cross-verified across the model cards, papers, and official repos (sources listed at the end).

---

## TL;DR Recommendation (read this first)

| Role | Model | HF id / source | Arch | Embedding dim | Why |
|---|---|---|---|---|---|
| **DEFAULT multimodal (SAR+MS+RGB → one space)** | **DOFA** | `XShadow/DOFA` (also mirrored `earthflow/DOFA`) | ViT-B/16 & ViT-L/16 | **768 (B) / 1024 (L)** | One model, any band-set via wavelength-conditioned patch embed. Feed S1, S2, RGB through the *same* weights → naturally shared space. Permissive-ish (CC-BY-4.0). |
| **Strong alt. multimodal (radar-optical native)** | **CROMA** | `antofuller/CROMA` | ViT-B & ViT-L | **768 (B) / 1024 (L)** | Purpose-built radar+optical MAE+contrastive; gives `joint_GAP` fused vector + per-modality GAP. **MIT license.** Only S1(2-band)+S2(12-band), fixed bands. |
| **SAR + MS specialist (per-modality)** | **SoftCon** | `wangyi111/softcon` | ViT-S/14 & ViT-B/14 | **384 (S) / 768 (B)** | Best dedicated S1(2-band) **and** S2(13-band) encoders; strong linear-probe features for SAR-to-SAR and MS-to-MS. Apache-2.0. |
| **RGB / optical specialist** | **RemoteCLIP** | `chendelong/RemoteCLIP` | RN50, ViT-B/32, ViT-L/14 | **1024 / 512 / 768** | SOTA RS image–image & image–text retrieval; OpenCLIP-compatible; great for optical-to-optical. |
| **General fallback #1 (always works)** | **OpenCLIP ViT-B/32 (LAION-2B)** | `laion/CLIP-ViT-B-32-laion2B-s34B-b79K` | ViT-B/32 | **512** | Tiny, ubiquitous, offline-cacheable, identical interface to RemoteCLIP/GeoRSCLIP/SkyCLIP. |
| **General fallback #2 (dense SSL)** | **DINOv2 ViT-B/14 (+registers)** | `facebook/dinov2-with-registers-base` | ViT-B/14 | **768** | Best generic frozen features for image retrieval; no text head needed. |

**FAISS index sizing:** standardize on **768-d** as the primary index width (DOFA-B = CROMA-B = SoftCon-B = DINOv2-B = RemoteCLIP-ViT-L all emit 768). Use a separate index per embedding-width if you mix model families. See §"FAISS / index design" at the end.

---

## How to read this catalogue

For each model: **name · architecture · accepted modalities/bands · pretraining dataset & objective · output embedding dim · weights availability + exact HF repo/GitHub · license · input size & preprocessing · retrieval-suitability note (and whether it yields a single shared space).**

"Shared space across modalities" means: can a query from modality A and a gallery item from modality B be compared *directly* (same vector space) **without** training a cross-modal aligner? Models marked **partial** produce per-modality encoders from joint pretraining, so cross-modal cosine similarity is meaningful but usually benefits from a small contrastive projection head (which the challenge explicitly allows).

---

## A. Multimodal / multi-sensor RS foundation models (the core of this challenge)

### 1. DOFA (Dynamic-One-For-All) ⭐ recommended default
- **Architecture:** Single shared ViT (ViT-B/16 and ViT-L/16) + a **wavelength-conditioned dynamic patch-embedding hypernetwork**. Neural-plasticity-inspired: one set of weights serves any number of channels.
- **Modalities/bands:** Truly band-agnostic. Verified examples for **Sentinel-1 SAR (2 bands, λ≈5.405 cm → encoded as wave value 5.405)**, **Sentinel-2 (passes the per-band central wavelengths in µm, e.g. 9-band subset [0.665,0.56,0.49,0.705,0.74,0.783,0.842,1.61,2.19])**, **RGB/NAIP (3 bands [0.665,0.56,0.49])**, Gaofen-2, EnMAP hyperspectral. You literally pass `wave_list=[...]` per input.
- **Pretraining data/objective:** Multimodal masked-image-modeling distillation across 5 sensors (Sentinel-1/2, Gaofen-2, NAIP, EnMAP).
- **Output embedding dim:** **768 (ViT-B), 1024 (ViT-L)** — use the CLS/pooled token from `forward_features`.
- **Weights:** PUBLIC. HF: **`XShadow/DOFA`** (file `DOFA_ViT_base_e100.pth`, plus large); also commonly mirrored as **`earthflow/DOFA`**; torchgeo has `DOFABase16_Weights`/`DOFALarge16_Weights`. PyTorch-Hub: `torch.hub.load('zhu-xlab/DOFA','vit_base_dofa',pretrained=True)`.
- **License:** **CC-BY-4.0** (model card). Commercial use generally OK with attribution — confirm for production.
- **Input/preproc:** 224×224. Per-channel standardization; you must supply central wavelengths in **micrometres** (SAR in cm as the repo does — match their convention exactly). Bands can be in any order as long as `wave_list` matches.
- **Retrieval note / shared space:** **YES — single shared space.** Because S1, S2 and RGB all flow through the *identical* backbone, their CLS embeddings live in one 768/1024-d space out of the box — ideal for cross-modal SAR↔optical retrieval. A short contrastive projection head sharpens it further. **This is the best default for PS11.**

### 2. CROMA (Contrastive Radar-Optical MAE, NeurIPS'23) ⭐ recommended alt-default
- **Architecture:** Two ViT encoders (one SAR, one optical) + a **third "radar-optical" fusion encoder** + multimodal decoder. ViT-B and ViT-L.
- **Modalities/bands:** **Sentinel-1 SAR = 2 channels (VV, VH)**; **Sentinel-2 = 12 channels (drop B10 cirrus from the 13)**. Fixed band layout (not band-agnostic).
- **Pretraining data/objective:** **SSL4EO-S12** (~1M co-registered S1 GRD + S2 L2A pairs). Objective = cross-modal **contrastive** alignment **+ MAE reconstruction**.
- **Output embedding dim:** **768 (base) / 1024 (large)**. Output dict keys: `SAR_encodings`, `optical_encodings`, `joint_encodings` (patch-level B×N×D) and pooled **`SAR_GAP`, `optical_GAP`, `joint_GAP`** (B×D). **Use `joint_GAP` (or modality GAP) as the retrieval vector.**
- **Weights:** PUBLIC. HF: **`antofuller/CROMA`** (`CROMA_base.pt`, `CROMA_large.pt`). GitHub: `antofuller/CROMA`.
- **License:** **MIT** (most permissive here — fully commercial-friendly).
- **Input/preproc:** default **120×120** (changeable). Normalization: per-channel mean/std, clip to mean±2σ then scale to [0,1] or [0,255] (helper provided in repo).
- **Retrieval note / shared space:** **YES (strong)** — contrastive objective explicitly aligns S1 and S2; `joint_GAP` is a fused multimodal vector and the per-modality GAPs are pulled together. Excellent for SAR↔optical. **Caveat:** only works for S1(2-band) + S2(12-band); RGB-only or hyperspectral inputs need adaptation. If your data is purely Sentinel-1/2, **CROMA is arguably the single best fit** (purpose-built, MIT-licensed).

### 3. DeCUR (Decoupling Common & Unique Reps, ECCV'24)
- **Architecture:** Two-branch SSL (Barlow-Twins-style) that splits each modality's embedding into **common** (cross-modal) + **unique** (intra-modal) dims. Backbones: **ResNet-50** and **ViT-S/16**.
- **Modalities/bands:** Radar-optical variant = **Sentinel-1 SAR (2-band) + Sentinel-2 MS**; also RGB-DEM and RGB-depth variants.
- **Pretraining data/objective:** SSL4EO-S12; decoupled common+unique self-supervision.
- **Output embedding dim:** RN50 → 2048; ViT-S/16 → 384.
- **Weights:** PUBLIC. HF: **`wangyi111/DeCUR`** (e.g. `rn50_ssl4eo-s12_joint_decur_ep100.pth`, plus per-modality `..._sar_...`, `..._ms_...`). GitHub: `zhu-xlab/DeCUR`.
- **License:** **Apache-2.0** (commercial-friendly).
- **Input/preproc:** SSL4EO-S12 normalization (mean±2σ clip → scale), 224×224.
- **Retrieval note / shared space:** **Partial→good.** The "common" subspace is explicitly the shared cross-modal space — slice the common dims (or use the joint checkpoint) for SAR↔optical retrieval; the "unique" dims help same-modal. A natural fit for a system that does *both* same- and cross-modal.

### 4. Galileo (NASA Harvest + AI2, ICML'25) — highly multimodal
- **Architecture:** Highly-multimodal ViT processing many modalities jointly across space & time; dual global+local contrastive objective. Sizes **nano / tiny / base**.
- **Modalities/bands:** **Sentinel-2 MS, Sentinel-1 SAR, elevation/DEM, ERA5 weather, NDVI/derived products, pseudo-labels** — 10+ products, space-time tensors `[B,H,W,T,bands]`.
- **Pretraining data/objective:** Large global EO corpus; masked + contrastive (global & local features).
- **Output embedding dim:** depends on size (nano ≈ 128, base larger; load and pool the encoder tokens — check the config in the released folder). Encoder via `Encoder.load_from_folder('models/base')`.
- **Weights:** PUBLIC. HF: **`nasaharvest/galileo`** (`hf download nasaharvest/galileo --include "models/**"`); GitHub `nasaharvest/galileo`.
- **License:** **MIT**.
- **Input/preproc:** sensor-stack tensors with masks; designed for patch/pixel time-series more than single RGB chips.
- **Retrieval note / shared space:** **Partial→good** for S1/S2/DEM. Very flexible but heavier to wire up; best if your gallery has rich multi-band/time-series data rather than plain RGB JPEGs. Mean-pool encoder tokens → L2-normalize.

### 5. TerraMind 1.0 (IBM + ESA + Jülich, ICCV'25) — any-to-any generative
- **Architecture:** Dual-scale (token-level + pixel-level) multimodal transformer; "Thinking-in-Modalities" generation. ViT sizes **tiny / small / base / large**.
- **Modalities/bands:** **S2L2A (12-band), S2L1C, S1GRD, S1RTC (SAR), DEM, RGB** + more (9 modalities total). You pass `modalities=['S2L2A','S1GRD',...]`.
- **Pretraining data/objective:** TerraMesh large multimodal corpus; correlated masked + generative pretraining.
- **Output embedding dim:** large emits patch tokens of shape `B×196×768` (→ **768** for base/large; pool tokens). `merge_method ∈ {mean,max,concat,dict}` controls multimodal fusion.
- **Weights:** PUBLIC. HF: **`ibm-esa-geospatial/TerraMind-1.0-base`** and **`-large`** (`-tiny`, `-small` also). Load via TerraTorch `BACKBONE_REGISTRY.build('terramind_v1_base', pretrained=True, modalities=[...])`. GitHub `IBM/terramind`.
- **License:** **Apache-2.0** (commercial-friendly).
- **Input/preproc:** 224×224; TerraTorch handles per-modality normalization (datamodule presets). Topped ESA's PANGAEA benchmark (+8% over 12 FMs).
- **Retrieval note / shared space:** **YES (strong)** — explicitly multimodal; mean-pool tokens across requested modalities → one vector. Newest & strongest, but the TerraTorch dependency is the heaviest integration here. Excellent SAR+MS+DEM+RGB option if you can take the dependency.

### 6. AnySat (CVPR'25) — any-resolution / any-modality JEPA
- **Architecture:** Scale-adaptive **JEPA**; single model, choose modalities + patch size at inference.
- **Modalities/bands:** 11 sensors — **Sentinel-2 (10-band TS), Sentinel-1 (2–3 band TS), aerial/SPOT/NAIP VHR RGB, Landsat, ALOS, MODIS**.
- **Pretraining/obj.:** GeoPlex (5 multimodal datasets); JEPA self-supervision.
- **Output embedding dim:** configurable `D` (model `output='tile'` → single `[D]` vector; `'patch'/'dense'` for maps). Check loaded config for D.
- **Weights:** PUBLIC. HF **`g-astruc/AnySat`**; `torch.hub.load('gastruc/anysat','anysat',pretrained=True)`. GitHub `gastruc/AnySat`.
- **License:** **MIT**.
- **Input/preproc:** `[B,C,H,W]` or `[B,T,C,H,W]` (+ `_dates`); patch size in multiples of 10 m.
- **Retrieval note / shared space:** **YES (partial)** — one model spanning all sensors → `tile` vectors comparable across modalities; very strong for cross-modal but more bespoke I/O (best for true S1/S2/aerial, not arbitrary RGB).

### 7. msGFM (Multisensor Geospatial FM, CVPR'24)
- **Architecture:** Cross-sensor masked-image-modeling; unifies 4 sensor modalities (RGB, MS, SAR, DSM/elevation), handles paired & unpaired data.
- **Status:** **WEIGHTS NOT RELEASED** — repo (`boranhan/Geospatial_Foundation_Models`) lists msGFM as *"To be released."* Conceptually ideal (RGB+MS+SAR+elevation in one space) but **not usable today**. Track for later; do not depend on it.

### 8. USat (Stanford, 2023)
- **Architecture:** ViT with modified patch-projection + positional encodings to mix bands of different GSDs from multiple sensors (MAE-style).
- **Modalities/bands:** Multi-sensor **multispectral** (Sentinel-2, NAIP, …). Optical/MS focus; not a primary SAR model.
- **Weights:** PUBLIC. GitHub **`stanfordmlgroup/USat`** (code + weights). No first-party HF card.
- **License:** check repo (research/MIT-style).
- **Retrieval note:** decent multi-sensor MS encoder; less battle-tested than CROMA/DOFA; SAR support limited. Lower priority.

---

## B. Optical / Multispectral RS foundation models (same-modal optical & MS)

### 9. Prithvi-EO-2.0 (IBM + NASA + Jülich, 2024)
- **Arch:** ViT + **MAE**, with **3D (spatiotemporal) patch & positional embeddings**; optional lat/lon + date conditioning (`-TL` variants). Sizes **300M / 600M** (and 100M-TL).
- **Bands:** **6 HLS bands** (Blue, Green, Red, Narrow-NIR, SWIR1, SWIR2) at 30 m; multi-temporal. **Optical/MS only — no SAR.**
- **Pretraining:** 4.2M HLS samples; MAE.
- **Embedding dim:** ViT-L-class (300M) → 1024; 600M larger. Pool encoder tokens.
- **Weights:** PUBLIC. HF **`ibm-nasa-geospatial/Prithvi-EO-2.0-300M`** / **`-600M`** (+ `-TL`). GitHub `NASA-IMPACT/Prithvi-EO-2.0`. Load via TerraTorch.
- **License:** **Apache-2.0**.
- **Input/preproc:** HLS reflectance; per-band normalization; multi-temporal stacks.
- **Retrieval note:** strong for HLS/Landsat-Sentinel optical & time-series; **not** for SAR or plain RGB.

### 10. SatMAE (NeurIPS'22)
- **Arch:** ViT-Large MAE with **spectral-group encoding** + temporal encoding.
- **Bands:** fMoW-RGB or **fMoW-Sentinel (multispectral, grouped)**.
- **Embedding dim:** **1024** (ViT-L; D=1024, +768 spatial PE + 256 spectral-group PE internally).
- **Weights:** PUBLIC. GitHub **`sustainlab-group/SatMAE`** (checkpoints, 200-epoch).
- **License:** research (check repo).
- **Retrieval note:** solid MS baseline; superseded by SatMAE++/Scale-MAE/DOFA but a good reference encoder.

### 11. SatMAE++ (CVPR'24)
- **Arch:** multi-scale MAE extending SatMAE/Scale-MAE to multispectral with conv upsampling, multi-scale reconstruction. ViT-L.
- **Bands:** multispectral (Sentinel-2 / fMoW-Sentinel).
- **Embedding dim:** **1024** (ViT-L).
- **Weights:** PUBLIC. GitHub **`techmn/satmae_pp`**.
- **License:** check repo.
- **Retrieval note:** stronger MS features than SatMAE; optical/MS only.

### 12. Scale-MAE (ICCV'23)
- **Arch:** MAE with **GSD-aware positional encoding** + Laplacian-pyramid decoder → scale-invariant features. ViT-Large.
- **Bands:** primarily **RGB** at varying GSD (scale-robust).
- **Embedding dim:** **1024** (ViT-L).
- **Weights:** PUBLIC. GitHub **`bair-climate-initiative/scale-mae`**.
- **License:** check repo.
- **Retrieval note:** excellent when gallery spans many resolutions/altitudes; RGB-centric.

### 13. SpectralGPT (TPAMI'24)
- **Arch:** **3D GPT/MAE** for spectral cubes; spatial-spectral 3D tokens; multi-target reconstruction. ~600M params.
- **Bands:** Sentinel-2 **multispectral** (spectral sequence modeling).
- **Embedding dim:** ViT-L-class (~1024; check checkpoint).
- **Weights:** PUBLIC (paper repo / `danfenghong` SpectralGPT). 
- **License:** check repo.
- **Retrieval note:** strongest *spectral* modeling; great MS-to-MS; no SAR, not for plain RGB.

### 14. Clay v1.5 (Clay Foundation)
- **Arch:** ViT MAE with a **Dynamic Embedding Block** (patches conditioned on band wavelengths) + metadata (GSD, lat/lon, time).
- **Bands:** multi-sensor — **Sentinel-2 (10), Landsat (6), Sentinel-1 SAR (2), NAIP (4), LINZ (3), MODIS (7)**; supports arbitrary bands.
- **Embedding dim:** **1024** (`[1,1024]` class embedding).
- **Weights:** PUBLIC. HF **`made-with-clay/Clay`** (`v1.5/clay-v1.5.ckpt`). GitHub `Clay-foundation/model`.
- **License:** **Apache-2.0** (code) / OpenRAIL-style for model — verify on card.
- **Input/preproc:** chips + metadata dict (wavelengths, GSD, lat/lon, time); 224-ish.
- **Retrieval note / shared space:** **partial multimodal** — like DOFA it conditions on wavelengths, so S2/S1/RGB pass through one model → comparable 1024-d embeddings. A strong *secondary* multimodal option (esp. if you want metadata conditioning). Heavier I/O (needs metadata) than DOFA.

---

## C. Vision–Language (CLIP-style) RS models — best for **optical** retrieval (+text query bonus)

> These are OpenCLIP-architecture, so the **image-encoder output (projection) dims are the standard CLIP dims:** ViT-B/32 → **512**, ViT-L/14 → **768**, ViT-H/14 → **1024**, RN50 → **1024**. All accept **3-band RGB**, 224×224, CLIP normalization (mean `[0.4815,0.4578,0.4082]`, std `[0.2686,0.2613,0.2758]`). Use `model.encode_image(x)` then L2-normalize.

### 15. RemoteCLIP (IEEE TGRS'24) ⭐ recommended optical specialist
- **Arch:** OpenCLIP RN50 / ViT-B/32 / ViT-L/14 fine-tuned on RS.
- **Embedding dim:** **1024 / 512 / 768**.
- **Weights:** PUBLIC. HF **`chendelong/RemoteCLIP`** (`RN50`, `ViT-B-32`, `ViT-L-14`). GitHub `ChenDelong1999/RemoteCLIP`.
- **License:** check card (research-permissive; built on OpenAI/LAION CLIP).
- **Retrieval note:** **SOTA RS image–image & image–text retrieval** (+9% mean-recall on RSITMD/RSICD). **Best RGB-optical specialist for PS11** same-modal optical-to-optical. RGB only.

### 16. GeoRSCLIP / RS5M
- **Arch:** CLIP ViT-B/32 (OpenAI init) & **ViT-H/14** (LAION init) fine-tuned on **RS5M (5M RS image–text)**.
- **Embedding dim:** **512 (B/32) / 1024 (H/14)**.
- **Weights:** PUBLIC. HF **`Zilun/GeoRSCLIP`** (`RS5M_ViT-B-32.pt`, `RS5M_ViT-H-14.pt`); dataset `Zilun/RS5M`.
- **License:** CC (check card).
- **Retrieval note:** very strong RS retrieval/zero-shot; ViT-H/14 gives a fat 1024-d optical embedding. RGB only.

### 17. SkyCLIP (SkyScript, AAAI'24)
- **Arch:** CLIP ViT-B/32 & **ViT-L/14** continually pretrained on **SkyScript (5.2M RS image–text)**.
- **Embedding dim:** 512 / **768**.
- **Weights:** PUBLIC. GitHub `wangzhecheng/SkyScript` (S3 ckpts, e.g. `SkyCLIP_ViT_L14_top50pct`).
- **License:** check repo.
- **Retrieval note:** broad semantic-tag coverage (29K tags); good optical retrieval alt to RemoteCLIP. RGB only.

### 18. DOFA-CLIP (2025)
- **Arch:** DOFA backbone + CLIP text alignment → vision-language over **multi-band** EO (not just RGB).
- **Weights:** emerging (see arXiv 2503.06312; check `earthflow`/`xshadow` HF orgs). 
- **Retrieval note:** interesting because it brings CLIP-style text + **multispectral** image space together; watch for stable weights. Lower priority than plain DOFA for now.

---

## D. SAR-specific / SAR-capable encoders (critical for the SAR side of PS11)

**Which catalogued models actually ingest Sentinel-1 SAR:**
- **CROMA** — S1 (2-band VV/VH) native, fused with S2. ✅ (best radar-optical)
- **DOFA** — S1 via `wave_list=[5.405,5.405]`. ✅ (band-agnostic)
- **DeCUR** — S1 (2-band) radar-optical SSL. ✅
- **SoftCon** — dedicated **S1 (2-band)** encoder checkpoint. ✅ (best pure-SAR features)
- **Clay v1.5** — S1 (2-band) among its supported sensors. ✅
- **Galileo / AnySat / TerraMind** — S1 GRD/RTC supported. ✅
- **SAR-JEPA** — SAR-only ATR model. ✅ (single-channel SAR amplitude)

### 19. SoftCon ⭐ recommended SAR & MS specialist
- **Arch:** ViT-S/14 & ViT-B/14 (also RN50), **soft-supervised contrastive** continual-pretraining from DINO/DINOv2.
- **Bands:** **Sentinel-1 SAR = 2 channels**; **Sentinel-2 = 13 channels** (separate checkpoints per modality).
- **Pretraining:** SSL4EO-S12 + Dynamic-World multi-label guidance.
- **Embedding dim:** **384 (ViT-S/14) / 768 (ViT-B/14)**; RN50 → 2048.
- **Weights:** PUBLIC. HF **`wangyi111/softcon`** (e.g. `B2_vits14_softcon.pth` for S1, `B13_vitb14_softcon.pth` for S2). GitHub `zhu-xlab/softcon`.
- **License:** **Apache-2.0**.
- **Input/preproc:** **224×224**; SSL4EO-S12 per-channel mean/std, clip mean±2σ → 0–255. Replace head with `nn.Identity()` to get features.
- **Retrieval note:** **best dedicated SAR-to-SAR and MS-to-MS encoders** (frozen features are excellent). Per-modality (not a shared space by itself), so pair with a contrastive head for cross-modal — but unbeatable for same-modal SAR/MS.

### 20. SAR-JEPA (ISPRS'24)
- **Arch:** Joint-Embedding Predictive Architecture predicting **multi-scale SAR gradient features**; ViT backbone.
- **Bands:** **single-channel SAR amplitude** (ATR-style: vehicles/ships/aircraft).
- **Weights:** PUBLIC. GitHub **`waterdisappear/SAR-JEPA`** (code + weights).
- **License:** check repo.
- **Retrieval note:** specialized SAR self-supervision; good if your SAR is fine-grained target chips. Narrower than SoftCon for scene-level S1.

### 21. SSL4EO-S12 model zoo (MoCo-v2 / DINO / MAE / data2vec, B13 ViT)
- **Arch:** ResNet-50 & **ViT-S/16** pretrained on SSL4EO-S12 with MoCo, DINO, MAE, data2vec; **separate S1 (2-band) and S2 (13-band, "B13") checkpoints**.
- **Embedding dim:** RN50 → 2048; ViT-S → 384.
- **Weights:** PUBLIC. GitHub **`zhu-xlab/SSL4EO-S12`** + **`DLR-MF-DAS/SSL4EO-S12-v1.1`** (download links on the cards).
- **License:** Apache-2.0-style (check).
- **Retrieval note:** the canonical S1/S2 SSL baselines; "DINO-S1/B13" make handy modality-specific encoders. SoftCon/CROMA usually beat them but these are reliable.

---

## E. General-purpose vision encoders — fallbacks, baselines & teachers

> Use these when (a) specialized RS weights can't be downloaded, or (b) you want a teacher for contrastive distillation, or (c) input is plain 3-band RGB. All take **3-channel RGB**. For multispectral/SAR you must reduce to 3 channels (e.g. RGB bands, or VV/VH/ratio for SAR) or replace the patch-embed conv.

### 22. DINOv2 (+ registers) ⭐ recommended generic fallback
- **Arch:** ViT-S/B/L/g, patch 14, self-distillation SSL; register-token variants reduce artifacts.
- **Embedding dim:** **S 384 / B 768 / L 1024 / g 1536**.
- **Weights:** PUBLIC. HF **`facebook/dinov2-base`**, **`facebook/dinov2-large`**, and registers **`facebook/dinov2-with-registers-base`** / **`-large`**.
- **License:** **Apache-2.0**.
- **Input/preproc:** 224 (multiples of 14); ImageNet mean/std. CLS token = global embedding.
- **Retrieval note:** **best generic frozen features for image retrieval**; no text head needed. Top fallback for optical-to-optical and as a teacher.

### 23. DINOv3 — incl. **satellite-specialized SAT-493M** variant
- **Arch:** ViT (S/B/L up to 7B), patch 16, RoPE, registers; SOTA dense features.
- **Embedding dim:** B 768 / L 1024 / **7B 4096**.
- **Weights:** PUBLIC (gated). HF **`facebook/dinov3-vitb16-pretrain-lvd1689m`**, **`-vitl16-...`**; **satellite**: **`facebook/dinov3-vit7b16-pretrain-sat493m`** and `timm/vit_7b_patch16_dinov3.sat493m` (Maxar 0.6 m RGB, 493M images; SOTA canopy height). Smaller SAT distillations exist in the DINOv3 release.
- **License:** **DINOv3 License** (gated; review terms for commercial use — more restrictive than Apache).
- **Input/preproc:** SAT variant uses *non-ImageNet* RGB mean/std (satellite stats — check release; an open issue tracks the exact values). 256×256 common.
- **Retrieval note:** the **SAT-493M** model is a very strong **RGB-aerial** retrieval backbone; license is the catch. ViT-L SAT, if accessible, is an excellent optical-RGB specialist.

### 24. OpenCLIP (LAION) ⭐ recommended always-works fallback
- **Arch:** ViT-B/32, ViT-L/14, ViT-H/14 (and bigG); contrastive image-text on LAION-2B / DataComp.
- **Embedding dim:** **512 / 768 / 1024**.
- **Weights:** PUBLIC. HF **`laion/CLIP-ViT-B-32-laion2B-s34B-b79K`**, **`laion/CLIP-ViT-L-14-laion2B-s32B-b82K`**, **`laion/CLIP-ViT-H-14-laion2B-s32B-b79K`**, plus DataComp `laion/CLIP-ViT-L-14-DataComp.XL-s13B-b90K`.
- **License:** **MIT** (open_clip) — fully commercial-friendly.
- **Input/preproc:** 224, CLIP mean/std; `encode_image` → L2-norm.
- **Retrieval note:** **the safe universal fallback** — identical interface to RemoteCLIP/GeoRSCLIP/SkyCLIP, so you can hot-swap weights. ViT-B/32 (512-d) is tiny and cache-friendly.

### 25. SigLIP / SigLIP 2 (Google, 2025)
- **Arch:** ViT base/large/so400m, sigmoid-loss contrastive; multilingual; FixRes & NaFlex (any-res) variants.
- **Embedding dim:** base ~768, so400m ~1152 (pooled image embedding).
- **Weights:** PUBLIC. HF **`google/siglip2-base-patch16-224`**, **`google/siglip2-so400m-patch14-384`**, etc.
- **License:** **Apache-2.0**.
- **Retrieval note:** stronger image–text retrieval & representations than CLIP at equal scale; excellent RGB fallback/teacher. NaFlex handles non-square satellite tiles nicely.

### 26. EVA-02-CLIP
- **Arch:** EVA-02 ViT (B/L/E) CLIP, trained on LAION-2B + COYO.
- **Embedding dim:** projection ~512 (B) / 768 (L) (image-text shared dim).
- **Weights:** PUBLIC. HF **`QuanSun/EVA-CLIP`** (e.g. `EVA02-CLIP-L-14`).
- **License:** MIT-style (check card).
- **Retrieval note:** strong CLIP-family alternative; good teacher. RGB only.

### 27. Presto (Lightweight RS Transformer, 2023) — pixel time-series
- **Arch:** tiny MAE transformer over **pixel time-series** (not image chips); ingests S1, S2, ERA5, DEM, location/time.
- **Embedding dim:** small (~128); single-file impl `nasaharvest/presto/single_file_presto.py`.
- **Weights:** PUBLIC. GitHub **`nasaharvest/presto`**.
- **License:** check repo.
- **Retrieval note:** brilliant if your "image" is actually a pixel/time-series (cropland-style), **not** for 2-D scene chips. Niche for PS11 unless data is time-series.

---

## Master comparison table

| # | Model | Arch | Modalities (SAR? MS? RGB? other) | Pretrain obj. | Emb. dim | Weights (HF id / repo) | License | Input | Shared cross-modal space? |
|--|--|--|--|--|--|--|--|--|--|
| 1 | **DOFA** ⭐ | ViT-B/L 16 + wavelength hypernet | S1✅(2) · S2/MS✅(any) · RGB✅ · hyperspectral · any bands | Multimodal MIM distill | **768/1024** | `XShadow/DOFA` (mirror `earthflow/DOFA`) | CC-BY-4.0 | 224, +wavelengths | **YES (one backbone)** |
| 2 | **CROMA** ⭐ | 2×ViT + fusion enc (B/L) | S1✅(2) · S2✅(12) · RGB❌ | Contrastive + MAE | **768/1024** | `antofuller/CROMA` | **MIT** | 120 (var) | **YES (`joint_GAP`)** |
| 3 | DeCUR | RN50 / ViT-S16 | S1✅(2) · S2✅ · DEM | Common+unique SSL | 2048 / 384 | `wangyi111/DeCUR` (gh `zhu-xlab/DeCUR`) | Apache-2.0 | 224 | Partial (common dims) |
| 4 | Galileo | Multimodal ViT (nano/tiny/base) | S1✅ · S2✅ · DEM · ERA5 · derived | Global+local contrastive | size-dep | `nasaharvest/galileo` | **MIT** | space-time tensors | Partial→good |
| 5 | TerraMind 1.0 | ViT (t/s/b/l) dual-scale | S2(12)✅ · S1✅ · DEM · RGB✅ (9 mod) | Generative + masked | **~768 (b/l)** | `ibm-esa-geospatial/TerraMind-1.0-base`/`-large` | Apache-2.0 | 224 | **YES** |
| 6 | AnySat | Scale-adaptive JEPA | S2✅ · S1✅ · aerial/SPOT/NAIP RGB✅ · Landsat/ALOS/MODIS | JEPA | config `D` | `g-astruc/AnySat` | **MIT** | [B,(T),C,H,W] | YES (partial) |
| 7 | msGFM | MIM, 4 sensors | RGB · MS · SAR · DSM | Cross-sensor MIM | n/a | **NOT released** | — | — | (would be yes) |
| 8 | USat | ViT (multi-GSD) | MS multi-sensor; SAR limited | MAE | — | gh `stanfordmlgroup/USat` | repo | — | Partial |
| 9 | Prithvi-EO-2.0 | ViT + 3D MAE (300M/600M) | HLS 6-band MS✅; **no SAR** | MAE (+geo/time) | 1024+ | `ibm-nasa-geospatial/Prithvi-EO-2.0-300M`/`-600M` | Apache-2.0 | multi-temporal | No (MS only) |
| 10 | SatMAE | ViT-L MAE | MS (fMoW-Sentinel) / RGB | MAE + spectral group | 1024 | gh `sustainlab-group/SatMAE` | repo | 224 | No |
| 11 | SatMAE++ | ViT-L multi-scale MAE | MS | Multi-scale MAE | 1024 | gh `techmn/satmae_pp` | repo | 224 | No |
| 12 | Scale-MAE | ViT-L MAE + GSD-PE | RGB (multi-scale) | MAE + Laplacian | 1024 | gh `bair-climate-initiative/scale-mae` | repo | var GSD | No |
| 13 | SpectralGPT | 3D GPT/MAE ~600M | S2 MS (spectral) | 3D masked recon | ~1024 | gh `danfenghong/SpectralGPT` | repo | var | No |
| 14 | Clay v1.5 | ViT MAE + dyn. embed | S2(10)✅ · S1✅(2) · RGB · Landsat · MODIS | MAE (+metadata) | **1024** | `made-with-clay/Clay` | Apache/OpenRAIL | 224 +meta | Partial (wavelength-cond.) |
| 15 | **RemoteCLIP** ⭐ | OpenCLIP RN50/B32/L14 | RGB only | Image-text contrastive | **1024/512/768** | `chendelong/RemoteCLIP` | repo | 224 CLIP | (text↔RGB) |
| 16 | GeoRSCLIP | CLIP B32 / H14 | RGB only | Contrastive (RS5M) | 512 / **1024** | `Zilun/GeoRSCLIP` | CC | 224 CLIP | (text↔RGB) |
| 17 | SkyCLIP | CLIP B32 / L14 | RGB only | Contrastive (SkyScript) | 512 / 768 | gh `wangzhecheng/SkyScript` | repo | 224 CLIP | (text↔RGB) |
| 18 | DOFA-CLIP | DOFA + text | multi-band + text | Contrastive | 768/1024 | emerging (`earthflow`) | check | 224 | YES (+text) |
| 19 | **SoftCon** ⭐ | ViT-S/14, ViT-B/14, RN50 | **S1✅(2)** · **S2✅(13)** | Soft contrastive | **384/768**/2048 | `wangyi111/softcon` | Apache-2.0 | 224 | Per-modality |
| 20 | SAR-JEPA | ViT JEPA | **SAR only** (1-ch) | JEPA (gradient pred) | — | gh `waterdisappear/SAR-JEPA` | repo | chips | No (SAR) |
| 21 | SSL4EO-S12 zoo | RN50 / ViT-S16 | S1✅(2) · S2✅(13) | MoCo/DINO/MAE/data2vec | 2048 / 384 | gh `zhu-xlab/SSL4EO-S12` | repo | 224 | Per-modality |
| 22 | **DINOv2(+reg)** ⭐ | ViT S/B/L/g p14 | RGB (3-ch) | Self-distill SSL | 384/768/1024/1536 | `facebook/dinov2-with-registers-base`/`-large` | Apache-2.0 | 224 | (generic) |
| 23 | DINOv3 / SAT | ViT B/L/7B p16 | RGB; **SAT=aerial RGB** | Self-distill | 768/1024/4096 | `facebook/dinov3-vit7b16-pretrain-sat493m` | DINOv3 (gated) | 256 | (generic) |
| 24 | **OpenCLIP** ⭐ | ViT-B32/L14/H14 | RGB (3-ch) | Image-text contrastive | **512/768/1024** | `laion/CLIP-ViT-B-32-laion2B-s34B-b79K` etc. | **MIT** | 224 CLIP | (generic) |
| 25 | SigLIP 2 | ViT b/l/so400m | RGB (3-ch) | Sigmoid contrastive | ~768/1152 | `google/siglip2-base-patch16-224` | Apache-2.0 | 224/NaFlex | (generic) |
| 26 | EVA-02-CLIP | EVA-02 ViT B/L/E | RGB (3-ch) | CLIP | ~512/768 | `QuanSun/EVA-CLIP` | repo | 224 | (generic) |
| 27 | Presto | tiny MAE (time-series) | S1 · S2 · ERA5 · DEM (pixel TS) | MAE | ~128 | gh `nasaharvest/presto` | repo | pixel-TS | Partial |

⭐ = recommended for this challenge. "repo" under License = permissive research license stated in the GitHub repo; verify the exact text before commercial use.

---

## Recommended backbone strategy for PS11

### (1) DEFAULT multimodal backbone (SAR + MS + RGB → one shared space)
**Primary choice: DOFA (ViT-B/16, 768-d).** HF id **`XShadow/DOFA`** (mirror `earthflow/DOFA`).
- **Why:** It is the *only* mature, public model where SAR, multispectral and RGB all pass through the **same weights** via wavelength-conditioned patch embedding, so their CLS embeddings are *already* in one 768-d space — exactly the "common representation space irrespective of sensor modality" the problem asks for. Band-agnostic = future-proof if hyperspectral/DEM appear. Apache-friendly CC-BY-4.0.
- **Load & embed:**
  ```python
  import torch
  model = torch.hub.load('zhu-xlab/DOFA', 'vit_base_dofa', pretrained=True).eval()
  # SAR (Sentinel-1 VV,VH): 2 bands
  z_sar = model.forward_features(s1_img,  wave_list=[5.405, 5.405])      # -> [B,768]
  # Multispectral (Sentinel-2 subset): pass each band's central wavelength (µm)
  z_ms  = model.forward_features(s2_img,  wave_list=[0.49,0.56,0.665,0.705,0.74,0.783,0.842,1.61,2.19])
  # RGB: 3 bands
  z_rgb = model.forward_features(rgb_img, wave_list=[0.665,0.56,0.49])
  z = torch.nn.functional.normalize(z_sar.mean(1) if z_sar.ndim==3 else z_sar, dim=-1)  # pool->L2
  ```
  (If `forward_features` returns patch tokens `[B,N,768]`, mean-pool over N; if it returns the pooled CLS, use directly. Always L2-normalize before FAISS.)

**Strong alternative / co-default if data is purely Sentinel-1+2: CROMA (768-d, MIT).** HF **`antofuller/CROMA`**. Use `joint_GAP` (fused) for cross-modal, `SAR_GAP`/`optical_GAP` for same-modal. CROMA's contrastive objective explicitly aligns radar↔optical, so cross-modal retrieval often works *better out-of-the-box* than DOFA — at the cost of fixed bands (S1=2, S2=12, no RGB-only). **Recommendation:** if the official dataset is S1/S2 pairs, prototype with **both DOFA and CROMA** and keep whichever scores higher on the cross-modal F1; CROMA's MIT license is a bonus.

> Both DOFA-B and CROMA-B output **768-d** → consistent FAISS index width. Use a small trainable **projection head (768→256, L2-norm)** with a contrastive/InfoNCE loss on paired SAR–optical samples to tighten the shared space (allowed by the rules and typically the single biggest cross-modal F1 win).

### (2) RGB / optical specialist (same-modal optical-to-optical)
**RemoteCLIP ViT-L/14 (768-d).** HF **`chendelong/RemoteCLIP`** (`ViT-L-14`).
- SOTA on RS image retrieval; OpenCLIP-compatible. For optical/RGB queries against an optical gallery this will beat the generic multimodal backbone.
- **Load:** `open_clip.create_model('ViT-L-14'); load chendelong/RemoteCLIP ckpt; feat = model.encode_image(x); feat = F.normalize(feat)`.
- Bonus: same 768-d as DOFA-B/CROMA-B → can share the index width. Alternative optical specialists: GeoRSCLIP-H14 (1024-d) or DINOv2-L (1024-d, no text) if you prefer SSL features.

### (3) Robust always-works fallback
**OpenCLIP ViT-B/32 (512-d), HF `laion/CLIP-ViT-B-32-laion2B-s34B-b79K` — MIT.** Tiny, reliably downloadable/cacheable, and *interface-identical* to RemoteCLIP/GeoRSCLIP/SkyCLIP so you can swap weights without code changes.
**Secondary fallback: DINOv2 ViT-B/14 (768-d), HF `facebook/dinov2-with-registers-base` — Apache-2.0** for the best generic *frozen* image-retrieval features (no text head). If even HF is unreachable, **`timm` ImageNet ViT-B/16** (768-d) is the last-resort baseline.

### (4) Per-modality input handling
| Modality | Bands | Feed to DOFA | Feed to CROMA | Feed to SoftCon | Feed to CLIP-family / DINOv2 |
|--|--|--|--|--|--|
| **SAR (Sentinel-1)** | VV, VH (2) | `wave_list=[5.405,5.405]`, 2-ch tensor | `modality='SAR'`, 2-ch (VV,VH) | S1 ckpt, 2-ch | make 3-ch: **[VV, VH, VV/VH ratio]** (or VV,VH,(VV+VH)/2), then CLIP/ImageNet norm |
| **Multispectral (Sentinel-2)** | up to 12–13 | pass per-band central λ in µm; any band subset | drop B10 cirrus → **12-ch**; `modality='optical'` | S2 "B13" ckpt, 13-ch | pick **RGB = B4,B3,B2**, CLIP/ImageNet norm |
| **Optical RGB** | R,G,B (3) | `wave_list=[0.665,0.56,0.49]` | (n/a — use RGB→optical adapter or use DOFA/CLIP) | (n/a) | native 3-ch, CLIP/ImageNet norm |
| **DEM / hyperspectral (optional)** | varies | DOFA: pass λ / single channel | — | — | reduce to 3-ch or skip |

**Normalization rules of thumb:**
- DOFA / CROMA / SoftCon / DeCUR / SSL4EO models → **SSL4EO-S12 per-channel mean/std**, clip to mean±2σ then scale to [0,1] (or [0,255] for SoftCon). SAR is in dB; clip typical S1 GRD range before standardizing.
- CLIP-family (RemoteCLIP/GeoRSCLIP/SkyCLIP/OpenCLIP) → **CLIP mean `[0.4815,0.4578,0.4082]`, std `[0.2686,0.2613,0.2758]`**, 224×224.
- DINOv2 → ImageNet mean/std; DINOv3-SAT → satellite-specific mean/std (per release).

### Embedding dimensions → FAISS / index design
- **Primary index width = 768** (DOFA-B, CROMA-B, SoftCon-B, DINOv2-B, RemoteCLIP-ViT-L all 768). Build `IndexFlatIP` on **L2-normalized** 768-d vectors (inner product = cosine) for exact top-k; switch to `IndexHNSWFlat` (M≈32) or `IVF-PQ` for large galleries / low latency.
- If you also use 512-d (OpenCLIP-B32 / CLIP-B32 RS models) or 1024-d (CROMA-L, DINOv2-L, GeoRSCLIP-H14, Prithvi, *MAE family), keep **one FAISS index per embedding width**, or unify everything to a common width with the **trainable projection head (→256-d)** — this also gives the cleanest single shared cross-modal space and the fastest search.
- **Plan A (simplest, strong):** DOFA-B (768) frozen → L2-norm → FlatIP. Same-modal & cross-modal from one index.
- **Plan B (best accuracy):** DOFA-B *or* CROMA-B backbone → trainable projection head (768→256, InfoNCE on SAR–optical pairs) → 256-d FlatIP/HNSW. Add RemoteCLIP-L for the optical-only sub-task and late-fuse scores.

---

## Key caveats / verification notes
- **msGFM weights are NOT released** ("To be released" in the official repo) — do not plan around it.
- **DOFA exact embedding dim (768/1024) and CROMA (768/1024)** confirmed from the papers/READMEs; for CROMA always use the **`*_GAP`** pooled outputs, not raw patch tokens, for retrieval.
- **TerraMind** base/large pool to ~768-d patch tokens (saw `B×196×768`); strongest but heaviest dependency (TerraTorch).
- **Galileo / AnySat** embedding dims are config-dependent — read the loaded model's hidden size; both are great for true S1/S2/aerial but more bespoke I/O than DOFA/CROMA.
- **DINOv3-SAT** is licence-gated (DINOv3 license) and uses non-ImageNet RGB normalization — verify before commercial use.
- CLIP-family RS models are **RGB-only**; they do **not** ingest raw 12-band MS or 2-band SAR — that's why the multimodal default (DOFA/CROMA) is essential for the SAR and MS legs of PS11.

---

## Sources
- CROMA — GitHub `antofuller/CROMA`; HF `antofuller/CROMA`; arXiv 2311.00566 (NeurIPS'23).
- DOFA — HF `earthflow/DOFA` & `XShadow/DOFA`; GitHub `zhu-xlab/DOFA`; arXiv 2403.15356; DOFA-CLIP arXiv 2503.06312.
- DeCUR — GitHub `zhu-xlab/DeCUR`; HF `wangyi111/DeCUR`; arXiv 2309.05300 (ECCV'24).
- Galileo — GitHub `nasaharvest/galileo`; HF `nasaharvest/galileo`; ICML'25.
- TerraMind — HF `ibm-esa-geospatial/TerraMind-1.0-{tiny,small,base,large}`; GitHub `IBM/terramind`; arXiv 2504.11171 (ICCV'25).
- AnySat — HF `g-astruc/AnySat`; GitHub `gastruc/AnySat`; arXiv 2412.14123 (CVPR'25).
- msGFM — GitHub `boranhan/Geospatial_Foundation_Models`; arXiv 2404.01260 (CVPR'24) — *weights not released.*
- USat — GitHub `stanfordmlgroup/USat`; arXiv 2312.02199.
- Prithvi-EO-2.0 — HF `ibm-nasa-geospatial/Prithvi-EO-2.0-{300M,600M}(-TL)`; GitHub `NASA-IMPACT/Prithvi-EO-2.0`; arXiv 2412.02732.
- SatMAE — GitHub `sustainlab-group/SatMAE`; arXiv 2207.08051. SatMAE++ — GitHub `techmn/satmae_pp`; arXiv 2403.05419. Scale-MAE — GitHub `bair-climate-initiative/scale-mae`; arXiv 2212.14532.
- SpectralGPT — arXiv 2311.07113 (TPAMI'24); repo `danfenghong/SpectralGPT`.
- Clay — HF `made-with-clay/Clay`; GitHub `Clay-foundation/model`; docs clay-foundation.github.io.
- RemoteCLIP — HF `chendelong/RemoteCLIP`; GitHub `ChenDelong1999/RemoteCLIP`; arXiv 2306.11029. GeoRSCLIP/RS5M — HF `Zilun/GeoRSCLIP`, `Zilun/RS5M`; arXiv 2306.11300. SkyCLIP/SkyScript — GitHub `wangzhecheng/SkyScript`; arXiv 2312.12856.
- SoftCon — HF `wangyi111/softcon`; GitHub `zhu-xlab/softcon`; arXiv 2405.20462. SSL4EO-S12 — GitHub `zhu-xlab/SSL4EO-S12`, `DLR-MF-DAS/SSL4EO-S12-v1.1`; arXiv 2211.07044 / 2503.00168. SAR-JEPA — GitHub `waterdisappear/SAR-JEPA`; arXiv 2311.15153.
- DINOv2 — HF `facebook/dinov2-*`, `facebook/dinov2-with-registers-*`. DINOv3 — HF `facebook/dinov3-*`, satellite `facebook/dinov3-vit7b16-pretrain-sat493m`, `timm/vit_7b_patch16_dinov3.sat493m`; GitHub `facebookresearch/dinov3`.
- OpenCLIP — GitHub `mlfoundations/open_clip`; HF `laion/CLIP-ViT-{B-32,L-14,H-14}-laion2B-*`. SigLIP 2 — HF `google/siglip2-*`. EVA-02-CLIP — HF `QuanSun/EVA-CLIP`. Presto — GitHub `nasaharvest/presto`; arXiv 2304.14065.
