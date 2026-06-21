# BAH 2026 PS-11 — Multi-Sensor Remote-Sensing Datasets for Cross-Modal Image Retrieval

**Problem:** Cross-Modal Satellite Image Retrieval Using Multi-Sensor Remote Sensing Data.
Build a retrieval system supporting **same-modal** (optical↔optical, SAR↔SAR, MS↔MS) and **cross-modal** (optical↔SAR, optical↔MS, SAR↔optical) retrieval, scored by **F1@5 / F1@10 (same- & cross-modal)** plus average retrieval time. Relevance is judged by **semantic class / geographic correspondence / predefined relevance labels**.

This document catalogues 30+ usable datasets, then gives concrete recommendations (primary + secondary), an evaluation protocol matched to the scoring, and a minimal prototyping subset.

---

## 0. What "good" looks like for this problem

Two distinct properties matter, and **different datasets give you different ones**:

1. **Pixel-aligned / co-registered multi-modal pairs** (same geo-location, same pixel grid across modalities). This is what makes *cross-modal* retrieval learnable via contrastive/metric learning (positive pair = the same place seen by two sensors). Critical datasets: SEN12MS, BigEarthNet-MM, So2Sat, SEN1-2, SEN12MS-CR(-TS), DFC2020, QXS-SAROPT, SARptical, WHU-OPT-SAR, SpaceNet 6, Major TOM (S1+S2+DEM), MMEarth, SSL4EO-S12, Copernicus-Pretrain, 3MOS, MRSSC.
2. **Clean semantic class labels** (for relevance-by-class scoring and same-modal retrieval quality). Strongest: EuroSAT, NWPU-RESISC45, AID, PatternNet, UC Merced, Million-AID, WHU-RS19, RSI-CB, BigEarthNet (multi-label CLC), So2Sat (LCZ), SEN12MS (IGBP).

The winning data strategy is to **cross-fill**: take ONE dataset that gives co-registered SAR+MS+labels (SEN12MS or BigEarthNet-MM) as the backbone, then borrow *clean RGB scene diversity* (NWPU-RESISC45 / AID / PatternNet) and *higher-resolution SAR-optical pairs* (QXS-SAROPT / SpaceNet 6) to verify generalization.

---

## 1. Comparison Table (all catalogued datasets)

Legend — Modalities: **O**=optical RGB, **MS**=multispectral, **HS**=hyperspectral, **SAR**, **DEM**, **LiDAR**, **TXT**=text captions. Align: **PA**=pixel-aligned/co-registered, **SA**=semantically associated only, **N/A**=single-modality.

| # | Dataset | Modalities | Sensors / Sources | #Images / Patches | Res. / Patch size | #Classes / Label type | Align | Size | License | Link |
|---|---------|-----------|-------------------|-------------------|-------------------|-----------------------|-------|------|---------|------|
| 1 | **SEN12MS** | SAR + MS + (RGB derivable) + landcover | Sentinel-1 (VV/VH), Sentinel-2 (13b), MODIS LC | 180,662 triplets | 10 m / 256×256 | IGBP 17→10 (simplified) land cover | **PA** | ~430 GB | CC-BY (TUM mediaTUM, nonexclusive-distrib) | mediatum.ub.tum.de/1474000 ; torchgeo `SEN12MS` |
| 2 | **BigEarthNet-MM (v1.0)** | SAR + MS | Sentinel-1 (VV/VH), Sentinel-2 (12b) | 590,326 pairs | 10/20/60 m / 120×120 (S2 10m) | 19 (also 43) CLC2018 multi-label | **PA** | ~120 GB | CDLA-Permissive 1.0 | bigearth.net ; torchgeo `BigEarthNet` |
| 3 | **BigEarthNet v2.0 (reBEN)** | SAR + MS | Sentinel-1, Sentinel-2 | 549,488 pairs | 10 m / 120×120 | 19 CLC2018 multi-label (+ ref maps) | **PA** | ~110 GB (S2 59 + S1 51 GiB) | CDLA-Permissive 1.0 | zenodo.org/records/10891137 |
| 4 | **So2Sat LCZ42** | SAR + MS | Sentinel-1 (8ch), Sentinel-2 (10ch) | 400,673 patches | 10 m / 32×32 | 17 Local Climate Zones | **PA** | ~55 GB | CC-BY-4.0 | mediatum.ub.tum.de/1454690 ; TFDS `so2sat`; torchgeo `So2Sat` |
| 5 | **SEN1-2 (SEN12)** | SAR + O(RGB) | Sentinel-1 (VV), Sentinel-2 (RGB) | 282,384 pairs | 10 m / 256×256 | none (pairs only) | **PA** | ~32 GB | CC-BY | mediatum.ub.tum.de/1436631 |
| 6 | **SEN12MS-CR** | SAR + MS (cloudy+clear) | Sentinel-1, Sentinel-2 | ~122,218 triplets | 10 m / 256×256 | none (cloud-removal) | **PA** | ~270 GB | CC-BY | patricktum.github.io/cloud_removal |
| 7 | **SEN12MS-CR-TS** | SAR + MS time-series | Sentinel-1, Sentinel-2 | ~15,000 TS ×30 steps | 10 m / 256×256 | none (multi-temporal) | **PA** | ~2 TB | CC-BY | patricktum.github.io/cloud_removal/sen12mscrts |
| 8 | **IEEE GRSS DFC2020** | SAR + MS + landcover | Sentinel-1, Sentinel-2, ref LC | 6,114 + val/test | 10 m / 256×256 | IGBP 10-class (high-res 10m val) | **PA** | ~25 GB | research (GRSS) | ieee-dataport / DFC2020; mirrors SEN12MS scheme |
| 9 | **IEEE GRSS DFC2023** | O(RGB) + SAR + DEM | SuperView-1, Gaofen-2 (RGB), Gaofen-3 (SAR) | ~3,000 tiles | 0.5–1 m | building roof type + height | **PA** | ~50 GB | research (GRSS) | ieee-dataport competitions/2023-…-fine-grained-building |
| 10 | **OSCD (Onera)** | MS bitemporal | Sentinel-2 (13b) | 24 pairs (14 labelled) | 10/20/60 m / varies | binary change | **PA** (temporal) | ~1 GB | research | ieee-dataport ; torchgeo `OSCD`; HF `blanchon/OSCD_MSI` |
| 11 | **WHU-OPT-SAR** | O(RGB+NIR) + SAR + landcover | GaoFen-1 (opt), GaoFen-3 (SAR) | 100 pairs (5556×3704) | 5 m | 7 land-cover (pixel) | **PA** | ~25 GB | research | github.com/AmberHen/WHU-OPT-SAR-dataset |
| 12 | **QXS-SAROPT** | SAR + O(RGB) | GaoFen-3 (SAR), Google Earth (opt) | 20,000 pairs | 1 m / 256×256 | none (matching) | **PA** | ~2 GB | research (cite paper) | github.com/yaoxu008/QXS-SAROPT |
| 13 | **SARptical** | SAR + O(RGB) (+3D) | TerraSAR-X VHR spotlight, UltraCAM aerial | ~10,000 pairs | VHR / 112×112 | none (3D-matched) | **PA** (3D) | ~3 GB | research | github.com/sarptical / TUM (Wang & Zhu) |
| 14 | **SpaceNet 6** | SAR + O(RGB/MS) + building | Capella SAR (X-band quad-pol), Maxar WV-2 | 202 strips → ~3,400 tiles | 0.5 m SAR / 450×450 m | building footprints (+height) | **PA** | ~100 GB (39 GB train) | CC-BY-SA-4.0 | spacenet.ai/sn6-challenge ; AWS s3://spacenet-dataset |
| 15 | **3MOS** | O + SAR multi-source | GF-3, ALOS, Sentinel-1, RadarSat, RCM + opt | ~155,000 pairs | 1.25–12.5 m | 8 scene types (urban…frozen) | **PA** | ~50 GB | research | github.com/3M-OS/3MOS |
| 16 | **MRSSC** | O(VIS/SWIR) + SAR | Tiangong-2 (WIS + InIRA) | 26,710 images | space-borne | 7 scene classes, 4 domains | **PA** (sim. acq.) | ~10 GB | research | ISPRS-Archives XLIII-B2-2021/785 |
| 17 | **EuroSAT** | MS (+RGB subset) | Sentinel-2 (13b) | 27,000 | 10 m / 64×64 | 10 land-use | N/A (single) | 2 GB (allBands) / 90 MB RGB | MIT | madm.dfki.de EuroSATallBands.zip ; torchgeo `EuroSAT`; HF |
| 18 | **NWPU-RESISC45** | O(RGB) | Google Earth (multi-sensor) | 31,500 | 0.2–30 m / 256×256 | 45 scene | N/A | ~0.4 GB | research (non-commercial) | gcheng-nwpu / HF `timm/resisc45`; torchgeo `RESISC45` |
| 19 | **AID** | O(RGB) | Google Earth | 10,000 | 0.5–8 m / 600×600 | 30 scene | N/A | ~2.6 GB | research | captain-whu AID ; HF `blanchon/AID` |
| 20 | **PatternNet** | O(RGB) | Google Earth / Maps API | 30,400 | 0.06–4.7 m / 256×256 | 38 scene (**retrieval benchmark**) | N/A | ~1.4 GB | research | sites.google.com/view/zhouwx/dataset ; torchgeo `PatternNet` |
| 21 | **UC Merced (UCM)** | O(RGB) | USGS aerial | 2,100 | 0.3 m / 256×256 | 21 land-use | N/A | ~0.3 GB | public domain (USGS) | weegee.vision.ucmerced.edu ; torchgeo `UCMerced` |
| 22 | **WHU-RS19** | O(RGB) | Google Earth | ~1,005 | ≤0.5 m / 600×600 | 19 scene | N/A | ~0.1 GB | research | HF `jonathan-roberts1/WHU-RS19` |
| 23 | **RSI-CB256** | O(RGB) | Google Earth + OSM crowdsource | ~24,000 | 0.3–3 m / 256×256 | 35 (6 super) | N/A | ~3 GB | research | github.com/lehaifeng/RSI-CB |
| 24 | **Million-AID** | O(RGB) | Google Earth (SPOT/IKONOS/WV/Landsat) | 1,000,848 | 0.5–153 m / 256² & 512² | 51-leaf / 8-top hierarchy | N/A | ~80 GB | research | captain-whu.github.io ; jin-pu.github.io/Million-AID |
| 25 | **fMoW** | O(RGB+MS) + metadata | DigitalGlobe/Maxar (4/8-band) | >1,000,000 | VHR / variable | 63 functional categories | N/A | ~3.5 TB (full) / ~200 GB rgb | fMoW license (research) | github.com/fMoW ; AWS s3://spacenet-dataset/Hosted-Datasets/fmow |
| 26 | **fMoW-Sentinel** | MS | Sentinel-2 (13b) | 882,779 | 10 m / variable | 62 functional categories | N/A (paired to fMoW geo) | ~80 GB | CC-BY-4.0 | purl.stanford.edu/vg497cb6002 |
| 27 | **SSL4EO-S12 (v1.0/1.1)** | SAR + MS (+DEM/LC v1.1) | Sentinel-1, Sentinel-2 (L1C+L2A) | ~1M patches (251k loc ×4 seas) | 10 m / 264×264 | none (SSL pretrain) | **PA** | ~500 GB (v1.0) | CC-BY-4.0 | HF `embed2scale/SSL4EO-S12-v1.1`; github zhu-xlab/SSL4EO-S12 |
| 28 | **MMEarth** | MS+SAR+DEM+LC+climate (12 mod, 6 pixel) | Sentinel-2/1, ASTER DEM, Dynamic World, ESA WorldCover… | 1.2M locations | 10 m / 128×128 (& 64²) | aligned multi-modal (SSL) | **PA** | ~600 GB (100k & 64 subsets) | CC-BY-4.0 | vishalned.github.io/mmearth |
| 29 | **SatlasPretrain** | O(RGB) + MS | Sentinel-2, NAIP (+Landsat) | ~856k S2 + 11.5M NAIP | 10 m / 1 m | 137 cat, 7 label types, 302M labels | SA (multi-task labels) | ~10 TB | ODC-BY | github.com/allenai/satlas ; HF `allenai/satlas-pretrain` |
| 30 | **Major TOM Core (S2L2A/S2L1C/S1RTC/DEM)** | O/MS + SAR + DEM | Sentinel-2 L1C/L2A, Sentinel-1 RTC, Copernicus DEM | ~60M+ patches (multi-TB) | 10 m / 1068×1068 | none (global grid) | **PA** (shared grid) | 50+ TB | CC-BY-SA-4.0 | huggingface.co/Major-TOM (Core-S2L2A, Core-S1RTC, Core-DEM) |
| 31 | **Copernicus-Pretrain** | S1+S2+S3+S5P+DEM (8 mod) | All Sentinel missions + Copernicus DEM | 18.7M images / ~310k grids | multi-res / grid | none (SSL, 0.25° grids) | **PA** (per grid) | multi-TB | CC-BY (see card) | HF `wangyi111/Copernicus-Pretrain` |
| 32 | **LoveDA** | O(RGB) + landcover | Google Earth (Spaceborne) | 5,987 (1024×1024) | 0.3 m | 7 land-cover (segmentation) | N/A | ~3 GB | CC-BY-NC-SA-4.0 | zenodo 5706578 ; torchgeo `LoveDA` |
| 33 | **OpenEarthMap** | O(RGB) + landcover | aerial + satellite (global) | 5,000 (1024×1024) | 0.25–0.5 m | 8 land-cover | N/A | ~7 GB | CC-BY-NC-SA-4.0 (some sources) | open-earth-map.org |
| 34 | **DOTA (v1/1.5/2.0)** | O(RGB) | Google Earth, GF-2, JL-1 | 2,806+ large images | varied / oriented | 15–18 object cat (detection) | N/A | ~30 GB | research | captain-whu.github.io/DOTA |
| 35 | **Houston2013 (DFC2013)** | HS + LiDAR | ITRES CASI HS + LiDAR | 1 scene 349×1905 | 2.5 m / 144 bands | 15 classes | **PA** | small | research | hyperspectral.ee.uh.edu |
| 36 | **Houston2018 (DFC2018)** | HS + MS-LiDAR + O(RGB) | CASI HS (48b), MS-LiDAR (7ch), VHR RGB (3ch) | 1 scene 1202×4172 | 0.5–1 m / 48 bands | 20 classes | **PA** | small | research | machinelearning.ee.uh.edu/2018-…-data-fusion |
| 37 | **RSICD** | O(RGB) + TXT | Google Earth/Baidu/MapABC/Tianditu | 10,921 imgs / 54,605 caps | varied / 224×224 | 5 captions/img (image–text) | SA (img-text) | ~1 GB | research | github.com/201528014227051/RSICD_optimal |
| 38 | **RSITMD** | O(RGB) + TXT | from RSICD/AID classes | 4,743 imgs / 23,715 caps | varied / 256×256 | fine-grained captions | SA (img-text) | ~0.4 GB | research | github.com/xiaoyuan1996/AMFMN |
| 39 | **UCM-Captions** | O(RGB) + TXT | from UC Merced | 2,100 imgs / 10,500 caps | 0.3 m / 256×256 | 21 cls + captions | SA (img-text) | small | research | github.com (Qu et al. 2016) |
| 40 | **Sydney-Captions** | O(RGB) + TXT | from Sydney scene set | 613 imgs / 3,065 caps | varied | 7 cls + captions | SA (img-text) | small | research | github.com (Qu et al. 2016) |
| 41 | **NWPU-Captions** | O(RGB) + TXT | from NWPU-RESISC45 | 31,500 imgs / 157,500 caps | 0.2–30 m / 256² | 45 cls + captions | SA (img-text) | ~0.5 GB | research | github.com/HaiyanHuang98/NWPU-Captions |

> Sizes are approximate "order-of-magnitude" figures aggregated from dataset cards / papers; verify exact bytes on the download page before provisioning storage.

---

## 2. Per-dataset notes & retrieval suitability

### Tier A — Co-registered SAR + MS/optical WITH usable labels (best for *both* same- & cross-modal)

- **SEN12MS** — 180,662 *triplets* of Sentinel-1 dual-pol SAR (VV/VH), Sentinel-2 13-band MS, and a MODIS-derived land-cover map, all at 10 m GSD, 256×256, fully georeferenced, all seasons, all inhabited continents. Labels follow the **simplified IGBP scheme (17→10 classes)** (the exact scheme used by DFC2020). Modalities are **pixel-aligned**. From S2 you derive an RGB modality (B4/B3/B2) for free, giving you **SAR + MS + RGB + class label in one co-registered package** — exactly the tri-modal setup PS-11 wants. *Suitability:* the single best **primary** dataset; supports SAR↔SAR, MS↔MS, RGB↔RGB (same-modal) and SAR↔optical, optical↔MS (cross-modal) with class-based relevance out-of-the-box. In **torchgeo** as `SEN12MS`.

- **BigEarthNet-MM / v2.0 (reBEN)** — ~550k Sentinel-1 + Sentinel-2 **pixel-aligned pairs**, 120×120 (S2 10 m), each with **19-class CLC2018 multi-label** annotations. Far cleaner, larger, and more class-rich than SEN12MS, *but* it is **S1+S2 only (no separate VHR RGB)** and labels are *multi-label* (a patch can be "urban+water+forest"), which complicates strict relevance-by-single-class. *Suitability:* outstanding for SAR↔MS cross-modal contrastive training and multi-label retrieval; pair with a single-label dataset for clean F1 scoring. In **torchgeo** as `BigEarthNet`. v2 download on Zenodo; CDLA-Permissive license = commercial-friendly.

- **So2Sat LCZ42** — ~400k **co-registered** S1 (8ch incl. real+imaginary) + S2 (10ch) patches, 32×32, expert-labelled into **17 Local Climate Zones**, 42+ cities globally. *Suitability:* high-quality single-label cross-modal pairs (SAR↔MS) with strong class purity; small patch size limits texture but ideal for fast retrieval prototyping and clean F1. In **torchgeo** as `So2Sat`, TFDS as `so2sat`.

- **IEEE GRSS DFC2020** — a curated subset/extension of SEN12MS with **high-resolution (10 m) semi-manual land-cover labels** on val/test (vs 500 m MODIS in train). Same S1+S2 co-registered format, 10-class IGBP. *Suitability:* excellent ready-made **held-out evaluation set** with trustworthy labels for cross-modal F1 — use it as the official test split.

### Tier B — High-resolution SAR-optical pairs (best for cross-modal SAR↔optical generalization, weak/no labels)

- **QXS-SAROPT** — 20,000 **1 m co-registered** GaoFen-3 SAR ↔ Google-Earth optical patches (256×256) over San Diego/Shanghai/Qingdao. First high-res co-registered SAR-optical set. No class labels (matching dataset). *Suitability:* strong **cross-modal SAR↔optical** fine-tuning / hard-negative source; relevance must come from geographic pair identity, not class.

- **SpaceNet 6** — Capella **X-band quad-pol SAR** + Maxar WorldView-2 optical over Rotterdam, co-registered tiles with building footprints (+height). *Suitability:* realistic VHR SAR↔optical cross-modal pairs; building-density can act as weak semantic signal. CC-BY-SA. AWS-hosted.

- **SARptical** — ~10k TerraSAR-X VHR ↔ aerial optical patches in Berlin, uniquely **3D-matched** at the center pixel. *Suitability:* hardest/most-accurate urban SAR↔optical correspondences; great stress-test for cross-modal robustness.

- **WHU-OPT-SAR** — 100 large GF-1 optical(RGB+NIR) ↔ GF-3 SAR pairs at 5 m with **7-class pixel land-cover**. Tile them into patches → co-registered SAR↔optical with class labels. *Suitability:* adds a labelled, higher-resolution-than-Sentinel SAR↔optical source for cross-modal verification.

- **SEN1-2** — 282,384 Sentinel-1(VV)↔Sentinel-2(RGB) **co-registered** pairs, no labels. *Suitability:* large pretraining pool for SAR↔RGB alignment; relevance = pair identity. Predecessor of SEN12MS.

- **3MOS** — 155k optical↔SAR pairs from **6 different SAR satellites** (GF-3, ALOS, Sentinel-1, RadarSat, RCM) and 8 scene types. *Suitability:* the best resource to test **cross-sensor / cross-resolution generalization** of a SAR↔optical retriever.

### Tier C — Cloud-removal / time-series multi-modal (auxiliary, co-registered)

- **SEN12MS-CR** and **SEN12MS-CR-TS** — co-registered S1 + (cloudy & clear) S2, single- and multi-temporal. *Suitability:* useful for robustness to clouds and for self-supervised cross-modal pretext tasks; not class-labelled.

### Tier D — Large-scale multi-modal SSL / foundation corpora (pretraining, mostly unlabeled)

- **SSL4EO-S12 (v1.0/1.1)** — ~1M co-registered S1+S2 patches (264×264) across 4 seasons; v1.1 adds DEM/land-cover/vegetation. CC-BY-4.0, HF-hosted. *Suitability:* premier **self-supervised pretraining** corpus to learn a shared SAR-MS encoder before fine-tuning on labelled retrieval data.
- **MMEarth** — 1.2M locations, **12 pixel-/image-level modalities all reprojected to the S2 10 m grid** (S2, S1, DEM, Dynamic World, ESA WorldCover, climate). *Suitability:* richest co-registered multi-modal pretraining; the 100k/64-px subsets are practical.
- **Major TOM Core** — global, ML-ready, **shared-grid** Sentinel-2 (L1C/L2A) + Sentinel-1 RTC + Copernicus DEM. Multi-TB, HF/Parquet. *Suitability:* near-global gallery + pretraining; pull a regional subset.
- **Copernicus-Pretrain / Copernicus-FM** — 18.7M images over ~310k 0.25° grids spanning **S1, S2, S3, S5P, DEM** (8 modalities). *Suitability:* widest sensor variety for foundation-model features.
- **SatlasPretrain** — Sentinel-2 + NAIP with 302M labels / 137 categories / 7 label types. ODC-BY. *Suitability:* strong supervised pretraining for optical features (no SAR).
- **fMoW / fMoW-Sentinel** — >1M VHR optical (fMoW) and 882,779 Sentinel-2 MS (fMoW-Sentinel) over the same 62–63 functional categories. *Suitability:* RGB↔MS cross-source retrieval and functional-class relevance at scale.

### Tier E — Optical-only scene-classification sets (clean labels for same-modal optical & relevance scoring)

- **PatternNet** (30,400 / 38 cls) — *purpose-built remote-sensing image-retrieval benchmark*; highest images-per-class, ideal for clean **optical↔optical F1@K** and as a relevance reference.
- **NWPU-RESISC45** (31,500 / 45 cls), **AID** (10,000 / 30 cls), **Million-AID** (1M / hierarchical), **UC Merced** (2,100 / 21 cls), **WHU-RS19** (~1k / 19 cls), **RSI-CB256** (24k / 35 cls) — multi-sensor Google-Earth RGB; provide **scene-diversity and clean single labels** to (a) train/evaluate optical-to-optical retrieval and (b) define a consistent semantic class taxonomy for relevance. *Note:* these are semantically associated to other modalities only via shared class names, not pixel-aligned.

### Tier F — Land-cover segmentation / detection (auxiliary class signal, optical)

- **LoveDA** (7 cls), **OpenEarthMap** (8 cls), **DOTA** (object detection) — high-res optical with dense labels; useful to derive scene-level class tags or domain-adaptation tests. LoveDA/OpenEarthMap are **CC-BY-NC** (non-commercial) — watch licensing.

### Tier G — Hyperspectral & HS-LiDAR fusion (covers the HS modality, pixel-aligned but local)

- **Houston2013** (144-band HS + LiDAR, 15 cls), **Houston2018** (48-band HS + MS-LiDAR + RGB, 20 cls) — **co-registered** HS/LiDAR/optical over one urban scene. *Suitability:* the practical way to include a **hyperspectral** modality and HS↔optical cross-modal experiments; patch-extract for retrieval. PRISMA/EnMAP scenes (240+ bands, open-access via ASI/DLR portals) can supply additional HS if needed, but require manual labelling.

### Tier H — Image–text (cross-modal anchors; optional but powerful)

- **RSICD, RSITMD, UCM-Captions, Sydney-Captions, NWPU-Captions** — RS image↔caption pairs. *Suitability:* not sensor-multimodal, but text is a **modality bridge**: a CLIP-style text encoder can align optical and SAR through shared captions/class names, and these sets let you add text→image retrieval if the brief is interpreted broadly.

---

## 3. Recommendations

### (1) PRIMARY dataset — **SEN12MS** (+ DFC2020 as its labelled test split)

**Why SEN12MS is the single best primary choice:**
- It is the only large, global dataset that delivers **all three target modalities co-registered at the pixel level in one sample**: Sentinel-1 SAR (VV/VH), Sentinel-2 **multispectral** (13 bands), and a derivable **RGB** (B4/B3/B2) — plus a **land-cover label** (simplified IGBP, 10 classes). That covers every PS-11 retrieval direction: optical↔optical, SAR↔SAR, MS↔MS, optical↔SAR, optical↔MS, SAR↔optical.
- Co-registration means a positive cross-modal pair is unambiguous (the *same place* in SAR and MS), which is exactly the supervision signal contrastive/triplet/metric-learning needs to build a shared embedding space.
- 180k samples is large enough to train transformers/Siamese nets, yet tractable on a single multi-GPU box.
- **First-class tooling**: available in **torchgeo** (`torchgeo.datasets.SEN12MS`) with ready data splits; widely benchmarked, so results are comparable.
- **DFC2020** provides a curated held-out set in the *same format* with **high-resolution (10 m) semi-manual labels**, giving a trustworthy, standard **evaluation split** for cross-modal F1 (avoids the 500 m MODIS noise in SEN12MS train).

**Caveat & fix:** SEN12MS labels are MODIS-derived (coarse) on the training split → use DFC2020's high-res labels for the official query/gallery evaluation, and optionally clean training labels via the published high-res reference maps.

### (2) SECONDARY datasets — fill gaps & verify generalization (pick 2–3)

1. **BigEarthNet-MM / v2.0** — *gap filled:* class richness + scale for SAR↔MS. Use its 19-class CLC labels and ~550k pairs to (a) pretrain the shared SAR-MS encoder and (b) test multi-label retrieval. Verifies that the method scales beyond SEN12MS's coarse labels. (torchgeo `BigEarthNet`; CDLA-Permissive.)
2. **QXS-SAROPT** (and/or **SpaceNet 6**) — *gap filled:* **high-resolution (1 m) SAR↔optical** cross-modal pairs. Sentinel data is 10 m; QXS/SN6 prove the retriever generalizes to VHR sensors and different SAR bands (X-band). Relevance = geographic pair identity (hardest cross-modal case).
3. **PatternNet** (or **NWPU-RESISC45 / AID**) — *gap filled:* **clean single-label optical scene diversity** for rigorous optical↔optical F1@K and to define a consistent semantic taxonomy for relevance scoring. PatternNet is purpose-built for RS retrieval.

*(Optional 4th)* **So2Sat LCZ42** for clean single-label SAR↔MS pairs, or **3MOS** for cross-sensor SAR↔optical stress-testing, or **Houston2018** to demonstrate the hyperspectral modality.

### (3) Query/Gallery split + relevance-by-class evaluation protocol (matched to F1@5 / F1@10)

**Modality channels per sample (from SEN12MS):**
- `OPT_RGB` = S2 bands B4,B3,B2 (8-bit stretch)
- `MS` = S2 all 13 bands (or 10 at 10 m)
- `SAR` = S1 VV,VH (dB-scaled, speckle-filtered)
- `label` = IGBP simplified class (DFC2020 high-res labels for eval)

**Splits (geographically disjoint to prevent leakage):**
- **Train** = SEN12MS scenes (exclude any scene that overlaps DFC2020 test ROIs). Optionally add BigEarthNet-MM for pretraining.
- **Query/Gallery (evaluation)** = DFC2020 val/test (high-res labels). Build, per modality, a **query set** (e.g., 20% of eval patches, stratified by class) and a **gallery** (remaining 80%). Construct **same-modal galleries** (SAR-only, MS-only, RGB-only) and a **mixed cross-modal gallery** (all modalities pooled).

**Relevance definition:** a retrieved gallery item is **relevant to a query iff it shares the query's semantic class** (IGBP class). For multi-label datasets (BigEarthNet) use ≥1 shared label or Jaccard-overlap threshold. This matches PS-11's "semantic class / predefined relevance labels."

**Retrieval directions to evaluate (report each):**
- Same-modal: O→O, SAR→SAR, MS→MS.
- Cross-modal: O→SAR, SAR→O, O→MS, MS→O (cross-modal weighted higher per the brief).

**Metric computation (per query, then averaged):**
- For K ∈ {5, 10}: retrieve top-K by cosine similarity in the shared embedding space.
- `Precision@K = (#relevant in top-K)/K`.
- `Recall@K = (#relevant in top-K)/min(K, total_relevant_in_gallery)` (cap by K so a perfect ranker reaches F1=1; standard for F1@K in retrieval).
- `F1@K = 2·P·R/(P+R)`; report **F1@5 and F1@10** for **same-modal** and **cross-modal** separately (4 headline numbers), macro-averaged over classes to handle imbalance.
- **Average retrieval time/query**: build a FAISS index over gallery embeddings (`IndexFlatIP` for exact, or `IVF/HNSW/PQ` for speed); time = embed-query + ANN-search, averaged over all queries. Report both exact and ANN.

**Embedding pipeline:** shared backbone (e.g., a multi-modal ViT or modality-specific encoders projecting to a common space) trained with **contrastive/triplet loss on co-registered pairs** (positive = same location across modalities; class-aware hard negatives). L2-normalize → cosine = inner product → FAISS.

### (4) Smallest viable subset to prototype quickly

- **EuroSAT (RGB + allBands)** — 27,000 / 10 classes / 64×64, ~90 MB (RGB) or 2 GB (13-band). *Use it day 1* to validate the **same-modal optical↔optical and MS↔MS** retrieval + FAISS + F1@K harness end-to-end (single-label, tiny, in torchgeo). No SAR, so it only exercises the optical/MS path.
- Add **So2Sat LCZ42 subset** (a few cities, 32×32, single-label LCZ) — smallest **co-registered SAR↔MS** data to validate the **cross-modal** path and contrastive training quickly (~GB-scale).
- Then scale to a **SEN12MS regional subset** (e.g., one season / a few ROIs, a few thousand triplets) to confirm tri-modal (SAR+MS+RGB) retrieval before full training.

This ladder — **EuroSAT → So2Sat subset → SEN12MS subset → full SEN12MS+DFC2020 (+BigEarthNet-MM pretrain, +QXS/SN6 for VHR cross-modal)** — lets you stand up the scoring pipeline in hours and grow to a competitive system without re-architecting.

---

## 4. Cross-fill cheat-sheet (which dataset supplies which gap)

| Need | Best source(s) |
|------|----------------|
| Co-registered SAR+MS+RGB + class labels (core training) | **SEN12MS** |
| Trustworthy high-res labels for evaluation | **DFC2020** |
| Scale + class-rich SAR↔MS pretraining | **BigEarthNet-MM/v2**, **SSL4EO-S12**, **MMEarth** |
| Clean single-label SAR↔MS pairs | **So2Sat LCZ42** |
| High-resolution SAR↔optical cross-modal (generalization) | **QXS-SAROPT**, **SpaceNet 6**, **SARptical**, **WHU-OPT-SAR** |
| Cross-sensor / cross-resolution SAR↔optical | **3MOS** |
| Clean optical scene diversity + retrieval benchmark | **PatternNet**, **NWPU-RESISC45**, **AID**, **Million-AID** |
| Hyperspectral modality | **Houston2013/2018**, PRISMA/EnMAP |
| Text bridge (optional cross-modal anchor) | **RSICD/RSITMD/NWPU-Captions** |
| Tiny prototyping harness | **EuroSAT** (+ So2Sat subset) |

---

## 5. Key download links (quick reference)

- SEN12MS — https://mediatum.ub.tum.de/1474000 · torchgeo `SEN12MS` · repo https://github.com/schmitt-muc/SEN12MS
- BigEarthNet (v1/v2/MM) — https://bigearth.net/ · v2 Zenodo https://zenodo.org/records/10891137 · torchgeo `BigEarthNet`
- So2Sat LCZ42 — https://mediatum.ub.tum.de/1454690 · TFDS `so2sat` · torchgeo `So2Sat`
- SEN1-2 — https://mediatum.ub.tum.de/1436631
- SEN12MS-CR / -CR-TS — https://patricktum.github.io/cloud_removal/
- DFC2020 — https://ieee-dataport.org/ (2020 GRSS DFC) · mirrors SEN12MS format
- DFC2023 — https://ieee-dataport.org/competitions/2023-ieee-grss-data-fusion-contest-large-scale-fine-grained-building-classification
- OSCD — https://ieee-dataport.org/open-access/oscd-onera-satellite-change-detection · torchgeo `OSCD` · HF `blanchon/OSCD_MSI`
- WHU-OPT-SAR — https://github.com/AmberHen/WHU-OPT-SAR-dataset
- QXS-SAROPT — https://github.com/yaoxu008/QXS-SAROPT
- SARptical — TUM / https://github.com/sarptical (Wang & Zhu)
- SpaceNet 6 — https://spacenet.ai/sn6-challenge/ · `s3://spacenet-dataset/spacenet/SN6_buildings/`
- 3MOS — https://github.com/3M-OS/3MOS
- EuroSAT — https://madm.dfki.de/files/sentinel/EuroSATallBands.zip · torchgeo `EuroSAT` · HF
- NWPU-RESISC45 — torchgeo `RESISC45` · HF `timm/resisc45`
- AID — https://captain-whu.github.io/AID/ · HF `blanchon/AID`
- PatternNet — https://sites.google.com/view/zhouwx/dataset · torchgeo `PatternNet`
- UC Merced — http://weegee.vision.ucmerced.edu/datasets/landuse.html · torchgeo `UCMerced`
- WHU-RS19 — HF `jonathan-roberts1/WHU-RS19`
- RSI-CB — https://github.com/lehaifeng/RSI-CB
- Million-AID — https://captain-whu.github.io/DiRS/ · https://jin-pu.github.io/Million-AID/
- fMoW — https://github.com/fMoW · `s3://spacenet-dataset/Hosted-Datasets/fmow/`
- fMoW-Sentinel — https://purl.stanford.edu/vg497cb6002
- SSL4EO-S12 — https://huggingface.co/datasets/embed2scale/SSL4EO-S12-v1.1 · https://github.com/zhu-xlab/SSL4EO-S12
- MMEarth — https://vishalned.github.io/mmearth/
- SatlasPretrain — https://github.com/allenai/satlas · HF `allenai/satlas-pretrain`
- Major TOM — https://huggingface.co/Major-TOM (Core-S2L2A, Core-S2L1C, Core-S1RTC, Core-DEM)
- Copernicus-Pretrain — https://huggingface.co/datasets/wangyi111/Copernicus-Pretrain
- LoveDA — https://zenodo.org/records/5706578 · torchgeo `LoveDA`
- OpenEarthMap — https://open-earth-map.org/
- DOTA — https://captain-whu.github.io/DOTA/
- Houston2013/2018 — https://hyperspectral.ee.uh.edu/ · https://machinelearning.ee.uh.edu/2018-ieee-grss-data-fusion-challenge-fusion-of-multispectral-lidar-and-hyperspectral-data/
- RSICD — https://github.com/201528014227051/RSICD_optimal · RSITMD — https://github.com/xiaoyuan1996/AMFMN · NWPU-Captions — https://github.com/HaiyanHuang98/NWPU-Captions

---

## 6. Sources

- SEN12MS — arXiv:1906.07789 ; ISPRS Annals IV-2-W7 ; repo github.com/schmitt-muc/SEN12MS
- BigEarthNet-MM — arXiv:2105.07921 ; bigearth.net ; reBEN/v2 Zenodo 10891137
- So2Sat LCZ42 — arXiv:1912.12171 ; TFDS catalog `so2sat`
- SEN1-2 — arXiv:1807.01569 ; ISPRS Annals IV-1
- SEN12MS-CR-TS — arXiv:2201.09613 ; patricktum.github.io/cloud_removal
- DFC2020/2023 — grss-ieee.org IADF ; ieee-dataport ; arXiv:2104.00704 (SEN12MS classification)
- OSCD — arXiv:1810.… ; torchgeo docs ; ieee-dataport
- WHU-OPT-SAR — github.com/AmberHen/WHU-OPT-SAR-dataset ; MDPI RS 16(2):431
- QXS-SAROPT — arXiv:2103.08259 ; github.com/yaoxu008/QXS-SAROPT
- SARptical — arXiv:1801.07532
- SpaceNet 6 — arXiv (Shermeyer et al.) ; spacenet.ai/sn6-challenge
- 3MOS — arXiv:2404.00838 ; github.com/3M-OS/3MOS
- MRSSC — ISPRS Archives XLIII-B2-2021/785
- EuroSAT — torchgeo eurosat docs ; madm.dfki.de
- NWPU-RESISC45 / AID / UC Merced / PatternNet — arXiv:1706.03424 (PatternNet) ; captain-whu AID ; standard scene-class refs
- Million-AID — arXiv:2006.12485 ; captain-whu.github.io/DiRS
- fMoW / fMoW-Sentinel — arXiv:1711.07846 ; purl.stanford.edu/vg497cb6002 ; SatMAE (arXiv:2207.08051)
- SSL4EO-S12 — arXiv:2211.07044 ; v1.1 arXiv:2503.00168 ; HF embed2scale
- MMEarth — arXiv:2405.02771 ; vishalned.github.io/mmearth
- SatlasPretrain — arXiv:2211.15660 ; github.com/allenai/satlas
- Major TOM — arXiv:2402.12095 ; huggingface.co/Major-TOM ; ESA Φ-lab
- Copernicus-Pretrain/FM — arXiv:2503.11849 ; HF wangyi111/Copernicus-Pretrain
- LoveDA — Zenodo 5706578 ; OpenEarthMap — open-earth-map.org ; DOTA — captain-whu.github.io/DOTA
- Houston2013/2018 — hyperspectral.ee.uh.edu ; machinelearning.ee.uh.edu (2018 DFC)
- RSICD/RSITMD/UCM-Captions/Sydney-Captions/NWPU-Captions — respective repos ; LuoJiaHOG survey arXiv:2403.10887
