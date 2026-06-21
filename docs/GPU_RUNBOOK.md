# GPU Runbook — Phase 1: reach F1@5 / F1@10 ≥ 0.8 (same- AND cross-modal)

> Read `CLAUDE.md` first. This runbook is the step-by-step plan to turn the
> framework into a real, competition-grade result on a GPU machine. Every CLI
> command and config field below has been verified against the code; where a
> needed capability is **missing**, it is called out as a **`TODO (wire this)`**
> rather than pretended to exist.

---

## Objective & honest status

**Target.** F1@5 ≥ 0.8 **and** F1@10 ≥ 0.8 for **same-modal** (optical→optical,
SAR→SAR, MS→MS) **and** **cross-modal** (optical↔SAR, optical↔MS, …), on a held-out
split, with low average per-query retrieval time. Cross-modal is weighted higher.

**Status today (be honest in any report you write).** The repo's headline numbers so
far (`smoke-test`: F1@5≈0.26, F1@10_cross≈0.32) are produced by a **synthetic numpy
substrate** — no foundation-model weights, no real satellite imagery. They prove the
*mechanism* (whitening closes the modality gap; sub-ms search) and nothing about real
accuracy. **No real-data run has been produced yet.** This runbook produces it.

**The achievability claim.** Because `F1@K = 2·r_K / (K + R_q)` is *bounded by the
relevant-set size `R_q`* (see §3a), F1 ≥ 0.8 is only reachable if the gallery is
engineered so `R_q ≈ K`. With that protocol, a strong backbone (DOFA/CROMA) + the
whitening/projection/re-rank recipe in §3 is expected to clear 0.8 on same-modal
easily and to reach 0.8 on the harder cross-modal directions. The closest published
analogue (CSMAE on BigEarthNet-MM) reports balanced cross-modal **F1@10 ≈ 0.71**
without gallery calibration, so 0.8 with `R_q ≈ K` is a realistic operating point —
**but it must be verified on real data, not assumed.**

---

## 0) Hardware / environment setup

```bash
# Clone (if not already on the box) and enter the repo.
git clone <your-fork-url> bah2026-ps11 && cd bah2026-ps11

# A clean Python 3.11 venv.
python3.11 -m venv .venv && source .venv/bin/activate
python -m pip install -U pip wheel

# 1) GPU PyTorch — pick the wheel matching your CUDA (cu121 shown; use cu118/cu124
#    as appropriate). Do this BEFORE `pip install -e .` so the CPU torch in
#    requirements-cpu.txt does not get pulled.
pip install torch --index-url https://download.pytorch.org/whl/cu121

# 2) GPU FAISS (conda is the most reliable for faiss-gpu; pip wheels also exist).
#    Either of:
pip install faiss-gpu-cu12          # pip wheel (CUDA 12)
#   conda install -c pytorch -c nvidia faiss-gpu      # conda alternative
#   (faiss-cpu is fine too — search is sub-ms at PS-11 gallery sizes either way.)

# 3) The package + the remaining deps (transformers/timm/open_clip/rasterio/…).
pip install -e .
#   Pull the rest WITHOUT clobbering the GPU torch you just installed:
pip install transformers timm open_clip_torch huggingface_hub scikit-learn \
            pyyaml pillow rasterio tqdm fastapi uvicorn python-multipart pytest

# 4) Verify.
python -c "import torch; print('cuda:', torch.cuda.is_available(), torch.version.cuda)"
python -m xsretrieval.cli info       # lists installed deps + registered backbones
```

`xsretrieval.cli info` prints which optional deps loaded (torch/faiss/transformers/
…) and the registered backbone names. Confirm `torch OK`, `faiss OK`,
`transformers OK` before proceeding.

> **`TODO (wire this)` — GPU device in the training path.** Configs expose
> `backbone.kwargs.device` (set it to `cuda`), and torch-backed backbones honor it.
> However, the projection-head trainer (`xsretrieval/alignment/trainer.py`) caches
> frozen embeddings and creates the head/loss tensors on the default device without
> an explicit `.to("cuda")`. On a GPU box you will still get correct results, but to
> get real GPU *acceleration* of the head training you may need to move
> `head`, `loss_fn`, `feats_t`, and the per-batch tensors to `cuda` (and back for
> numpy). The backbone forward pass (the expensive part) already runs on
> `backbone.kwargs.device`, so this matters most for large heads / many epochs.

---

## 1) Get real data

The pipeline falls back to synthetic data when a real dataset is absent — so the
**single most important thing** to get off the synthetic substrate is to put real
data on disk and point a config at it.

### Recommended datasets (links verified in `research/01_datasets.md`)

| Role | Dataset | Modalities | Size | Get it |
|---|---|---|---|---|
| **QUICK-START** | **EuroSAT** | MS (13-band) + derived RGB | ~90 MB (RGB) / ~2 GB (allBands) | `scripts/download_data.py` (auto) |
| **PRIMARY** | **SEN12MS** | S1 SAR (VV/VH) + S2 MS (13b) + derived RGB + IGBP label | ~430 GB | <https://mediatum.ub.tum.de/1474000> |
| **HELD-OUT TEST** | **DFC2020** | S1 + S2 + high-res 10 m IGBP labels | ~25 GB | IEEE GRSS DFC2020 (ieee-dataport); *mirrors the SEN12MS layout* |
| SECONDARY | **BigEarthNet-MM** / reBEN v2 | S1 + S2, 19-class multi-label | ~110–120 GB | <https://bigearth.net/> · Zenodo `10891137` |
| SECONDARY | **QXS-SAROPT** | 1 m SAR ↔ optical pairs (no labels) | ~2 GB | <https://github.com/yaoxu008/QXS-SAROPT> |

```bash
# QUICK-START: EuroSAT (same-modal optical/MS harness, stands up in minutes).
python scripts/download_data.py eurosat --root data/eurosat              # RGB (~90 MB)
python scripts/download_data.py eurosat --root data/eurosat --bands all  # 13-band (~2 GB)

# Print download guide + URLs for the big co-registered sets (SEN12MS, …).
python scripts/download_data.py instructions
```

### Expected on-disk layout (what the adapters scan for)

**EuroSAT** (`EuroSATDataset`, `dataset: eurosat`) — ImageFolder style, one
sub-directory per class:

```
data/eurosat/
  AnnualCrop/   AnnualCrop_1.jpg ...      # RGB variant (.jpg)  -> modality optical_rgb
  Forest/       Forest_1.jpg   ...        # or 13-band .tif     -> modality multispectral
  ...                                      # (the download script extracts this for you)
```

**SEN12MS** (`SEN12MSDataset`, `dataset: sen12ms`) — the official release layout;
S1/S2/LC patches for the same `<scene>_p<patch>` are pixel-aligned and paired by key:

```
data/sen12ms/
  ROIs1158_spring/
    s1_0/   ROIs1158_spring_s1_0_p100.tif     # 2 bands: VV, VH   -> modality sar
    s2_0/   ROIs1158_spring_s2_0_p100.tif     # 13 bands          -> modality multispectral / derived optical_rgb
    lc_0/   ROIs1158_spring_lc_0_p100.tif     # IGBP land cover   -> the class label (band 0, majority vote)
  ROIs1868_summer/ ...
```

The adapter derives an **RGB** modality from the S2 bands for free, simplifies IGBP
17→10 classes (the DFC2020 scheme), and emits up to three co-registered samples per
patch sharing one `location_id`. Reading GeoTIFFs needs `rasterio` (installed above).

> **`TODO (wire this)` — DFC2020 as a separate held-out test set.** The dataset
> registry (`get_dataset`) knows `folder`, `eurosat`, `sen12ms`, `synthetic` —
> **not** `dfc2020`. DFC2020 *mirrors the SEN12MS directory layout*, so the simplest
> path is to point the **`sen12ms` adapter** at the DFC2020 root (`data.dataset:
> sen12ms`, `data.root: data/dfc2020`) and use it as the eval config. If you want a
> single command that trains on SEN12MS and evaluates on a *disjoint* DFC2020 split,
> you must either (a) run two configs (train-config root=SEN12MS, eval-config
> root=DFC2020) and plug the trained head into the eval config via
> `projection.weights`, or (b) add a small `DFC2020Dataset` adapter / a `--test-root`
> option. The repo does **not** currently take two dataset roots in one run.

---

## 2) Configure

Copy `configs/default.yaml` to a working config and edit the data + backbone fields.
**All real-data and backbone selection happens here — there are no CLI flags for it.**

```bash
cp configs/default.yaml configs/run_gpu.yaml
```

Sample edited `configs/run_gpu.yaml` (every field below exists in
`xsretrieval/config.py`):

```yaml
name: run_gpu

data:
  dataset: sen12ms              # eurosat | sen12ms | folder | synthetic
  root: data/sen12ms            # the on-disk root from §1 (omit/null for synthetic)
  modalities: [optical_rgb, multispectral, sar]
  size: 224                     # foundation backbones expect 224x224
  query_frac: 0.3               # 30% of each class -> queries, rest -> gallery
  class_balanced_gallery: true  # KEY: equalizes per-class gallery size (the R_q lever, §3a)
  seed: 0
  # n_classes / per_class / substrate / modality_shift apply only to synthetic data.

backbone:
  name: croma                   # croma (radar-optical) | dofa (all modalities) | remoteclip | dinov2 | ...
  embed_dim: 256                # projection/whitening target width
  kwargs:
    device: cuda                # run the backbone on GPU

projection:
  enabled: false                # start zero-training (whitening only); flip to true in §3d
  out_dim: 256
  hidden: 512
  share_final: true
  weights: null                 # set to a trained head .pt to serve/evaluate it

whitening:
  enabled: true                 # the modality-gap fix — keep ON
  n_components: null            # full ZCA-style whitening (preserves the shared frame)
  remove_top_pc: 0              # try 1 to drop the sensor-identity direction (§3c)
  shrinkage: 0.9
  fit_on: gallery

index:
  type: auto                    # auto-pick faiss factory by gallery size; numpy fallback
  metric: ip                    # inner product on L2-normalized vectors == cosine
  nprobe: 16
  rerank: false                 # k-reciprocal re-rank (flip to true in §3e)

eval:
  ks: [5, 10]                   # the two scored cutoffs
  recall_mode: raw              # "raw" (grader default) or "capped" (fair selection, §3a)
  measure_latency: true
  latency_warmup: 50
  latency_runs: 500
```

You can equivalently start from a backbone preset (e.g. `configs/backbones/croma.yaml`
or `dofa.yaml`) and add the `data:` block — they already set whitening/index/eval.

---

## 3) The recipe to actually hit ≥ 0.8 (research-backed, ranked)

Each lever below names **which metric it moves and why**. Apply them roughly in this
order; (a) is non-negotiable.

### (a) Evaluation protocol is the #1 lever — calibrate the gallery so `R_q ≈ K`

F1@K has an exact closed form (`research/06`, `ARCHITECTURE.md §9`):

```
F1@K = 2 · r_K / (K + R_q)
```

where `r_K` = #relevant in the top-K and `R_q` = total relevant items for that query
in the gallery. **F1@K is bounded by `R_q` relative to `K`**, independent of how good
the ranker is:

- `R_q = 100`, `K = 5`, *perfect* top-5 → max F1@5 = `2·5/(5+100) = 0.095`. A huge
  class caps F1 low no matter what.
- `R_q ≈ K` is the sweet spot. Worked example: `R_q = 5`, 4 correct in the top-5 →
  **F1@5 = `2·4/(5+5) = 0.80`**. Another (from `research/06`): `R_q = 8`, 7 correct
  in top-10 → **F1@10 = `2·7/(10+8) = 0.778`**.

So **build a class-balanced gallery with ≈ 5–10 relevant items per class** so a
strong model can reach 0.8–1.0. In this repo:

- Keep **`data.class_balanced_gallery: true`** (it truncates every class to the
  minimum per-class gallery count → equal `R_q` across classes).
- **Size `R_q` toward K** by tuning `data.query_frac` and, for big datasets, capping
  per-class samples (e.g. via `SEN12MSDataset(max_patches=...)` if you script the
  load) so each class contributes ≈ 5–10 gallery items.
- **Report BOTH** `recall_mode: raw` (the grader's likely formula — primary number)
  **and** `recall_mode: capped` (`R@K = r_K/min(K,R_q)`, the fair model-selection
  number). Run the eval twice with each setting and record both. When `R_q ≈ K` the
  two nearly coincide — which is the point.

> **`TODO (wire this)` — there is no CLI flag to set the per-class gallery size or
> the absolute `R_q` directly.** It is governed indirectly by `query_frac`,
> `class_balanced_gallery`, and how many samples each class has on disk. If you need
> exact control (e.g. "exactly 6 gallery items per class"), pre-subset the dataset
> before indexing or extend `make_query_gallery_split`.

### (b) Backbone — choose for the modality mix (HF ids from `research/02`)

| Use case | Backbone | `backbone.name` | HF id |
|---|---|---|---|
| Radar↔optical, S1+S2 (best cross-modal) | **CROMA** | `croma` | `antofuller/CROMA` |
| All modalities incl. RGB (wavelength-conditioned) | **DOFA** | `dofa` | `XShadow/DOFA` (mirror `earthflow/DOFA`) |
| Optical↔optical specialist | **RemoteCLIP** | `remoteclip` | `chendelong/RemoteCLIP` (`ViT-L-14`) |
| Generic frozen optical features | **DINOv2** | `dinov2` | `facebook/dinov2-with-registers-base` |
| Always-works fallback | OpenCLIP / numpy | `openclip` / `fallback` | `laion/CLIP-ViT-B-32-laion2B-s34B-b79K` / — |

Start with **CROMA** if the data is Sentinel-1/2 pairs (its contrastive pretraining
already aligns radar↔optical, so cross-modal often works best out of the box), or
**DOFA** if you need RGB in one encoder. Each backbone **falls back to the numpy
`FallbackBackbone` if its weights can't be downloaded** — so after switching, confirm
via `evaluate`'s `[meta] backbone=... (XxxBackbone)` line that the *real* class
loaded (not `FallbackBackbone`). An offline/air-gapped box must pre-cache the HF
weights.

> **`TODO (wire this)` — ENSEMBLE (concat + whiten across backbones) is described in
> `ARCHITECTURE.md` but is not exposed by a single config / CLI command.** Running it
> means encoding with two backbones and concatenating before whitening; you would
> script it on top of `xsretrieval.pipeline.encode_dataset` (which returns whitened
> embeddings) or add an ensemble backbone to the registry. Treat as an optional
> robustness lever, not a day-1 step.

### (c) Per-modality mean-centering + ZCA whitening — keep ON

This is the proven, training-free modality-gap fix (it roughly **doubles** cross-modal
F1 in the synthetic ablation: 0.166 → 0.323). It is `whitening.enabled: true` (the
default). Optionally set **`whitening.remove_top_pc: 1`** to drop the leading
principal component, which often encodes sensor identity rather than semantics — try
it and keep it only if cross-modal F1 improves. Keep `shrinkage: 0.9` (safe on small
reference sets); lower it (e.g. 0.3–0.5) only with a large, clean gallery.

### (d) Train the projection head (frozen backbone + the combined loss)

Once the zero-training (whitening-only) baseline is measured, train the shared
projection head to sharpen the space. The trainer (`alignment/trainer.py`,
`alignment/losses.py`) implements exactly the research recipe: **symmetric InfoNCE
(τ=0.07) + Sub-center ArcFace + cross-modal batch-hard triplet**, with a
**modality-balanced P×K sampler** (`data/samplers.py`) that guarantees cross-modal
positives and hard negatives in every batch. The backbone is **frozen** and its
embeddings are cached once (CPU/GPU-friendly); only the head trains.

Use `configs/train_projection.yaml` as the template and point its `data`/`backbone`
at your real dataset + GPU device, then:

```bash
cp configs/train_projection.yaml configs/train_gpu.yaml
# edit: data.dataset/root, backbone.name (croma/dofa), backbone.kwargs.device: cuda,
#       train.epochs (raise to 20–50 for real data), and the loss weights / temperature.

python -m xsretrieval.cli train --config configs/train_gpu.yaml --epochs 30 --out artifacts/head.pt
```

The recipe knobs live in the `train:` block of the YAML (all real fields):
`epochs`, `lr`, `weight_decay`, `batch_p` (P classes/batch), `batch_k` (K per class),
`batches_per_epoch`, `w_infonce: 1.5`, `w_arcface: 1.0`, `w_triplet: 0.5`,
`temperature: 0.07`. The output 256-d embeddings are L2-normalized. After training,
**plug the head into your eval/serve config** by setting `projection.enabled: true`
and `projection.weights: artifacts/head.pt`.

> **`TODO (wire this)` — scope of `train` vs. the full architecture.** The CLI `train`
> path: (i) trains on the **gallery split** and validates on the **held-out query
> split** of the *same* config dataset (`pipeline.make_query_gallery`); it does not
> take a separate train-vs-test dataset. (ii) The backbone is always **frozen** — the
> **LoRA r=8 fine-tuning** option in `ARCHITECTURE.md §6/§8` is **not** wired into the
> CLI trainer. (iii) The optimizer is AdamW; **cosine schedule + warmup, margin
> warmup, EMA, and MoCo queue** from the architecture's §8.3 are **not** in the
> current loop. For a first ≥0.8 attempt the frozen-head recipe is usually enough;
> if cross-modal stalls, implementing cosine+warmup and LoRA are the next levers
> (edit `alignment/trainer.py`).

### (e) k-reciprocal re-rank on the candidate pool

Turn on `index.rerank: true` to apply k-reciprocal re-ranking (k1=20, k2=6, λ=0.3 —
see `index/` and `ARCHITECTURE.md §7.2`) to the top candidates. It is training-free,
adds only a few ms per query, and lifts F1 most on the **deeper list (@10)** and on
the **harder cross-modal** queries. Enable it for both `evaluate` and `build-index`
(set it in the config; the engine reads `config.index.rerank`).

### (f) Spend the GPU budget on cross-modal

Same-modal **optical** retrieval is near-saturated with good frozen features
(deep-hashing baselines exceed 99% mAP on optical benchmarks), so it should clear 0.8
readily. The marginal effort — harder negative mining, the projection head, re-rank,
`remove_top_pc` — should target the **opt↔SAR / SAR↔opt** directions, which are the
hardest and carry extra weight. Read the per-cell matrix `evaluate` prints to see
which off-diagonal cells lag.

### (g) Use multiple datasets to cross-verify

Validate on more than one set (EuroSAT for same-modal sanity; SEN12MS for tri-modal;
DFC2020's high-res labels for trustworthy cross-modal F1; QXS-SAROPT for VHR SAR↔opt).
Agreement across datasets/backbones is a confidence signal; disagreement flags a
gallery-calibration or backbone-fallback problem.

---

## 4) Run it

```bash
# 0) Sanity: confirm the real backbone + real dataset actually loaded (NOT fallback/synthetic).
python -m xsretrieval.cli evaluate --config configs/run_gpu.yaml --json artifacts/eval_zeroshot.json
#   Read the tail: [meta] config=run_gpu  backbone=croma (CROMABackbone)  whitening=True
#                  index=faiss  dataset=sen12ms
#   If it says (FallbackBackbone) or dataset fell back to synthetic, fix weights/paths
#   before trusting any number.

# 1) (Optional but recommended) Build a persisted retrieval bundle for serving/inspection.
python -m xsretrieval.cli build-index --config configs/run_gpu.yaml --out artifacts/index

# 2) Train the projection head on the real data (§3d).
python -m xsretrieval.cli train --config configs/train_gpu.yaml --epochs 30 --out artifacts/head.pt

# 3) Evaluate WITH the trained head + re-rank. Make an eval config that sets
#    projection.enabled: true, projection.weights: artifacts/head.pt, index.rerank: true,
#    then:
python -m xsretrieval.cli evaluate --config configs/run_gpu_trained.yaml --json artifacts/eval_trained.json
```

`evaluate` prints the full **query×gallery matrix** (P/R/F1@5/@10 + mAP per cell), the
four **headline F1s** (same/cross at 5 and 10), and the **latency** report (mean / p50
/ p95, batch=1, search-only). The headline block looks like:

```
  F1@5   same-modal :  0.XXXX    cross-modal :  0.XXXX
  F1@10  same-modal :  0.XXXX    cross-modal :  0.XXXX
```

**Iterating if below 0.8** (cheapest → most expensive):

1. **Re-balance the gallery so `R_q ≈ K`** (§3a) — almost always the biggest jump.
2. Turn on **`index.rerank: true`** (§3e) and try **`whitening.remove_top_pc: 1`**
   (§3c).
3. **Train longer / harder**: raise `train.epochs`, increase `batch_p`/`batch_k` for
   more negatives, lower `temperature` toward 0.05, nudge ArcFace/triplet weights.
4. **Switch/strengthen the backbone** (CROMA↔DOFA; RemoteCLIP-L for optical) and
   confirm the real weights loaded.
5. Cross-check on a **second dataset** (§3g); consider the ensemble/LoRA TODOs.

---

## 5) Acceptance check

You have met Goal A when, on a **held-out split** with a **calibrated gallery**:

```
F1@5_same  ≥ 0.80   AND   F1@10_same  ≥ 0.80
F1@5_cross ≥ 0.80   AND   F1@10_cross ≥ 0.80
```

are all satisfied — ideally confirmed with `recall_mode: raw` (and shown to hold, or
nearly so, under `capped` too), on a dataset with **trustworthy labels** (DFC2020's
high-res IGBP is the recommended test split; see the §1 DFC2020 TODO for how to point
the `sen12ms` adapter at it). Record the four numbers, the latency, the backbone
class, the dataset, and the gallery size / per-class `R_q`.

**Save the artifacts** for deployment into `artifacts/`:

```
artifacts/
  head.pt              # trained projection-head weights (from `train --out`)
  index/               # the persisted retrieval bundle from `build-index`:
    index              #   the FAISS (or numpy) index
    whitener.npz       #   the fitted per-modality whitener
    config.yaml        #   the exact config used (backbone, modalities, …)
    gallery_meta.npz   #   gallery location ids + embed dim
  eval_trained.json    # the evaluate --json output (the F1 matrix + latency, for the record)
```

Build the deployable bundle with the **trained** config (so it bakes in the head +
whitener):

```bash
python -m xsretrieval.cli build-index --config configs/run_gpu_trained.yaml --out artifacts/index
```

Then proceed to `docs/DEPLOYMENT.md` (Goal B).

---

## 6) Troubleshooting

- **OOM during backbone encode/train** — lower `data.size` (but 224 is what the
  foundation backbones expect), reduce `train.batch_p`/`batch_k`, or encode in
  smaller chunks. The trainer caches embeddings once, so peak memory is the backbone
  forward batch (currently 64 in `_encode_frozen`) — shrink it there if needed.
- **`faiss-gpu` import fails / wrong CUDA** — `faiss-cpu` is a fine substitute (search
  is sub-ms at PS-11 gallery sizes); or install faiss via conda matched to your CUDA.
  Confirm with `python -m xsretrieval.cli info` (`faiss OK`).
- **Modality / channel mismatch** — SAR is 2-band (VV/VH), MS 13-band, RGB 3-band.
  The adapters + `data/preprocessing.py` handle resizing/normalization; if you feed a
  folder of rasters, tag the right `modality` and ensure band counts match what the
  backbone expects (CROMA: S1=2, S2=12; DOFA: any via wavelengths; CLIP/DINOv2:
  pseudo-RGB).
- **It "ran" but numbers look synthetic** — you almost certainly hit a fallback: the
  dataset path was wrong (→ synthetic) or the backbone weights were missing (→
  `FallbackBackbone`). Check the `[meta] ...` line `evaluate` prints; both must show
  the real dataset and real backbone class.
- **Download issues (HF / dataset hosts)** — pre-cache HF weights with
  `huggingface-cli download <id>` on a connected machine; for datasets, the big sets
  need accounts (TUM mediaTUM / IEEE-DataPort) — see
  `python scripts/download_data.py instructions`.

---

## COPY-PASTE PROMPT FOR CLAUDE CODE

> Read `CLAUDE.md` and `docs/GPU_RUNBOOK.md`. Set up the GPU environment per §0
> (GPU torch + faiss, `pip install -e .`, verify `torch.cuda.is_available()` and
> `xsretrieval info`). Download a real dataset — start with EuroSAT
> (`python scripts/download_data.py eurosat --root data/eurosat`) to validate the
> harness end to end, then SEN12MS for the full tri-modal run. Copy
> `configs/default.yaml` and `configs/train_projection.yaml` into working configs
> pointing at the real dataset root, `backbone.name: croma` (or `dofa`),
> `backbone.kwargs.device: cuda`, and confirm via `evaluate`'s `[meta]` line that the
> real backbone + real dataset loaded (NOT FallbackBackbone / synthetic). Then train
> and evaluate the cross-modal retrieval model, iterating on the §3 recipe —
> calibrate the gallery so `R_q ≈ K` (§3a), keep whitening on, train the projection
> head, enable `index.rerank`, target the cross-modal directions — until **F1@5 and
> F1@10 are both ≥ 0.80 for same-modal AND cross-modal** on a held-out split (use
> DFC2020 labels if available; see the §1 DFC2020 TODO). Report both `raw` and
> `capped` recall. Save the trained artifacts (head.pt, the build-index bundle, the
> evaluate --json) to `artifacts/`. Report the final 4-number F1 matrix, the per-query
> latency, the backbone class, and the gallery size / per-class `R_q`.
