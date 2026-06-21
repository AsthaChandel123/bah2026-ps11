# xsretrieval — Cross-Modal Satellite Image Retrieval

**BAH 2026 · Problem Statement 11 — Cross-Modal Satellite Image Retrieval Using Multi-Sensor Remote Sensing Data**

Given a query image from *any* sensor modality — optical RGB, multispectral
(Sentinel-2, 13 bands), or Synthetic Aperture Radar (Sentinel-1 VV/VH) — return
a ranked top-5 / top-10 of the most semantically-relevant images from a gallery
that may hold the **same** modality (optical→optical, SAR→SAR, MS→MS) **or a
different** one (optical→SAR, SAR→optical, optical→MS, …). Relevance is judged by
semantic class and/or geographic correspondence.

`xsretrieval` is a **CPU-first**, **zero-training-by-default** package that does
this end to end: pluggable foundation backbones → a shared cross-modal embedding
space (per-modality whitening, optional trainable projection) → an O(1) FAISS
index (with an exact numpy fallback) → F1@K evaluation, plus a CLI, a REST API,
and a demo.

---

## Why it scores

PS-11 is graded on **five numbers**: F1@5 and F1@10 for **same-modal** and
**cross-modal** retrieval, plus the **average retrieval time per query**.

| # | Metric | What it measures |
|---|--------|------------------|
| 1 | **F1@5 (same-modal)** | accuracy within one modality |
| 2 | **F1@10 (same-modal)** | accuracy within one modality, deeper list |
| 3 | **F1@5 (cross-modal)** | accuracy *across* modalities (the hard part) |
| 4 | **F1@10 (cross-modal)** | accuracy across modalities, deeper list |
| 5 | **avg query time** | retrieval latency (batch=1, search-only) |

The single highest-ROI lever for the cross-modal numbers is **closing the
modality gap** — foundation encoders place each sensor in its own offset cone on
the hypersphere, so a naive shared index returns mostly same-modality neighbours
and cross-modal F1 collapses. Per-modality **whitening** (mean-centering + ZCA)
removes that offset; in the bundled smoke test it nearly **doubles** cross-modal
F1@10 (0.166 → 0.323) with no training. See [Evaluation](#evaluation-methodology).

---

## Architecture at a glance

```
 ┌── DATA ──────────────┐   ┌── BACKBONE ──────────┐   ┌── ALIGNMENT ─────────────┐
 │ optical RGB          │   │ DOFA / CROMA         │   │ projection head (opt.)   │
 │ multispectral (13b)  │──►│ RemoteCLIP / OpenCLIP│──►│ per-modality WHITENING   │
 │ SAR (VV/VH)          │   │ DINOv2 / timm        │   │  mean-center + ZCA + L2  │
 │ (synthetic fallback) │   │ numpy fallback       │   │  → ONE shared space      │
 └──────────────────────┘   └──────────────────────┘   └────────────┬─────────────┘
                                                                     ▼
 ┌── EVALUATION ────────┐   ┌── RETRIEVAL ─────────┐   ┌── INDEX ─────────────────┐
 │ query×gallery matrix │◄──│ modality-filtered    │◄──│ ONE FAISS index over all │
 │ F1@5/@10 same+cross  │   │ search + k-recip.    │   │ modalities (+ side array)│
 │ avg query latency    │   │ re-rank (optional)   │   │ exact numpy fallback     │
 └──────────────────────┘   └──────────────────────┘   └──────────────────────────┘
```

The gallery is embedded **once, offline**; the query path is online and
sub-millisecond. Full design rationale (data, backbones, losses, complexity,
30+ methods catalogue, references) lives in **[ARCHITECTURE.md](ARCHITECTURE.md)**
and the rendered flowchart in
[`docs/diagrams/system_overview.mmd`](docs/diagrams/system_overview.mmd).

### The convergent stack

- **Datasets** — primary tri-modal **SEN12MS** (S1 SAR + S2 MS + derived RGB +
  IGBP labels), **DFC2020** as the high-res labelled test split, **EuroSAT** for
  the day-1 same-modal harness, **QXS-SAROPT / BigEarthNet-MM** for VHR and
  scale. A solvable **synthetic** multi-modal generator stands in when no data is
  on disk, so everything runs out of the box.
- **Backbones** — **DOFA** (wavelength-conditioned, one encoder for all sensors)
  and **CROMA** (radar-optical pretrained) as defaults; **RemoteCLIP**,
  **OpenCLIP**, **DINOv2**, **timm** as alternates; a deterministic **numpy
  fallback** that always works offline. Every backbone exposes
  `embed(images, modality) -> (B, D)` L2-normalized.
- **Alignment** — per-modality **whitening** (the modality-gap fix) plus an
  optional trainable **projection head** with a weight-shared final layer,
  trained with `1.5·InfoNCE + 1·SubCenterArcFace + 0.5·batch-hard-triplet`.
- **Retrieval** — **one FAISS index** over all-modality embeddings with a
  modality side-array for same-/cross-modal filtering, an exact numpy
  brute-force fallback, and optional **k-reciprocal re-ranking**. Inner product
  on L2-normalized vectors == cosine similarity.
- **Evaluation** — the full **query×gallery F1@K matrix**, same/cross
  aggregates, and correctly-measured latency (warmup, batch=1, search-only).

---

## Quickstart

```bash
# 1) Install (CPU-only, reproducible — torch CPU wheels, faiss-cpu, ...)
make install-cpu          # or: pip install -e . && pip install -r requirements-cpu.txt

# The numpy retrieval path needs no heavy deps; for that minimal install:
pip install -e .          # just numpy

# 2) Run the end-to-end smoke test (synthetic, with whitening)
make smoke                # or: python -m xsretrieval.cli smoke-test
#                          or: bash scripts/run_smoke_test.sh

# 3) Full evaluation on a config (synthetic fallback if no dataset present)
python -m xsretrieval.cli evaluate --config configs/zero_shot.yaml

# 4) Environment / backbone availability
python -m xsretrieval.cli info
```

### Smoke-test output (actual run, CPU, FAISS backend)

The smoke test runs the **default whitened pipeline** on a synthetic tri-modal
benchmark that carries a realistic modality gap, then ablates whitening on/off.
Numbers below are a verbatim run (`python -m xsretrieval.cli smoke-test`):

```
PS-11 headline F1 scores
----------------------------------------
  F1@5   same-modal :  0.2614    cross-modal :  0.2453
  F1@10  same-modal :  0.3403    cross-modal :  0.3231
----------------------------------------

Average retrieval time per query (batch=1, search-only)
------------------------------------------------------------------------------
  mean : 0.166 ms   p50 : 0.153 ms   p95 : 0.230 ms

Whitening ablation (headline F1):
  metric          whitening OFF    whitening ON
  F1@5_same            0.2351          0.2614
  F1@5_cross           0.1129          0.2453      ←  +117%
  F1@10_same           0.3162          0.3403
  F1@10_cross          0.1660          0.3231      ←   +95%

[check] cross-modal F1@10: whitening OFF=0.1660  ON=0.3231  (random≈0.1000)
[smoke-test] PASS — whitened cross-modal F1 is clearly above chance and improved by whitening.
```

Full per-cell query×gallery matrix (same run):

```
query         gallery       type         P@5      R@5     F1@5     P@10     R@10    F1@10      mAP
multispectral multispectral same      0.5067   0.1810   0.2667   0.4217   0.3012   0.3514   0.2133
optical_rgb   optical_rgb   same      0.4833   0.1726   0.2544   0.4083   0.2917   0.3403   0.2004
sar           sar           same      0.5000   0.1786   0.2632   0.3950   0.2821   0.3292   0.1969
multispectral optical_rgb   cross     0.4800   0.1714   0.2526   0.4150   0.2964   0.3458   0.2012
multispectral sar           cross     0.4800   0.1714   0.2526   0.3933   0.2810   0.3278   0.2067
optical_rgb   multispectral cross     0.4533   0.1619   0.2386   0.3917   0.2798   0.3264   0.1918
optical_rgb   sar           cross     0.4967   0.1774   0.2614   0.4117   0.2940   0.3431   0.2098
sar           multispectral cross     0.4400   0.1571   0.2316   0.3433   0.2452   0.2861   0.1534
sar           optical_rgb   cross     0.4467   0.1595   0.2351   0.3717   0.2655   0.3097   0.1824
```

> These numbers are produced by the **deterministic numpy substrate** (no
> foundation-model download) so the smoke test is fast and reproducible
> anywhere. They demonstrate the *mechanism* (whitening closes the modality gap;
> sub-ms retrieval). A real foundation backbone on a real dataset scores far
> higher — DINOv2 / DOFA / CROMA produce much stronger features (see
> [Plugging in a real backbone](#plug-in-a-real-backbone-or-dataset)).

---

## Dataset download

```bash
# EuroSAT (small, optical/MS, 10 classes) — auto-download:
python scripts/download_data.py eurosat --root data/eurosat            # RGB  (~90 MB)
python scripts/download_data.py eurosat --root data/eurosat --bands all # 13-band (~2 GB)

# Download guide + URLs for the large co-registered multi-modal sets:
python scripts/download_data.py instructions
```

| Dataset | Modalities | Labels | Size | Get it |
|---|---|---|---|---|
| **SEN12MS** (primary) | SAR + MS + RGB | IGBP land cover | ~430 GB | mediatum.ub.tum.de/1474000 |
| **DFC2020** (test) | SAR + MS | high-res IGBP | ~25 GB | IEEE GRSS DFC2020 |
| **BigEarthNet-MM** | SAR + MS | 19-class CLC (multi) | ~120 GB | bigearth.net |
| **QXS-SAROPT** | SAR + optical | none (geo pairs) | ~2 GB | github.com/yaoxu008/QXS-SAROPT |
| **EuroSAT** | MS (+RGB) | 10 land-use | ~90 MB / 2 GB | madm.dfki.de |

Point a config at a downloaded dataset via `data.dataset` / `data.root`. If the
path is missing the pipeline logs a warning and falls back to synthetic data, so
runs never fail.

---

## Plug in a real backbone or dataset

Switch the backbone with a config — everything downstream is unchanged:

```bash
# Self-supervised ViT features (verified to load & embed on CPU):
python -m xsretrieval.cli evaluate --config configs/backbones/dinov2.yaml

# Multi-sensor foundation encoders (fall back to numpy if weights unavailable):
python -m xsretrieval.cli evaluate --config configs/backbones/dofa.yaml
python -m xsretrieval.cli evaluate --config configs/backbones/croma.yaml
python -m xsretrieval.cli evaluate --config configs/backbones/remoteclip.yaml
```

A real backbone is a one-line config change (`backbone.name` +
optional `backbone.kwargs.model_name`). Any backbone that cannot be built — a
missing extra, a failed download — **automatically falls back** to the numpy
`FallbackBackbone`, so the pipeline never hard-crashes. To use a real dataset,
set `data.dataset: eurosat` (or `sen12ms` / `folder`) and `data.root`.

To **train** the optional projection head on top of a frozen backbone:

```bash
python -m xsretrieval.cli train --config configs/train_projection.yaml --out head.pt
# then set projection.weights: head.pt in your eval config.
```

---

## Serving — REST API & demo

```bash
# Build a retrieval bundle (index + whitener + config) once:
python -m xsretrieval.cli build-index --config configs/zero_shot.yaml --out artifacts/index

# REST API (FastAPI):
XSRETRIEVAL_INDEX=artifacts/index uvicorn apps.api:app --host 0.0.0.0 --port 8000
curl -F "file=@query.npy" -F "k=10" -F "gallery_modality=sar" http://localhost:8000/query

# Interactive demo (Gradio UI, or a pure-CLI loop if gradio is absent):
XSRETRIEVAL_INDEX=artifacts/index python apps/demo.py
```

`GET /health`, `GET /info`, and `POST /query` (image upload + `k` +
optional `gallery_modality`) return JSON top-k results. Or query from the CLI:

```bash
python -m xsretrieval.cli query --index artifacts/index --image query.npy \
    --modality optical_rgb --gallery-modality sar --k 10
```

---

## Evaluation methodology

Relevance is **class-equality** (a gallery item is relevant to a query iff it
shares the query's semantic class), with **leave-one-out** so a query never
retrieves itself or its co-located twin. For each query of class *c* with *Rq*
relevant gallery items and *rK* relevant items in the top *K*:

```
P@K  = rK / K
R@K  = rK / Rq           (raw — the literal reading used for reporting)
F1@K = 2·P@K·R@K / (P@K + R@K)
```

Evaluation builds the full **query-modality × gallery-modality matrix**: the
diagonal cells are same-modal, the off-diagonal cells are cross-modal. Each cell
is macro-averaged over its queries; the four **headline F1s** are the macro
averages of the diagonal (same) and off-diagonal (cross) cells at K=5 and K=10.
Latency is measured correctly — warmup, batch=1, search-only, with p50/p95.

> Note on the recall denominator: F1@K is bounded by *Rq* relative to *K*. With a
> large balanced gallery, *Rq* ≫ K caps the achievable F1@K (e.g. *Rq*=100, K=5
> ⇒ max F1@5 ≈ 0.095). The harness supports `recall_mode="capped"`
> (`R@K = rK / min(K, Rq)`) for fair model selection; `"raw"` is reported.

---

## Project structure

```
xsretrieval/
├── __init__.py          # lazy top-level exports (fast `import xsretrieval`)
├── config.py            # dataclass Config + YAML loader
├── pipeline.py          # build_pipeline / run_evaluation / build_index / encode
├── cli.py               # smoke-test | evaluate | build-index | query | train | info
├── data/                # Modality, Sample, datasets, synthetic, samplers, preprocessing
├── models/              # get_backbone registry, backbones (DOFA/CROMA/CLIP/DINOv2/...),
│                        #   projection heads, ensembling, numpy fallback, precomputed
├── alignment/           # losses, per-modality whitening, projection-head trainer
├── index/               # RetrievalIndex (FAISS + numpy), ITQ hashing, re-ranking
├── retrieval/           # RetrievalEngine (encode → whiten → index → query)
└── eval/                # metrics (P/R/F1/nDCG/mAP) + benchmark (matrix, latency, report)
configs/                 # default / zero_shot / train_projection + backbones/*.yaml
apps/                    # api.py (FastAPI), demo.py (Gradio / CLI fallback)
scripts/                 # download_data.py, run_smoke_test.sh
tests/                   # pytest suite (metrics, whitening, index, losses, pipeline)
docs/                    # USAGE.md, diagrams/system_overview.mmd
research/                # 01..06 research reports
ARCHITECTURE.md          # the definitive design (single source of truth)
```

---

## Research & design

The system synthesizes six research reports (all under [`research/`](research/)):

1. [Datasets](research/01_datasets.md) — multi-sensor co-registered data strategy.
2. [Foundation models](research/02_foundation_models.md) — backbone survey.
3. [Cross-modal alignment](research/03_crossmodal_alignment.md) — the modality-gap fix.
4. [Fast retrieval](research/04_fast_retrieval.md) — FAISS, hashing, re-ranking.
5. [Training losses](research/05_training_losses.md) — the combined recipe.
6. [SOTA evaluation](research/06_sota_evaluation.md) — F1@K methodology.

→ **[ARCHITECTURE.md](ARCHITECTURE.md)** ties them into one design.
→ **[docs/USAGE.md](docs/USAGE.md)** is the detailed CLI / API / config guide.

---

## Notes

- **CPU-only**: the whole system targets zero-GPU deployment. CPU PyTorch +
  faiss-cpu wheels are pinned in `requirements-cpu.txt`.
- **Zero-training by default**: the recommended pipeline is frozen backbone +
  per-modality whitening — no training required. A projection-head trainer is
  provided for when labelled co-registered data is available.
- **Graceful degradation**: missing torch/faiss → numpy paths; missing
  backbone weights → numpy fallback; missing dataset → synthetic. A bare
  `pip install -e .` (numpy only) can `import xsretrieval`, run the smoke test,
  and evaluate.
- **Tested**: `make test` runs the full pytest suite (67 tests).

## License

MIT — see [LICENSE](LICENSE).
