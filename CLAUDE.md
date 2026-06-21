# CLAUDE.md — project memory for `xsretrieval`

> Auto-loaded by Claude Code. Read this first, then the two runbooks it links to.

## What this is

`xsretrieval` is a **cross-modal satellite image retrieval** system for **BAH 2026
Problem Statement 11**. Given a query image from *any* sensor modality — optical
RGB, multispectral (Sentinel-2, 13 bands), or SAR (Sentinel-1 VV/VH) — it returns
a ranked top-5 / top-10 of the most semantically-relevant gallery images, where the
gallery may hold the **same** modality (optical→optical, SAR→SAR, MS→MS) **or a
different** one (optical→SAR, SAR→optical, optical→MS, …). The pipeline is:
data → pluggable foundation backbone → shared embedding space (per-modality
mean-centering + ZCA whitening, optional trainable projection head) → one FAISS
index (with an exact numpy fallback) → F1@K evaluation. It ships a CLI, a FastAPI
service (`apps/api.py`), and a demo (`apps/demo.py`).

PS-11 is graded on **five numbers**: F1@5 and F1@10 for **same-modal** and
**cross-modal** retrieval, plus **average retrieval time per query**. Cross-modal is
explicitly weighted higher ("more challenging").

## Repo layout

- `xsretrieval/` — the package. Sub-packages: `data/` (modalities, dataset
  adapters, synthetic generator, samplers, preprocessing), `models/` (backbone
  registry + DOFA/CROMA/RemoteCLIP/OpenCLIP/DINOv2/timm/fallback/precomputed,
  projection heads), `alignment/` (losses, whitening, projection-head trainer),
  `index/` (FAISS + numpy index, re-ranking), `retrieval/` (engine), `eval/`
  (metrics + benchmark). Entry point: `xsretrieval/cli.py`.
- `configs/` — `default.yaml`, `zero_shot.yaml`, `train_projection.yaml`,
  `backbones/{dofa,croma,remoteclip,openclip,dinov2,fallback}.yaml`.
- `apps/` — `api.py` (FastAPI; uvicorn target `apps.api:app`), `demo.py` (Gradio /
  CLI fallback).
- `scripts/download_data.py` — EuroSAT auto-download + instructions for the large
  multi-modal sets.
- **`ARCHITECTURE.md`** — the definitive design (single source of truth: data,
  backbones, losses, retrieval, evaluation math, references).
- **`research/01..06_*.md`** — the six research reports behind the design
  (01 datasets, 02 foundation models, 03 cross-modal alignment, 04 fast retrieval,
  05 training losses, 06 SOTA + F1@K methodology).
- `docs/USAGE.md` — detailed CLI / API / config guide.

## Current state (honest)

- **Framework is complete and tested** — the pytest suite passes (`make test`) and
  `python -m xsretrieval.cli smoke-test` runs end to end.
- **The accuracy you have seen so far is a SYNTHETIC mechanism-demo, NOT a real
  result.** The smoke test and the default/zero-shot configs run on a deterministic
  numpy substrate (no foundation-model download, no real satellite data). It
  *demonstrates the mechanism* — per-modality whitening closing the modality gap
  (cross-modal F1@10 0.166 → 0.323) and sub-millisecond retrieval — but the headline
  F1 numbers (~0.25–0.34) are from synthetic vectors and **say nothing about
  competition accuracy**.
- **No real-data training run has been produced yet.** Real backbones (DOFA/CROMA/
  RemoteCLIP/DINOv2) and real datasets (EuroSAT/SEN12MS) are wired in but **fall
  back to numpy/synthetic** when weights or data are absent — which is the case on
  this checkout.

## The owner's goal (in priority order)

- **Goal A — reach competition-grade accuracy on real data.** Run real-data
  training/eval until **F1@5 ≥ 0.8 and F1@10 ≥ 0.8 for BOTH same-modal AND
  cross-modal**, on a held-out split, with low per-query latency. This is the first
  priority. The detailed, research-backed plan is in **`docs/GPU_RUNBOOK.md`**.
- **Goal B — deploy.** Once Goal A is met, deploy a **frontend on Google Cloud Run**
  with the trained model hosted appropriately. The detailed plan is in
  **`docs/DEPLOYMENT.md`**.

## Key commands (these are the REAL CLI — verified)

```bash
python -m xsretrieval.cli info                                   # env + backbone availability
python -m xsretrieval.cli smoke-test                            # synthetic end-to-end + whitening ablation
python -m xsretrieval.cli evaluate    --config <cfg.yaml> [--json out.json] [--no-latency]
python -m xsretrieval.cli train       --config <cfg.yaml> [--epochs N] [--out head.pt]
python -m xsretrieval.cli build-index --config <cfg.yaml> --out artifacts/index
python -m xsretrieval.cli query       --index artifacts/index --image q.npy \
                                      --modality optical_rgb --gallery-modality sar --k 10
```

## One-command REAL-DATA proof (automated — start here for real numbers)

`make real-proof` (or `python scripts/run_real_proof.py`) is the **automated,
hands-off entrypoint that produces GENUINE F1 on REAL satellite imagery** — it
downloads EuroSAT (DFKI zip, or the HuggingFace parquet mirror if that host is
down), embeds it with a **real foundation backbone** (auto-tries DINOv2 /
OpenCLIP, falling back to the numpy backbone only if weights truly can't load —
and it *says so*), fits per-modality whitening, evaluates the class-balanced
gallery, and writes `docs/REAL_PROOF_RESULTS.md` + `artifacts/results.json`.
Unlike `smoke-test`, these numbers are real (not the synthetic substrate).

```bash
make real-proof                                  # EuroSAT, auto backbone, CPU-sane (subset 2000)
python scripts/run_real_proof.py --help          # all flags
# CPU same-modal optical proof (the guaranteed real result):
python scripts/run_real_proof.py --dataset eurosat --backbone auto --device cpu \
    --subset 2000 --gallery-per-class 10 --query-per-class 5 --no-train
# GPU full same- AND cross-modal proof toward F1 ≥ 0.8 (see docs/GPU_RUNBOOK.md):
python scripts/run_real_proof.py --dataset sen12ms --backbone dofa --device cuda --train --epochs 30
```

It self-contains the whole pipeline via the real Python API (no missing CLI
flags). The actual run executed on this checkout — real EuroSAT, CPU, real
backbone, zero-shot + whitened — is recorded in **`docs/REAL_PROOF_RESULTS.md`**.

Important: **the dataset, backbone, device, whitening, projection head, index type,
gallery balancing and eval cutoffs are all set in the YAML config — not via CLI
flags.** There is no `--dataset`, `--backbone`, `--device`, or `--gallery-balanced`
flag. To change any of those, edit (a copy of) a config in `configs/`. See
`xsretrieval/config.py` for every field.

## When the user asks you to run training / get the score up / deploy

**Follow `docs/GPU_RUNBOOK.md` (Goal A) and `docs/DEPLOYMENT.md` (Goal B) step by
step.** They contain the exact, verified command sequences, the config edits, the
research-backed recipe to actually reach F1 ≥ 0.8 (evaluation protocol, backbone
choice, whitening, projection-head training, k-reciprocal re-rank), the acceptance
check, and the Cloud Run deployment. Do not invent CLI flags — if a capability the
plan needs is missing, the runbook flags it as an explicit TODO to wire.
