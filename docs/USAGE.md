# xsretrieval — Usage Guide

Detailed reference for the CLI, the configuration system, the Python API, and
the serving surfaces. For the project overview and the headline numbers see the
top-level [README](../README.md); for the design see
[ARCHITECTURE.md](../ARCHITECTURE.md).

---

## 1. Installation

```bash
# Minimal (numpy only) — enough to import the package, run the smoke test,
# and evaluate on the numpy brute-force index:
pip install -e .

# CPU compute stack (validated real path): torch CPU + faiss-cpu + sklearn ...
make install-cpu
# equivalently:
pip install -e . && pip install -r requirements-cpu.txt

# Optional extras (declared in pyproject.toml):
pip install -e ".[compute]"   # torch, faiss-cpu, scikit-learn, scipy, tqdm
pip install -e ".[models]"    # transformers, timm, open_clip_torch, huggingface_hub
pip install -e ".[io]"        # pyyaml, pillow, rasterio
pip install -e ".[serve]"     # fastapi, uvicorn, gradio, python-multipart
pip install -e ".[all]"       # everything
```

After installation the `xsretrieval` console command and
`python -m xsretrieval.cli` are equivalent.

---

## 2. Command-line interface

```bash
python -m xsretrieval.cli <command> [options]
# or, once installed:
xsretrieval <command> [options]
```

### `smoke-test`
Run the synthetic end-to-end pipeline **with whitening**, print the
query×gallery matrix, the four headline F1s, latency, and a whitening on/off
ablation. Exits 0 on success.

```bash
xsretrieval smoke-test                       # default synthetic benchmark
xsretrieval smoke-test --config configs/zero_shot.yaml
```

### `evaluate`
Full evaluation on a configured dataset (synthetic fallback if absent).

```bash
xsretrieval evaluate --config configs/default.yaml
xsretrieval evaluate --config configs/zero_shot.yaml --no-latency
xsretrieval evaluate --config configs/backbones/dinov2.yaml --json results.json
```

| Option | Meaning |
|---|---|
| `--config PATH` | config YAML (required) |
| `--json PATH` | also write the full results dict as JSON |
| `--no-latency` | skip the latency benchmark (faster) |

### `build-index`
Build a retrieval **bundle** (FAISS index + fitted whitener + config + gallery
metadata) and persist it to a directory for serving.

```bash
xsretrieval build-index --config configs/zero_shot.yaml --out artifacts/index
```

Produces `artifacts/index/{index.npz, index.faiss?, whitener.npz, config.yaml,
gallery_meta.npz}`.

### `query`
Query a persisted bundle with an image (or a `.npy` embedding) and print top-k.

```bash
xsretrieval query --index artifacts/index --image query.png --k 10
xsretrieval query --index artifacts/index --image query.npy \
    --modality optical_rgb --gallery-modality sar --k 5 --json hits.json
```

| Option | Meaning |
|---|---|
| `--index DIR` | bundle directory from `build-index` (required) |
| `--image PATH` | query image, or `.npy` array / embedding (required) |
| `--k N` | number of results (default 10) |
| `--modality M` | query modality (default: first configured) |
| `--gallery-modality M` | restrict results to this modality (default: all) |
| `--json PATH` | write hits as JSON |

### `train`
Train the optional projection head on top of a frozen backbone (requires torch).

```bash
xsretrieval train --config configs/train_projection.yaml --epochs 5 --out head.pt
```

Reports per-epoch loss and validation headline F1; saves the head's
`state_dict`. Re-use it via `projection.weights: head.pt` in an eval config.

### `info`
Print the package version, optional-dependency availability, the registered
backbones, and the default pipeline settings.

```bash
xsretrieval info
```

---

## 3. Configuration

Configs are YAML files parsed into a typed `Config` dataclass
(`xsretrieval/config.py`). Any omitted field falls back to a sane default, so a
config can override just the parts it cares about (the `configs/backbones/*.yaml`
files set only the `backbone` section, for instance).

```yaml
name: my_run

data:
  dataset: synthetic           # synthetic | eurosat | sen12ms | folder
  root: null                   # dataset root on disk (real datasets)
  modalities: [optical_rgb, multispectral, sar]
  size: 64                     # synthetic image size
  n_classes: 10                # synthetic class count
  per_class: 16                # synthetic locations per class per modality
  query_frac: 0.3              # fraction of each class held out as queries
  class_balanced_gallery: true # equal gallery items per class (macro-fair F1)
  substrate: image             # image (full encode path) | embedding (modality-gap vectors)
  modality_shift: 2.5          # modality-gap size for the 'embedding' substrate
  seed: 0

backbone:
  name: dofa                   # dofa|croma|remoteclip|openclip|dinov2|timm|fallback|precomputed
  embed_dim: 256
  kwargs:                      # forwarded to the backbone constructor (unknown keys dropped)
    device: cpu
    # model_name: facebook/dinov2-small

projection:
  enabled: false               # trainable shared head (torch); off for zero-training
  out_dim: 256
  hidden: 512
  dropout: 0.1
  share_final: true            # weight-tie the final layer across modalities
  weights: null                # path to a trained head's state_dict

whitening:
  enabled: true                # the modality-gap fix — ON by default
  n_components: null            # null => full ZCA whitening (cross-modal safe)
  remove_top_pc: 0             # drop leading PCs (sensor-identity directions)
  shrinkage: 0.9               # covariance shrinkage; high => safe (≈ mean-centering)
  fit_on: gallery              # gallery | all

index:
  type: auto                   # auto | Flat | HNSW32,Flat | IVF...,PQ... (faiss factory)
  metric: ip                   # ip (cosine on L2-normalized) | l2
  nprobe: 16
  rerank: false                # k-reciprocal re-ranking

eval:
  ks: [5, 10]
  recall_mode: raw             # raw (reported) | capped (fair model selection)
  measure_latency: true
  latency_warmup: 50
  latency_runs: 500

train:                         # used by `train` only
  epochs: 5
  lr: 0.001
  weight_decay: 0.0001
  batch_p: 8                   # classes per batch
  batch_k: 4                   # instances per class (batch = p*k)
  batches_per_epoch: null
  w_infonce: 1.5
  w_arcface: 1.0
  w_triplet: 0.5
  temperature: 0.07
  seed: 0
```

Provided configs:

| File | Purpose |
|---|---|
| `configs/default.yaml` | DOFA backbone (fallback), whitening on, image substrate |
| `configs/zero_shot.yaml` | frozen + whitening only, embedding substrate (whitening demo) |
| `configs/train_projection.yaml` | projection-head training recipe |
| `configs/backbones/{dofa,croma,remoteclip,openclip,dinov2,fallback}.yaml` | per-backbone |

---

## 4. Python API

```python
from xsretrieval import Config, run_evaluation, build_pipeline
from xsretrieval.eval import format_report

# A) One-shot evaluation from a config (loads data, fits whitener, indexes, scores).
cfg = Config.from_yaml("configs/zero_shot.yaml")
results = run_evaluation(cfg)
print(format_report(results))
print(results["headline"])   # {'F1@5_same':..., 'F1@10_cross':..., ...}

# B) Build an engine and drive it manually.
from xsretrieval import make_synthetic_multimodal, RetrievalEngine, get_backbone
from xsretrieval import PerModalityWhitener
from xsretrieval.data import make_query_gallery_split, Modality
from collections import defaultdict

samples = make_synthetic_multimodal(n_classes=8, per_class_per_modality=12, size=32)
by_mod = defaultdict(list)
for s in samples:
    by_mod[s.modality].append(s)
queries, gallery = [], []
for items in by_mod.values():
    q, g = make_query_gallery_split(items, query_frac=0.3, seed=0)
    queries += q; gallery += g

engine = RetrievalEngine(get_backbone("fallback", embed_dim=256),
                         whitener=PerModalityWhitener(shrinkage=0.9))
engine.fit_whitener(gallery)          # fit the modality-gap fix on the gallery
engine.index_gallery(gallery)         # build the shared index

hits = engine.query(queries[0], k=10, gallery_modality=Modality.SAR)
for h in hits:
    print(h["rank"], h["id"], h["modality"], round(h["score"], 4))

# C) Plug in a real foundation backbone.
bb = get_backbone("dinov2", model_name="facebook/dinov2-small", device="cpu")
emb = bb.embed(images_bchw, Modality.OPTICAL_RGB)   # (B, 384) L2-normalized
```

Key building blocks:

| Symbol | Role |
|---|---|
| `get_backbone(name, **kw)` | construct a backbone (graceful fallback) |
| `RetrievalEngine(backbone, projection=, whitener=, index_cfg=, rerank=)` | encode→whiten→index→query |
| `PerModalityWhitener(shrinkage=, remove_top_pc=, n_components=)` | the modality-gap fix |
| `RetrievalIndex(dim, index_type=, metric=)` | FAISS / numpy shared index |
| `evaluate(engine, queries, gallery, ks=, ...)` | the full F1@K matrix + latency |
| `train_projection(backbone, train_samples, config, val_samples=)` | train the head |

---

## 5. Serving

### REST API (FastAPI)

```bash
xsretrieval build-index --config configs/zero_shot.yaml --out artifacts/index
XSRETRIEVAL_INDEX=artifacts/index uvicorn apps.api:app --host 0.0.0.0 --port 8000
```

| Endpoint | Description |
|---|---|
| `GET /health` | liveness + gallery size + index backend |
| `GET /info` | backbone / whitening / modality metadata |
| `POST /query` | multipart `file` + `k` + optional `query_modality` / `gallery_modality` → top-k JSON |

```bash
curl -F "file=@query.npy" -F "k=10" -F "gallery_modality=sar" \
    http://localhost:8000/query
```

### Demo

```bash
XSRETRIEVAL_INDEX=artifacts/index python apps/demo.py
```

Launches a Gradio UI (upload a query, pick the gallery modality, see top-5/10
with scores). If Gradio is not installed it drops to a pure-CLI prompt loop, so
the demo always runs.

---

## 6. Troubleshooting

- **`ImportError: ... requires PyTorch`** — a torch-only path (losses,
  projection head, training) was invoked without torch. Install it
  (`pip install -r requirements-cpu.txt`); the numpy retrieval/whitening path
  does not need torch.
- **`index_backend=numpy` when you expected faiss** — `faiss-cpu` is not
  installed, or the build fell back. Results are identical (the numpy path is
  exact); install `faiss-cpu` for the accelerated path.
- **A real backbone "falls back"** — its weights/dependency could not be loaded
  (offline, missing extra, download error). `xsretrieval info` shows what is
  installed; the pipeline keeps running on the numpy fallback by design.
- **Cross-modal F1 is low on the `image` substrate / `fallback` backbone** —
  expected: the numpy fallback's hand-crafted features lack a clean modality
  gap, so whitening has little to remove. Use a real backbone, or the
  `embedding` substrate, to see the whitening win.
```
