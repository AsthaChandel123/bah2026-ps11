# Deployment — Phase 2: frontend + model on Google Cloud Run

> Prerequisite: **Goal A is met** (see `docs/GPU_RUNBOOK.md`) and the trained
> artifacts exist in `artifacts/` (projection head, FAISS gallery index, fitted
> whitener, config). This guide ships the model as a FastAPI service and a frontend
> on **Google Cloud Run**. Every command/path below matches the real code
> (`apps/api.py`, `apps/demo.py`, the `xsretrieval` CLI, the repo `Dockerfile`).

---

## Architecture

```
                         Google Cloud Run
  ┌──────────────────────────────────────────────────────────────────────┐
  │                                                                        │
  │   ┌───────────────────────────┐        ┌──────────────────────────┐   │
  │   │  FRONTEND service          │  HTTP  │  API service              │   │
  │   │  apps/demo.py (Gradio)     │ ─────► │  apps.api:app (FastAPI)   │   │
  │   │  OR a static SPA           │  /query│  uvicorn, $PORT           │   │
  │   │  -> calls the API URL      │ ◄───── │                           │   │
  │   └───────────────────────────┘  JSON  │  loads the bundle:        │   │
  │                                         │   index + whitener +      │   │
  │                                         │   config + gallery_meta   │   │
  │                                         │   (XSRETRIEVAL_INDEX dir) │   │
  │                                         │  + backbone (HF weights)  │   │
  │                                         └──────────────────────────┘   │
  └──────────────────────────────────────────────────────────────────────┘
        trained artifacts/ (head.pt, index/, whitener.npz, config.yaml)
        ──► baked into the image (demo gallery)  OR  pulled from a GCS bucket
        backbone weights ──► from HuggingFace at build time  OR  cached in GCS
```

The API loads a **persisted retrieval bundle** (produced by
`xsretrieval build-index --out <dir>`) on first request, reconstructs the engine
(backbone + whitener + index), and answers `/query`. The frontend (Gradio demo or a
static page) calls the API. Both run as independent Cloud Run services.

**Endpoints the API exposes** (verified in `apps/api.py`):

- `GET /health` — liveness + gallery size + index backend (`faiss`/`numpy`).
- `GET /info` — backbone class, embed dim, whitening flag, gallery size, modalities.
- `POST /query` — multipart form: `file` (image or `.npy` embedding), `k` (int,
  default 10), optional `query_modality`, optional `gallery_modality`. Returns JSON
  `{query_modality, gallery_modality, k, results:[{rank,id,modality,label,score}]}`.

The bundle directory is read from the **`XSRETRIEVAL_INDEX`** env var (default
`artifacts/index`); the server port is **`PORT`** (default 8000) — both line up with
Cloud Run's conventions.

---

## Where the model lives ("wherever necessitated")

The trained artifacts are small (projection head ~MB, whitener ~MB, a demo-sized FAISS
index ~MB–tens of MB). Backbone weights are the large part (hundreds of MB) and come
from HuggingFace. Three options:

| Option | What | When |
|---|---|---|
| **(i) Bake artifacts into the image** | Copy `artifacts/index/` (index + whitener + config + gallery_meta) into the container at build time. | **Recommended for a demo / small fixed gallery.** Simplest, fastest cold start, fully self-contained. |
| **(ii) Artifacts in a GCS bucket** | Store a larger `index/` (or a production gallery) in `gs://<bucket>/...`; download at container start via env `MODEL_URI=gs://...`. | **Production / large galleries** that you do not want to rebuild into every image, or that update independently of the code. |
| **(iii) Backbone from HuggingFace** | Pull the backbone weights (`croma` / `dofa` / `remoteclip` / `dinov2`) at **image build time** and **pin the revision**; optionally cache them in GCS for air-gapped/faster builds. | Always — the backbone is needed to embed the query. Pin the HF revision for reproducibility. |

**Recommendation:** **(i) + (iii)** for a demo (bake the small artifacts, pull the
pinned backbone at build); **(ii) + (iii)** for production with a large gallery
(artifacts in GCS, backbone pinned/cached).

> **`TODO (wire this)` — `MODEL_URI` GCS download is a deployment convention, not yet
> code.** `apps/api.py` reads the bundle from the local `XSRETRIEVAL_INDEX` directory
> only. For option (ii), add a tiny startup step that, if `MODEL_URI` is set,
> `gsutil cp -r "$MODEL_URI" "$XSRETRIEVAL_INDEX"` (or uses `google-cloud-storage`)
> *before* uvicorn starts — e.g. in a small entrypoint wrapper or at the top of the
> container `CMD`. Until then, use option (i) and bake `artifacts/index/` into the
> image.

---

## Containerize

The repo ships a serving **`Dockerfile`** (python:3.11-slim, CPU torch + faiss-cpu +
the serving deps, `pip install -e .`, runs `uvicorn apps.api:app`) and a
**`.dockerignore`**. Build it locally first:

```bash
# from the repo root, with artifacts/index/ present (baked in by the Dockerfile)
docker build -t xsretrieval-api:latest .

# smoke-test locally
docker run --rm -p 8000:8000 -e XSRETRIEVAL_INDEX=/app/artifacts/index xsretrieval-api:latest
curl http://localhost:8000/health
```

For **GPU inference** on Cloud Run (only if the backbone is too slow on CPU at your
latency target), build from a CUDA base instead (e.g. `nvidia/cuda:12.1.1-runtime-
ubuntu22.04` + a Python install + GPU torch wheels); the `Dockerfile` documents this
alternative in comments. CPU is usually sufficient — search is sub-ms and the backbone
runs one forward pass per query.

---

## Deploy commands (concrete gcloud)

Set your project/region once:

```bash
export PROJECT_ID=your-gcp-project
export REGION=us-central1
export REPO=xsretrieval                       # Artifact Registry repo name
export IMAGE="${REGION}-docker.pkg.dev/${PROJECT_ID}/${REPO}/xsretrieval-api:latest"

gcloud config set project "$PROJECT_ID"
gcloud services enable run.googleapis.com artifactregistry.googleapis.com cloudbuild.googleapis.com
gcloud artifacts repositories create "$REPO" --repository-format=docker --location="$REGION" || true
```

**Build + push the image** (either Cloud Build or local docker):

```bash
# Option A — Cloud Build (no local docker needed):
gcloud builds submit --tag "$IMAGE" .

# Option B — local docker:
gcloud auth configure-docker "${REGION}-docker.pkg.dev"
docker build -t "$IMAGE" .
docker push "$IMAGE"
```

**Deploy the API service:**

```bash
gcloud run deploy xsretrieval-api \
  --image "$IMAGE" \
  --region "$REGION" \
  --platform managed \
  --allow-unauthenticated \
  --memory 4Gi --cpu 2 \
  --port 8000 \
  --set-env-vars XSRETRIEVAL_INDEX=/app/artifacts/index
#   For option (ii) GCS artifacts (after wiring the MODEL_URI TODO above):
#     --set-env-vars XSRETRIEVAL_INDEX=/tmp/index,MODEL_URI=gs://your-bucket/index
#
#   Cloud Run GPU (only if needed for inference latency):
#     --gpu 1 --gpu-type nvidia-l4 --max-instances 1
```

Note: Cloud Run honors the **`PORT`** env var it injects; `apps/api.py` already reads
`PORT` (default 8000), and `--port 8000` makes it explicit. Capture the service URL:

```bash
export API_URL=$(gcloud run services describe xsretrieval-api --region "$REGION" \
  --format='value(status.url)')
echo "$API_URL"
```

**Deploy the frontend.** Two choices:

- **(A) Gradio demo as a second service** (`apps/demo.py`). It reads the same bundle
  and serves a UI on `PORT` (default 7860 locally; Cloud Run injects `PORT`). Build a
  second image whose `CMD` runs `python apps/demo.py` (or override the entrypoint),
  bake/point it at the bundle, and deploy:

  ```bash
  gcloud run deploy xsretrieval-demo \
    --image "$IMAGE" \
    --region "$REGION" --allow-unauthenticated \
    --memory 4Gi --cpu 2 --port 8080 \
    --command python --args apps/demo.py \
    --set-env-vars XSRETRIEVAL_INDEX=/app/artifacts/index,PORT=8080
  ```

  > **`TODO (wire this)`** — `apps/demo.py` is **self-contained** (it loads the bundle
  > and runs retrieval in-process); it does **not** call the API service. That is fine
  > for a single combined demo. If you specifically want the frontend to call the
  > **API** over HTTP (decoupled services), point a thin client / SPA at
  > `POST $API_URL/query` instead — see (B).

- **(B) Static SPA** that POSTs to `$API_URL/query`. Host any static page (Cloud Run
  with an nginx image, Firebase Hosting, or GCS static site) and wire its fetch base
  URL to `$API_URL`. The `/query` contract is the multipart form documented above.

---

## Smoke test the deployment

```bash
# health (no bundle load required)
curl "$API_URL/health"
# -> {"status":"ok","gallery_size":<n>,"index_backend":"faiss"|"numpy"}

# info (loads the bundle)
curl "$API_URL/info"

# query with an image file (or a .npy embedding) — multipart form fields match apps/api.py
curl -F "file=@query.png" -F "k=10" -F "query_modality=optical_rgb" \
     -F "gallery_modality=sar" "$API_URL/query"
# -> {"query_modality":"optical_rgb","gallery_modality":"sar","k":10,"results":[...]}
```

---

## Cost / scaling notes

- **Cold starts vs. cost** — the bundle + backbone load on the first request. For a
  responsive demo set **`--min-instances 1`** (keeps one warm; costs idle CPU/RAM);
  for cheap/bursty use leave it at 0 and accept a slower first query.
- **Concurrency** — the engine is read-only after load, so a single instance can serve
  concurrent queries. Tune `--concurrency` (default 80) down if backbone inference is
  CPU-heavy and you see latency under load; scale out with `--max-instances`.
- **Memory** — `--memory 4Gi` comfortably holds CPU torch + a foundation backbone + a
  demo-sized index. A large production index may need more; or move it to GCS
  (option ii) and a bigger instance.
- **CPU vs GPU inference** — CPU is the default and is usually enough (sub-ms search,
  one backbone forward per query). Use Cloud Run **GPU** (`--gpu 1 --gpu-type
  nvidia-l4`) only if you measure the backbone forward pass as the latency bottleneck
  at your target QPS; it raises cost and constrains `--max-instances`.
- **The 5th scored metric is latency** — measure it the way `evaluate` does
  (warmup, batch=1, search-only) and report the deployed end-to-end number
  separately; do not conflate batched throughput with per-query latency.

See `deploy/deploy_cloudrun.sh` for a parameterized wrapper of the build + deploy
steps above (review it before running).
