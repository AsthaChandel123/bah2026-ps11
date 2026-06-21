#!/usr/bin/env bash
#
# deploy_cloudrun.sh — build + push + deploy the xsretrieval FastAPI service to
# Google Cloud Run. Convenience wrapper around the steps in docs/DEPLOYMENT.md.
#
# !!! REVIEW BEFORE RUNNING !!!  This creates/updates cloud resources and may incur
# cost. Read it, then run it. It is intentionally explicit and fail-fast.
#
# Usage:
#   PROJECT_ID=my-proj REGION=us-central1 ./deploy/deploy_cloudrun.sh
#
# Required env vars:
#   PROJECT_ID   GCP project id
# Optional env vars (defaults shown):
#   REGION       us-central1        Cloud Run + Artifact Registry region
#   REPO         xsretrieval        Artifact Registry repository name
#   SERVICE      xsretrieval-api    Cloud Run service name
#   IMAGE        <derived>          full image ref; default:
#                                   ${REGION}-docker.pkg.dev/${PROJECT_ID}/${REPO}/${SERVICE}:latest
#   INDEX_DIR    /app/artifacts/index   bundle dir inside the container (baked-in)
#   MODEL_URI    (unset)            gs:// bundle to download at start (requires the
#                                   MODEL_URI startup TODO in docs/DEPLOYMENT.md)
#   MEMORY       4Gi
#   CPU          2
#   PORT         8000
#   ALLOW_UNAUTH true              pass --allow-unauthenticated when "true"
#   USE_GPU      false             when "true", add --gpu 1 --gpu-type nvidia-l4
#   BUILD        cloudbuild        "cloudbuild" (gcloud builds submit) or "docker"

set -euo pipefail

# --- required / defaults ----------------------------------------------------
: "${PROJECT_ID:?Set PROJECT_ID (your GCP project id)}"
REGION="${REGION:-us-central1}"
REPO="${REPO:-xsretrieval}"
SERVICE="${SERVICE:-xsretrieval-api}"
IMAGE="${IMAGE:-${REGION}-docker.pkg.dev/${PROJECT_ID}/${REPO}/${SERVICE}:latest}"
INDEX_DIR="${INDEX_DIR:-/app/artifacts/index}"
MEMORY="${MEMORY:-4Gi}"
CPU="${CPU:-2}"
PORT="${PORT:-8000}"
ALLOW_UNAUTH="${ALLOW_UNAUTH:-true}"
USE_GPU="${USE_GPU:-false}"
BUILD="${BUILD:-cloudbuild}"

# Run from the repo root regardless of where this is invoked.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

echo "==> Project : ${PROJECT_ID}"
echo "==> Region  : ${REGION}"
echo "==> Image   : ${IMAGE}"
echo "==> Service : ${SERVICE}"
echo "==> Build   : ${BUILD}"
echo "==> Repo dir: ${REPO_ROOT}"
echo

if [[ ! -f "${REPO_ROOT}/Dockerfile" ]]; then
  echo "ERROR: Dockerfile not found in ${REPO_ROOT}" >&2
  exit 1
fi
if [[ ! -d "${REPO_ROOT}/artifacts/index" ]]; then
  echo "WARNING: ${REPO_ROOT}/artifacts/index not found — the image will have no"
  echo "         baked-in bundle. Build it first with:"
  echo "           python -m xsretrieval.cli build-index --config <cfg> --out artifacts/index"
  echo "         (or supply MODEL_URI for runtime download; see docs/DEPLOYMENT.md)."
fi

# --- 1) enable APIs + ensure the Artifact Registry repo exists --------------
echo "==> Enabling required GCP services ..."
gcloud config set project "${PROJECT_ID}" >/dev/null
gcloud services enable \
  run.googleapis.com \
  artifactregistry.googleapis.com \
  cloudbuild.googleapis.com

echo "==> Ensuring Artifact Registry repo '${REPO}' in ${REGION} ..."
gcloud artifacts repositories describe "${REPO}" --location "${REGION}" >/dev/null 2>&1 \
  || gcloud artifacts repositories create "${REPO}" \
       --repository-format=docker --location "${REGION}" \
       --description="xsretrieval serving images"

# --- 2) build + push the image ----------------------------------------------
if [[ "${BUILD}" == "docker" ]]; then
  echo "==> Building locally with docker ..."
  gcloud auth configure-docker "${REGION}-docker.pkg.dev" --quiet
  docker build -t "${IMAGE}" .
  echo "==> Pushing ${IMAGE} ..."
  docker push "${IMAGE}"
else
  echo "==> Building with Cloud Build ..."
  gcloud builds submit --tag "${IMAGE}" .
fi

# --- 3) deploy to Cloud Run -------------------------------------------------
ENV_VARS="XSRETRIEVAL_INDEX=${INDEX_DIR},PORT=${PORT}"
if [[ -n "${MODEL_URI:-}" ]]; then
  ENV_VARS="${ENV_VARS},MODEL_URI=${MODEL_URI}"
fi

DEPLOY_ARGS=(
  run deploy "${SERVICE}"
  --image "${IMAGE}"
  --region "${REGION}"
  --platform managed
  --memory "${MEMORY}"
  --cpu "${CPU}"
  --port "${PORT}"
  --set-env-vars "${ENV_VARS}"
)
if [[ "${ALLOW_UNAUTH}" == "true" ]]; then
  DEPLOY_ARGS+=(--allow-unauthenticated)
fi
if [[ "${USE_GPU}" == "true" ]]; then
  # Cloud Run GPU — only if inference latency requires it (see docs/DEPLOYMENT.md).
  DEPLOY_ARGS+=(--gpu 1 --gpu-type nvidia-l4 --max-instances 1)
fi

echo "==> Deploying: gcloud ${DEPLOY_ARGS[*]}"
gcloud "${DEPLOY_ARGS[@]}"

# --- 4) report the URL + a health hint --------------------------------------
URL="$(gcloud run services describe "${SERVICE}" --region "${REGION}" \
        --format='value(status.url)')"
echo
echo "==> Deployed: ${URL}"
echo "==> Smoke test:"
echo "      curl ${URL}/health"
echo "      curl ${URL}/info"
echo "      curl -F 'file=@query.png' -F 'k=10' -F 'gallery_modality=sar' ${URL}/query"
