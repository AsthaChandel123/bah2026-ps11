#!/usr/bin/env bash
# Run the xsretrieval automated REAL-DATA proof.
#
# Downloads real satellite imagery (EuroSAT by default), embeds it with a real
# foundation backbone, and produces genuine F1@5/@10 + latency. Sets PYTHONPATH
# to the repo root so the flat `xsretrieval` package is importable without an
# editable install, then forwards all arguments to scripts/run_real_proof.py.
#
#   bash scripts/run_real_proof.sh                       # EuroSAT, auto backbone, CPU-sane
#   bash scripts/run_real_proof.sh --subset 1500 --no-train
#   bash scripts/run_real_proof.sh --device cuda --dataset sen12ms --backbone dofa --train
set -euo pipefail

# Resolve the repo root (this script lives in <repo>/scripts).
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"
PYTHON="${PYTHON:-python}"

echo "[run_real_proof] repo root : ${REPO_ROOT}"
echo "[run_real_proof] python     : $(${PYTHON} --version 2>&1)"
echo "[run_real_proof] running real-data proof ..."
echo

cd "${REPO_ROOT}"

# Default to EuroSAT when no dataset/positional args are given; pass everything
# the caller supplied straight through.
if [ "$#" -eq 0 ]; then
  exec "${PYTHON}" scripts/run_real_proof.py --dataset eurosat
fi
exec "${PYTHON}" scripts/run_real_proof.py "$@"
