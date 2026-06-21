#!/usr/bin/env bash
# Run the xsretrieval end-to-end smoke test (synthetic, with whitening).
#
# Sets PYTHONPATH to the repo root so the flat `xsretrieval` package is
# importable without an editable install, then runs the CLI smoke-test.
#
#   bash scripts/run_smoke_test.sh            # default synthetic smoke test
#   bash scripts/run_smoke_test.sh --config configs/zero_shot.yaml
set -euo pipefail

# Resolve the repo root (this script lives in <repo>/scripts).
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"
PYTHON="${PYTHON:-python}"

echo "[run_smoke_test] repo root : ${REPO_ROOT}"
echo "[run_smoke_test] python     : $(${PYTHON} --version 2>&1)"
echo "[run_smoke_test] running smoke-test ..."
echo

cd "${REPO_ROOT}"
exec "${PYTHON}" -m xsretrieval.cli smoke-test "$@"
