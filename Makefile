# xsretrieval — developer Makefile.
#
# Common targets for installing, testing, and running the cross-modal satellite
# image retrieval pipeline. The numpy retrieval path needs no heavy deps; the
# CPU compute stack (torch/faiss) is installed via `make install-cpu`.

PYTHON ?= python
PIP ?= $(PYTHON) -m pip

.PHONY: help install install-cpu test smoke evaluate train build-index info lint clean

help:  ## Show this help.
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) \
		| sort \
		| awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-14s\033[0m %s\n", $$1, $$2}'

install:  ## Editable install of the package (general requirements).
	$(PIP) install -e .
	$(PIP) install -r requirements.txt

install-cpu:  ## Editable install + CPU-pinned deps (torch CPU, faiss-cpu, ...).
	$(PIP) install -e .
	$(PIP) install -r requirements-cpu.txt

test:  ## Run the full test suite.
	$(PYTHON) -m pytest -q

smoke:  ## Run the end-to-end synthetic smoke test (with whitening).
	$(PYTHON) -m xsretrieval.cli smoke-test

evaluate:  ## Evaluate the default config (synthetic fallback). CONFIG overridable.
	$(PYTHON) -m xsretrieval.cli evaluate --config $(or $(CONFIG),configs/zero_shot.yaml)

train:  ## Train a projection head (torch). CONFIG overridable.
	$(PYTHON) -m xsretrieval.cli train --config $(or $(CONFIG),configs/train_projection.yaml) --out head.pt

build-index:  ## Build a retrieval bundle. CONFIG / OUT overridable.
	$(PYTHON) -m xsretrieval.cli build-index --config $(or $(CONFIG),configs/zero_shot.yaml) --out $(or $(OUT),artifacts/index)

info:  ## Print environment + backbone availability.
	$(PYTHON) -m xsretrieval.cli info

lint:  ## Lint with ruff if available (no-op otherwise).
	@$(PYTHON) -c "import ruff" 2>/dev/null && ruff check xsretrieval tests || \
		echo "ruff not installed; skipping (pip install ruff)"

clean:  ## Remove caches and build artifacts.
	rm -rf build dist *.egg-info .pytest_cache .ruff_cache artifacts
	find . -type d -name __pycache__ -prune -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name '*.pyc' -delete 2>/dev/null || true
