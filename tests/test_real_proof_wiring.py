"""Light, network-free wiring tests for ``scripts/run_real_proof.py``.

These confirm the automated real-data proof orchestration is wired end to end
**without any download or heavy dependency**: the script imports, its argparse
CLI builds, the device resolver behaves, the class-balanced split sizes ``R_q``
correctly, and the full ``run_real_proof`` pipeline runs on *injected* synthetic
samples with the numpy ``FallbackBackbone`` and writes a results dict + a
``results.json`` artifact.

Everything here runs on bare numpy (the ``FallbackBackbone`` + numpy/faiss index
path); torch/faiss beyond what CI installs is not required (faiss-cpu is used if
present, else the numpy index fallback). No network access.
"""

from __future__ import annotations

import importlib.util
import json
import os
import sys
from pathlib import Path

import numpy as np
import pytest

# Make both the repo root and the scripts/ dir importable so the script (which
# is not part of the installed package) can be imported as a module.
_REPO_ROOT = Path(__file__).resolve().parents[1]
for p in (str(_REPO_ROOT), str(_REPO_ROOT / "scripts")):
    if p not in sys.path:
        sys.path.insert(0, p)


def _load_module():
    """Import scripts/run_real_proof.py as a module (no install needed)."""
    spec = importlib.util.spec_from_file_location(
        "run_real_proof", str(_REPO_ROOT / "scripts" / "run_real_proof.py")
    )
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


rp = _load_module()


def _synthetic_rgb_samples(n_classes: int = 5, per_class: int = 10, size: int = 16):
    """A small single-modality (optical_rgb) synthetic Sample list (no I/O)."""
    from xsretrieval.data.modalities import Modality
    from xsretrieval.data.synthetic import make_synthetic_multimodal

    return make_synthetic_multimodal(
        n_classes=n_classes,
        per_class_per_modality=per_class,
        modalities=[Modality.OPTICAL_RGB],
        size=size,
        seed=0,
    )


# ---------------------------------------------------------------------------
# Import + parser
# ---------------------------------------------------------------------------
def test_module_imports_and_exposes_entrypoints() -> None:
    assert hasattr(rp, "run_real_proof")
    assert hasattr(rp, "build_parser")
    assert hasattr(rp, "main")


def test_arg_parser_builds_with_defaults() -> None:
    parser = rp.build_parser()
    args = parser.parse_args([])
    assert args.dataset == "eurosat"
    assert args.backbone == "auto"
    assert args.device == "auto"
    assert args.train is False  # zero-shot by default
    assert args.ks == [5, 10]


def test_arg_parser_train_and_subset_flags() -> None:
    parser = rp.build_parser()
    args = parser.parse_args(
        ["--train", "--subset", "1500", "--gallery-per-class", "8",
         "--backbone", "dinov2", "--device", "cpu"]
    )
    assert args.train is True
    assert args.subset == 1500
    assert args.gallery_per_class == 8
    assert args.backbone == "dinov2"


def test_train_and_no_train_are_mutually_exclusive() -> None:
    # --train / --no-train form a mutually-exclusive group: passing both is a
    # usage error (argparse exits non-zero) rather than silently ambiguous.
    with pytest.raises(SystemExit):
        rp.build_parser().parse_args(["--train", "--no-train"])
    # Each on its own works.
    assert rp.build_parser().parse_args(["--train"]).train is True
    assert rp.build_parser().parse_args(["--no-train"]).train is False


# ---------------------------------------------------------------------------
# Device resolution (no cuda in CI)
# ---------------------------------------------------------------------------
def test_resolve_device_cpu_and_auto() -> None:
    assert rp.resolve_device("cpu") == "cpu"
    # auto resolves to cpu when cuda is unavailable (CI is CPU-only).
    assert rp.resolve_device("auto") in ("cpu", "cuda")
    # A cuda request on a CPU box degrades to cpu.
    assert rp.resolve_device("cuda") == "cpu"


# ---------------------------------------------------------------------------
# Class-balanced split sizes R_q correctly
# ---------------------------------------------------------------------------
def test_class_balanced_split_sizes_rq() -> None:
    samples = _synthetic_rgb_samples(n_classes=4, per_class=12)
    queries, gallery = rp.class_balanced_split(
        samples, gallery_per_class=5, query_per_class=3, seed=0
    )
    assert queries and gallery
    # Every class contributes exactly gallery_per_class gallery items.
    from collections import Counter

    gcounts = Counter(int(s.label) for s in gallery)
    assert set(gcounts.values()) == {5}, gcounts
    qcounts = Counter(int(s.label) for s in queries)
    assert all(c <= 3 for c in qcounts.values())
    # Disjoint by sample id.
    assert not ({s.id for s in queries} & {s.id for s in gallery})


def test_subset_class_balanced_caps_total() -> None:
    samples = _synthetic_rgb_samples(n_classes=5, per_class=20)
    sub = rp._subset_class_balanced(samples, subset=30, seed=0)
    assert len(sub) == 30
    # Round-robin keeps every class represented.
    from collections import Counter

    counts = Counter(int(s.label) for s in sub)
    assert len(counts) == 5


# ---------------------------------------------------------------------------
# Backbone resolution falls back to numpy when asked for fallback
# ---------------------------------------------------------------------------
def test_build_backbone_fallback_info() -> None:
    bb, info = rp.build_backbone("fallback", embed_dim=64, device="cpu")
    assert info["class"] == "FallbackBackbone"
    assert info["is_fallback"] is True
    assert info["multispectral_aware"] is False
    assert info["embed_dim"] == 64


# ---------------------------------------------------------------------------
# Full orchestration on injected synthetic samples (no downloads)
# ---------------------------------------------------------------------------
def test_run_real_proof_end_to_end_fallback(tmp_path) -> None:
    samples = _synthetic_rgb_samples(n_classes=5, per_class=10, size=16)
    out_dir = tmp_path / "artifacts"
    results = rp.run_real_proof(
        samples=samples,
        backbone="fallback",
        device="cpu",
        gallery_per_class=5,
        query_per_class=3,
        train=False,
        out=str(out_dir),
        report=None,
        write_report=False,
        latency_runs=10,
        latency_warmup=2,
        embed_dim=64,
        seed=0,
    )

    # A results dict with the four headline keys present and in [0, 1].
    head = results["headline"]
    for key in ("F1@5_same", "F1@10_same", "F1@5_cross", "F1@10_cross"):
        assert key in head
        assert 0.0 <= head[key] <= 1.0
    # Same-modal optical retrieval on the fallback descriptor must beat chance.
    assert head["F1@10_same"] > 1.0 / 5

    # Meta + backbone provenance recorded.
    assert results["meta"]["index_backend"] in ("faiss", "numpy")
    assert results["backbone"]["is_fallback"] is True
    assert results["backbone"]["class"] == "FallbackBackbone"
    assert results["cross_modal_real"] is False  # single modality, fallback

    # results.json artifact written and parseable.
    rj = out_dir / "results.json"
    assert rj.exists()
    payload = json.loads(rj.read_text())
    assert "headline" in payload and "meta" in payload
    assert payload["backbone"]["class"] == "FallbackBackbone"


def test_run_real_proof_unsupported_dataset_without_samples() -> None:
    # sen12ms download is not automated here; without injected samples it must
    # raise a clear error rather than silently going synthetic.
    with pytest.raises(NotImplementedError):
        rp.run_real_proof(
            dataset="sen12ms",
            backbone="fallback",
            device="cpu",
            report=None,
            write_report=False,
        )
