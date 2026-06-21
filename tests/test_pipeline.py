"""Integration tests for the orchestration layer (config + pipeline + CLI).

These cover the wiring added by the integration lead on top of the sub-modules:

* :class:`xsretrieval.config.Config` — defaults, dict/YAML round-trip, partial
  override merging.
* :func:`xsretrieval.pipeline.build_pipeline` / :func:`run_evaluation` — the
  config → engine → evaluation path on synthetic data, including the synthetic
  fallback and the **whitening win** on the embedding substrate.
* :func:`xsretrieval.cli.cmd_smoke_test` — the smoke-test command exits 0.

Everything here runs on bare numpy (the ``precomputed`` / ``fallback`` backbones
and the numpy index path); no torch/faiss required.
"""

from __future__ import annotations

import numpy as np
import pytest

from xsretrieval.config import Config


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
def test_config_defaults_enable_whitening() -> None:
    cfg = Config.default()
    assert cfg.whitening.enabled is True, "the default pipeline must whiten"
    assert cfg.eval.ks == [5, 10]
    assert cfg.backbone.embed_dim > 0
    assert 0.0 < cfg.data.query_frac < 1.0


def test_config_dict_roundtrip() -> None:
    cfg = Config.default()
    again = Config.from_dict(cfg.to_dict())
    assert again.to_dict() == cfg.to_dict()


def test_config_partial_override_merges_defaults() -> None:
    cfg = Config.from_dict(
        {"name": "partial", "backbone": {"name": "fallback", "embed_dim": 64}}
    )
    assert cfg.name == "partial"
    assert cfg.backbone.name == "fallback"
    assert cfg.backbone.embed_dim == 64
    # Untouched sections keep their defaults.
    assert cfg.whitening.enabled is True
    assert cfg.eval.ks == [5, 10]


def test_config_yaml_roundtrip(tmp_path) -> None:
    pytest.importorskip("yaml", reason="pyyaml not installed")
    cfg = Config.default()
    cfg.name = "yaml_run"
    path = tmp_path / "cfg.yaml"
    cfg.to_yaml(str(path))
    loaded = Config.from_yaml(str(path))
    assert loaded.name == "yaml_run"
    assert loaded.backbone.name == cfg.backbone.name
    assert loaded.whitening.shrinkage == cfg.whitening.shrinkage


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------
def _embedding_config() -> Config:
    cfg = Config.default()
    cfg.data.substrate = "embedding"
    cfg.data.dataset = "synthetic"
    cfg.data.n_classes = 8
    cfg.data.per_class = 16
    cfg.backbone.name = "precomputed"
    cfg.backbone.embed_dim = 64
    cfg.eval.measure_latency = False
    return cfg


def test_build_pipeline_returns_engine() -> None:
    from xsretrieval.pipeline import build_pipeline
    from xsretrieval.retrieval.engine import RetrievalEngine

    cfg = _embedding_config()
    engine = build_pipeline(cfg)
    assert isinstance(engine, RetrievalEngine)
    assert engine.whitener is not None  # whitening enabled by default


def test_run_evaluation_headline_above_chance() -> None:
    from xsretrieval.pipeline import run_evaluation

    cfg = _embedding_config()
    res = run_evaluation(cfg)
    h = res["headline"]
    # All four headline F1s present and in [0, 1].
    for key in ("F1@5_same", "F1@10_same", "F1@5_cross", "F1@10_cross"):
        assert key in h
        assert 0.0 <= h[key] <= 1.0
    chance = 1.0 / cfg.data.n_classes
    assert h["F1@10_same"] > chance, "same-modal F1@10 should beat chance"
    assert h["F1@10_cross"] > chance, "cross-modal F1@10 should beat chance"
    assert res["meta"]["whitening"] is True
    assert res["meta"]["backbone_class"] == "PrecomputedBackbone"


def test_whitening_improves_cross_modal() -> None:
    """The modality-gap fix must lift cross-modal F1 on the embedding substrate."""
    from xsretrieval.pipeline import run_evaluation

    cfg_on = _embedding_config()
    cfg_on.data.modality_shift = 2.5
    cfg_off = Config.from_dict(cfg_on.to_dict())
    cfg_off.whitening.enabled = False

    res_on = run_evaluation(cfg_on)
    res_off = run_evaluation(cfg_off)
    cross_on = res_on["headline"]["F1@10_cross"]
    cross_off = res_off["headline"]["F1@10_cross"]
    assert cross_on > cross_off, (
        f"whitening should lift cross-modal F1@10: off={cross_off:.4f} "
        f"on={cross_on:.4f}"
    )


def test_run_evaluation_image_substrate_runs() -> None:
    """The full image encode path (FallbackBackbone) produces valid headlines."""
    from xsretrieval.pipeline import run_evaluation

    cfg = Config.default()
    cfg.data.substrate = "image"
    cfg.data.n_classes = 6
    cfg.data.per_class = 10
    cfg.data.size = 24
    cfg.backbone.name = "fallback"
    cfg.backbone.embed_dim = 128
    cfg.eval.measure_latency = False
    res = run_evaluation(cfg)
    assert res["headline"]["F1@10_same"] > 1.0 / cfg.data.n_classes


def test_real_dataset_falls_back_to_synthetic() -> None:
    """A missing real dataset path degrades to synthetic rather than crashing."""
    from xsretrieval.pipeline import load_samples

    cfg = Config.default()
    cfg.data.dataset = "eurosat"
    cfg.data.root = "/nonexistent/path/does/not/exist"
    cfg.data.n_classes = 5
    cfg.data.per_class = 6
    samples = load_samples(cfg)
    assert len(samples) > 0  # synthetic fallback kicked in


def test_encode_dataset_shapes() -> None:
    from xsretrieval.pipeline import encode_dataset

    cfg = _embedding_config()
    out = encode_dataset(cfg)
    n = len(out["labels"])
    assert out["embeddings"].shape[0] == n
    assert out["embeddings"].ndim == 2
    assert len(out["modalities"]) == n
    assert len(out["ids"]) == n


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def test_cli_smoke_test_exits_zero(capsys) -> None:
    from xsretrieval.cli import main

    code = main(["smoke-test"])
    captured = capsys.readouterr()
    assert code == 0
    assert "PASS" in captured.out
    assert "cross-modal" in captured.out


def test_cli_info_exits_zero(capsys) -> None:
    from xsretrieval.cli import main

    code = main(["info"])
    captured = capsys.readouterr()
    assert code == 0
    assert "xsretrieval" in captured.out
    assert "Registered backbones" in captured.out
