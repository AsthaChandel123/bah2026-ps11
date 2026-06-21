"""Dataclass-based configuration for the ``xsretrieval`` pipeline (+ YAML loader).

A single :class:`Config` object describes an end-to-end retrieval run — which
dataset to load, which backbone to build, whether to project / whiten, how to
index, and how to evaluate (and, optionally, train). It is intentionally small,
fully typed, and has **sane defaults that match the system architecture**, so
``Config.default()`` already yields a strong zero-training pipeline (foundation
backbone w/ graceful fallback + per-modality whitening + FAISS-or-numpy index).

YAML is imported lazily inside :meth:`Config.from_yaml`, so importing this module
needs only the standard library — the numpy retrieval path never requires
``pyyaml`` unless a YAML config is actually loaded.

Layout::

    Config
    ├── data        DataConfig        (dataset, modalities, query/gallery split)
    ├── backbone    BackboneConfig    (name + kwargs + embed_dim)
    ├── projection  ProjectionConfig  (trainable shared head)
    ├── whitening   WhiteningConfig   (the modality-gap fix — ON by default)
    ├── index       IndexConfig       (faiss type / metric / re-rank)
    ├── eval        EvalConfig        (cutoffs, recall mode, latency)
    └── train       TrainConfig       (epochs, lr, loss weights)
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, fields, is_dataclass
from typing import Any, Optional

__all__ = [
    "DataConfig",
    "BackboneConfig",
    "ProjectionConfig",
    "WhiteningConfig",
    "IndexConfig",
    "EvalConfig",
    "TrainConfig",
    "Config",
]


# ---------------------------------------------------------------------------
# Sub-configs
# ---------------------------------------------------------------------------
@dataclass
class DataConfig:
    """Dataset selection and the query/gallery evaluation split.

    Attributes
    ----------
    dataset:
        Dataset key (``"synthetic"``, ``"eurosat"``, ``"sen12ms"``,
        ``"folder"``). When the real dataset is not present on disk the pipeline
        transparently falls back to a synthetic multi-modal dataset so a run
        always completes.
    root:
        Filesystem root of the real dataset (ignored for ``"synthetic"``).
    modalities:
        Modalities to include (string values of
        :class:`~xsretrieval.data.modalities.Modality`).
    size:
        Square image size used for synthetic generation / resizing.
    n_classes / per_class:
        Synthetic dataset shape (classes, and locations per class per modality).
    query_frac:
        Fraction of each class assigned to the query set (rest is the gallery).
    class_balanced_gallery:
        Truncate the gallery so each class contributes equally (macro-fair F1).
    substrate:
        Synthetic substrate: ``"image"`` renders multi-modal images (exercises
        the full encode path), ``"embedding"`` emits pre-encoded vectors that
        carry an explicit **modality gap** (the regime where per-modality
        whitening demonstrably lifts cross-modal F1 — used by the smoke test).
    modality_shift:
        Magnitude of the synthetic per-modality offset (the modality gap) for
        the ``"embedding"`` substrate. Larger ⇒ bigger gap ⇒ bigger whitening
        win.
    seed:
        Master RNG seed for data generation / splitting.
    """

    dataset: str = "synthetic"
    root: Optional[str] = None
    modalities: list[str] = field(
        default_factory=lambda: ["optical_rgb", "multispectral", "sar"]
    )
    size: int = 64
    n_classes: int = 10
    per_class: int = 16
    query_frac: float = 0.3
    class_balanced_gallery: bool = True
    substrate: str = "image"
    modality_shift: float = 2.5
    seed: int = 0


@dataclass
class BackboneConfig:
    """Feature-extraction backbone selection.

    Attributes
    ----------
    name:
        Registry key: ``"dofa"``, ``"croma"``, ``"remoteclip"``, ``"openclip"``,
        ``"dinov2"``, ``"timm"``, ``"fallback"``, or ``"precomputed"`` (serve
        pre-encoded embedding "images" verbatim — used by the embedding
        substrate). Any backbone that fails to build (missing extra / failed
        download) falls back to the numpy ``FallbackBackbone`` automatically.
    embed_dim:
        Embedding width requested from the backbone (and the fallback).
    kwargs:
        Extra keyword arguments forwarded to the backbone constructor (e.g.
        ``model_name``, ``device``, ``image_size``). Unknown keys are dropped by
        ``get_backbone`` rather than triggering the fallback.
    """

    name: str = "dofa"
    embed_dim: int = 256
    kwargs: dict[str, Any] = field(default_factory=dict)


@dataclass
class ProjectionConfig:
    """Optional trainable per-modality projection head (torch).

    Disabled by default (the zero-training pipeline relies on whitening alone).
    When enabled, a :class:`~xsretrieval.models.projection.ProjectionHeads` maps
    backbone features into a shared ``out_dim`` space; ``share_final`` weight-ties
    the last layer across modalities (a strong cross-modal aligner).
    """

    enabled: bool = False
    out_dim: int = 256
    hidden: int = 512
    dropout: float = 0.1
    share_final: bool = True
    weights: Optional[str] = None


@dataclass
class WhiteningConfig:
    """Per-modality mean-center + whitening — the modality-gap fix.

    **Enabled by default**: this is the single highest-ROI cross-modal lever, so
    the default pipeline always fits a whitener on the gallery and applies it to
    queries and gallery alike.

    Attributes
    ----------
    enabled:
        Apply per-modality whitening (default ``True``).
    n_components:
        PCA components to keep (``None`` = full, ZCA-style symmetric whitening,
        which preserves the shared coordinate frame across modalities — required
        for cross-modal comparability). Set an integer only for same-modal /
        dimensionality-reduction use.
    remove_top_pc:
        Leading principal components to drop before whitening (often encode
        sensor identity / global energy). ``0`` keeps everything.
    shrinkage:
        Covariance shrinkage in ``[0, 1]`` (Ledoit-Wolf style). Higher values
        damp over-whitening of noisy low-variance directions; as it approaches
        ``1`` the transform degenerates to robust per-modality mean-centering.
        The default ``0.9`` is deliberately high so whitening is *safe* (never
        worse than mean-centering) on small / noisy reference sets.
    fit_on:
        Where to fit the whitener: ``"gallery"`` (default) or ``"all"`` (queries
        + gallery; more reference vectors per modality).
    """

    enabled: bool = True
    n_components: Optional[int] = None
    remove_top_pc: int = 0
    shrinkage: float = 0.9
    fit_on: str = "gallery"


@dataclass
class IndexConfig:
    """Shared cross-modal vector index (FAISS w/ exact numpy fallback)."""

    type: str = "auto"
    metric: str = "ip"
    nprobe: int = 16
    rerank: bool = False


@dataclass
class EvalConfig:
    """Evaluation protocol (the four headline F1s + latency)."""

    ks: list[int] = field(default_factory=lambda: [5, 10])
    recall_mode: str = "raw"
    measure_latency: bool = True
    latency_warmup: int = 50
    latency_runs: int = 500


@dataclass
class TrainConfig:
    """Projection-head training recipe (torch; used by the ``train`` command).

    ``device`` controls where the projection head, the ArcFace/loss parameters,
    the cached frozen embeddings, and the per-batch tensors live. ``None`` /
    ``"auto"`` auto-detects (cuda if available, else cpu) and also honors
    ``backbone.kwargs.device``; set it to ``"cuda"`` on a GPU box to actually
    accelerate head training. A requested ``"cuda"`` that is unavailable falls
    back to cpu, so CPU-only runs are unaffected.
    """

    epochs: int = 5
    lr: float = 1e-3
    weight_decay: float = 1e-4
    batch_p: int = 8
    batch_k: int = 4
    batches_per_epoch: Optional[int] = None
    w_infonce: float = 1.5
    w_arcface: float = 1.0
    w_triplet: float = 0.5
    temperature: float = 0.07
    seed: int = 0
    device: Optional[str] = None


# ---------------------------------------------------------------------------
# Top-level config
# ---------------------------------------------------------------------------
_SUBCONFIGS: dict[str, type] = {
    "data": DataConfig,
    "backbone": BackboneConfig,
    "projection": ProjectionConfig,
    "whitening": WhiteningConfig,
    "index": IndexConfig,
    "eval": EvalConfig,
    "train": TrainConfig,
}


@dataclass
class Config:
    """Full pipeline configuration (composed of the sub-configs above)."""

    data: DataConfig = field(default_factory=DataConfig)
    backbone: BackboneConfig = field(default_factory=BackboneConfig)
    projection: ProjectionConfig = field(default_factory=ProjectionConfig)
    whitening: WhiteningConfig = field(default_factory=WhiteningConfig)
    index: IndexConfig = field(default_factory=IndexConfig)
    eval: EvalConfig = field(default_factory=EvalConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    name: str = "default"

    # -- constructors -----------------------------------------------------
    @classmethod
    def default(cls) -> "Config":
        """Return the default pipeline config (foundation backbone + whitening)."""
        return cls()

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Config":
        """Build a :class:`Config` from a (possibly partial) nested ``dict``.

        Unknown top-level keys are ignored with no error; within each sub-config,
        only recognised fields are consumed so forward-compatible YAML files do
        not crash older code. ``backbone.kwargs`` (a free-form dict) is preserved
        verbatim.
        """
        data = dict(data or {})
        kwargs: dict[str, Any] = {}
        for key, sub_cls in _SUBCONFIGS.items():
            if key in data and data[key] is not None:
                kwargs[key] = _build_subconfig(sub_cls, data[key])
        if "name" in data:
            kwargs["name"] = str(data["name"])
        return cls(**kwargs)

    @classmethod
    def from_yaml(cls, path: str) -> "Config":
        """Load a :class:`Config` from a YAML file (``pyyaml`` imported lazily)."""
        try:
            import yaml  # local, lazy
        except ImportError as exc:  # pragma: no cover - environment dependent
            raise ImportError(
                "Config.from_yaml requires PyYAML. Install it with "
                "`pip install pyyaml` (it is in requirements-cpu.txt)."
            ) from exc
        with open(path, "r", encoding="utf-8") as fh:
            raw = yaml.safe_load(fh) or {}
        if not isinstance(raw, dict):
            raise ValueError(f"config file {path!r} must contain a mapping")
        cfg = cls.from_dict(raw)
        if cfg.name == "default" and "name" not in raw:
            # Name the config after its file stem for nicer reports.
            import os

            cfg.name = os.path.splitext(os.path.basename(path))[0]
        return cfg

    # -- serialisation ----------------------------------------------------
    def to_dict(self) -> dict[str, Any]:
        """Return a plain nested ``dict`` (JSON / YAML friendly)."""
        return asdict(self)

    def to_yaml(self, path: str) -> None:
        """Write this config to a YAML file (``pyyaml`` imported lazily)."""
        import yaml  # local, lazy

        with open(path, "w", encoding="utf-8") as fh:
            yaml.safe_dump(self.to_dict(), fh, sort_keys=False)


def _build_subconfig(sub_cls: type, value: Any) -> Any:
    """Instantiate a sub-config dataclass from a dict, ignoring unknown keys."""
    if is_dataclass(value) and isinstance(value, sub_cls):
        return value
    if not isinstance(value, dict):
        raise ValueError(f"expected a mapping for {sub_cls.__name__}, got {type(value)}")
    allowed = {f.name for f in fields(sub_cls)}
    kwargs = {k: v for k, v in value.items() if k in allowed}
    return sub_cls(**kwargs)
