"""Train a shared projection head on top of a frozen backbone (torch).

:func:`train_projection` fits a
:class:`~xsretrieval.models.projection.ProjectionHeads` on **frozen** backbone
embeddings so that all modalities land in one comparable space, using the
research-recommended combined objective
(:func:`~xsretrieval.alignment.losses.CrossModalRetrievalLoss` —
``1.5·InfoNCE + 1·SubCenterArcFace + 0.5·batch-hard-triplet``) and the
modality-balanced P×K batch sampler
(:class:`~xsretrieval.data.samplers.PKModalitySampler`), which guarantees each
batch carries cross-modal positives and hard negatives.

Why freeze the backbone? It keeps training cheap and CPU-friendly: we encode the
whole (small) training set **once** with the frozen backbone, then iterate the
P×K sampler over those cached embeddings, training only the lightweight head.
This is enough to learn the cross-modal alignment on top of strong frozen
features (research ``05_training_losses.md`` §13 / §15).

torch policy: torch is imported lazily inside :func:`train_projection`; a clear
:class:`ImportError` is raised if it is missing. The numpy retrieval / whitening
path never needs this module.
"""

from __future__ import annotations

import logging
from typing import Any, Optional

import numpy as np

from xsretrieval.data.modalities import Modality, Sample

logger = logging.getLogger(__name__)

__all__ = ["train_projection", "TrainResult"]


class TrainResult:
    """Result of :func:`train_projection`.

    Attributes
    ----------
    head:
        The trained ``ProjectionHeads`` (torch module, in ``eval`` mode).
    history:
        Per-epoch list of dicts with the (mean) total loss and components.
    val_metrics:
        Optional dict of validation headline metrics (if ``val_samples`` given).
    """

    def __init__(self, head: Any, history: list[dict], val_metrics: Optional[dict]):
        self.head = head
        self.history = history
        self.val_metrics = val_metrics

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        last = self.history[-1] if self.history else {}
        return (
            f"TrainResult(epochs={len(self.history)}, "
            f"final_loss={last.get('total', float('nan')):.4f}, "
            f"val={self.val_metrics})"
        )


def _torch():
    """Lazily import torch (clear error if missing)."""
    try:
        import torch  # noqa: F401
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise ImportError(
            "xsretrieval.alignment.trainer requires PyTorch. Install the training "
            "extras (`pip install torch`) to train a projection head; the numpy "
            "retrieval/whitening path does not need it."
        ) from exc
    return torch


def _encode_frozen(backbone: Any, samples: list[Sample], batch_size: int) -> np.ndarray:
    """Encode all samples with the frozen backbone, grouped by modality.

    Returns a ``(N, D)`` float32 array aligned with ``samples`` order.
    """
    groups: dict[str, list[int]] = {}
    for i, s in enumerate(samples):
        key = s.modality.value if isinstance(s.modality, Modality) else str(s.modality)
        groups.setdefault(key, []).append(i)
    out: Optional[np.ndarray] = None
    for key, idxs in groups.items():
        mod = Modality(key)
        imgs = [np.asarray(samples[i].image, dtype=np.float32) for i in idxs]
        embs: list[np.ndarray] = []
        for start in range(0, len(imgs), batch_size):
            chunk = np.stack(imgs[start : start + batch_size], axis=0)
            embs.append(np.asarray(backbone.embed(chunk, mod), dtype=np.float32))
        emb = np.concatenate(embs, axis=0)
        if out is None:
            out = np.zeros((len(samples), emb.shape[1]), dtype=np.float32)
        out[idxs] = emb
    assert out is not None
    return out


def train_projection(
    backbone: Any,
    train_samples: list[Sample],
    config: Any,
    val_samples: Optional[list[Sample]] = None,
) -> TrainResult:
    """Train a shared :class:`ProjectionHeads` on frozen backbone embeddings.

    Parameters
    ----------
    backbone:
        A frozen backbone exposing ``embed(images, modality) -> (B, D)``.
    train_samples:
        Training samples (mixed modalities, each with a class ``label`` and a
        ``location_id`` so cross-modal InfoNCE pairs can be formed).
    config:
        A :class:`~xsretrieval.config.Config` (uses ``config.train`` for the
        recipe and ``config.projection`` for the head geometry / ``backbone``
        for ``embed_dim``).
    val_samples:
        Optional held-out samples; if given, a quick same/cross F1 evaluation is
        run after training and returned in :attr:`TrainResult.val_metrics`.

    Returns
    -------
    TrainResult
        The trained head (eval mode), the loss history, and optional val metrics.
    """
    torch = _torch()
    from torch import optim

    from xsretrieval.alignment.losses import CrossModalRetrievalLoss
    from xsretrieval.data.samplers import PKModalitySampler
    from xsretrieval.models.projection import ProjectionHeads

    tc = config.train
    pc = config.projection
    rng_seed = int(getattr(tc, "seed", 0))
    torch.manual_seed(rng_seed)
    np.random.seed(rng_seed)

    modalities = sorted(
        {(s.modality.value if isinstance(s.modality, Modality) else str(s.modality)) for s in train_samples}
    )
    mod_enums = [Modality(m) for m in modalities]

    # 1) Cache frozen embeddings for the whole training set (CPU-friendly).
    feats = _encode_frozen(backbone, train_samples, batch_size=64)
    in_dim = feats.shape[1]
    labels = np.array([int(s.label) for s in train_samples], dtype=np.int64)
    mod_codes = np.array(
        [(s.modality.value if isinstance(s.modality, Modality) else str(s.modality)) for s in train_samples],
        dtype=object,
    )
    mod_to_int = {m: i for i, m in enumerate(modalities)}
    mod_ints = np.array([mod_to_int[m] for m in mod_codes], dtype=np.int64)
    location_ids = [
        ("" if s.location_id is None else str(s.location_id)) for s in train_samples
    ]
    n_classes = int(labels.max()) + 1 if len(labels) else 1

    feats_t = torch.from_numpy(feats)

    # 2) Build the head + loss + optimizer.
    head = ProjectionHeads(
        in_dim=in_dim,
        out_dim=pc.out_dim,
        modalities=mod_enums,
        share_final=pc.share_final,
        hidden=pc.hidden,
        dropout=pc.dropout,
    )
    head.train()
    loss_fn = CrossModalRetrievalLoss(
        in_dim=pc.out_dim,
        n_classes=n_classes,
        w_infonce=tc.w_infonce,
        w_arcface=tc.w_arcface,
        w_triplet=tc.w_triplet,
        temperature=tc.temperature,
    )
    params = list(head.parameters()) + list(loss_fn.parameters())
    optimizer = optim.AdamW(params, lr=tc.lr, weight_decay=tc.weight_decay)

    sampler = PKModalitySampler(
        labels=labels,
        modalities=[Modality(m) for m in mod_codes],
        p=tc.batch_p,
        k=tc.batch_k,
        num_batches=tc.batches_per_epoch,
        seed=rng_seed,
        require_multimodal=True,
    )

    def _project_batch(idx: np.ndarray) -> "torch.Tensor":
        """Project a batch of cached features per modality and stack (B, out_dim)."""
        out = torch.empty((len(idx), pc.out_dim), dtype=torch.float32)
        idx_mods = mod_ints[idx]
        for mi, m in enumerate(mod_enums):
            sel = np.where(idx_mods == mi)[0]
            if sel.size == 0:
                continue
            rows = idx[sel]
            z = head.forward(feats_t[rows], m)
            out[sel] = z
        return out

    # 3) Training loop.
    history: list[dict] = []
    for epoch in range(int(tc.epochs)):
        sampler.set_epoch(epoch)
        epoch_logs: list[dict] = []
        for batch_idx in sampler:
            idx = np.asarray(batch_idx, dtype=np.int64)
            z = _project_batch(idx)
            y = torch.from_numpy(labels[idx])
            m = torch.from_numpy(mod_ints[idx])
            locs = [location_ids[i] for i in idx]
            total, comps = loss_fn(z, y, modalities=m, location_ids=locs)
            optimizer.zero_grad()
            total.backward()
            optimizer.step()
            epoch_logs.append(comps)
        agg = _mean_logs(epoch_logs)
        history.append(agg)
        logger.info(
            "epoch %d/%d  loss=%.4f (nce=%.3f arc=%.3f tri=%.3f)",
            epoch + 1,
            int(tc.epochs),
            agg.get("total", float("nan")),
            agg.get("infonce", 0.0),
            agg.get("arcface", 0.0),
            agg.get("triplet", 0.0),
        )

    head.eval()

    val_metrics: Optional[dict] = None
    if val_samples:
        val_metrics = _quick_eval(head, backbone, val_samples, config, mod_enums)

    return TrainResult(head=head, history=history, val_metrics=val_metrics)


def _mean_logs(logs: list[dict]) -> dict:
    """Average the per-batch component dicts of an epoch."""
    if not logs:
        return {}
    keys = set().union(*[set(d.keys()) for d in logs])
    return {k: float(np.mean([d.get(k, 0.0) for d in logs])) for k in keys}


def _quick_eval(
    head: Any,
    backbone: Any,
    val_samples: list[Sample],
    config: Any,
    mod_enums: list[Modality],
) -> dict:
    """Run a quick same/cross F1 eval with the trained head wired into an engine."""
    from xsretrieval.eval.benchmark import evaluate
    from xsretrieval.pipeline import make_query_gallery
    from xsretrieval.retrieval.engine import RetrievalEngine

    # A modality-aware backbone+projection: the backbone stashes the current
    # modality on a shared slot, and the projection reads it to route the shared
    # head. The engine encodes one modality at a time, so the slot is always
    # correct at projection time. This keeps the engine's duck-typed contract.
    slot: dict[str, Any] = {"mod": None}
    engine = RetrievalEngine(
        _ModalityAwareBackbone(backbone, slot),
        projection=_RoutingProjection(head, slot),
    )
    queries, gallery = make_query_gallery(config, val_samples)
    if not queries or not gallery:
        return {}
    res = evaluate(engine, queries, gallery, ks=tuple(config.eval.ks), measure_latency=False)
    return res["headline"]


class _ModalityAwareBackbone:
    """Wrap a backbone, stashing the current modality on a shared slot."""

    def __init__(self, backbone: Any, slot: dict) -> None:
        self._backbone = backbone
        self._slot = slot
        self.name = getattr(backbone, "name", "backbone")
        self.embed_dim = getattr(backbone, "embed_dim", 0)

    def embed(self, images: Any, modality: Any = None) -> np.ndarray:
        self._slot["mod"] = modality
        return self._backbone.embed(images, modality)


class _RoutingProjection:
    """Project a numpy batch with a torch head, routing by the shared slot."""

    def __init__(self, head: Any, slot: dict) -> None:
        self._head = head
        self._slot = slot

    def forward(self, emb: np.ndarray) -> np.ndarray:
        import torch  # local, lazy

        with torch.no_grad():
            t = torch.from_numpy(np.ascontiguousarray(emb, dtype=np.float32))
            out = self._head.forward(t, self._slot.get("mod"))
        return out.detach().cpu().numpy().astype(np.float32)
