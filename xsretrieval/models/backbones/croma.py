"""CROMA (Contrastive Radar-Optical MAE) radar-optical backbone.

CROMA (Fuller et al., NeurIPS 2023, arXiv:2311.00566) is purpose-built for
**Sentinel-1 SAR + Sentinel-2 optical** and is a strong alternative default for
PS-11. Its contrastive objective explicitly aligns radar and optical, so its
per-modality embeddings live in a shared space and it additionally exposes a
*fused* ``joint_GAP`` vector -- ideal for cross-modal SAR<->optical retrieval
when the data is purely Sentinel-1/2.

* Weights: HF ``antofuller/CROMA`` (``CROMA_base.pt`` / ``CROMA_large.pt``);
  GitHub ``antofuller/CROMA``.
* License: **MIT** (most permissive of the RS foundation models here).
* Arch: a SAR ViT encoder + an optical ViT encoder + a radar-optical fusion
  encoder. ViT-B -> **768-d**, ViT-L -> 1024-d.
* Inputs / bands (fixed layout):
    - **Sentinel-1 SAR = 2 channels** (VV, VH);
    - **Sentinel-2 = 12 channels** (the 13 bands with B10 cirrus dropped).
* Outputs: patch-level ``SAR_encodings`` / ``optical_encodings`` /
  ``joint_encodings`` and pooled **``SAR_GAP`` / ``optical_GAP`` / ``joint_GAP``**
  (``B x D``). For retrieval use the GAP vectors.
* Input size: default 120x120 (changeable); per-channel mean/std with mean+-2σ
  clipping then scaling (a helper ships in the repo).

Routing
-------
Which embedding is returned depends on the input:

* SAR-only input (2-band, or modality says SAR) -> ``SAR_GAP``.
* Optical-only input (12-band, or modality says optical/MS) -> ``optical_GAP``.
* A paired call (see :meth:`CROMABackbone.embed_joint`) -> ``joint_GAP``.

Because the gallery in PS-11 is embedded one modality at a time, the standard
:meth:`embed` returns the appropriate per-modality GAP; the fused ``joint_GAP``
is available via :meth:`embed_joint` for paired SAR+optical inputs.

Robust loading / defensive forward
----------------------------------
CROMA's reference loader expects a specific module; construction tries the
``croma`` package and a raw checkpoint load, and *documents* the expected
``model(SAR_images=..., optical_images=...)`` dict interface. If the package /
weights are unavailable, construction raises and
:func:`~xsretrieval.models.backbones.base.get_backbone` falls back. All heavy
imports are lazy.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import numpy as np

from .base import Backbone, _is_sar, register_backbone

if TYPE_CHECKING:  # pragma: no cover - typing only
    import torch

    from xsretrieval.data.modalities import Modality

logger = logging.getLogger(__name__)

__all__ = ["CROMABackbone"]

#: Sentinel-2 band count CROMA expects (B10 cirrus dropped from the 13).
CROMA_OPTICAL_CHANNELS = 12
#: Sentinel-1 band count CROMA expects (VV, VH).
CROMA_SAR_CHANNELS = 2


@register_backbone("croma", aliases=["croma-base", "croma_vit_base"])
class CROMABackbone(Backbone):
    """CROMA ViT-B radar-optical backbone (768-d).

    Parameters
    ----------
    size:
        ``"base"`` (768-d, default) or ``"large"`` (1024-d).
    hf_repo:
        HF repo for the checkpoint. Default ``"antofuller/CROMA"``.
    hf_filename:
        Checkpoint filename. Defaults to ``CROMA_base.pt`` / ``CROMA_large.pt``
        per ``size``.
    image_size:
        Square input size. Default ``120`` (CROMA's native size).
    device:
        Torch device. Default ``"cpu"``.
    """

    def __init__(
        self,
        size: str = "base",
        hf_repo: str = "antofuller/CROMA",
        hf_filename: str | None = None,
        image_size: int = 120,
        device: str = "cpu",
    ) -> None:
        self.size = size
        self.name = f"croma:{size}"
        self.hf_repo = hf_repo
        self.hf_filename = hf_filename or (
            "CROMA_large.pt" if size == "large" else "CROMA_base.pt"
        )
        self.image_size = int(image_size)
        self.device = device
        # CROMA ingests S1 SAR + S2 optical (not arbitrary RGB / hyperspectral).
        # We still advertise "all" and adapt channel counts defensively so the
        # pipeline can route any modality here; non-S1/S2 inputs are coerced.
        self.supported_modalities = set()
        self.embed_dim = 1024 if size == "large" else 768
        self._model = None
        self._build_model()

    # -- model construction -------------------------------------------------
    def _build_model(self) -> None:
        """Load CROMA via its reference package + HF checkpoint. Lazy imports.

        Raises on failure so ``get_backbone`` can fall back.
        """
        from huggingface_hub import hf_hub_download  # local, lazy

        ckpt_path = hf_hub_download(self.hf_repo, self.hf_filename)

        model = None
        errors: list[str] = []
        import importlib

        for mod_name, attr in (
            ("croma.pretrain_croma", "PretrainedCROMA"),
            ("CROMA", "PretrainedCROMA"),
            ("pretrain_croma", "PretrainedCROMA"),
        ):
            try:
                mod = importlib.import_module(mod_name)
                ctor = getattr(mod, attr)
                model = ctor(
                    pretrained_path=ckpt_path,
                    size=self.size,
                    modality="both",
                    image_resolution=self.image_size,
                )
                break
            except Exception as exc:  # pragma: no cover - env dependent
                errors.append(f"{mod_name}.{attr}: {type(exc).__name__}: {exc}")

        if model is None:
            raise RuntimeError(
                "Could not construct CROMA from the reference package. Install "
                "the CROMA repo (`antofuller/CROMA`) so `PretrainedCROMA` is "
                "importable. Tried: " + " | ".join(errors)
            )

        model.eval()
        for p in model.parameters():
            p.requires_grad_(False)
        model.to(self.device)
        self._model = model

    # -- preprocessing ------------------------------------------------------
    def _prep_modality(
        self, batch: np.ndarray, target_channels: int
    ) -> "torch.Tensor":
        """Clip-normalize, fix channel count, and resize for CROMA.

        Per-channel mean+-2σ clipping then min-max to ``[0, 1]`` (CROMA recipe),
        channel count coerced to ``target_channels`` (truncate or tile), resized
        to ``image_size``.
        """
        import torch  # local, lazy
        import torch.nn.functional as F  # local, lazy

        x = np.asarray(batch, dtype=np.float32)
        if not np.isfinite(x).all():
            x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
        # Per-image, per-channel clip to mean +- 2σ then scale to [0, 1].
        mean = x.mean(axis=(2, 3), keepdims=True)
        std = x.std(axis=(2, 3), keepdims=True) + 1e-6
        lo, hi = mean - 2.0 * std, mean + 2.0 * std
        x = np.clip(x, lo, hi)
        x = (x - lo) / np.maximum(hi - lo, 1e-6)

        x = _coerce_channels(x, target_channels)
        t = torch.from_numpy(np.ascontiguousarray(x)).to(self.device)
        if t.shape[-1] != self.image_size or t.shape[-2] != self.image_size:
            t = F.interpolate(
                t,
                size=(self.image_size, self.image_size),
                mode="bilinear",
                align_corners=False,
            )
        return t

    # -- forward ------------------------------------------------------------
    def _run(
        self,
        sar: "torch.Tensor | None",
        optical: "torch.Tensor | None",
    ) -> dict:
        """Call CROMA defensively; return the output dict of encodings/GAPs.

        Expected reference interface:
        ``model(SAR_images=sar, optical_images=optical)`` -> dict with keys
        ``SAR_GAP`` / ``optical_GAP`` / ``joint_GAP`` (and patch encodings).
        """
        model = self._model
        assert model is not None
        kwargs: dict[str, object] = {}
        if sar is not None:
            kwargs["SAR_images"] = sar
        if optical is not None:
            kwargs["optical_images"] = optical

        try:
            out = model(**kwargs)
        except TypeError:
            # Alternative positional/keyword spellings.
            try:
                out = model(
                    sar_images=sar, optical_images=optical
                )  # type: ignore[arg-type]
            except TypeError:
                out = model(optical if optical is not None else sar)
        if not isinstance(out, dict):
            out = {"joint_GAP": out}
        return out

    # -- public: joint embedding -------------------------------------------
    def embed_joint(
        self, sar_images: object, optical_images: object
    ) -> np.ndarray:
        """Embed paired SAR + optical inputs and return fused ``joint_GAP``.

        Parameters
        ----------
        sar_images, optical_images:
            Batches (numpy / torch / list) of co-registered SAR (2-band) and
            optical (12-band) images, same batch size and order.

        Returns
        -------
        numpy.ndarray
            ``(B, embed_dim)`` ``float32`` L2-normalized fused embeddings.
        """
        import torch  # local, lazy

        from .base import l2_normalize

        sar_batch = self._coerce_to_numpy_batch(sar_images)
        opt_batch = self._coerce_to_numpy_batch(optical_images)
        sar_t = self._prep_modality(sar_batch, CROMA_SAR_CHANNELS)
        opt_t = self._prep_modality(opt_batch, CROMA_OPTICAL_CHANNELS)
        with torch.no_grad():
            out = self._run(sar_t, opt_t)
        feats = _select_gap(out, prefer="joint")
        return l2_normalize(
            feats.detach().cpu().numpy().astype(np.float32), axis=1
        )

    # -- Backbone API -------------------------------------------------------
    def _embed_batch(
        self, batch: np.ndarray, modality: "Modality | str | None"
    ) -> np.ndarray:
        import torch  # local, lazy

        assert self._model is not None
        num_channels = int(batch.shape[1])
        sar_like = _is_sar(modality) or num_channels == CROMA_SAR_CHANNELS

        if sar_like:
            sar_t = self._prep_modality(batch, CROMA_SAR_CHANNELS)
            with torch.no_grad():
                out = self._run(sar_t, None)
            feats = _select_gap(out, prefer="SAR")
        else:
            opt_t = self._prep_modality(batch, CROMA_OPTICAL_CHANNELS)
            with torch.no_grad():
                out = self._run(None, opt_t)
            feats = _select_gap(out, prefer="optical")
        return feats.detach().cpu().numpy().astype(np.float32)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _coerce_channels(x: np.ndarray, target: int) -> np.ndarray:
    """Truncate or tile channels of ``(B, C, H, W)`` to exactly ``target``."""
    c = x.shape[1]
    if c == target:
        return x
    if c > target:
        return np.ascontiguousarray(x[:, :target], dtype=np.float32)
    reps = int(np.ceil(target / max(c, 1)))
    tiled = np.tile(x, (1, reps, 1, 1))[:, :target]
    return np.ascontiguousarray(tiled, dtype=np.float32)


def _select_gap(out: dict, prefer: str) -> "torch.Tensor":
    """Pick the best available GAP vector from a CROMA output dict.

    Tries ``{prefer}_GAP`` first, then the other GAPs, then any patch-level
    encoding (mean-pooled), in a sensible priority order.
    """
    import torch  # local, lazy

    order = {
        "joint": ["joint_GAP", "optical_GAP", "SAR_GAP"],
        "optical": ["optical_GAP", "joint_GAP", "SAR_GAP"],
        "SAR": ["SAR_GAP", "joint_GAP", "optical_GAP"],
    }[prefer]
    for key in order:
        if key in out and out[key] is not None:
            return out[key]
    # Fall back to mean-pooling a patch-level encoding.
    for key in ("joint_encodings", "optical_encodings", "SAR_encodings"):
        if key in out and out[key] is not None:
            t = out[key]
            return t.mean(dim=1) if t.ndim == 3 else t
    # Last resort: first tensor-like value.
    for v in out.values():
        if isinstance(v, torch.Tensor):
            return v.mean(dim=1) if v.ndim == 3 else v
    raise RuntimeError("CROMA output contained no usable embedding")
