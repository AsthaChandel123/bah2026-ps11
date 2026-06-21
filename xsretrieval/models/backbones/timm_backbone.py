"""Generic ImageNet backbone via ``timm`` (ViT / ResNet).

:class:`TimmBackbone` wraps any `timm <https://github.com/huggingface/
pytorch-image-models>`_ model (default ``vit_base_patch16_224``, 768-d) as a
frozen feature extractor. It is the **dependable mid-tier fallback** between the
pure-numpy :class:`~xsretrieval.models.backbones.fallback.FallbackBackbone` and
the remote-sensing foundation models (DOFA / CROMA / RemoteCLIP): when real RS
weights cannot be downloaded but ``torch``/``timm`` *are* installed and ImageNet
weights are cached, this gives genuine deep features.

Non-RGB modalities are reduced to a 3-channel **pseudo-RGB** image
(:meth:`Backbone.to_pseudo_rgb`) before encoding (SAR -> ``[VV, VH, VV/VH]``;
multispectral -> ``[B4, B3, B2]``). ImageNet mean/std normalization is applied.
Embeddings are the model's global-pooled features (``num_classes=0`` +
``global_pool='avg'``), then L2-normalized by the base class.

Dependencies (``torch``, ``timm``) are imported lazily inside methods, so
``import xsretrieval.models.backbones.timm_backbone`` works on bare numpy; only
*constructing* / *embedding* requires them.

License: ``timm`` is Apache-2.0; individual ImageNet weights carry their own
licenses (typically permissive research/Apache).
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import numpy as np

from .base import Backbone, register_backbone

if TYPE_CHECKING:  # pragma: no cover - typing only
    import torch

    from xsretrieval.data.modalities import Modality

logger = logging.getLogger(__name__)

__all__ = ["TimmBackbone"]

# ImageNet normalization constants (RGB).
_IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

# Known embedding widths for common defaults (used only as a hint before the
# model is loaded; the true value is read from the model after construction).
_KNOWN_DIMS: dict[str, int] = {
    "vit_base_patch16_224": 768,
    "vit_small_patch16_224": 384,
    "vit_large_patch16_224": 1024,
    "resnet50": 2048,
    "resnet18": 512,
}


@register_backbone("timm", aliases=["imagenet", "vit_imagenet"])
class TimmBackbone(Backbone):
    """Frozen ImageNet feature extractor via ``timm``.

    Parameters
    ----------
    model_name:
        Any ``timm`` model id. Default ``"vit_base_patch16_224"`` (768-d).
    pretrained:
        Load pretrained ImageNet weights. Default ``True``. Construction raises
        if weights cannot be fetched while ``pretrained=True``;
        :func:`~xsretrieval.models.backbones.base.get_backbone` catches that and
        falls back.
    image_size:
        Square input size fed to the model. Default ``224``.
    device:
        Torch device. Default ``"cpu"`` (package is CPU-only).
    """

    def __init__(
        self,
        model_name: str = "vit_base_patch16_224",
        pretrained: bool = True,
        image_size: int = 224,
        device: str = "cpu",
    ) -> None:
        self.name = f"timm:{model_name}"
        self.model_name = model_name
        self.pretrained = pretrained
        self.image_size = int(image_size)
        self.device = device
        # timm/ImageNet models accept any modality via pseudo-RGB adaptation.
        self.supported_modalities = set()
        self.embed_dim = _KNOWN_DIMS.get(model_name, 768)
        self._model = None  # lazily built torch module
        self._build_model()

    # -- model construction -------------------------------------------------
    def _build_model(self) -> None:
        """Create and freeze the timm model. Lazy-imports timm."""
        import timm  # local, lazy

        model = timm.create_model(
            self.model_name,
            pretrained=self.pretrained,
            num_classes=0,  # drop the classification head -> features
            global_pool="avg",  # global-average-pooled feature vector
        )
        model.eval()
        for p in model.parameters():
            p.requires_grad_(False)
        model.to(self.device)
        self._model = model
        # Read the true feature dimensionality from the model.
        num_features = getattr(model, "num_features", None)
        if isinstance(num_features, int) and num_features > 0:
            self.embed_dim = int(num_features)

    # -- preprocessing ------------------------------------------------------
    def _preprocess(
        self, batch: np.ndarray, modality: "Modality | str | None"
    ) -> "torch.Tensor":
        """Pseudo-RGB -> [0,1] scaling -> resize -> ImageNet-normalize.

        Returns a ``(B, 3, S, S)`` ``float32`` tensor on :attr:`device`.
        """
        import torch  # local, lazy
        import torch.nn.functional as F  # local, lazy

        rgb = self.to_pseudo_rgb(batch, modality)  # (B, 3, H, W) numpy
        rgb = _to_unit_range(rgb)
        t = torch.from_numpy(np.ascontiguousarray(rgb)).to(self.device)
        if t.shape[-1] != self.image_size or t.shape[-2] != self.image_size:
            t = F.interpolate(
                t,
                size=(self.image_size, self.image_size),
                mode="bilinear",
                align_corners=False,
            )
        mean = torch.tensor(_IMAGENET_MEAN, device=self.device).view(1, 3, 1, 1)
        std = torch.tensor(_IMAGENET_STD, device=self.device).view(1, 3, 1, 1)
        t = (t - mean) / std
        return t

    # -- Backbone API -------------------------------------------------------
    def _embed_batch(
        self, batch: np.ndarray, modality: "Modality | str | None"
    ) -> np.ndarray:
        import torch  # local, lazy

        assert self._model is not None
        x = self._preprocess(batch, modality)
        with torch.no_grad():
            feats = self._model(x)  # (B, num_features) after global_pool='avg'
        if feats.ndim > 2:  # safety: pool any stray spatial dims
            feats = feats.flatten(1)
        return feats.detach().cpu().numpy().astype(np.float32)


def _to_unit_range(rgb: np.ndarray) -> np.ndarray:
    """Per-image min-max scale a pseudo-RGB batch into ``[0, 1]``.

    Inputs to the package can be raw reflectance, dB SAR, or already-normalized
    arrays; ImageNet stats assume ``[0, 1]`` inputs, so we robustly rescale per
    image (per sample, across all channels jointly to preserve color balance).
    """
    rgb = np.asarray(rgb, dtype=np.float32)
    b = rgb.shape[0]
    flat = rgb.reshape(b, -1)
    lo = flat.min(axis=1).reshape(b, 1, 1, 1)
    hi = flat.max(axis=1).reshape(b, 1, 1, 1)
    scale = np.maximum(hi - lo, 1e-6)
    return ((rgb - lo) / scale).astype(np.float32)
