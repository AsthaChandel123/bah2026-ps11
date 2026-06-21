"""DINOv2 (with registers) backbone via HuggingFace ``transformers``.

:class:`DINOv2Backbone` wraps Meta's **DINOv2-with-registers** ViT-B/14
(HF id ``facebook/dinov2-with-registers-base``, **768-d**, Apache-2.0) as a
frozen image-retrieval feature extractor. DINOv2 produces the best *generic*
frozen features for image retrieval and needs no text head, making it the
recommended generic fallback (and a useful contrastive-distillation teacher) for
optical-to-optical retrieval.

Modality handling
-----------------
DINOv2 is an RGB (3-channel) model. Non-RGB inputs are reduced to a 3-channel
**pseudo-RGB** image before encoding (documented mapping, via
:meth:`Backbone.to_pseudo_rgb`):

* **SAR (Sentinel-1)**: ``[VV, VH, VV/VH-ratio]`` -- the VV/VH ratio gives the
  third channel genuine structural signal.
* **Multispectral (Sentinel-2)**: ``[B4 (red), B3 (green), B2 (blue)]``
  (0-based band indices ``[3, 2, 1]``).
* **Optical RGB**: used as-is.

Preprocessing: scale to ``[0, 1]``, resize to a multiple of the patch size
(14 -> default 224), ImageNet mean/std normalization. The **CLS token** of the
last hidden state is the global embedding (then L2-normalized by the base
class).

Dependencies (``torch``, ``transformers``) are imported lazily inside methods,
so ``import xsretrieval.models.backbones.dinov2`` works on bare numpy. Weights
download from the HuggingFace Hub on first construction; if that fails,
:func:`~xsretrieval.models.backbones.base.get_backbone` falls back.
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

__all__ = ["DINOv2Backbone"]

_IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


@register_backbone("dinov2", aliases=["dino", "dinov2-base", "dinov2_reg"])
class DINOv2Backbone(Backbone):
    """Frozen DINOv2-with-registers ViT-B/14 feature extractor.

    Parameters
    ----------
    model_name:
        HF model id. Default ``"facebook/dinov2-with-registers-base"`` (768-d).
    image_size:
        Square input size; must be a multiple of the patch size (14). Default
        ``224``.
    pretrained:
        Load pretrained weights from the Hub. Default ``True``.
    device:
        Torch device. Default ``"cpu"``.
    """

    #: Patch size of the ViT-B/14 backbone (input must be a multiple of this).
    PATCH_SIZE = 14

    def __init__(
        self,
        model_name: str = "facebook/dinov2-with-registers-base",
        image_size: int = 224,
        pretrained: bool = True,
        device: str = "cpu",
    ) -> None:
        self.name = f"dinov2:{model_name.split('/')[-1]}"
        self.model_name = model_name
        self.pretrained = pretrained
        # Round image size to a multiple of the patch size.
        self.image_size = max(
            self.PATCH_SIZE,
            int(round(image_size / self.PATCH_SIZE)) * self.PATCH_SIZE,
        )
        self.device = device
        # RGB model + pseudo-RGB adaptation -> supports all modalities.
        self.supported_modalities = set()
        self.embed_dim = 768  # ViT-B hidden size; refined after load
        self._model = None
        self._build_model()

    # -- model construction -------------------------------------------------
    def _build_model(self) -> None:
        """Load the DINOv2 model via ``transformers.AutoModel``. Lazy imports."""
        from transformers import AutoModel  # local, lazy

        if self.pretrained:
            model = AutoModel.from_pretrained(self.model_name)
        else:
            from transformers import AutoConfig

            cfg = AutoConfig.from_pretrained(self.model_name)
            model = AutoModel.from_config(cfg)
        model.eval()
        for p in model.parameters():
            p.requires_grad_(False)
        model.to(self.device)
        self._model = model
        hidden = getattr(getattr(model, "config", None), "hidden_size", None)
        if isinstance(hidden, int) and hidden > 0:
            self.embed_dim = int(hidden)

    # -- preprocessing ------------------------------------------------------
    def _preprocess(
        self, batch: np.ndarray, modality: "Modality | str | None"
    ) -> "torch.Tensor":
        import torch  # local, lazy
        import torch.nn.functional as F  # local, lazy

        rgb = self.to_pseudo_rgb(batch, modality)  # (B, 3, H, W)
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
        return (t - mean) / std

    # -- Backbone API -------------------------------------------------------
    def _embed_batch(
        self, batch: np.ndarray, modality: "Modality | str | None"
    ) -> np.ndarray:
        import torch  # local, lazy

        assert self._model is not None
        x = self._preprocess(batch, modality)
        with torch.no_grad():
            out = self._model(pixel_values=x)
        # Prefer the dedicated pooler output (CLS); else take CLS (index 0) of
        # the last hidden state.
        pooled = getattr(out, "pooler_output", None)
        if pooled is None:
            last_hidden = out.last_hidden_state  # (B, 1+reg+patches, hidden)
            pooled = last_hidden[:, 0]  # CLS token
        return pooled.detach().cpu().numpy().astype(np.float32)


def _to_unit_range(rgb: np.ndarray) -> np.ndarray:
    """Per-image joint-channel min-max scale into ``[0, 1]``."""
    rgb = np.asarray(rgb, dtype=np.float32)
    b = rgb.shape[0]
    flat = rgb.reshape(b, -1)
    lo = flat.min(axis=1).reshape(b, 1, 1, 1)
    hi = flat.max(axis=1).reshape(b, 1, 1, 1)
    scale = np.maximum(hi - lo, 1e-6)
    return ((rgb - lo) / scale).astype(np.float32)
