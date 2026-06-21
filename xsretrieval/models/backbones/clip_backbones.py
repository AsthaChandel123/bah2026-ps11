"""CLIP-family image-encoder backbones via ``open_clip``.

Two backbones, both using the **image encoder** of an OpenCLIP-architecture
model and L2-normalizing the result:

* :class:`OpenCLIPBackbone` -- general always-works CLIP fallback. Default
  LAION-2B **ViT-B/32** (HF ``laion/CLIP-ViT-B-32-laion2B-s34B-b79K``,
  **512-d**, MIT). Tiny, ubiquitous, offline-cacheable, and *interface-identical*
  to the RS CLIP variants so weights can be hot-swapped.
* :class:`RemoteCLIPBackbone` -- remote-sensing optical specialist. **ViT-L/14**
  with RemoteCLIP weights (HF ``chendelong/RemoteCLIP``, file
  ``RemoteCLIP-ViT-L-14.pt``, **768-d**). SOTA on RS image-image / image-text
  retrieval; best for optical-to-optical. Built on OpenAI/LAION CLIP (research-
  permissive license -- verify the model card before commercial use).

CLIP models are RGB (3-channel). Non-optical modalities are reduced to a
3-channel **pseudo-RGB** image (SAR -> ``[VV, VH, VV/VH]``; multispectral ->
``[B4, B3, B2]``) via :meth:`Backbone.to_pseudo_rgb`, then CLIP-normalized
(mean ``[0.4815, 0.4578, 0.4082]``, std ``[0.2686, 0.2613, 0.2758]``) at
224x224. Image-encoder output dims follow the standard CLIP projection widths:
ViT-B/32 -> 512, ViT-L/14 -> 768.

Dependencies (``torch``, ``open_clip``, ``huggingface_hub``) are imported lazily
inside methods; ``import xsretrieval.models.backbones.clip_backbones`` works on
bare numpy. Weight download failures are caught by
:func:`~xsretrieval.models.backbones.base.get_backbone` (fallback).
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

__all__ = ["OpenCLIPBackbone", "RemoteCLIPBackbone"]

# CLIP image normalization (OpenAI/OpenCLIP standard).
_CLIP_MEAN = np.array([0.48145466, 0.4578275, 0.40821073], dtype=np.float32)
_CLIP_STD = np.array([0.26862954, 0.26130258, 0.27577711], dtype=np.float32)


class _BaseCLIPBackbone(Backbone):
    """Shared machinery for OpenCLIP-architecture image encoders.

    Subclasses set the architecture name, pretrained tag / checkpoint source,
    expected embedding dim, and a friendly name.
    """

    #: open_clip architecture name (e.g. "ViT-B-32", "ViT-L-14").
    arch: str = "ViT-B-32"
    #: open_clip ``pretrained`` tag, or ``None`` when weights load from a file.
    pretrained_tag: str | None = None
    #: HuggingFace repo id to fetch a checkpoint from (RemoteCLIP), or ``None``.
    hf_repo: str | None = None
    #: Filename within ``hf_repo`` to load (RemoteCLIP), or ``None``.
    hf_filename: str | None = None

    def __init__(
        self,
        image_size: int = 224,
        device: str = "cpu",
        embed_dim: int | None = None,
    ) -> None:
        self.image_size = int(image_size)
        self.device = device
        # RGB model + pseudo-RGB adaptation -> all modalities.
        self.supported_modalities = set()
        if embed_dim is not None:
            self.embed_dim = int(embed_dim)
        self._model = None
        self._build_model()

    # -- model construction -------------------------------------------------
    def _build_model(self) -> None:
        """Create the open_clip model and load weights. Lazy imports."""
        import open_clip  # local, lazy
        import torch  # local, lazy

        if self.pretrained_tag is not None:
            model = open_clip.create_model(
                self.arch, pretrained=self.pretrained_tag
            )
        else:
            model = open_clip.create_model(self.arch)
            if self.hf_repo is not None and self.hf_filename is not None:
                from huggingface_hub import hf_hub_download  # local, lazy

                ckpt_path = hf_hub_download(self.hf_repo, self.hf_filename)
                state = torch.load(ckpt_path, map_location="cpu")
                if isinstance(state, dict) and "state_dict" in state:
                    state = state["state_dict"]
                model.load_state_dict(state, strict=False)

        model.eval()
        for p in model.parameters():
            p.requires_grad_(False)
        model.to(self.device)
        self._model = model
        # Infer the image-embedding dim from the visual projection if possible.
        self.embed_dim = _infer_clip_dim(model, fallback=self.embed_dim)

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
        mean = torch.tensor(_CLIP_MEAN, device=self.device).view(1, 3, 1, 1)
        std = torch.tensor(_CLIP_STD, device=self.device).view(1, 3, 1, 1)
        return (t - mean) / std

    # -- Backbone API -------------------------------------------------------
    def _embed_batch(
        self, batch: np.ndarray, modality: "Modality | str | None"
    ) -> np.ndarray:
        import torch  # local, lazy

        assert self._model is not None
        x = self._preprocess(batch, modality)
        with torch.no_grad():
            feats = self._model.encode_image(x)  # (B, proj_dim)
        return feats.detach().cpu().numpy().astype(np.float32)


@register_backbone("openclip", aliases=["clip", "openclip-b32", "laion-clip"])
class OpenCLIPBackbone(_BaseCLIPBackbone):
    """LAION-2B OpenCLIP image encoder (default ViT-B/32, 512-d).

    Parameters
    ----------
    arch:
        open_clip architecture name. Default ``"ViT-B-32"``.
    pretrained_tag:
        open_clip pretrained tag. Default ``"laion2b_s34b_b79k"`` (the
        ``laion/CLIP-ViT-B-32-laion2B-s34B-b79K`` weights).
    image_size:
        Input size. Default ``224``.
    device:
        Torch device. Default ``"cpu"``.
    """

    def __init__(
        self,
        arch: str = "ViT-B-32",
        pretrained_tag: str = "laion2b_s34b_b79k",
        image_size: int = 224,
        device: str = "cpu",
    ) -> None:
        self.arch = arch
        self.pretrained_tag = pretrained_tag
        self.hf_repo = None
        self.hf_filename = None
        self.name = f"openclip:{arch}"
        # 512 for ViT-B/32; refined from the model after load.
        self.embed_dim = 512
        super().__init__(image_size=image_size, device=device)


@register_backbone("remoteclip", aliases=["remote-clip", "rclip"])
class RemoteCLIPBackbone(_BaseCLIPBackbone):
    """RemoteCLIP image encoder (ViT-L/14, 768-d) -- RS optical specialist.

    Weights: HF ``chendelong/RemoteCLIP``, file ``RemoteCLIP-ViT-L-14.pt``
    (loaded into an open_clip ``ViT-L-14`` model). SOTA remote-sensing image
    retrieval; best for optical-to-optical.

    Parameters
    ----------
    arch:
        open_clip architecture name. Default ``"ViT-L-14"``.
    hf_repo:
        HF repo holding the checkpoint. Default ``"chendelong/RemoteCLIP"``.
    hf_filename:
        Checkpoint filename. Default ``"RemoteCLIP-ViT-L-14.pt"``.
    image_size:
        Input size. Default ``224``.
    device:
        Torch device. Default ``"cpu"``.
    """

    def __init__(
        self,
        arch: str = "ViT-L-14",
        hf_repo: str = "chendelong/RemoteCLIP",
        hf_filename: str = "RemoteCLIP-ViT-L-14.pt",
        image_size: int = 224,
        device: str = "cpu",
    ) -> None:
        self.arch = arch
        self.pretrained_tag = None
        self.hf_repo = hf_repo
        self.hf_filename = hf_filename
        self.name = f"remoteclip:{arch}"
        # 768 for ViT-L/14; refined from the model after load.
        self.embed_dim = 768
        super().__init__(image_size=image_size, device=device)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _infer_clip_dim(model: object, fallback: int) -> int:
    """Best-effort read of the image-embedding width from an open_clip model."""
    visual = getattr(model, "visual", None)
    if visual is not None:
        out_dim = getattr(visual, "output_dim", None)
        if isinstance(out_dim, int) and out_dim > 0:
            return out_dim
        proj = getattr(visual, "proj", None)
        if proj is not None and hasattr(proj, "shape"):
            try:
                return int(proj.shape[1])
            except Exception:  # pragma: no cover - defensive
                pass
    return int(fallback)


def _to_unit_range(rgb: np.ndarray) -> np.ndarray:
    """Per-image joint-channel min-max scale into ``[0, 1]``."""
    rgb = np.asarray(rgb, dtype=np.float32)
    b = rgb.shape[0]
    flat = rgb.reshape(b, -1)
    lo = flat.min(axis=1).reshape(b, 1, 1, 1)
    hi = flat.max(axis=1).reshape(b, 1, 1, 1)
    scale = np.maximum(hi - lo, 1e-6)
    return ((rgb - lo) / scale).astype(np.float32)
