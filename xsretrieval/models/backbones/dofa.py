"""DOFA (Dynamic-One-For-All) wavelength-conditioned multimodal backbone.

DOFA is the **default multimodal backbone** for this challenge. A single ViT
with a *wavelength-conditioned dynamic patch-embedding hypernetwork* serves any
number/type of spectral bands: Sentinel-1 SAR, Sentinel-2 multispectral, RGB,
hyperspectral -- all flow through the *same* weights, so their CLS embeddings
land in **one shared 768-d space out of the box** (ViT-B/16). This is exactly
the "common representation space irrespective of sensor modality" PS-11 asks
for, and is the strongest default for cross-modal SAR<->optical retrieval.

* Weights: HF ``XShadow/DOFA`` (file ``DOFA_ViT_base_e100.pth``), commonly
  mirrored as ``earthflow/DOFA``; also exposed via torchgeo
  (``DOFABase16_Weights``) and PyTorch Hub
  (``torch.hub.load('zhu-xlab/DOFA', 'vit_base_dofa', pretrained=True)``).
* Paper: Xiong et al., *Neural Plasticity-Inspired Multimodal Foundation Model
  for Earth Observation* (arXiv:2403.15356).
* License: CC-BY-4.0 (model card). Confirm for production use.
* Embedding dim: **768** (ViT-B/16) / 1024 (ViT-L/16).
* Input: 224x224, per-channel standardized; you supply each channel's central
  **wavelength**. DOFA's convention (matched here): optical/MS wavelengths in
  **micrometres**, Sentinel-1 SAR encoded as the value **5.405** (the C-band
  ~5.405 cm radar wavelength used by the upstream repo verbatim).

Wavelength inputs
-----------------
The ``wave_list`` passed alongside an image must have one entry per channel, in
the same channel order. Defaults are provided as module constants:

* :data:`RGB_WAVELENGTHS` -- ``[0.665, 0.560, 0.490]`` (R, G, B; µm).
* :data:`SENTINEL2_WAVELENGTHS` -- the 13 Sentinel-2 band central wavelengths in
  µm (B1..B12 incl. B8A). A 12-band input (B10 dropped, CROMA-style) uses
  :data:`SENTINEL2_12_WAVELENGTHS`.
* :data:`SENTINEL1_WAVELENGTHS` -- ``[5.405, 5.405]`` (VV, VH).

Robust loading / defensive forward
----------------------------------
Because the exact upstream module/class layout varies across releases (torchgeo
vs. the reference repo vs. raw ``state_dict``), construction tries several
import paths and *documents* the expected ``forward_features(x, wave_list=...)``
interface. If the DOFA package/weights are not importable on this machine,
construction raises a clear error and
:func:`~xsretrieval.models.backbones.base.get_backbone` gracefully falls back to
the offline backbone. The forward path is defensive: it adapts to either a
pooled CLS return or patch-token return (mean-pooled), and to a few known
``forward`` signatures.

All heavy imports (``torch``, ``huggingface_hub``, ``torchgeo``) are lazy.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Sequence

import numpy as np

from .base import (
    Backbone,
    _is_multispectral,
    _is_rgb,
    _is_sar,
    register_backbone,
)

if TYPE_CHECKING:  # pragma: no cover - typing only
    import torch

    from xsretrieval.data.modalities import Modality

logger = logging.getLogger(__name__)

__all__ = [
    "DOFABackbone",
    "RGB_WAVELENGTHS",
    "SENTINEL1_WAVELENGTHS",
    "SENTINEL2_WAVELENGTHS",
    "SENTINEL2_12_WAVELENGTHS",
]

# -- Default central wavelengths (micrometres unless noted) -----------------
# RGB (red, green, blue).
RGB_WAVELENGTHS: list[float] = [0.665, 0.560, 0.490]

# Sentinel-2 MSI, all 13 bands in native order
# B1, B2, B3, B4, B5, B6, B7, B8, B8A, B9, B10, B11, B12 (central λ in µm).
SENTINEL2_WAVELENGTHS: list[float] = [
    0.443,  # B1  coastal aerosol
    0.490,  # B2  blue
    0.560,  # B3  green
    0.665,  # B4  red
    0.705,  # B5  red edge 1
    0.740,  # B6  red edge 2
    0.783,  # B7  red edge 3
    0.842,  # B8  NIR
    0.865,  # B8A narrow NIR
    0.945,  # B9  water vapour
    1.375,  # B10 cirrus
    1.610,  # B11 SWIR 1
    2.190,  # B12 SWIR 2
]

# Sentinel-2 with the B10 cirrus band dropped (12-band, CROMA-style ordering).
SENTINEL2_12_WAVELENGTHS: list[float] = [
    w for i, w in enumerate(SENTINEL2_WAVELENGTHS) if i != 10
]

# Sentinel-1 SAR (VV, VH). C-band radar wavelength ~5.405 cm; DOFA uses the
# scalar 5.405 for each SAR channel (match the upstream convention exactly).
SENTINEL1_WAVELENGTHS: list[float] = [5.405, 5.405]


def default_wavelengths(
    modality: "Modality | str | None", num_channels: int
) -> list[float]:
    """Return sensible default central wavelengths for *modality*/channel count.

    Falls back to a linear spread across the optical range when the modality is
    unknown, so DOFA always receives a ``wave_list`` of the right length.
    """
    if _is_sar(modality):
        return [SENTINEL1_WAVELENGTHS[0]] * num_channels
    if _is_multispectral(modality):
        if num_channels == 13:
            return list(SENTINEL2_WAVELENGTHS)
        if num_channels == 12:
            return list(SENTINEL2_12_WAVELENGTHS)
        # Subset/superset: take the first ``num_channels`` S2 wavelengths,
        # padding by repeating the last if necessary.
        base = list(SENTINEL2_WAVELENGTHS)
        if num_channels <= len(base):
            return base[:num_channels]
        return base + [base[-1]] * (num_channels - len(base))
    if _is_rgb(modality) or num_channels == 3:
        if num_channels == 3:
            return list(RGB_WAVELENGTHS)
        return (RGB_WAVELENGTHS * (num_channels // 3 + 1))[:num_channels]
    if num_channels == 2:  # assume SAR-like VV/VH
        return list(SENTINEL1_WAVELENGTHS)
    if num_channels == 1:
        return [0.560]
    # Unknown multi-band: spread across the visible-NIR range.
    return list(np.linspace(0.49, 2.19, num_channels).astype(float))


@register_backbone("dofa", aliases=["dofa-base", "dofa_vit_base", "oneforall"])
class DOFABackbone(Backbone):
    """DOFA ViT-B/16 wavelength-conditioned multimodal backbone (768-d).

    Parameters
    ----------
    model_name:
        Variant key. ``"vit_base_dofa"`` (768-d, default) or ``"vit_large_dofa"``
        (1024-d). Used for the torchgeo / torch.hub entry points.
    hf_repo:
        HF repo to fetch weights from. Default ``"XShadow/DOFA"`` (mirror
        ``"earthflow/DOFA"``).
    hf_filename:
        Weight filename in ``hf_repo``. Default ``"DOFA_ViT_base_e100.pth"``.
    image_size:
        Square input size. Default ``224``.
    wavelengths:
        Optional explicit ``{modality_name: [λ, ...]}`` overrides. When absent,
        :func:`default_wavelengths` is used per call.
    device:
        Torch device. Default ``"cpu"``.
    """

    def __init__(
        self,
        model_name: str = "vit_base_dofa",
        hf_repo: str = "XShadow/DOFA",
        hf_filename: str = "DOFA_ViT_base_e100.pth",
        image_size: int = 224,
        wavelengths: dict[str, Sequence[float]] | None = None,
        device: str = "cpu",
    ) -> None:
        self.name = f"dofa:{model_name}"
        self.model_name = model_name
        self.hf_repo = hf_repo
        self.hf_filename = hf_filename
        self.image_size = int(image_size)
        self.device = device
        self.wavelengths = {
            k: list(v) for k, v in (wavelengths or {}).items()
        }
        # Truly multimodal: supports all modalities natively.
        self.supported_modalities = set()
        self.embed_dim = 1024 if "large" in model_name else 768
        self._model = None
        self._build_model()

    # -- model construction -------------------------------------------------
    def _build_model(self) -> None:
        """Load DOFA, trying torchgeo -> reference repo -> torch.hub. Lazy.

        Raises on total failure so ``get_backbone`` can fall back.
        """
        import torch  # local, lazy

        model = None
        errors: list[str] = []

        # (1) torchgeo exposes DOFA with pretrained Weights enums.
        try:
            model = self._build_from_torchgeo(torch)
        except Exception as exc:  # pragma: no cover - env dependent
            errors.append(f"torchgeo: {type(exc).__name__}: {exc}")

        # (2) Reference implementation (``dofa`` / ``DOFA`` package), loading the
        #     HF checkpoint into a freshly constructed model.
        if model is None:
            try:
                model = self._build_from_reference(torch)
            except Exception as exc:  # pragma: no cover - env dependent
                errors.append(f"reference: {type(exc).__name__}: {exc}")

        # (3) PyTorch Hub entry point.
        if model is None:
            try:
                model = torch.hub.load(
                    "zhu-xlab/DOFA", self.model_name, pretrained=True
                )
            except Exception as exc:  # pragma: no cover - env dependent
                errors.append(f"torch.hub: {type(exc).__name__}: {exc}")

        if model is None:
            raise RuntimeError(
                "Could not load DOFA from torchgeo / reference repo / torch.hub. "
                "Install the DOFA package or torchgeo, or allow network access. "
                "Tried: " + " | ".join(errors)
            )

        model.eval()
        for p in model.parameters():
            p.requires_grad_(False)
        model.to(self.device)
        self._model = model

    def _build_from_torchgeo(self, torch) -> object:
        """Construct DOFA via torchgeo's model + Weights enum."""
        from torchgeo.models import dofa as tg_dofa  # type: ignore

        if "large" in self.model_name:
            weights = tg_dofa.DOFALarge16_Weights.DOFA_MAE  # type: ignore[attr-defined]
            model = tg_dofa.dofa_large_patch16_224(weights=weights)
        else:
            weights = tg_dofa.DOFABase16_Weights.DOFA_MAE  # type: ignore[attr-defined]
            model = tg_dofa.dofa_base_patch16_224(weights=weights)
        return model

    def _build_from_reference(self, torch) -> object:
        """Construct the reference DOFA model and load the HF checkpoint."""
        from huggingface_hub import hf_hub_download  # local, lazy

        # The reference package may expose a constructor under a few names.
        ctor = None
        import importlib

        for mod_name, attr in (
            ("dofa.models_dwv", "vit_base_patch16"),
            ("dofa", "vit_base_dofa"),
            ("DOFA.models_dwv", "vit_base_patch16"),
        ):
            try:
                mod = importlib.import_module(mod_name)
                ctor = getattr(mod, attr)
                break
            except Exception:
                continue
        if ctor is None:
            raise ImportError("reference DOFA constructor not importable")

        model = ctor()
        ckpt_path = hf_hub_download(self.hf_repo, self.hf_filename)
        state = torch.load(ckpt_path, map_location="cpu")
        if isinstance(state, dict) and "model" in state:
            state = state["model"]
        model.load_state_dict(state, strict=False)
        return model

    # -- wavelengths --------------------------------------------------------
    def _wave_list(
        self, modality: "Modality | str | None", num_channels: int
    ) -> list[float]:
        from .base import _modality_name

        key = _modality_name(modality)
        if key in self.wavelengths:
            wl = list(self.wavelengths[key])
            # Adjust length to the actual channel count.
            if len(wl) == num_channels:
                return wl
            if len(wl) > num_channels:
                return wl[:num_channels]
            return wl + [wl[-1]] * (num_channels - len(wl))
        return default_wavelengths(modality, num_channels)

    # -- preprocessing ------------------------------------------------------
    def _preprocess(self, batch: np.ndarray) -> "torch.Tensor":
        """Standardize per channel and resize to ``image_size``.

        DOFA expects per-channel standardized inputs. We z-score each channel
        per image (robust to unknown upstream scaling) then resize.
        """
        import torch  # local, lazy
        import torch.nn.functional as F  # local, lazy

        x = np.asarray(batch, dtype=np.float32)
        if not np.isfinite(x).all():
            x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
        # Per-image, per-channel z-score.
        mean = x.mean(axis=(2, 3), keepdims=True)
        std = x.std(axis=(2, 3), keepdims=True) + 1e-6
        x = (x - mean) / std
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
    def _forward_features(
        self, model, x: "torch.Tensor", wavelengths: list[float]
    ) -> "torch.Tensor":
        """Call DOFA defensively across known forward signatures.

        Expected reference interface:
        ``model.forward_features(x, wave_list=[λ_per_channel])`` returning either
        a pooled ``(B, D)`` CLS embedding or patch tokens ``(B, N, D)``.
        torchgeo's ``forward(x, wavelengths=...)`` is also handled.
        """
        import torch  # local, lazy

        wl_tensor = torch.tensor(
            wavelengths, dtype=torch.float32, device=self.device
        )

        candidates = []
        if hasattr(model, "forward_features"):
            ff = model.forward_features
            candidates += [
                lambda: ff(x, wave_list=wavelengths),
                lambda: ff(x, wavelengths=wl_tensor),
                lambda: ff(x, wl_tensor),
            ]
        candidates += [
            lambda: model(x, wave_list=wavelengths),
            lambda: model(x, wavelengths=wl_tensor),
            lambda: model(x, wl_tensor),
            lambda: model(x),
        ]

        last_err: Exception | None = None
        for call in candidates:
            try:
                out = call()
            except TypeError as exc:
                last_err = exc
                continue
            return out
        raise RuntimeError(
            f"DOFA forward failed for all known signatures: {last_err}"
        )

    # -- Backbone API -------------------------------------------------------
    def _embed_batch(
        self, batch: np.ndarray, modality: "Modality | str | None"
    ) -> np.ndarray:
        import torch  # local, lazy

        assert self._model is not None
        num_channels = int(batch.shape[1])
        wl = self._wave_list(modality, num_channels)
        x = self._preprocess(batch)
        with torch.no_grad():
            out = self._forward_features(self._model, x, wl)
        feats = _pool_output(out)
        return feats.detach().cpu().numpy().astype(np.float32)


def _pool_output(out: object) -> "torch.Tensor":
    """Reduce a DOFA forward output to a ``(B, D)`` tensor.

    Handles: a dict (prefer pooled/CLS keys), patch tokens ``(B, N, D)``
    (mean-pool over N), or an already-pooled ``(B, D)`` tensor.
    """
    import torch  # local, lazy

    if isinstance(out, dict):
        for key in ("pooled", "cls", "cls_token", "x_norm_clstoken", "logits"):
            if key in out:
                out = out[key]
                break
        else:
            out = next(iter(out.values()))
    if isinstance(out, (tuple, list)):
        out = out[0]
    t = out
    if not isinstance(t, torch.Tensor):  # pragma: no cover - defensive
        t = torch.as_tensor(t)
    if t.ndim == 3:  # (B, N, D) -> mean-pool tokens
        t = t.mean(dim=1)
    elif t.ndim > 3:
        t = t.flatten(1)
    return t
