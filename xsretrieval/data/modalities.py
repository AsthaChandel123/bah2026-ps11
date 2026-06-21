"""Modality definitions, normalization statistics and band helpers.

This module is the single source of truth for the ``Modality`` enum, the
``Sample`` container and per-modality normalization constants used everywhere in
``xsretrieval``.  It is the *only* place where the canonical channel layout of
each remote-sensing sensor is encoded, so downstream code (preprocessing,
augmentation, models) never hard-codes band orders.

Design notes / remote-sensing rationale
----------------------------------------
* **OPTICAL_RGB** -- true-colour 3-band imagery (Sentinel-2 B4/B3/B2, or any
  Google-Earth style RGB scene). Normalized with ImageNet statistics so that
  ImageNet/timm/DINOv2 backbones receive inputs in their expected range.
* **MULTISPECTRAL** -- the full Sentinel-2 13-band stack (L1C ordering
  ``B1,B2,B3,B4,B5,B6,B7,B8,B8A,B9,B10,B11,B12``). Per-band mean/std are
  *placeholders* derived from typical Sentinel-2 surface-reflectance ranges;
  they should be replaced by dataset-specific statistics when available.
* **SAR** -- Sentinel-1 dual-polarisation (VV, VH) in decibels. The constants
  encode the usual clipping window and the (mean, std) of the clipped dB values.
* **HYPERSPECTRAL** -- variable channel count (sensor dependent), hence
  ``MODALITY_CHANNELS`` maps it to ``0`` ("variable"). Normalization falls back
  to a generic per-band standardisation computed on the fly.
* **DEM** -- single-band elevation; standardised with a coarse global
  elevation prior.

The module performs **no heavy imports** at top level (only ``numpy``), so
``import xsretrieval.data.modalities`` is always cheap and dependency-free.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

import numpy as np

__all__ = [
    "Modality",
    "MODALITY_CHANNELS",
    "Sample",
    "RGB_MEAN",
    "RGB_STD",
    "S2_BANDS",
    "S2_RGB_BAND_INDICES",
    "MS_MEAN",
    "MS_STD",
    "SAR_DB_CLIP",
    "SAR_MEAN",
    "SAR_STD",
    "DEM_MEAN",
    "DEM_STD",
    "NORMALIZATION_STATS",
    "to_rgb",
    "get_normalization",
]


class Modality(str, Enum):
    """Sensor modality of a remote-sensing image.

    Subclasses ``str`` so that members can be used directly as dictionary keys,
    serialised to JSON, compared against raw strings (``m == "sar"``) and printed
    cleanly, while still behaving as a proper enumeration.
    """

    OPTICAL_RGB = "optical_rgb"
    MULTISPECTRAL = "multispectral"
    SAR = "sar"
    HYPERSPECTRAL = "hyperspectral"
    DEM = "dem"

    def __str__(self) -> str:  # pragma: no cover - cosmetic
        return self.value


#: Canonical channel count per modality. ``0`` means "variable / sensor
#: dependent" (hyperspectral sensors range from tens to hundreds of bands).
MODALITY_CHANNELS: dict[Modality, int] = {
    Modality.OPTICAL_RGB: 3,
    Modality.MULTISPECTRAL: 13,
    Modality.SAR: 2,
    Modality.HYPERSPECTRAL: 0,
    Modality.DEM: 1,
}


@dataclass
class Sample:
    """A single remote-sensing image plus its metadata.

    This is the **plain-numpy boundary object** exchanged between every module
    of ``xsretrieval`` (data, models, alignment, index, retrieval, eval). Keeping
    it framework-free (no torch tensors) is what allows the data layer to be
    imported and unit-tested without any deep-learning dependency installed.

    Attributes
    ----------
    id:
        Unique identifier for the sample (e.g. ``"sen12ms_roi1_p007_s1"``).
    image:
        Image array with shape ``(C, H, W)`` and dtype ``float32``. ``C`` must
        match :data:`MODALITY_CHANNELS` for the modality (except hyperspectral,
        which is variable).
    modality:
        Which sensor produced the image.
    label:
        Integer semantic class id, or ``-1`` if unknown/unlabeled. Used for
        relevance-by-class retrieval scoring.
    location_id:
        Identifier shared across modalities that observe the *same* geographic
        location. This is the supervision signal for cross-modal positive pairs
        (the same place seen by SAR and optical). ``None`` if not applicable.
    meta:
        Optional free-form metadata (season, sensor params, original band list,
        normalization stats actually applied, etc.).
    """

    id: str
    image: np.ndarray
    modality: Modality
    label: int = -1
    location_id: Optional[str] = None
    meta: Optional[dict] = field(default=None)

    def __post_init__(self) -> None:
        # Light, cheap validation: do not copy or cast the array here (callers
        # may pass large arrays); only sanity-check rank so bugs surface early.
        if not isinstance(self.image, np.ndarray):
            raise TypeError(
                f"Sample.image must be a numpy.ndarray, got {type(self.image)!r}"
            )
        if self.image.ndim != 3:
            raise ValueError(
                "Sample.image must have shape (C, H, W); got array with "
                f"ndim={self.image.ndim} and shape={self.image.shape}"
            )
        # Coerce the modality to the enum so callers may pass the raw string.
        if not isinstance(self.modality, Modality):
            self.modality = Modality(self.modality)

    @property
    def num_channels(self) -> int:
        """Number of channels of the stored image (``image.shape[0]``)."""
        return int(self.image.shape[0])

    @property
    def hw(self) -> tuple[int, int]:
        """Spatial size ``(H, W)`` of the stored image."""
        return int(self.image.shape[1]), int(self.image.shape[2])


# ---------------------------------------------------------------------------
# Normalization statistics (module constants).
# ---------------------------------------------------------------------------
# Optical RGB -- ImageNet statistics (inputs assumed already scaled to [0, 1]).
RGB_MEAN: np.ndarray = np.array([0.485, 0.456, 0.406], dtype=np.float32)
RGB_STD: np.ndarray = np.array([0.229, 0.224, 0.225], dtype=np.float32)

#: Sentinel-2 L1C band names in canonical order (13 bands).
S2_BANDS: tuple[str, ...] = (
    "B1",   # coastal aerosol (60 m)
    "B2",   # blue (10 m)
    "B3",   # green (10 m)
    "B4",   # red (10 m)
    "B5",   # red edge 1 (20 m)
    "B6",   # red edge 2 (20 m)
    "B7",   # red edge 3 (20 m)
    "B8",   # NIR (10 m)
    "B8A",  # narrow NIR (20 m)
    "B9",   # water vapour (60 m)
    "B10",  # cirrus (60 m)
    "B11",  # SWIR 1 (20 m)
    "B12",  # SWIR 2 (20 m)
)

#: Indices into the 13-band stack that yield a true-colour RGB image
#: (B4=red, B3=green, B2=blue).
S2_RGB_BAND_INDICES: tuple[int, int, int] = (3, 2, 1)

# Multispectral (Sentinel-2, 13 bands) per-band mean/std *placeholders*.
# Values are expressed in surface-reflectance-like units scaled to roughly
# [0, 1] (i.e. raw DN / 10000). They are intentionally conservative defaults;
# replace with dataset statistics for best results.
MS_MEAN: np.ndarray = np.array(
    [
        0.130, 0.110, 0.100, 0.095, 0.110, 0.160, 0.185,
        0.195, 0.205, 0.130, 0.040, 0.155, 0.105,
    ],
    dtype=np.float32,
)
MS_STD: np.ndarray = np.array(
    [
        0.045, 0.050, 0.055, 0.070, 0.070, 0.080, 0.090,
        0.095, 0.100, 0.060, 0.030, 0.090, 0.075,
    ],
    dtype=np.float32,
)

# SAR (Sentinel-1 VV/VH) dB clipping windows and post-clip standardisation.
# Typical land backscatter ranges: VV in roughly [-23, 0] dB, VH in [-28, -5] dB.
#: ``SAR_DB_CLIP[c]`` is the ``(min_db, max_db)`` clip window for channel ``c``
#: (channel 0 = VV, channel 1 = VH).
SAR_DB_CLIP: np.ndarray = np.array(
    [[-23.0, 0.0], [-28.0, -5.0]], dtype=np.float32
)
#: Mean/std of clipped VV/VH dB values (used to standardise to ~N(0, 1)).
SAR_MEAN: np.ndarray = np.array([-11.5, -18.0], dtype=np.float32)
SAR_STD: np.ndarray = np.array([5.0, 5.0], dtype=np.float32)

# DEM (Copernicus DEM style) -- coarse global elevation prior (metres).
DEM_MEAN: np.ndarray = np.array([500.0], dtype=np.float32)
DEM_STD: np.ndarray = np.array([700.0], dtype=np.float32)


#: Lookup table of ``(mean, std)`` numpy arrays for the modalities that have a
#: fixed channel count. Hyperspectral is variable and therefore standardised on
#: the fly (see :func:`get_normalization`).
NORMALIZATION_STATS: dict[Modality, tuple[np.ndarray, np.ndarray]] = {
    Modality.OPTICAL_RGB: (RGB_MEAN, RGB_STD),
    Modality.MULTISPECTRAL: (MS_MEAN, MS_STD),
    Modality.SAR: (SAR_MEAN, SAR_STD),
    Modality.DEM: (DEM_MEAN, DEM_STD),
}


def get_normalization(
    modality: Modality, num_channels: Optional[int] = None
) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(mean, std)`` arrays for standardising a modality.

    For fixed-channel modalities the module constants are returned. For
    hyperspectral (variable channel count) -- or any modality whose requested
    ``num_channels`` does not match the stored stats -- a neutral
    ``mean=0, std=1`` fallback of the requested length is returned, signalling
    that per-image standardisation should be applied downstream.

    Parameters
    ----------
    modality:
        The sensor modality.
    num_channels:
        Optional number of channels the caller will normalise. Used to size the
        fallback and to detect a mismatch with the stored statistics.

    Returns
    -------
    (mean, std):
        1-D float32 arrays broadcastable against a ``(C, H, W)`` image as
        ``(C, 1, 1)``.
    """
    stats = NORMALIZATION_STATS.get(modality)
    if stats is not None:
        mean, std = stats
        if num_channels is None or num_channels == mean.shape[0]:
            return mean, std
    # Variable / mismatched: neutral stats so the caller can standardise itself.
    n = num_channels if num_channels is not None else 1
    return np.zeros(n, dtype=np.float32), np.ones(n, dtype=np.float32)


def to_rgb(
    ms_image: np.ndarray,
    *,
    band_indices: tuple[int, int, int] = S2_RGB_BAND_INDICES,
    percentile_stretch: tuple[float, float] = (2.0, 98.0),
) -> np.ndarray:
    """Derive a true-colour RGB image from a Sentinel-2 multispectral stack.

    Selects the red/green/blue bands (B4/B3/B2 by default) and applies a robust
    percentile contrast stretch so the result is a pleasant, display-ready RGB
    image in ``[0, 1]``. This is the standard way to obtain a *free* optical-RGB
    modality from Sentinel-2 imagery (used by SEN12MS-style pipelines).

    Parameters
    ----------
    ms_image:
        Multispectral image, shape ``(C, H, W)`` with ``C >= max(band_indices)+1``.
        Any numeric dtype; raw DN, reflectance or already-normalised values are
        all accepted (the percentile stretch is scale-robust).
    band_indices:
        Indices of the (red, green, blue) bands within ``ms_image``.
    percentile_stretch:
        ``(low, high)`` percentiles used for the contrast stretch. Set to
        ``(0, 100)`` to disable stretching (pure min-max).

    Returns
    -------
    rgb:
        Float32 array of shape ``(3, H, W)`` in ``[0, 1]``.

    Raises
    ------
    ValueError
        If ``ms_image`` is not 3-D or does not contain the requested bands.
    """
    if ms_image.ndim != 3:
        raise ValueError(
            f"to_rgb expects a (C, H, W) array, got shape {ms_image.shape}"
        )
    c = ms_image.shape[0]
    if max(band_indices) >= c:
        raise ValueError(
            f"to_rgb needs at least {max(band_indices) + 1} bands to extract "
            f"RGB indices {band_indices}, but image has only {c} channels"
        )
    rgb = ms_image[list(band_indices)].astype(np.float32, copy=True)
    lo_p, hi_p = percentile_stretch
    out = np.empty_like(rgb)
    for i in range(3):
        band = rgb[i]
        lo = float(np.percentile(band, lo_p))
        hi = float(np.percentile(band, hi_p))
        if hi <= lo:
            # Degenerate band (constant); fall back to a flat mid-grey channel.
            out[i] = np.zeros_like(band)
        else:
            out[i] = np.clip((band - lo) / (hi - lo), 0.0, 1.0)
    return out
