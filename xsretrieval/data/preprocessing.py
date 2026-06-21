"""Per-modality preprocessing for cross-modal satellite image retrieval.

This module converts raw, heterogeneous sensor arrays into clean, fixed-size,
normalised ``(C, H, W)`` float32 tensors ready for a feature extractor. Each
modality is handled according to its physics:

* **SAR** (Sentinel-1 VV/VH): clip linear intensity, convert to decibels
  (``10*log10``) -- in which multiplicative speckle becomes additive and the
  dynamic range is compressed -- then standardise per polarisation. Optional
  refined-Lee despeckling is provided.
* **MULTISPECTRAL** (Sentinel-2, 13 bands): select / pad to the canonical band
  set, scale raw DN to reflectance, then standardise per band. Missing bands are
  handled gracefully (zero-filled and flagged).
* **OPTICAL_RGB**: scale to ``[0, 1]`` then ImageNet-standardise so pretrained
  backbones see their expected input distribution.

All resizing uses PIL (lazy import) with an optional ``cv2`` fast path; no torch
or rasterio is imported at module load time, so ``import
xsretrieval.data.preprocessing`` works in a bare numpy environment.
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from .modalities import (
    MODALITY_CHANNELS,
    SAR_DB_CLIP,
    SAR_MEAN,
    SAR_STD,
    Modality,
    get_normalization,
)

__all__ = [
    "preprocess",
    "resize_image",
    "despeckle_lee",
    "to_uint8_preview",
    "ensure_channels",
    "sar_to_db",
]

# Small constant to avoid log/division by zero in radiometric conversions.
_EPS = 1e-6


# ---------------------------------------------------------------------------
# Generic helpers
# ---------------------------------------------------------------------------
def ensure_channels(
    image: np.ndarray, target: int, *, pad_value: float = 0.0
) -> np.ndarray:
    """Coerce a ``(C, H, W)`` image to exactly ``target`` channels.

    Robustness helper for the very common real-world case where an input has the
    wrong number of bands (e.g. a 12-band Sentinel-2 product when 13 are
    expected, or an RGBA tile when 3 are expected). Extra channels are dropped;
    missing channels are appended as constant ``pad_value`` planes.

    Parameters
    ----------
    image:
        Array of shape ``(C, H, W)``.
    target:
        Desired number of channels. ``target <= 0`` is treated as "keep as is"
        (used for variable-channel modalities such as hyperspectral).
    pad_value:
        Value used to fill newly created channels.

    Returns
    -------
    np.ndarray
        Array of shape ``(target, H, W)`` (or unchanged if ``target <= 0``).
    """
    if image.ndim != 3:
        raise ValueError(f"ensure_channels expects (C, H, W), got {image.shape}")
    if target <= 0:
        return image
    c, h, w = image.shape
    if c == target:
        return image
    if c > target:
        return image[:target]
    pad = np.full((target - c, h, w), pad_value, dtype=image.dtype)
    return np.concatenate([image, pad], axis=0)


def resize_image(
    image: np.ndarray, size: int, *, interpolation: str = "bilinear"
) -> np.ndarray:
    """Resize a ``(C, H, W)`` image to ``(C, size, size)``.

    Works for an arbitrary number of channels by resizing each channel
    independently. Uses ``cv2`` when available (fast, handles many channels in
    one call), otherwise falls back to PIL band-by-band. Both are lazily
    imported so neither is a hard dependency.

    Parameters
    ----------
    image:
        Input array, shape ``(C, H, W)``, any numeric dtype.
    size:
        Target spatial size (square ``size x size``). ``size <= 0`` returns the
        input unchanged.
    interpolation:
        One of ``"nearest"``, ``"bilinear"``, ``"bicubic"``. Use ``"nearest"``
        for label/index maps to avoid introducing fractional classes.

    Returns
    -------
    np.ndarray
        Resized float32 array of shape ``(C, size, size)``.
    """
    if image.ndim != 3:
        raise ValueError(f"resize_image expects (C, H, W), got {image.shape}")
    c, h, w = image.shape
    if size <= 0 or (h == size and w == size):
        return image.astype(np.float32, copy=False)

    img = image.astype(np.float32, copy=False)

    # Fast path: OpenCV (optional).
    try:  # pragma: no cover - depends on optional dependency
        import cv2  # type: ignore

        cv2_interp = {
            "nearest": cv2.INTER_NEAREST,
            "bilinear": cv2.INTER_LINEAR,
            "bicubic": cv2.INTER_CUBIC,
        }[interpolation]
        # cv2 works on (H, W, C); it supports up to 4 channels at once, so do
        # it in band-chunks to be safe for MS/HS.
        hwc = np.transpose(img, (1, 2, 0))
        out_chunks = []
        for start in range(0, c, 4):
            chunk = hwc[:, :, start : start + 4]
            resized = cv2.resize(
                chunk, (size, size), interpolation=cv2_interp
            )
            if resized.ndim == 2:  # single-channel chunk collapses last dim
                resized = resized[:, :, None]
            out_chunks.append(resized)
        out = np.concatenate(out_chunks, axis=2)
        return np.ascontiguousarray(np.transpose(out, (2, 0, 1)), dtype=np.float32)
    except Exception:
        pass

    # Fallback: PIL, band by band. PIL needs per-band scaling to uint-ish range;
    # we resize in float via the 'F' mode to preserve dynamic range.
    from PIL import Image  # lazy

    pil_interp = {
        "nearest": Image.NEAREST,
        "bilinear": Image.BILINEAR,
        "bicubic": Image.BICUBIC,
    }[interpolation]
    out = np.empty((c, size, size), dtype=np.float32)
    for i in range(c):
        pil = Image.fromarray(img[i], mode="F")
        pil = pil.resize((size, size), resample=pil_interp)
        out[i] = np.asarray(pil, dtype=np.float32)
    return out


# ---------------------------------------------------------------------------
# SAR
# ---------------------------------------------------------------------------
def sar_to_db(
    sar: np.ndarray, *, assume_db: Optional[bool] = None
) -> np.ndarray:
    """Convert SAR backscatter to decibels (``10*log10``) if needed.

    SAR amplitude/intensity is best modelled in the log (dB) domain, where the
    multiplicative speckle becomes additive and the heavy-tailed distribution is
    compressed. This helper auto-detects whether the input is already in dB
    (heuristic: contains substantial negative values) and only converts linear
    inputs.

    Parameters
    ----------
    sar:
        SAR array of shape ``(C, H, W)`` (linear intensity or already dB).
    assume_db:
        Force the interpretation: ``True`` = input already dB (returned as-is),
        ``False`` = input linear (always converted). ``None`` = auto-detect.

    Returns
    -------
    np.ndarray
        dB-scaled float32 array, same shape.
    """
    arr = sar.astype(np.float32, copy=False)
    if assume_db is None:
        # Linear intensity is non-negative; a meaningful fraction of negatives
        # strongly implies the data is already in dB.
        assume_db = bool(np.mean(arr < -0.5) > 0.05)
    if assume_db:
        return arr
    return (10.0 * np.log10(np.maximum(arr, _EPS))).astype(np.float32)


def despeckle_lee(sar: np.ndarray, size: int = 7) -> np.ndarray:
    """Refined-Lee-style adaptive speckle filter (pure numpy).

    Implements a local-statistics Lee filter -- the workhorse of SAR speckle
    reduction. In homogeneous regions it averages strongly (suppressing speckle);
    near edges/point targets the adaptive weight ``k`` approaches 1, preserving
    structure. This is a faithful, dependency-free approximation of the refined
    Lee filter suitable for both preprocessing and as a training augmentation.

    The estimator is::

        out = mean + k * (x - mean)
        k   = max(0, (var_x - mean^2 * Cu^2)) / (var_x + eps)

    where ``Cu^2 = 1 / L`` is the noise variation coefficient for an ``L``-look
    image (``L`` is inferred from the local statistics).

    Parameters
    ----------
    sar:
        SAR array of shape ``(C, H, W)`` (operates per channel). dB or linear --
        the filter is applied directly to the provided values.
    size:
        Odd window side length (default 7). Even values are bumped to the next
        odd number.

    Returns
    -------
    np.ndarray
        Filtered float32 array, same shape as ``sar``.
    """
    if sar.ndim != 3:
        raise ValueError(f"despeckle_lee expects (C, H, W), got {sar.shape}")
    if size < 3:
        size = 3
    if size % 2 == 0:
        size += 1

    arr = sar.astype(np.float32, copy=False)
    c, h, w = arr.shape
    pad = size // 2
    out = np.empty_like(arr)

    for ch in range(c):
        band = arr[ch]
        padded = np.pad(band, pad, mode="reflect")
        # Box-filter mean and mean-of-squares via summed-area (integral image)
        # for O(1) per-pixel window statistics regardless of window size.
        local_mean = _box_mean(padded, size)
        local_sqmean = _box_mean(padded * padded, size)
        local_var = np.maximum(local_sqmean - local_mean * local_mean, 0.0)

        # Estimate the noise variation coefficient Cu^2 = 1/L from the ratio of
        # global std to mean (overall speckle level), clamped to a sane range.
        gmean = float(np.mean(np.abs(band))) + _EPS
        gstd = float(np.std(band))
        cu2 = float(np.clip((gstd / gmean) ** 2, 1e-3, 1.0))

        denom = local_var + _EPS
        k = (local_var - (local_mean * local_mean) * cu2) / denom
        k = np.clip(k, 0.0, 1.0)
        out[ch] = local_mean + k * (band - local_mean)

    return out.astype(np.float32)


def _box_mean(padded: np.ndarray, size: int) -> np.ndarray:
    """Mean over a ``size x size`` window using an integral image.

    ``padded`` must already be reflect-padded by ``size // 2`` on each side; the
    returned array has the original (unpadded) spatial shape.
    """
    # Integral image with a zero top/left border.
    integral = np.zeros(
        (padded.shape[0] + 1, padded.shape[1] + 1), dtype=np.float64
    )
    integral[1:, 1:] = np.cumsum(np.cumsum(padded, axis=0), axis=1)
    h = padded.shape[0] - (size - 1)
    w = padded.shape[1] - (size - 1)
    # Window sum via the four corners of the integral image.
    total = (
        integral[size : size + h, size : size + w]
        - integral[0:h, size : size + w]
        - integral[size : size + h, 0:w]
        + integral[0:h, 0:w]
    )
    return (total / float(size * size)).astype(np.float32)


def _preprocess_sar(
    image: np.ndarray,
    size: int,
    *,
    despeckle: bool,
    despeckle_size: int,
    assume_db: Optional[bool],
) -> np.ndarray:
    """SAR pipeline: dB-scale -> (optional despeckle) -> clip -> standardise."""
    n_target = MODALITY_CHANNELS[Modality.SAR]
    img = ensure_channels(image, n_target)
    db = sar_to_db(img, assume_db=assume_db)
    if despeckle:
        db = despeckle_lee(db, size=despeckle_size)
    db = resize_image(db, size, interpolation="bilinear")

    # Per-polarisation clip then standardise to ~N(0, 1).
    out = np.empty_like(db)
    for ch in range(db.shape[0]):
        clip_lo, clip_hi = SAR_DB_CLIP[min(ch, SAR_DB_CLIP.shape[0] - 1)]
        clipped = np.clip(db[ch], clip_lo, clip_hi)
        mean = SAR_MEAN[min(ch, SAR_MEAN.shape[0] - 1)]
        std = SAR_STD[min(ch, SAR_STD.shape[0] - 1)]
        out[ch] = (clipped - mean) / (std + _EPS)
    return out.astype(np.float32)


# ---------------------------------------------------------------------------
# Multispectral
# ---------------------------------------------------------------------------
def _preprocess_ms(
    image: np.ndarray,
    size: int,
    *,
    reflectance_scale: float,
) -> np.ndarray:
    """MS pipeline: select/pad to 13 bands -> reflectance -> standardise.

    Missing bands (input has < 13 channels) are zero-padded; the zero planes are
    standardised to roughly ``-mean/std`` which keeps them finite and lets the
    network learn to ignore them (consistent with band-dropout augmentation).
    """
    n_target = MODALITY_CHANNELS[Modality.MULTISPECTRAL]
    img = ensure_channels(image, n_target)
    img = resize_image(img, size, interpolation="bilinear")

    # Heuristic: if values look like raw DN (>> 1), scale to reflectance.
    if float(np.nanmax(img)) > 1.5:
        img = img / float(reflectance_scale)

    mean, std = get_normalization(Modality.MULTISPECTRAL, img.shape[0])
    out = (img - mean[:, None, None]) / (std[:, None, None] + _EPS)
    return np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)


# ---------------------------------------------------------------------------
# Optical RGB
# ---------------------------------------------------------------------------
def _preprocess_rgb(image: np.ndarray, size: int) -> np.ndarray:
    """RGB pipeline: to 3 channels -> [0, 1] -> ImageNet standardise."""
    n_target = MODALITY_CHANNELS[Modality.OPTICAL_RGB]
    img = ensure_channels(image, n_target)
    img = resize_image(img, size, interpolation="bilinear").astype(np.float32)

    # Scale to [0, 1]. uint8-style data (0..255) is divided; data already in
    # [0, 1] is left alone; arbitrary float ranges are min-max stretched.
    vmax = float(np.nanmax(img)) if img.size else 0.0
    if vmax > 1.5:
        img = img / (255.0 if vmax <= 255.0 + 1e-3 else vmax)
    img = np.clip(img, 0.0, 1.0)

    mean, std = get_normalization(Modality.OPTICAL_RGB, 3)
    out = (img - mean[:, None, None]) / (std[:, None, None] + _EPS)
    return out.astype(np.float32)


def _preprocess_generic(image: np.ndarray, size: int) -> np.ndarray:
    """Fallback for DEM / hyperspectral: resize + standardisation.

    For DEM the fixed elevation prior from :func:`get_normalization` is used;
    for variable-channel hyperspectral data each band is standardised by its own
    empirical mean/std, which is the safe choice without sensor-specific
    radiometric calibration.
    """
    img = resize_image(image, size, interpolation="bilinear").astype(np.float32)
    c = img.shape[0]

    if c == MODALITY_CHANNELS[Modality.DEM]:
        mean, std = get_normalization(Modality.DEM, c)
        out = (img - mean[:, None, None]) / (std[:, None, None] + _EPS)
    else:
        # Hyperspectral / unknown: per-band self-standardisation.
        per_band_mean = img.reshape(c, -1).mean(axis=1)
        per_band_std = img.reshape(c, -1).std(axis=1)
        out = (img - per_band_mean[:, None, None]) / (
            per_band_std[:, None, None] + _EPS
        )
    return np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)


def preprocess(
    image: np.ndarray,
    modality: Modality,
    size: int = 224,
    *,
    despeckle: bool = False,
    despeckle_size: int = 7,
    sar_assume_db: Optional[bool] = None,
    ms_reflectance_scale: float = 10000.0,
) -> np.ndarray:
    """Preprocess a raw sensor image into a normalised ``(C, H, W)`` tensor.

    Dispatches to the modality-specific pipeline and always returns a
    contiguous float32 array of shape ``(C, size, size)`` with ``C`` equal to the
    canonical channel count for the modality (variable for hyperspectral). The
    function is robust to wrong input channel counts and odd value ranges.

    Parameters
    ----------
    image:
        Raw image of shape ``(C, H, W)`` (or ``(H, W)`` for a single band, which
        is promoted to ``(1, H, W)``).
    modality:
        Sensor modality controlling the pipeline.
    size:
        Output spatial size. ``size <= 0`` keeps the native resolution.
    despeckle:
        SAR only -- apply :func:`despeckle_lee` before standardisation.
    despeckle_size:
        Window size for the despeckle filter.
    sar_assume_db:
        SAR only -- override dB auto-detection (see :func:`sar_to_db`).
    ms_reflectance_scale:
        MS only -- divisor applied to raw DN to obtain reflectance (Sentinel-2
        uses ``10000``).

    Returns
    -------
    np.ndarray
        Normalised float32 image, shape ``(C, size, size)``.

    Raises
    ------
    ValueError
        If ``image`` cannot be interpreted as a ``(C, H, W)`` array.
    """
    if image.ndim == 2:
        image = image[None, :, :]
    if image.ndim != 3:
        raise ValueError(
            f"preprocess expects (C, H, W) or (H, W), got shape {image.shape}"
        )
    modality = Modality(modality)

    if modality == Modality.SAR:
        return _preprocess_sar(
            image,
            size,
            despeckle=despeckle,
            despeckle_size=despeckle_size,
            assume_db=sar_assume_db,
        )
    if modality == Modality.MULTISPECTRAL:
        return _preprocess_ms(image, size, reflectance_scale=ms_reflectance_scale)
    if modality == Modality.OPTICAL_RGB:
        return _preprocess_rgb(image, size)
    # DEM and hyperspectral.
    return _preprocess_generic(image, size)


# ---------------------------------------------------------------------------
# Preview / visualisation
# ---------------------------------------------------------------------------
def to_uint8_preview(image: np.ndarray, modality: Modality) -> np.ndarray:
    """Render any modality as a display-ready ``uint8`` RGB preview.

    Produces an ``(H, W, 3)`` uint8 image suitable for the demo / report. Inputs
    may be either raw or already-preprocessed (the function applies a robust
    percentile stretch, so normalised tensors render correctly too).

    Mapping per modality:

    * RGB -> the three channels directly.
    * MS  -> false-colour from B4/B3/B2 (via percentile stretch).
    * SAR -> a 3-plane composite ``[VV, VH, VV/VH-ratio]`` (a common SAR
      visualisation that highlights polarimetric contrast).
    * DEM -> greyscale elevation replicated to RGB.
    * Hyperspectral -> first three (or repeated) bands.

    Parameters
    ----------
    image:
        Image of shape ``(C, H, W)``.
    modality:
        Sensor modality controlling the colour mapping.

    Returns
    -------
    np.ndarray
        ``(H, W, 3)`` uint8 array.
    """
    if image.ndim == 2:
        image = image[None, :, :]
    if image.ndim != 3:
        raise ValueError(f"to_uint8_preview expects (C, H, W), got {image.shape}")
    modality = Modality(modality)
    c = image.shape[0]
    img = image.astype(np.float32, copy=False)

    if modality == Modality.SAR and c >= 2:
        vv, vh = img[0], img[1]
        ratio = vv - vh  # difference in dB ~ log-ratio (robust composite plane)
        planes = np.stack([vv, vh, ratio], axis=0)
    elif modality == Modality.MULTISPECTRAL and c >= 4:
        planes = img[[3, 2, 1]]
    elif modality == Modality.DEM or c == 1:
        planes = np.repeat(img[:1], 3, axis=0)
    elif c >= 3:
        planes = img[:3]
    else:
        planes = np.repeat(img[:1], 3, axis=0)

    out = np.empty((planes.shape[1], planes.shape[2], 3), dtype=np.uint8)
    for i in range(3):
        band = planes[i]
        lo = float(np.percentile(band, 2.0))
        hi = float(np.percentile(band, 98.0))
        if hi <= lo:
            out[:, :, i] = 0
        else:
            stretched = np.clip((band - lo) / (hi - lo), 0.0, 1.0)
            out[:, :, i] = (stretched * 255.0 + 0.5).astype(np.uint8)
    return out
