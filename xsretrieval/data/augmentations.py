"""Per-modality, co-registration-preserving augmentations (pure numpy).

Augmentation is *critical* for cross-modal satellite retrieval because each
sensor has a very different noise / radiometric character, and because paired
modalities must stay pixel-aligned for the cross-modal positive signal to remain
valid. This module follows the golden rule from the training-recipe research:

    Apply the SAME geometric transform to all co-registered modalities of a
    pair (shared RNG seed for flip/rot/crop/rotation), but apply
    modality-specific photometric/radiometric augmentations INDEPENDENTLY.

Every augmentation is a **pure function of (image, rng)** -- deterministic given
a seed, and a no-op-safe when its probability does not fire. Callables operate on
``(C, H, W)`` float32 numpy arrays and return the same shape (geometry-only
exception: random-resized-crop changes ``H, W`` consistently). No torchvision is
imported; an optional lazy torch path is *not* required for any function here.

Public API
----------
* :func:`build_augmentation` -- factory returning a single modality-appropriate
  callable (photometric + geometric) for a given ``strength``.
* :class:`PairedAugmentor` -- augments a multi-modal tuple with a shared
  geometric seed (preserving co-registration) and independent photometrics.
* :func:`modality_dropout`, :func:`cross_modal_mixup` -- batch-level helpers that
  force reliance on the shared embedding space.
* Individual transforms (``sar_gamma_speckle``, ``ms_band_dropout``,
  ``geom_random_resized_crop`` ...) are exported for fine-grained control.
"""

from __future__ import annotations

from typing import Callable, Optional, Sequence

import numpy as np

from .modalities import MODALITY_CHANNELS, Modality
from .preprocessing import despeckle_lee

__all__ = [
    # types
    "Augmentation",
    "Compose",
    # strength config
    "STRENGTH_PRESETS",
    # SAR
    "sar_gamma_speckle",
    "sar_additive_speckle",
    "sar_db_jitter",
    "sar_pol_dropout",
    "sar_random_despeckle",
    # optical
    "optical_color_jitter",
    "optical_random_grayscale",
    "optical_gaussian_blur",
    "optical_random_erasing",
    # multispectral
    "ms_band_dropout",
    "ms_spectral_tube_mask",
    "ms_band_gain_bias",
    "ms_ndvi_jitter",
    # geometric (shared)
    "geom_hflip",
    "geom_vflip",
    "geom_rot90",
    "geom_small_rotation",
    "geom_random_resized_crop",
    "apply_geometric",
    # builders / helpers
    "build_augmentation",
    "PairedAugmentor",
    "modality_dropout",
    "cross_modal_mixup",
    "as_rng",
]

#: An augmentation is a callable ``(image, rng) -> image``.
Augmentation = Callable[[np.ndarray, np.random.Generator], np.ndarray]


# ---------------------------------------------------------------------------
# Strength presets -- probabilities/magnitudes per intensity level.
# ---------------------------------------------------------------------------
STRENGTH_PRESETS: dict[str, dict[str, float]] = {
    "none": {
        "p_geom": 0.0, "p_photo": 0.0, "magnitude": 0.0,
    },
    "light": {
        "p_geom": 0.5, "p_photo": 0.3, "magnitude": 0.5,
    },
    "medium": {
        "p_geom": 0.7, "p_photo": 0.5, "magnitude": 1.0,
    },
    "strong": {
        "p_geom": 0.9, "p_photo": 0.8, "magnitude": 1.5,
    },
}


def as_rng(seed: Optional[int | np.random.Generator]) -> np.random.Generator:
    """Coerce ``seed`` into a numpy ``Generator``.

    Accepts ``None`` (fresh entropy), an ``int`` seed (reproducible) or an
    existing ``Generator`` (passed through). Centralising this lets every
    transform share one deterministic stream when a paired augmentor seeds it.
    """
    if isinstance(seed, np.random.Generator):
        return seed
    return np.random.default_rng(seed)


class Compose:
    """Sequentially apply a list of augmentations sharing one RNG stream.

    Using a single ``Generator`` for the whole chain means a given seed exactly
    reproduces the full augmentation. ``Compose`` is itself an ``Augmentation``
    (callable with ``(image, rng)``) so it can be nested.
    """

    def __init__(self, transforms: Sequence[Augmentation]) -> None:
        self.transforms = list(transforms)

    def __call__(
        self, image: np.ndarray, rng: Optional[np.random.Generator] = None
    ) -> np.ndarray:
        rng = as_rng(rng)
        out = image
        for t in self.transforms:
            out = t(out, rng)
        return out

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        names = ", ".join(getattr(t, "__name__", type(t).__name__)
                          for t in self.transforms)
        return f"Compose([{names}])"


def _prob(rng: np.random.Generator, p: float) -> bool:
    """Return True with probability ``p`` (no-op-safe at the extremes)."""
    if p <= 0.0:
        return False
    if p >= 1.0:
        return True
    return bool(rng.random() < p)


# ===========================================================================
# SAR augmentations -- speckle is the defining characteristic.
# ===========================================================================
def sar_gamma_speckle(
    image: np.ndarray,
    rng: np.random.Generator,
    *,
    looks: Sequence[int] = (1, 2, 4, 8),
    p: float = 0.5,
    db_domain: bool = True,
) -> np.ndarray:
    """Apply realistic multiplicative Gamma speckle (the key SAR augmentation).

    SAR speckle is multiplicative and Gamma-distributed with unit mean and
    variance ``1/L`` where ``L`` is the number of looks. Sampling ``L`` from
    ``{1,2,4,8}`` spans the noisy-to-clean range, training speckle-invariant
    features. When the input is in the dB (log) domain the multiplicative field
    is applied additively in dB (``+10*log10(field)``), which is the physically
    correct homomorphic formulation.

    Parameters
    ----------
    image:
        SAR image ``(C, H, W)`` (dB or linear -- see ``db_domain``).
    rng:
        Random generator (shared stream).
    looks:
        Candidate equivalent-number-of-looks values.
    p:
        Probability of applying the augmentation.
    db_domain:
        If ``True`` the image is assumed to be in dB and speckle is added in the
        log domain; otherwise multiplied directly.

    Returns
    -------
    np.ndarray
        Augmented float32 image (same shape).
    """
    if not _prob(rng, p):
        return image
    out = image.astype(np.float32, copy=True)
    L = int(looks[rng.integers(0, len(looks))])
    # Gamma(shape=L, scale=1/L) has mean 1, variance 1/L.
    field = rng.gamma(shape=L, scale=1.0 / L, size=out.shape).astype(np.float32)
    field = np.maximum(field, 1e-6)
    if db_domain:
        out = out + 10.0 * np.log10(field)
    else:
        out = out * field
    return out.astype(np.float32)


def sar_additive_speckle(
    image: np.ndarray,
    rng: np.random.Generator,
    *,
    sigma_db: float = 1.0,
    p: float = 0.3,
) -> np.ndarray:
    """Add small zero-mean Gaussian noise (residual/thermal noise in dB).

    Complements the multiplicative Gamma speckle by modelling additive sensor
    noise and minor calibration jitter in the dB domain.
    """
    if not _prob(rng, p):
        return image
    out = image.astype(np.float32, copy=True)
    noise = rng.normal(0.0, sigma_db, size=out.shape).astype(np.float32)
    return out + noise


def sar_db_jitter(
    image: np.ndarray,
    rng: np.random.Generator,
    *,
    offset_db: float = 1.5,
    gain: float = 0.05,
    p: float = 0.5,
) -> np.ndarray:
    """Per-channel radiometric (dB) offset + gain jitter.

    Mimics calibration differences between acquisitions/sensors. Each
    polarisation channel gets an independent additive dB offset and a small
    multiplicative gain.
    """
    if not _prob(rng, p):
        return image
    out = image.astype(np.float32, copy=True)
    c = out.shape[0]
    off = rng.uniform(-offset_db, offset_db, size=(c, 1, 1)).astype(np.float32)
    g = (1.0 + rng.uniform(-gain, gain, size=(c, 1, 1))).astype(np.float32)
    return out * g + off


def sar_pol_dropout(
    image: np.ndarray,
    rng: np.random.Generator,
    *,
    p: float = 0.2,
) -> np.ndarray:
    """Randomly zero one polarisation (VV-only / VH-only robustness).

    Gallery / query SAR may carry only one polarisation. Dropping VV or VH (but
    never both) teaches the encoder to cope with whichever channel is available.
    The dropped channel is replaced by a copy of the surviving one so downstream
    statistics stay sane (rather than an all-zero plane).
    """
    if image.shape[0] < 2 or not _prob(rng, p):
        return image
    out = image.astype(np.float32, copy=True)
    drop = int(rng.integers(0, 2))
    keep = 1 - drop
    out[drop] = out[keep]
    return out


def sar_random_despeckle(
    image: np.ndarray,
    rng: np.random.Generator,
    *,
    p: float = 0.2,
    size: int = 7,
) -> np.ndarray:
    """Randomly apply refined-Lee despeckling (despeckle augmentation).

    Showing the model both speckled and smoothed versions of the same scene
    encourages speckle-invariant features. Uses the numpy Lee filter from
    :mod:`xsretrieval.data.preprocessing`.
    """
    if not _prob(rng, p):
        return image
    return despeckle_lee(image, size=size)


# ===========================================================================
# Optical RGB augmentations.
# ===========================================================================
def optical_color_jitter(
    image: np.ndarray,
    rng: np.random.Generator,
    *,
    brightness: float = 0.3,
    contrast: float = 0.3,
    saturation: float = 0.2,
    p: float = 0.5,
) -> np.ndarray:
    """Brightness / contrast / saturation jitter (operates on any C, RGB-tuned).

    Models illumination, sun-angle and seasonal colour variation. Implemented in
    pure numpy: brightness scales values, contrast scales around the per-image
    mean, saturation interpolates between the image and its greyscale version.
    Inputs are assumed roughly in ``[0, 1]`` (optical preprocessing scale) but
    the op is value-range agnostic.
    """
    if not _prob(rng, p):
        return image
    out = image.astype(np.float32, copy=True)
    # Brightness.
    if brightness > 0:
        out = out * (1.0 + rng.uniform(-brightness, brightness))
    # Contrast around the global mean.
    if contrast > 0:
        mean = float(out.mean())
        out = (out - mean) * (1.0 + rng.uniform(-contrast, contrast)) + mean
    # Saturation (towards luminance).
    if saturation > 0 and out.shape[0] >= 3:
        gray = out[:3].mean(axis=0, keepdims=True)
        factor = 1.0 + rng.uniform(-saturation, saturation)
        out[:3] = gray + (out[:3] - gray) * factor
    return out.astype(np.float32)


def optical_random_grayscale(
    image: np.ndarray,
    rng: np.random.Generator,
    *,
    p: float = 0.2,
) -> np.ndarray:
    """Convert to greyscale (forces structure over colour; nudges toward SAR).

    In cross-modal training, occasionally desaturating the optical image pushes
    the encoder to rely on structural cues that are also present in SAR, helping
    bridge the modality gap.
    """
    if image.shape[0] < 3 or not _prob(rng, p):
        return image
    out = image.astype(np.float32, copy=True)
    gray = (0.299 * out[0] + 0.587 * out[1] + 0.114 * out[2])
    out[0] = out[1] = out[2] = gray
    return out


def optical_gaussian_blur(
    image: np.ndarray,
    rng: np.random.Generator,
    *,
    sigma_max: float = 1.2,
    p: float = 0.3,
) -> np.ndarray:
    """Separable Gaussian blur (resolution / focus robustness).

    Implemented as a separable convolution in pure numpy (no scipy). The kernel
    radius adapts to the sampled ``sigma``.
    """
    if not _prob(rng, p):
        return image
    sigma = float(rng.uniform(0.3, sigma_max))
    if sigma <= 1e-3:
        return image
    radius = max(1, int(round(3.0 * sigma)))
    xs = np.arange(-radius, radius + 1, dtype=np.float32)
    kernel = np.exp(-(xs**2) / (2.0 * sigma * sigma))
    kernel /= kernel.sum()
    out = image.astype(np.float32, copy=True)
    for ch in range(out.shape[0]):
        out[ch] = _sep_conv2d(out[ch], kernel)
    return out


def _sep_conv2d(plane: np.ndarray, kernel_1d: np.ndarray) -> np.ndarray:
    """Apply a 1-D kernel along rows then columns with reflect padding."""
    r = len(kernel_1d) // 2
    padded = np.pad(plane, ((0, 0), (r, r)), mode="reflect")
    # Horizontal pass via sliding windows.
    tmp = np.zeros_like(plane)
    for k, w in enumerate(kernel_1d):
        tmp += w * padded[:, k : k + plane.shape[1]]
    padded = np.pad(tmp, ((r, r), (0, 0)), mode="reflect")
    out = np.zeros_like(plane)
    for k, w in enumerate(kernel_1d):
        out += w * padded[k : k + plane.shape[0], :]
    return out


def optical_random_erasing(
    image: np.ndarray,
    rng: np.random.Generator,
    *,
    area: tuple[float, float] = (0.02, 0.2),
    p: float = 0.3,
) -> np.ndarray:
    """Cutout / random-erasing -- mask a rectangular patch (occlusion / clouds).

    Robustness to occlusion and cloud cover, a frequent real-world degradation
    in optical satellite imagery. The erased region is filled with the per-image
    mean (a neutral value after standardisation).
    """
    if not _prob(rng, p):
        return image
    out = image.astype(np.float32, copy=True)
    _, h, w = out.shape
    target = rng.uniform(*area) * h * w
    aspect = rng.uniform(0.3, 3.3)
    eh = int(round(np.sqrt(target * aspect)))
    ew = int(round(np.sqrt(target / aspect)))
    eh = min(eh, h)
    ew = min(ew, w)
    if eh < 1 or ew < 1:
        return out
    top = int(rng.integers(0, h - eh + 1))
    left = int(rng.integers(0, w - ew + 1))
    fill = out.reshape(out.shape[0], -1).mean(axis=1)[:, None, None]
    out[:, top : top + eh, left : left + ew] = fill
    return out


# ===========================================================================
# Multispectral augmentations.
# ===========================================================================
def ms_band_dropout(
    image: np.ndarray,
    rng: np.random.Generator,
    *,
    max_frac: float = 0.3,
    p: float = 0.4,
) -> np.ndarray:
    """Randomly zero a subset of bands (missing-band / cross-sensor robustness).

    Gallery MS images may have different band availability; randomly dropping up
    to ``max_frac`` of the bands trains the encoder to be robust to which bands
    are present. Dropped bands are set to zero (post-standardisation neutral).
    """
    if not _prob(rng, p):
        return image
    out = image.astype(np.float32, copy=True)
    c = out.shape[0]
    n_drop = int(rng.integers(0, max(1, int(c * max_frac)) + 1))
    if n_drop == 0:
        return out
    idx = rng.choice(c, size=n_drop, replace=False)
    out[idx] = 0.0
    return out


def ms_spectral_tube_mask(
    image: np.ndarray,
    rng: np.random.Generator,
    *,
    frac_range: tuple[float, float] = (0.1, 0.3),
    p: float = 0.3,
) -> np.ndarray:
    """Mask a CONTIGUOUS block of bands ("spectral tube masking").

    Unlike random band dropout, masking a contiguous spectral window (e.g. the
    red-edge group) better mimics whole-region sensor gaps and is the MS analogue
    of MAE-style tube masking. Masks ``frac_range`` of the spectrum.
    """
    if not _prob(rng, p):
        return image
    out = image.astype(np.float32, copy=True)
    c = out.shape[0]
    frac = float(rng.uniform(*frac_range))
    width = max(1, int(round(frac * c)))
    width = min(width, c)
    start = int(rng.integers(0, c - width + 1))
    out[start : start + width] = 0.0
    return out


def ms_band_gain_bias(
    image: np.ndarray,
    rng: np.random.Generator,
    *,
    gain: float = 0.1,
    bias: float = 0.05,
    p: float = 0.5,
) -> np.ndarray:
    """Independent per-band multiplicative gain + additive bias (spectral jitter).

    Models radiometric variation across acquisitions and sensors -- the spectral
    counterpart of optical colour jitter, applied independently to every band so
    the spectral signature is perturbed but not destroyed.
    """
    if not _prob(rng, p):
        return image
    out = image.astype(np.float32, copy=True)
    c = out.shape[0]
    g = (1.0 + rng.uniform(-gain, gain, size=(c, 1, 1))).astype(np.float32)
    b = rng.uniform(-bias, bias, size=(c, 1, 1)).astype(np.float32)
    return out * g + b


def ms_ndvi_jitter(
    image: np.ndarray,
    rng: np.random.Generator,
    *,
    strength: float = 0.1,
    red_index: int = 3,
    nir_index: int = 7,
    p: float = 0.3,
) -> np.ndarray:
    """Perturb the red/NIR pair to jitter vegetation indices (NDVI prior).

    NDVI ``=(NIR-Red)/(NIR+Red)`` is the dominant vegetation cue and a key bridge
    between MS and optical semantics. Slightly and jointly perturbing the red and
    NIR bands injects this domain prior as an augmentation without adding a
    derived channel (keeping the canonical 13-band layout intact).
    """
    if not _prob(rng, p):
        return image
    c = image.shape[0]
    if red_index >= c or nir_index >= c:
        return image
    out = image.astype(np.float32, copy=True)
    out[nir_index] *= 1.0 + rng.uniform(-strength, strength)
    out[red_index] *= 1.0 + rng.uniform(-strength, strength)
    return out


# ===========================================================================
# Geometric augmentations -- SHARED across paired modalities (seed-coupled).
# ===========================================================================
def geom_hflip(
    image: np.ndarray, rng: np.random.Generator, *, p: float = 0.5
) -> np.ndarray:
    """Horizontal flip. RS scenes have no canonical orientation, so this is safe."""
    if not _prob(rng, p):
        return image
    return np.ascontiguousarray(image[:, :, ::-1])


def geom_vflip(
    image: np.ndarray, rng: np.random.Generator, *, p: float = 0.5
) -> np.ndarray:
    """Vertical flip (safe for orientation-agnostic remote-sensing imagery)."""
    if not _prob(rng, p):
        return image
    return np.ascontiguousarray(image[:, ::-1, :])


def geom_rot90(
    image: np.ndarray, rng: np.random.Generator, *, p: float = 0.75
) -> np.ndarray:
    """Random 0/90/180/270 degree rotation (8x effective data with flips)."""
    if not _prob(rng, p):
        return image
    k = int(rng.integers(1, 4))  # 1..3 quarter turns (0 handled by no-op prob)
    return np.ascontiguousarray(np.rot90(image, k=k, axes=(1, 2)))


def geom_small_rotation(
    image: np.ndarray,
    rng: np.random.Generator,
    *,
    max_deg: float = 12.0,
    p: float = 0.4,
) -> np.ndarray:
    """Small arbitrary-angle rotation about the centre (reflect-filled).

    Pure-numpy bilinear rotation by an angle in ``[-max_deg, max_deg]``. Out-of-
    frame pixels are filled by reflecting the nearest valid coordinates so no
    hard border is introduced.
    """
    if not _prob(rng, p):
        return image
    angle = float(rng.uniform(-max_deg, max_deg))
    if abs(angle) < 1e-3:
        return image
    return _rotate_bilinear(image, angle)


def _rotate_bilinear(image: np.ndarray, angle_deg: float) -> np.ndarray:
    """Rotate ``(C, H, W)`` by ``angle_deg`` with bilinear sampling + reflect."""
    c, h, w = image.shape
    theta = np.deg2rad(angle_deg)
    cos_t, sin_t = np.cos(theta), np.sin(theta)
    cy, cx = (h - 1) / 2.0, (w - 1) / 2.0

    ys, xs = np.meshgrid(
        np.arange(h, dtype=np.float32), np.arange(w, dtype=np.float32),
        indexing="ij",
    )
    yc, xc = ys - cy, xs - cx
    # Inverse map (sample source coords for each destination pixel).
    src_x = cos_t * xc + sin_t * yc + cx
    src_y = -sin_t * xc + cos_t * yc + cy

    # Reflect coordinates into range, then bilinear interpolate.
    src_x = _reflect_coord(src_x, w)
    src_y = _reflect_coord(src_y, h)
    x0 = np.floor(src_x).astype(np.int64)
    y0 = np.floor(src_y).astype(np.int64)
    x1 = np.clip(x0 + 1, 0, w - 1)
    y1 = np.clip(y0 + 1, 0, h - 1)
    x0 = np.clip(x0, 0, w - 1)
    y0 = np.clip(y0, 0, h - 1)
    wx = src_x - x0
    wy = src_y - y0

    out = np.empty_like(image)
    for ch in range(c):
        plane = image[ch]
        top = plane[y0, x0] * (1 - wx) + plane[y0, x1] * wx
        bot = plane[y1, x0] * (1 - wx) + plane[y1, x1] * wx
        out[ch] = top * (1 - wy) + bot * wy
    return out.astype(np.float32)


def _reflect_coord(coord: np.ndarray, size: int) -> np.ndarray:
    """Reflect floating coordinates into ``[0, size-1]`` (mirror padding)."""
    if size == 1:
        return np.zeros_like(coord)
    period = 2 * (size - 1)
    c = np.mod(coord, period)
    c = np.where(c >= size, period - c, c)
    return c


def geom_random_resized_crop(
    image: np.ndarray,
    rng: np.random.Generator,
    *,
    size: Optional[int] = None,
    scale: tuple[float, float] = (0.6, 1.0),
    ratio: tuple[float, float] = (0.85, 1.18),
    p: float = 0.5,
) -> np.ndarray:
    """Random-resized-crop (scale/aspect jitter), output resized back to square.

    Crops a random sub-region (``scale`` fraction of the area, aspect within
    ``ratio``) and resizes it to ``size x size`` (default: original ``H``). This
    is the standard contrastive-learning spatial augmentation; when applied with
    a shared seed across paired modalities the crop window is identical, keeping
    them co-registered.
    """
    if not _prob(rng, p):
        return image
    from .preprocessing import resize_image  # lazy to avoid cycle at import

    c, h, w = image.shape
    out_size = size if size is not None else h
    area = h * w
    for _ in range(10):
        target_area = area * float(rng.uniform(*scale))
        log_ratio = (np.log(ratio[0]), np.log(ratio[1]))
        ar = float(np.exp(rng.uniform(*log_ratio)))
        cw = int(round(np.sqrt(target_area * ar)))
        ch_ = int(round(np.sqrt(target_area / ar)))
        if 0 < cw <= w and 0 < ch_ <= h:
            left = int(rng.integers(0, w - cw + 1))
            top = int(rng.integers(0, h - ch_ + 1))
            crop = image[:, top : top + ch_, left : left + cw]
            return resize_image(crop, out_size, interpolation="bilinear")
    # Fallback: centre crop to square then resize.
    s = min(h, w)
    top = (h - s) // 2
    left = (w - s) // 2
    crop = image[:, top : top + s, left : left + s]
    return resize_image(crop, out_size, interpolation="bilinear")


def apply_geometric(
    image: np.ndarray,
    rng: np.random.Generator,
    *,
    strength: str = "medium",
    rrc_size: Optional[int] = None,
    enable_rrc: bool = True,
) -> np.ndarray:
    """Apply the standard shared geometric chain with one RNG stream.

    The order (flip -> flip -> rot90 -> small rotation -> RRC) and the single
    ``rng`` guarantee that, given the same seed, two different modalities undergo
    the *identical* geometric transform -- the core requirement for preserving
    cross-modal co-registration.
    """
    cfg = STRENGTH_PRESETS.get(strength, STRENGTH_PRESETS["medium"])
    pg = cfg["p_geom"]
    mag = cfg["magnitude"]
    out = geom_hflip(image, rng, p=pg)
    out = geom_vflip(out, rng, p=pg)
    out = geom_rot90(out, rng, p=pg)
    out = geom_small_rotation(out, rng, max_deg=12.0 * mag, p=0.4 * pg)
    if enable_rrc:
        out = geom_random_resized_crop(
            out, rng, size=rrc_size, scale=(1.0 - 0.4 * mag, 1.0), p=0.5 * pg
        )
    return out


# ===========================================================================
# Builders.
# ===========================================================================
def _build_photometric(modality: Modality, strength: str) -> list[Augmentation]:
    """Return the list of modality-specific photometric/radiometric transforms."""
    cfg = STRENGTH_PRESETS.get(strength, STRENGTH_PRESETS["medium"])
    pp = cfg["p_photo"]
    mag = cfg["magnitude"]
    modality = Modality(modality)

    if modality == Modality.SAR:
        return [
            lambda x, r: sar_gamma_speckle(x, r, p=pp),
            lambda x, r: sar_additive_speckle(x, r, sigma_db=1.0 * mag, p=0.6 * pp),
            lambda x, r: sar_db_jitter(x, r, offset_db=1.5 * mag, p=pp),
            lambda x, r: sar_pol_dropout(x, r, p=0.4 * pp),
            lambda x, r: sar_random_despeckle(x, r, p=0.3 * pp),
        ]
    if modality == Modality.OPTICAL_RGB:
        return [
            lambda x, r: optical_color_jitter(
                x, r, brightness=0.3 * mag, contrast=0.3 * mag,
                saturation=0.2 * mag, p=pp,
            ),
            lambda x, r: optical_random_grayscale(x, r, p=0.3 * pp),
            lambda x, r: optical_gaussian_blur(x, r, sigma_max=1.2 * mag, p=0.5 * pp),
            lambda x, r: optical_random_erasing(x, r, p=0.5 * pp),
        ]
    if modality == Modality.MULTISPECTRAL:
        return [
            lambda x, r: ms_band_dropout(x, r, p=0.6 * pp),
            lambda x, r: ms_spectral_tube_mask(x, r, p=0.5 * pp),
            lambda x, r: ms_band_gain_bias(
                x, r, gain=0.1 * mag, bias=0.05 * mag, p=pp,
            ),
            lambda x, r: ms_ndvi_jitter(x, r, strength=0.1 * mag, p=0.4 * pp),
        ]
    # DEM / hyperspectral: only a mild generic gain/bias jitter is meaningful.
    return [
        lambda x, r: ms_band_gain_bias(x, r, gain=0.05 * mag, bias=0.03 * mag, p=pp),
    ]


def build_augmentation(
    modality: Modality,
    strength: str = "medium",
    *,
    include_geometric: bool = True,
    rrc_size: Optional[int] = None,
) -> Augmentation:
    """Build a single ``(image, rng) -> image`` augmentation for a modality.

    Composes the modality's photometric/radiometric transforms followed by the
    shared geometric chain (so a standalone call still augments geometry). For
    *paired* training prefer :class:`PairedAugmentor`, which couples the geometry
    across modalities via a shared seed.

    Parameters
    ----------
    modality:
        Sensor modality.
    strength:
        One of ``"none"``, ``"light"``, ``"medium"``, ``"strong"``.
    include_geometric:
        Append the geometric chain. Set ``False`` when the geometry is applied
        externally (e.g. by :class:`PairedAugmentor`).
    rrc_size:
        Output size for random-resized-crop (defaults to the input height).

    Returns
    -------
    Augmentation
        A deterministic-given-seed callable.
    """
    transforms = _build_photometric(modality, strength)
    if include_geometric:
        transforms.append(
            lambda x, r: apply_geometric(
                x, r, strength=strength, rrc_size=rrc_size
            )
        )
    return Compose(transforms)


class PairedAugmentor:
    """Augment a multi-modal tuple, sharing the geometric seed across modalities.

    This is the workhorse for contrastive cross-modal training. Given a tuple of
    co-registered images (one per modality), it:

    1. draws a single geometric seed and applies the **identical** geometric
       transform to every modality (preserving pixel correspondence), and
    2. applies each modality's photometric/radiometric augmentations with an
       **independent** stream (so SAR speckle, optical colour jitter and MS band
       dropout differ, as they should).

    Example
    -------
    >>> aug = PairedAugmentor(
    ...     [Modality.OPTICAL_RGB, Modality.SAR], strength="medium"
    ... )
    >>> opt2, sar2 = aug((opt, sar), seed=123)
    """

    def __init__(
        self,
        modalities: Sequence[Modality],
        strength: str = "medium",
        *,
        rrc_size: Optional[int] = None,
        enable_rrc: bool = True,
    ) -> None:
        self.modalities = [Modality(m) for m in modalities]
        self.strength = strength
        self.rrc_size = rrc_size
        self.enable_rrc = enable_rrc
        # Pre-build the per-modality photometric chains (geometry handled jointly).
        self._photo = [
            Compose(_build_photometric(m, strength)) for m in self.modalities
        ]

    def __call__(
        self,
        images: Sequence[np.ndarray],
        seed: Optional[int | np.random.Generator] = None,
    ) -> tuple[np.ndarray, ...]:
        """Augment a tuple of images (one per configured modality).

        Parameters
        ----------
        images:
            Sequence of ``(C, H, W)`` arrays matching ``self.modalities`` in
            order and length.
        seed:
            Base seed / generator. The geometric seed is derived from it so the
            whole operation is reproducible.

        Returns
        -------
        tuple of np.ndarray
            Augmented images in the same order.
        """
        if len(images) != len(self.modalities):
            raise ValueError(
                f"PairedAugmentor configured for {len(self.modalities)} "
                f"modalities but received {len(images)} images"
            )
        base = as_rng(seed)
        # One integer drives the shared geometry; independent ints drive photo.
        geom_seed = int(base.integers(0, 2**31 - 1))
        photo_seeds = [int(base.integers(0, 2**31 - 1)) for _ in images]

        out: list[np.ndarray] = []
        for img, photo, pseed in zip(images, self._photo, photo_seeds):
            # Photometric first (independent), then shared geometry.
            x = photo(img, as_rng(pseed))
            x = apply_geometric(
                x,
                as_rng(geom_seed),
                strength=self.strength,
                rrc_size=self.rrc_size,
                enable_rrc=self.enable_rrc,
            )
            out.append(x)
        return tuple(out)


# ===========================================================================
# Batch-level cross-modal helpers.
# ===========================================================================
def modality_dropout(
    images: Sequence[Optional[np.ndarray]],
    rng: np.random.Generator,
    *,
    p: float = 0.3,
    min_keep: int = 1,
) -> list[Optional[np.ndarray]]:
    """Randomly drop whole modalities from a multi-modal sample.

    Forces the model to produce consistent embeddings from any subset of
    modalities (modality-agnostic representation) and to handle missing
    modalities at query time. At least ``min_keep`` modalities always survive.
    Dropped entries are returned as ``None``.

    Parameters
    ----------
    images:
        Sequence of per-modality images (``None`` already-missing entries are
        preserved as missing).
    rng:
        Random generator.
    p:
        Per-modality drop probability.
    min_keep:
        Minimum number of present modalities to retain.

    Returns
    -------
    list
        New list with some entries possibly set to ``None``.
    """
    present = [i for i, im in enumerate(images) if im is not None]
    out: list[Optional[np.ndarray]] = list(images)
    if len(present) <= min_keep:
        return out
    # Decide drops, but never go below min_keep present.
    droppable = present.copy()
    rng.shuffle(droppable)
    keep_needed = min_keep
    survivors = set(present)
    for idx in droppable:
        if len(survivors) <= keep_needed:
            break
        if _prob(rng, p):
            survivors.discard(idx)
            out[idx] = None
    return out


def cross_modal_mixup(
    image_a: np.ndarray,
    image_b: np.ndarray,
    rng: np.random.Generator,
    *,
    alpha: float = 0.2,
) -> tuple[np.ndarray, float]:
    """Convex-combine two (same-shape) modality images -> populate inter-modal manifold.

    Cross-modal mixup smooths the shared embedding space by interpolating between
    modalities (or between samples), as in Multimodal-Mixup-Contrastive. Requires
    matching shapes (e.g. two single-band or two RGB images, or two views resized
    alike); raises otherwise so callers handle channel mismatches explicitly.

    Parameters
    ----------
    image_a, image_b:
        Same-shape ``(C, H, W)`` arrays.
    rng:
        Random generator.
    alpha:
        Beta-distribution parameter; the mixing weight ``lam ~ Beta(alpha, alpha)``.

    Returns
    -------
    (mixed, lam):
        The blended image ``lam*a + (1-lam)*b`` and the weight ``lam`` (so the
        caller can mix the corresponding labels/targets accordingly).
    """
    if image_a.shape != image_b.shape:
        raise ValueError(
            "cross_modal_mixup requires identical shapes, got "
            f"{image_a.shape} vs {image_b.shape}"
        )
    lam = float(rng.beta(alpha, alpha)) if alpha > 0 else 0.5
    mixed = (lam * image_a + (1.0 - lam) * image_b).astype(np.float32)
    return mixed, lam
